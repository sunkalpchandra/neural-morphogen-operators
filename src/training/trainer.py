"""Training loop shared by the NMO and every baseline.

The task
--------
**Masked spatial reconstruction.** At each step a set of locations is hidden
from the encoder; the model must predict expression there from the surrounding
field. Held-out *evaluation* regions (the ``val`` / ``test`` blocks) are hidden
for the entire run and never contribute to any gradient.

During training we additionally re-mask a random subset of the visible training
locations each step. This is what forces the operator to learn to *propagate*
information rather than to memorise an interpolation of its own inputs: if the
model could always see every training location it would never need dynamics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..evaluation.metrics import evaluate_prediction
from ..losses.objectives import LossWeights, NMOLoss, masked_mse, pearson_loss
from ..utils.common import ExperimentLogger, count_parameters, save_checkpoint, set_seed
from .dataset import SpatialSection


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    epochs: int = 400
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup: int = 20
    scheduler: str = "cosine"           # {'cosine', 'none'}
    eval_every: int = 20
    patience: int = 120                 # early stopping on val Pearson
    # fraction of *visible training* locations additionally hidden each step
    train_mask_frac: float = 0.30
    mask_kind: str = "block"
    seed: int = 0
    amp: bool = False
    log_every: int = 20


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        section: SpatialSection,
        cfg: TrainConfig,
        loss_weights: Optional[LossWeights] = None,
        logger: Optional[ExperimentLogger] = None,
        device: torch.device = torch.device("cpu"),
        is_nmo: bool = True,
    ):
        self.model = model.to(device)
        self.sec = section.to(device)
        self.cfg = cfg
        self.device = device
        self.is_nmo = is_nmo
        self.logger = logger
        self.criterion = NMOLoss(loss_weights) if is_nmo else None

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.sched = self._make_scheduler()

        # Permanent visibility masks. val/test blocks are never visible.
        self.train_visible = self.sec.mask("train")
        self.val_mask = self.sec.mask("val")
        self.test_mask = self.sec.mask("test")
        self.rng = np.random.default_rng(cfg.seed)

        self.history: List[Dict] = []
        self.best = {"score": -np.inf, "epoch": -1, "state": None}

    def _make_scheduler(self):
        if self.cfg.scheduler != "cosine":
            return None

        def fn(step: int) -> float:
            if step < self.cfg.warmup:
                return (step + 1) / max(self.cfg.warmup, 1)
            p = (step - self.cfg.warmup) / max(self.cfg.epochs - self.cfg.warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.opt, fn)

    # -- masking ------------------------------------------------------------ #

    def _step_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(encoder_visible, loss_evaluated_on) for one training step."""
        vis = self.train_visible.clone()
        idx = torch.nonzero(vis > 0).squeeze(1)
        n_hide = int(self.cfg.train_mask_frac * idx.numel())
        if n_hide > 0:
            sel = torch.from_numpy(
                self.rng.choice(idx.cpu().numpy(), n_hide, replace=False)
            ).to(vis.device)
            vis[sel] = 0.0
            hidden = torch.zeros_like(vis)
            hidden[sel] = 1.0
        else:
            hidden = vis.clone()
        # Score on the re-masked locations (the genuine prediction task) plus a
        # little weight on visible ones to keep the read-out calibrated.
        eval_mask = torch.clamp(hidden + 0.25 * vis, 0, 1)
        return vis, eval_mask

    # -- one epoch ---------------------------------------------------------- #

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        vis, eval_mask = self._step_masks()
        s = self.sec

        out = self.model(
            s.coords, s.expr * vis.view(-1, 1),
            query_coords=s.coords, edge_index=s.edge_index, point_mask=vis,
        )

        if self.is_nmo:
            terms = self.criterion(out, s.expr, self.model.operator, eval_mask, vis)
            loss = terms["total"]
        else:
            l_mse = masked_mse(out["pred"], s.expr, eval_mask)
            l_p = pearson_loss(out["pred"], s.expr, eval_mask)
            loss = l_mse + 0.5 * l_p
            terms = {"mse": l_mse.detach(), "pearson_loss": l_p.detach(), "total": loss}

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
        return {k: float(v) for k, v in terms.items()}

    # -- evaluation --------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, visible: torch.Tensor, query: Optional[torch.Tensor] = None) -> np.ndarray:
        """Predict expression at ``query`` seeing only ``visible`` locations."""
        self.model.eval()
        s = self.sec
        q = s.coords if query is None else query
        out = self.model(
            s.coords, s.expr * visible.view(-1, 1),
            query_coords=q, edge_index=s.edge_index, point_mask=visible,
        )
        return s.denormalise(out["pred"]).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, split: str = "val", compute_ssim: bool = True) -> Dict[str, float]:
        """Score on a held-out spatial block, with only training data visible."""
        s = self.sec
        target_mask = self.val_mask if split == "val" else self.test_mask
        sel = target_mask.bool().cpu().numpy()
        if sel.sum() == 0:
            return {"pearson_mean": float("nan")}

        pred = self.predict(self.train_visible)
        true = s.numpy_expr(denorm=True)
        coords = s.coords.cpu().numpy()
        return evaluate_prediction(
            pred[sel], true[sel], coords[sel],
            gene_names=s.gene_names, compute_ssim=compute_ssim,
        )

    # -- driver ------------------------------------------------------------- #

    def fit(self, checkpoint_path: Optional[Path] = None) -> Dict:
        cfg = self.cfg
        set_seed(cfg.seed)
        n_params = count_parameters(self.model)
        if self.logger:
            self.logger.info(
                f"training {type(self.model).__name__} ({n_params:,} params) "
                f"on {self.sec.name} for {cfg.epochs} epochs"
            )
        t0 = time.time()
        bad = 0

        for ep in range(cfg.epochs):
            terms = self.train_epoch()

            if self.logger and (ep % cfg.log_every == 0 or ep == cfg.epochs - 1):
                self.logger.log_metrics(ep, "train", **terms)

            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1:
                m = self.evaluate("val", compute_ssim=False)
                score = m.get("pearson_mean", float("nan"))
                if self.logger:
                    self.logger.log_metrics(ep, "val", **m)
                    self.logger.info(
                        f"  ep {ep+1:>4}/{cfg.epochs}  loss {terms['total']:.4f}  "
                        f"val r {score:.4f}  rmse {m.get('rmse', float('nan')):.4f}"
                    )
                self.history.append({"epoch": ep, **terms, **{f"val_{k}": v for k, v in m.items()}})

                if np.isfinite(score) and score > self.best["score"]:
                    self.best = {
                        "score": float(score), "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                    }
                    bad = 0
                else:
                    bad += cfg.eval_every
                    if bad >= cfg.patience:
                        if self.logger:
                            self.logger.info(f"  early stop at epoch {ep+1} (best {self.best['score']:.4f})")
                        break

        if self.best["state"] is not None:
            self.model.load_state_dict(self.best["state"])

        wall = time.time() - t0
        test = self.evaluate("test", compute_ssim=True)
        result = {
            "model": type(self.model).__name__,
            "section": self.sec.name,
            "n_params": n_params,
            "best_epoch": self.best["epoch"],
            "best_val_pearson": self.best["score"],
            "train_seconds": round(wall, 1),
            "test": test,
        }
        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, self.model, self.opt, self.best["epoch"], result)
        if self.logger:
            self.logger.info(
                f"  done in {wall:.0f}s | test r={test.get('pearson_mean', float('nan')):.4f} "
                f"rmse={test.get('rmse', float('nan')):.4f} ssim={test.get('ssim_mean', float('nan')):.4f}"
            )
        return result

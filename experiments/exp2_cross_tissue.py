"""Experiment 2 -- cross-tissue / cross-species transfer.

Train on adult mouse brain (Visium), evaluate **zero-shot** on human breast
carcinoma (Visium): a different species, a different organ, and a completely
different spatial architecture (laminar/nuclear brain vs. tumour nests and
stroma).

Comparisons reported
--------------------
1. ``zero_shot``   -- operator fitted on mouse, applied to human unchanged.
2. ``oracle``      -- the same architecture fitted directly on human breast.
                      This is the in-domain ceiling, not a competitor.
3. ``mean``        -- predict each gene's training-set mean. The floor.
4. Baselines under exactly the same zero-shot protocol.

We also report ``decoder_only`` fine-tuning (encoder + operator frozen, decoder
re-fitted on the target). If the operator is genuinely transferable, refitting
only the read-out should recover a large fraction of the oracle gap while the
learned dynamics stay fixed -- that is the concrete, falsifiable claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import evaluate_prediction
from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import align_sections, load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed


def _build(model_type: str, cfg: Config, n_genes: int):
    if model_type == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    return build_baseline(model_type, n_genes=n_genes, hidden=128,
                          latent=cfg.model.get("latent_channels", 32))


@torch.no_grad()
def _predict_all(model, sec, visible: torch.Tensor) -> np.ndarray:
    model.eval()
    out = model(sec.coords, sec.expr * visible.view(-1, 1),
                query_coords=sec.coords, edge_index=sec.edge_index, point_mask=visible)
    return sec.denormalise(out["pred"]).cpu().numpy()


def evaluate_zero_shot(model, target_sec, visible_frac: float = 0.5, seed: int = 0,
                       protocol: str = "block") -> Dict:
    """Condition the model on part of the target tissue; score it on the rest.

    A zero-shot spatial model still needs *some* observations to condition on --
    it predicts a field, not a constant -- so the question is which part to
    reveal.

    ``protocol='block'`` reveals the target's own training blocks and scores its
    held-out blocks, which is the split every other experiment in this paper
    uses and is the only setting in which a zero-shot number is comparable with
    the in-domain oracle.

    ``protocol='random'`` reveals a random half. This was the original choice and
    it is retained for reproducing earlier numbers, but it is not comparable to
    the oracle: on the Xenium target the median distance from a scored location
    to the nearest visible one is 19.5 um under the random split against 116.6 um
    under blocks (``experiments/exp12_split_geometry.py``), so it permits exactly
    the neighbour leakage that motivated the block protocol in the first place.
    """
    n = target_sec.n_obs
    if protocol == "block" and hasattr(target_sec, "split"):
        vis_np = (target_sec.split == "train").astype(np.float32)
        held = target_sec.split == "test"
        if vis_np.sum() < 1 or held.sum() < 1:      # no usable split; fall back
            protocol = "random"
    if protocol != "block" or not hasattr(target_sec, "split"):
        rng = np.random.default_rng(seed)
        vis_np = np.zeros(n, dtype=np.float32)
        vis_np[rng.choice(n, int(visible_frac * n), replace=False)] = 1.0
        held = vis_np == 0
    visible = torch.from_numpy(vis_np).to(target_sec.coords.device)

    pred = _predict_all(model, target_sec, visible)
    true = target_sec.numpy_expr(denorm=True)
    coords = target_sec.coords.cpu().numpy()
    return evaluate_prediction(pred[held], true[held], coords[held],
                               gene_names=target_sec.gene_names)


def mean_predictor(target_sec, visible_frac: float = 0.5, seed: int = 0,
                   protocol: str = "block") -> Dict:
    """Training-mean floor, scored under the same protocol as everything else."""
    n = target_sec.n_obs
    if protocol == "block" and hasattr(target_sec, "split"):
        vis = target_sec.split == "train"
        if vis.sum() < 1 or (target_sec.split == "test").sum() < 1:
            protocol = "random"
    if protocol != "block" or not hasattr(target_sec, "split"):
        rng = np.random.default_rng(seed)
        vis = np.zeros(n, dtype=bool)
        vis[rng.choice(n, int(visible_frac * n), replace=False)] = True
    true = target_sec.numpy_expr(denorm=True)
    pred = np.repeat(true[vis].mean(0, keepdims=True), (~vis).sum(), axis=0)
    coords = target_sec.coords.cpu().numpy()
    return evaluate_prediction(pred, true[~vis], coords[~vis])


def finetune_decoder(model, target_sec, cfg: Config, epochs: int, seed: int, logger) -> Dict:
    """Freeze encoder + operator; refit only the read-out head on the target.

    Not every baseline has a separable read-out. The exact GP, for instance, is
    three kernel hyperparameters and no decoder at all, and freezing by name
    leaves nothing trainable -- which used to reach ``loss.backward()`` and fail
    with an opaque autograd error. Such a model is reported as not supporting
    the setting rather than as having scored badly in it.
    """
    trainable = [n for n, _ in model.named_parameters()
                 if n.startswith("decoder") or n.startswith("dec")]
    if not trainable:
        for p_ in model.parameters():
            p_.requires_grad = True
        return {"unsupported": True,
                "reason": "no separable read-out head to refit"}
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("decoder") or name.startswith("dec")
    tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": epochs})
    tr = Trainer(model, target_sec, tcfg, LossWeights(**cfg.loss.to_dict()),
                 logger, target_sec.coords.device, is_nmo=hasattr(model, "operator"))
    r = tr.fit()
    for p in model.parameters():
        p.requires_grad = True
    return r["test"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--source", default="visium_mouse_brain")
    p.add_argument("--target", default="visium_human_breast")
    p.add_argument("--models", nargs="+", default=["nmo", "gnn", "stagate", "gp"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--finetune-epochs", type=int, default=150)
    p.add_argument("--out-dir", default="results/exp2")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{a.source}__to__{a.target}.json"
    results: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["model"], r["seed"], r["setting"]) for r in results}

    src_full = load_section(Path(cfg.data.processed_dir) / f"{a.source}.h5ad", device=device)
    tgt_full = load_section(Path(cfg.data.processed_dir) / f"{a.target}.h5ad", device=device)
    src, tgt, shared = align_sections(src_full, tgt_full)
    print(f"shared vocabulary: {len(shared)} genes")

    for seed in a.seeds:
        if ("mean", seed, "floor") not in done:
            m = mean_predictor(tgt, seed=seed)
            results.append({"model": "mean", "display": "Training-mean predictor",
                            "seed": seed, "setting": "floor", "n_shared_genes": len(shared), **m})

        for model_type in a.models:
            set_seed(seed)
            key = (model_type, seed)
            logger = ExperimentLogger(out / "runs" / f"{model_type}__seed{seed}", cfg.to_dict())

            if (*key, "zero_shot") in done and (*key, "oracle") in done:
                print(f"[skip] {model_type} seed {seed}")
                continue

            # --- fit on the source tissue --------------------------------- #
            model = _build(model_type, cfg, src.n_genes)
            tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": a.epochs})
            tr = Trainer(model, src, tcfg, LossWeights(**cfg.loss.to_dict()),
                         logger, device, is_nmo=(model_type == "nmo"))
            src_res = tr.fit()
            results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                            "seed": seed, "setting": "source_in_domain",
                            "n_shared_genes": len(shared), **src_res["test"]})

            # --- zero-shot on the target tissue --------------------------- #
            zs = evaluate_zero_shot(model, tgt, seed=seed)
            results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                            "seed": seed, "setting": "zero_shot", "n_shared_genes": len(shared), **zs})
            print(f"[zero-shot] {model_type} seed {seed}: r={zs['pearson_mean']:.4f}", flush=True)

            # --- decoder-only fine-tuning --------------------------------- #
            if a.finetune_epochs > 0:
                ft = finetune_decoder(model, tgt, cfg, a.finetune_epochs, seed, logger)
                results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                                "seed": seed, "setting": "decoder_finetune",
                                "n_shared_genes": len(shared), **ft})

            # --- in-domain oracle ----------------------------------------- #
            set_seed(seed)
            oracle = _build(model_type, cfg, tgt.n_genes)
            tr_o = Trainer(oracle, tgt, tcfg, LossWeights(**cfg.loss.to_dict()),
                           logger, device, is_nmo=(model_type == "nmo"))
            o_res = tr_o.fit()
            results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                            "seed": seed, "setting": "oracle", "n_shared_genes": len(shared),
                            **o_res["test"]})

            out_path.write_text(json.dumps(results, indent=2, default=float))

    out_path.write_text(json.dumps(results, indent=2, default=float))
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n", df.groupby(["setting", "model"])[["pearson_mean", "rmse", "ssim_mean"]].agg(["mean", "std"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

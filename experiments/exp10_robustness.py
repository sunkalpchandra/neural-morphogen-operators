"""Experiment 10 -- robustness to the perturbations real data actually exhibits.

Spatial transcriptomics is noisy, sparsely sampled and acquired at whatever
resolution the assay provides. A method that only works at the exact density and
signal-to-noise of one benchmark is of limited use, so each axis below perturbs
the *input* and re-measures held-out reconstruction, with every model exposed to
identical corruptions and identical splits.

Axes
----
``noise``        multiplicative log-normal noise on the observed expression,
                 which is the standard model for count-depth variation, at
                 several magnitudes.
``dropout``      random removal of observed locations, simulating tissue damage,
                 failed spots and partial capture.
``density``      uniform subsampling of the section, which changes the sampling
                 density without changing the anatomy -- the axis on which a
                 continuous operator ought to degrade most gracefully.
``knn``          the neighborhood size used to build the graph, which every
                 graph-based model depends on and which is usually tuned.

Models are re-trained under each corruption rather than merely evaluated on it,
since the realistic deployment is training on the data one has.

    python experiments/exp10_robustness.py --axes noise dropout density knn
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.layers import knn_graph
from src.models.nmo import build_nmo
from src.training.dataset import SpatialSection, load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

AXES = {
    "noise":   [0.0, 0.25, 0.5, 1.0],        # sigma of log-normal multiplicative noise
    "dropout": [0.0, 0.2, 0.4, 0.6],         # fraction of observed locations removed
    "density": [1.0, 0.5, 0.25, 0.125],      # fraction of the section retained
    "knn":     [4, 8, 16, 32],               # graph neighborhood size
}


def corrupt(sec: SpatialSection, axis: str, level: float, seed: int) -> SpatialSection:
    """Return a corrupted copy of the section. Splits are preserved so the
    held-out blocks are identical across corruption levels."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if axis == "noise" and level > 0:
        # multiplicative log-normal on the standardized values; applied to the
        # inputs only, never to the evaluation targets
        eps = torch.randn(sec.expr.shape, generator=g).to(sec.expr.device) * level
        sec = SpatialSection(**{**sec.__dict__, "expr": sec.expr * torch.exp(eps)})
    elif axis == "dropout" and level > 0:
        keep = torch.rand(sec.n_obs, generator=g).to(sec.coords.device) > level
        keep[torch.from_numpy((sec.split != "train")).to(keep.device)] = True  # keep eval blocks
        idx = torch.nonzero(keep).squeeze(1)
        sec = _subset(sec, idx)
    elif axis == "density" and level < 1.0:
        n = max(int(level * sec.n_obs), 200)
        sec = subsample_section(sec, n, seed=seed)
    elif axis == "knn":
        sec = SpatialSection(**{**sec.__dict__,
                                "edge_index": knn_graph(sec.coords, int(level))})
    return sec


def _subset(sec: SpatialSection, idx: torch.Tensor) -> SpatialSection:
    coords = sec.coords[idx]
    return SpatialSection(
        name=sec.name, coords=coords, expr=sec.expr[idx],
        gene_names=sec.gene_names, split=sec.split[idx.cpu().numpy()],
        edge_index=knn_graph(coords, 8), gene_mean=sec.gene_mean, gene_std=sec.gene_std,
        coord_scale_um=sec.coord_scale_um, meta=sec.meta)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--models", nargs="+", default=["nmo", "stagate", "gnn", "gp_multiscale"])
    p.add_argument("--axes", nargs="+", default=list(AXES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-locations", type=int, default=0,
                   help="cap the section before corrupting it. The density axis "
                        "needs a base small enough to be affordable and large "
                        "enough that the sparsest level still leaves an "
                        "estimable held-out set")
    p.add_argument("--out-dir", default="results/exp10")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"robustness_shard{a.shard}.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["axis"], r["level"], r["model"], r["seed"]) for r in rows}
    for f in out.glob("robustness_shard*.json"):
        try:
            for r in json.loads(f.read_text()):
                done.add((r["axis"], r["level"], r["model"], r["seed"]))
        except Exception:
            pass

    jobs = [(ax, lv, m, sd) for ax in a.axes for lv in AXES[ax]
            for m in a.models for sd in a.seeds]
    jobs = [j for i, j in enumerate(jobs) if i % a.n_shards == a.shard]
    todo = [j for j in jobs if j not in done]
    print(f"shard {a.shard}: {len(todo)}/{len(jobs)} jobs")

    for k, (axis, level, model_type, seed) in enumerate(todo, 1):
        try:
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad", device=device)
            if a.max_locations and sec.n_obs > a.max_locations:
                sec = subsample_section(sec, a.max_locations, seed=0).to(device)
            sec = corrupt(sec, axis, level, seed)
            model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes) if model_type == "nmo"
                     else build_baseline(model_type, n_genes=sec.n_genes, hidden=128,
                                         latent=cfg.model.get("latent_channels", 32)))
            lg = ExperimentLogger(out / "runs" / f"{axis}{level}__{model_type}__s{seed}",
                                  cfg.to_dict())
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                   "epochs": a.epochs}),
                         LossWeights(**cfg.loss.to_dict()), lg, device,
                         is_nmo=(model_type == "nmo"))
            r = tr.fit()
            rows.append(dict(axis=axis, level=float(level), model=model_type,
                             display=DISPLAY_NAMES.get(model_type, model_type), seed=seed,
                             n_obs=sec.n_obs, **r["test"]))
            print(f"[{k}/{len(todo)}] {axis}={level:<6} {model_type:<14} s{seed} "
                  f"r={r['test']['pearson_mean']:.4f}", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL {axis}={level} {model_type}: {exc}", flush=True)
            rows.append(dict(axis=axis, level=float(level), model=model_type, seed=seed,
                             failed=True, error=str(exc)[:140]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"shard {a.shard} complete: {len(rows)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

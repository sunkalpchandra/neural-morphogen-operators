"""Experiment 14 -- does the reduced-budget ordering survive convergence?

The multi-section benchmark trains every model for 200 epochs on subsampled
sections, because 190 runs on CPU is what was affordable. Absolute scores are
correspondingly low, and Section 6 argues that undertraining compresses the
differences between models. That argument cuts both ways: if no model has
converged, the observed ordering is a fact about the budget as much as about the
models, and nothing establishes that it is the converged ordering.

There is direct evidence that this matters. On ``visium_mouse_brain`` the
ordering *inverts* with budget: at 200 epochs NMO scores 0.218 against 0.225 for
the STAGATE-style baseline, and at 500 epochs on the full section it scores
0.247 against 0.234. One of those two orderings is an artifact.

This settles it on the section where the inversion occurs, which is the sharpest
available test: every model is trained to convergence under a shared early-
stopping rule on a held-out validation split, on the full section rather than a
subsample, with several seeds. If the converged ordering matches the benchmark,
the benchmark's budget is defensible; if it does not, that is a limitation of
the benchmark and is reported as one.

Cheapest configuration that answers the question: one section, the three
comparators that actually matter (the strongest graph baseline, a plain GCN, and
the non-spatial autoencoder that ties them), and a generous epoch ceiling that
early stopping is expected to reach well before.

    python experiments/exp14_converged.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

METRICS = ["pearson_mean", "spearman_mean", "rmse", "ssim_mean",
           "morans_i_abs_error", "gearys_c_abs_error"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--models", nargs="+",
                   default=["nmo", "stagate", "gnn", "autoencoder"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=2000,
                   help="ceiling; early stopping is expected to fire first")
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--out-dir", default="results/exp14")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "converged.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["model"], r["seed"]) for r in rows if not r.get("failed")}

    todo = [(m, s) for m in a.models for s in a.seeds if (m, s) not in done]
    print(f"{len(todo)} of {len(a.models) * len(a.seeds)} runs remaining "
          f"(ceiling {a.epochs} epochs, patience {a.patience})")

    for k, (model_type, seed) in enumerate(todo, 1):
        t0 = time.time()
        try:
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad",
                               device=device)          # full section, no subsample
            model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes)
                     if model_type == "nmo"
                     else build_baseline(model_type, n_genes=sec.n_genes, hidden=128,
                                         latent=cfg.model.get("latent_channels", 32)))
            lg = ExperimentLogger(out / "runs" / f"{model_type}__s{seed}", cfg.to_dict())
            tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                  "epochs": a.epochs, "patience": a.patience})
            tr = Trainer(model, sec, tcfg, LossWeights(**cfg.loss.to_dict()), lg,
                         device, is_nmo=(model_type == "nmo"))
            res = tr.fit()
            rows.append(dict(model=model_type, display=DISPLAY_NAMES.get(model_type, model_type),
                             seed=seed, section=a.section, n_obs=int(sec.n_obs),
                             n_params=res.get("n_params"), epochs_ceiling=a.epochs,
                             wall_s=round(time.time() - t0, 1),
                             **{m: float(res["test"][m]) for m in METRICS
                                if m in res["test"]}))
            print(f"[{k}/{len(todo)}] {model_type:<12} s{seed}  "
                  f"r={res['test']['pearson_mean']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL {model_type} s{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(model=model_type, seed=seed, failed=True,
                             error=str(exc)[:160]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))

    ok = [r for r in rows if not r.get("failed")]
    if ok:
        import pandas as pd
        df = pd.DataFrame(ok)
        print("\nconverged ordering (" + a.section + "):")
        print(df.groupby("display")["pearson_mean"].agg(["mean", "std", "count"])
              .sort_values("mean", ascending=False).round(4).to_string())
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

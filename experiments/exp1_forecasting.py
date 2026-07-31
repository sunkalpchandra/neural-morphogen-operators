"""Experiment 1 -- spatial forecasting / masked-region reconstruction.

Contiguous tissue blocks are hidden from every model; each must predict the full
expression profile at those unobserved coordinates from the surrounding field.
This is the main benchmark table.

    python experiments/exp1_forecasting.py --section visium_mouse_brain \
        --models nmo gnn spagcn stagate graph_transformer gp autoencoder \
        --seeds 0 1 2

Every model is trained by the identical loop with the identical masks, so the
comparison isolates the model. Results are appended to
``results/exp1/<section>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

ALL_MODELS = ["nmo", "gnn", "spagcn", "stagate", "graph_transformer", "gp", "autoencoder"]


def build(model_type: str, cfg: Config, n_genes: int):
    if model_type == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    kw = dict(hidden=128, latent=cfg.model.get("latent_channels", 32))
    return build_baseline(model_type, n_genes=n_genes, **kw)


def run(section_name: str, models: List[str], seeds: List[int], cfg: Config,
        out_dir: Path, epochs: int, max_locations: int | None) -> List[Dict]:
    device = get_device(cfg.experiment.get("device", "auto"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    out_path = out_dir / f"{section_name}.json"
    if out_path.exists():
        results = json.loads(out_path.read_text())

    done = {(r["model"], r["seed"]) for r in results}

    for model_type in models:
        for seed in seeds:
            if (model_type, seed) in done:
                print(f"[skip] {model_type} seed {seed} already done")
                continue
            set_seed(seed)
            sec = load_section(
                Path(cfg.data.processed_dir) / f"{section_name}.h5ad",
                device=device, knn_k=cfg.data.get("knn_k", 8),
                stats_from=cfg.data.get("stats_from", "train"),
            )
            if max_locations:
                sec = subsample_section(sec, max_locations, seed=seed).to(device)

            logger = ExperimentLogger(
                out_dir / "runs" / f"{section_name}__{model_type}__seed{seed}", cfg.to_dict()
            )
            model = build(model_type, cfg, sec.n_genes)
            tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": epochs})
            tr = Trainer(model, sec, tcfg, LossWeights(**cfg.loss.to_dict()),
                         logger, device, is_nmo=(model_type == "nmo"))
            t0 = time.time()
            r = tr.fit(checkpoint_path=out_dir / "runs" / f"{section_name}__{model_type}__seed{seed}" / "best.pt")
            rec = {
                "section": section_name,
                "model": model_type,
                "display": DISPLAY_NAMES.get(model_type, model_type),
                "seed": seed,
                "n_params": r["n_params"],
                "wall_s": round(time.time() - t0, 1),
                "best_epoch": r["best_epoch"],
                **{k: v for k, v in r["test"].items()},
            }
            results.append(rec)
            out_path.write_text(json.dumps(results, indent=2, default=float))
            print(f"[done] {model_type} seed {seed}: r={rec.get('pearson_mean'):.4f}", flush=True)
    return results


def aggregate(results: List[Dict]) -> "object":
    import pandas as pd

    df = pd.DataFrame(results)
    metrics = ["pearson_mean", "pearson_median", "spearman_mean", "rmse", "mae",
               "ssim_mean", "morans_i_abs_error", "morans_i_corr", "pearson_per_location"]
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(["section", "model", "display"])[metrics].agg(["mean", "std"])
    return g


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--max-locations", type=int, default=None)
    p.add_argument("--out-dir", default="results/exp1")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    res = run(a.section, a.models, a.seeds, cfg, Path(a.out_dir), a.epochs, a.max_locations)
    print("\n" + str(aggregate(res)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

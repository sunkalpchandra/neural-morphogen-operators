"""Experiment 3 -- cross-resolution transfer (Visium -> Xenium / MERFISH).

Train the operator on 55 um multi-cell Visium spots on a 100 um hexagonal
lattice; evaluate it on **single-cell** Xenium and MERFISH sampling of the same
organ. The sampling geometry, the cell-vs-spot semantics and the count
distributions all differ; only the underlying anatomy is shared.

This is the sharpest test of the continuous-operator claim. A discrete graph
model is defined on the lattice it was fitted to, so transferring it requires
re-instantiating the graph; a neural operator is defined on the *field* and can
in principle be evaluated at any sampling density.

Gene vocabulary
---------------
Rather than accept the small HVG intersection (134 genes for Xenium), we rebuild
the Visium object with the target panel force-included. All 248 Xenium panel
genes and the MERFISH panel are present in the raw Visium matrix, so the shared
vocabulary is the full panel -- a much fairer and more informative comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_raw
from src.data.preprocess import QCConfig, QC_PRESETS, preprocess
from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import align_sections, load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed
from exp2_cross_tissue import evaluate_zero_shot, mean_predictor  # noqa: E402


def build_visium_with_panel(panel: List[str], raw_dir: str, out_path: Path, seed: int = 0) -> Path:
    """Rebuild the Visium mouse-brain object force-including a target panel."""
    if out_path.exists():
        return out_path
    a = load_raw("visium_mouse_brain", raw_dir)
    a = preprocess(a, "visium_mouse_brain", qc=QCConfig(**QC_PRESETS["visium_mouse_brain"]),
                   seed=seed, keep_genes=panel)
    a.uns["nmo"] = {k: v for k, v in a.uns["nmo"].items()
                    if isinstance(v, (str, int, float, bool, list, dict, np.ndarray, type(None)))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(out_path, compression="gzip")
    print(f"built {out_path.name}: {a.n_obs} x {a.n_vars}")
    return out_path


def _build(model_type: str, cfg: Config, n_genes: int):
    if model_type == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    return build_baseline(model_type, n_genes=n_genes, hidden=128,
                          latent=cfg.model.get("latent_channels", 32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--source", default="visium_mouse_brain")
    p.add_argument("--targets", nargs="+", default=["xenium_mouse_brain"])
    p.add_argument("--models", nargs="+", default=["nmo", "gnn", "stagate", "gp"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--max-target-locations", type=int, default=15000)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="results/exp3")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for target_name in a.targets:
        out_path = out / f"{a.source}__to__{target_name}.json"
        results: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
        done = {(r["model"], r["seed"], r["setting"]) for r in results}

        tgt_path = Path(cfg.data.processed_dir) / f"{target_name}.h5ad"
        if not tgt_path.exists():
            print(f"[skip] {target_name}: not built")
            continue
        tgt_full = load_section(tgt_path, device=device)
        panel = [g for g in tgt_full.gene_names]

        src_path = build_visium_with_panel(
            panel, a.raw_dir,
            Path(cfg.data.processed_dir) / f"{a.source}__panel_{target_name}.h5ad",
        )
        src_full = load_section(src_path, device=device)
        src, tgt, shared = align_sections(src_full, tgt_full)
        if a.max_target_locations and tgt.n_obs > a.max_target_locations:
            tgt = subsample_section(tgt, a.max_target_locations, seed=0).to(device)
        print(f"{a.source} -> {target_name}: {len(shared)} shared genes, "
              f"{src.n_obs} source locations, {tgt.n_obs} target locations")

        for seed in a.seeds:
            if ("mean", seed, "floor") not in done:
                results.append({"model": "mean", "display": "Training-mean predictor", "seed": seed,
                                "setting": "floor", "target": target_name,
                                "n_shared_genes": len(shared), **mean_predictor(tgt, seed=seed)})
            for model_type in a.models:
                if (model_type, seed, "zero_shot") in done:
                    print(f"[skip] {model_type} seed {seed}")
                    continue
                set_seed(seed)
                logger = ExperimentLogger(out / "runs" / f"{target_name}__{model_type}__seed{seed}",
                                          cfg.to_dict())
                model = _build(model_type, cfg, src.n_genes)
                tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": a.epochs})
                tr = Trainer(model, src, tcfg, LossWeights(**cfg.loss.to_dict()),
                             logger, device, is_nmo=(model_type == "nmo"))
                src_res = tr.fit()
                results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                                "seed": seed, "setting": "source_in_domain", "target": target_name,
                                "n_shared_genes": len(shared), **src_res["test"]})

                zs = evaluate_zero_shot(model, tgt, seed=seed)
                results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                                "seed": seed, "setting": "zero_shot", "target": target_name,
                                "n_shared_genes": len(shared), **zs})
                print(f"[zero-shot] {target_name} {model_type} seed {seed}: "
                      f"r={zs['pearson_mean']:.4f}", flush=True)

                set_seed(seed)
                oracle = _build(model_type, cfg, tgt.n_genes)
                tr_o = Trainer(oracle, tgt, tcfg, LossWeights(**cfg.loss.to_dict()),
                               logger, device, is_nmo=(model_type == "nmo"))
                results.append({"model": model_type, "display": DISPLAY_NAMES.get(model_type, model_type),
                                "seed": seed, "setting": "oracle", "target": target_name,
                                "n_shared_genes": len(shared), **tr_o.fit()["test"]})
                out_path.write_text(json.dumps(results, indent=2, default=float))

        out_path.write_text(json.dumps(results, indent=2, default=float))
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n", df.groupby(["setting", "model"])[["pearson_mean", "rmse"]].agg(["mean", "std"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

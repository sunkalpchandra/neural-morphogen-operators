"""Experiment 6 -- developmental forecasting on Stereo-seq (E9.5 -> E10.5).

Every other experiment in this project fits a *stationary* operator, because
standard spatial transcriptomics measures a tissue once. The MOSTA developmental
series is the one dataset here with a genuine temporal axis, so it is the only
place we can ask the model to act as a forward-time predictor rather than as a
relaxation operator.

What is and is not being measured
---------------------------------
Consecutive MOSTA stages are **different embryos**, not the same embryo imaged
twice. There is therefore no cell-to-cell correspondence between E9.5 and E10.5,
and no amount of registration creates one. We consequently evaluate at the level
of the *rasterised expression field*: both sections are isotropically normalised
to a common frame and binned to a common lattice, and we score how well the
evolved E9.5 field predicts the E10.5 field. This is a distributional,
field-level comparison and it inherits whatever error the coarse affine
registration introduces.

Baselines make the comparison interpretable:

  ``persistence``  predict E10.5 = E9.5 (the operator does nothing). If the
                   learned dynamics carry no temporal information, NMO cannot
                   beat this.
  ``mean``         predict each gene's spatial mean at E9.5.
  ``nmo_T``        the operator integrated for T steps.

We sweep T so that the horizon is a reported quantity rather than a tuned one.
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

from src.evaluation.metrics import pearson_per_gene, rasterise, rmse
from src.losses.objectives import LossWeights
from src.models.nmo import build_nmo
from src.training.dataset import align_sections, load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed


def _register(sec) -> np.ndarray:
    """Coordinates in the shared frame.

    Both sections are already centred and isotropically scaled by the
    preprocessing pipeline, which is the extent of the registration we perform.
    We state this rather than implying a landmark-based alignment.
    """
    return sec.coords.detach().cpu().numpy()


def _field(sec, values: np.ndarray, grid: int) -> np.ndarray:
    img, _ = rasterise(_register(sec), values, grid=grid)
    return img.reshape(img.shape[0], -1).T          # (pixels, genes)


def evaluate_forecast(pred_pix: np.ndarray, true_pix: np.ndarray) -> Dict[str, float]:
    r = pearson_per_gene(pred_pix, true_pix)
    return {
        "field_pearson_mean": float(np.nanmean(r)),
        "field_pearson_median": float(np.nanmedian(r)),
        "field_rmse": rmse(pred_pix, true_pix),
        "frac_genes_positive": float(np.nanmean(r > 0)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--t0", default="mosta_embryo_E9.5")
    p.add_argument("--t1", default="mosta_embryo_E10.5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--horizons", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    p.add_argument("--out-dir", default="results/exp6")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "forecast.json"
    results: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["model"], r["seed"], r.get("horizon")) for r in results}

    pdir = Path(cfg.data.processed_dir)
    for f in (pdir / f"{a.t0}.h5ad", pdir / f"{a.t1}.h5ad"):
        if not f.exists():
            print(f"[skip] missing {f}")
            return 0

    s0_full = load_section(pdir / f"{a.t0}.h5ad", device=device)
    s1_full = load_section(pdir / f"{a.t1}.h5ad", device=device)
    s0, s1, shared = align_sections(s0_full, s1_full)
    print(f"{a.t0} -> {a.t1}: {len(shared)} shared genes, "
          f"{s0.n_obs} -> {s1.n_obs} locations")

    true_pix = _field(s1, s1.numpy_expr(denorm=True), a.grid)
    src_pix = _field(s0, s0.numpy_expr(denorm=True), a.grid)

    # ---- reference points ------------------------------------------------ #
    if ("persistence", -1, None) not in done:
        results.append({"model": "persistence", "display": "Persistence (no dynamics)",
                        "seed": -1, "horizon": None, "n_shared_genes": len(shared),
                        **evaluate_forecast(src_pix, true_pix)})
    if ("mean", -1, None) not in done:
        mean_pix = np.repeat(src_pix.mean(0, keepdims=True), true_pix.shape[0], axis=0)
        results.append({"model": "mean", "display": "Spatial-mean predictor",
                        "seed": -1, "horizon": None, "n_shared_genes": len(shared),
                        **evaluate_forecast(mean_pix, true_pix)})

    # ---- NMO fitted on E9.5, then integrated forward --------------------- #
    for seed in a.seeds:
        set_seed(seed)
        logger = ExperimentLogger(out / "runs" / f"seed{seed}", cfg.to_dict())
        model = build_nmo(cfg.model.to_dict(), n_genes=s0.n_genes)
        tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": a.epochs})
        Trainer(model, s0, tcfg, LossWeights(**cfg.loss.to_dict()),
                logger, device, is_nmo=True).fit()

        visible = s0.mask(["train", "val", "test"])
        with torch.no_grad():
            z0, _ = model.encode(s0.coords, s0.expr * visible.view(-1, 1),
                                 s0.edge_index, visible)
            for T in a.horizons:
                if ("nmo", seed, T) in done:
                    continue
                zT = model.evolve(z0, n_steps=T) if T > 0 else z0
                pred = s0.denormalise(model.decode(zT, s0.coords)).cpu().numpy()
                pred_pix = _field(s0, pred, a.grid)
                m = evaluate_forecast(pred_pix, true_pix)
                results.append({"model": "nmo", "display": "NMO (ours)", "seed": seed,
                                "horizon": T, "n_shared_genes": len(shared), **m})
                print(f"[forecast] seed{seed} T={T:>2}: "
                      f"field r={m['field_pearson_mean']:.4f}", flush=True)
        out_path.write_text(json.dumps(results, indent=2, default=float))

    out_path.write_text(json.dumps(results, indent=2, default=float))
    import pandas as pd
    df = pd.DataFrame(results)
    print("\n", df.groupby(["model", "horizon"], dropna=False)["field_pearson_mean"]
          .agg(["mean", "std", "count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Single-run training entry point.

    python -m src.training.train --config configs/nmo_visium_mouse.yaml
    python -m src.training.train --config configs/base.yaml model.type=gnn train.epochs=300
    python -m src.training.train --config configs/base.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.baselines import build_baseline
from ..models.nmo import build_nmo
from ..utils.common import Config, ExperimentLogger, get_device, set_seed
from .dataset import load_section, subsample_section
from .trainer import Trainer, TrainConfig
from ..losses.objectives import LossWeights


def build_model(cfg: Config, n_genes: int) -> torch.nn.Module:
    mtype = cfg.model.get("type", "nmo")
    if mtype == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    return build_baseline(mtype, n_genes=n_genes, **{
        k: v for k, v in cfg.model.to_dict().items()
        if k in ("hidden", "latent", "n_layers", "heads", "n_neighbors", "n_inducing")
    })


def get_section(cfg: Config, device: torch.device):
    path = Path(cfg.data.processed_dir) / f"{cfg.data.section}.h5ad"
    sec = load_section(
        path, device=device, knn_k=cfg.data.get("knn_k", 8),
        standardise=cfg.data.get("standardise", True),
        stats_from=cfg.data.get("stats_from", "train"),
    )
    n_max = cfg.data.get("max_locations")
    if n_max:
        sec = subsample_section(sec, int(n_max), seed=cfg.experiment.get("seed", 0))
        sec = sec.to(device)
    return sec


def run_single(cfg: Config, seed: int, out_root: Path) -> Dict:
    set_seed(seed)
    device = get_device(cfg.experiment.get("device", "auto"))
    mtype = cfg.model.get("type", "nmo")
    run_dir = out_root / f"{cfg.experiment.name}__{mtype}__seed{seed}"
    logger = ExperimentLogger(run_dir, cfg.to_dict())
    logger.info(f"device={device}  model={mtype}  seed={seed}")

    sec = get_section(cfg, device)
    model = build_model(cfg, sec.n_genes)

    tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed})
    trainer = Trainer(
        model, sec, tcfg,
        loss_weights=LossWeights(**cfg.loss.to_dict()),
        logger=logger, device=device, is_nmo=(mtype == "nmo"),
    )
    result = trainer.fit(checkpoint_path=run_dir / "best.pt")
    result.update({"seed": seed, "model_type": mtype, "config_name": cfg.experiment.name})

    # Physics diagnostics are only meaningful for the operator model.
    if mtype == "nmo":
        with torch.no_grad():
            z0, _ = model.encode(sec.coords, sec.expr * trainer.train_visible.view(-1, 1),
                                 sec.edge_index, trainer.train_visible)
            zT = model.evolve(z0)
        rep = model.stability_report(zT, coord_scale_um=sec.coord_scale_um)
        result["physics"] = {
            "turing_unstable": bool(rep["turing_unstable"]),
            "k_max": rep["k_max"],
            "growth_max": rep["growth_max"],
            "pattern_wavelength_um": (
                float(2 * np.pi / rep["k_max"] * sec.coord_scale_um) if rep["k_max"] > 1e-6 else None
            ),
            "diffusion_length_um": model.diffusion_length_um(sec.coord_scale_um).tolist(),
        }
        np.savez_compressed(
            run_dir / "stability.npz",
            **{k: v for k, v in rep.items() if isinstance(v, np.ndarray)},
        )

    logger.save_json("result.json", result)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("overrides", nargs="*", default=[])
    args = p.parse_args(argv)

    cfg = Config.load(args.config).override(args.overrides)
    out_root = Path(args.out_dir or cfg.experiment.get("out_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else [cfg.experiment.get("seed", 0)]

    results = [run_single(cfg, s, out_root) for s in seeds]

    if len(results) > 1:
        rs = [r["test"]["pearson_mean"] for r in results]
        print(f"\n{cfg.experiment.name}: test Pearson {np.mean(rs):.4f} +/- {np.std(rs):.4f} over {len(rs)} seeds")
    summary = out_root / f"{cfg.experiment.name}__{cfg.model.get('type','nmo')}__summary.json"
    summary.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

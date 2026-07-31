"""Experiment 5 -- ablations.

Each variant removes exactly one ingredient, holding everything else (data,
masks, seeds, optimiser, epochs) fixed:

  full                  the complete model
  no_pde                PDE / steady-state, splitting and mass terms disabled
  no_diffusion          the diffusion operator removed (reaction only)
  no_reaction           the reaction network removed (pure anisotropic diffusion)
  no_bio_reg            smoothness, mass and Jacobian regularisers disabled
  isotropic_diffusion   D constrained to d*I -- tests whether anisotropy matters
  discrete_gnn          operator replaced by a discrete GNN of matched capacity
  latent_{8,16,32,64}   latent width sweep
  no_dynamics           n_steps = 0: encoder+decoder only, operator never applied

``no_dynamics`` is the single most important control. If it matches the full
model, the PDE is decorative and the paper's central claim fails. We report it
prominently rather than burying it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.baselines import build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed


def variants() -> Dict[str, Dict]:
    """name -> {'model': overrides, 'loss': overrides}"""
    return {
        "full": {},
        "no_pde": {"loss": {"pde": 0.0, "split": 0.0, "mass": 0.0}},
        "no_diffusion": {"model": {"dynamics": {"use_diffusion": False}}},
        "no_reaction": {"model": {"dynamics": {"use_reaction": False}}},
        "no_bio_reg": {"loss": {"smooth": 0.0, "mass": 0.0, "jacobian": 0.0}},
        "isotropic_diffusion": {"model": {"dynamics": {"isotropic": True}}},
        "no_dynamics": {"model": {"dynamics": {"n_steps": 0}}},
        "state_dependent_diffusion": {"model": {"dynamics": {"state_dependent_diffusion": True}}},
        "latent_8": {"model": {"latent_channels": 8, "dynamics": {"channels": 8}}},
        "latent_16": {"model": {"latent_channels": 16, "dynamics": {"channels": 16}}},
        "latent_32": {"model": {"latent_channels": 32, "dynamics": {"channels": 32}}},
        "latent_64": {"model": {"latent_channels": 64, "dynamics": {"channels": 64}}},
        "discrete_gnn": {"special": "gnn"},
    }


def _deep_update(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--max-locations", type=int, default=None)
    p.add_argument("--out-dir", default="results/exp5")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{a.section}.json"
    results: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["variant"], r["seed"]) for r in results}

    V = variants()
    names = a.variants or list(V)

    for name in names:
        spec = V[name]
        for seed in a.seeds:
            if (name, seed) in done:
                print(f"[skip] {name} seed {seed}")
                continue
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad", device=device)
            if a.max_locations:
                sec = subsample_section(sec, a.max_locations, seed=seed).to(device)

            model_cfg = _deep_update(cfg.model.to_dict(), spec.get("model", {}))
            loss_cfg = _deep_update(cfg.loss.to_dict(), spec.get("loss", {}))
            logger = ExperimentLogger(out / "runs" / f"{name}__seed{seed}",
                                      {"model": model_cfg, "loss": loss_cfg})

            is_nmo = "special" not in spec
            if is_nmo:
                model = build_nmo(model_cfg, n_genes=sec.n_genes)
            else:
                model = build_baseline(spec["special"], n_genes=sec.n_genes, hidden=128,
                                       latent=model_cfg.get("latent_channels", 32))

            tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": a.epochs})
            tr = Trainer(model, sec, tcfg, LossWeights(**loss_cfg), logger, device, is_nmo=is_nmo)
            r = tr.fit()

            rec = {"variant": name, "seed": seed, "section": a.section,
                   "n_params": r["n_params"], "best_epoch": r["best_epoch"], **r["test"]}

            if is_nmo and model_cfg.get("dynamics", {}).get("n_steps", 8) > 0:
                with torch.no_grad():
                    z0, _ = model.encode(sec.coords, sec.expr * tr.train_visible.view(-1, 1),
                                         sec.edge_index, tr.train_visible)
                    zT = model.evolve(z0)
                rep = model.stability_report(zT, coord_scale_um=sec.coord_scale_um)
                rec["turing_unstable"] = bool(rep["turing_unstable"])
                rec["growth_max"] = float(rep["growth_max"])
                rec["diffusion_length_um_mean"] = float(
                    np.mean(model.diffusion_length_um(sec.coord_scale_um))
                ) if model.operator.diffusion is not None else None

            results.append(rec)
            out_path.write_text(json.dumps(results, indent=2, default=float))
            print(f"[done] {name} seed {seed}: r={rec['pearson_mean']:.4f} "
                  f"rmse={rec['rmse']:.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(results)
    agg = df.groupby("variant")[["pearson_mean", "rmse", "ssim_mean", "n_params"]].agg(["mean", "std"])
    print("\n", agg)
    agg.to_csv(out / f"{a.section}_aggregate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

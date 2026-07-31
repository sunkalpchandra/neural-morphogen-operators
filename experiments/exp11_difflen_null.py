"""Experiment 11 -- is the recovered diffusion length a property of the tissue?

The paper reports a median diffusion length of ~374 um and notes that it is the
same order as measured morphogen gradients. That observation has no null
distribution behind it, and there are two entirely mundane mechanisms that could
produce a number in that range without any biology being involved:

``lattice pitch``
    the field is evolved on an ``H x W`` lattice spanning the section, so the
    smallest resolvable length is one cell. A diffusion length of a few cells is
    what an unconstrained fit would produce if the data said nothing at all.

``splat bandwidth``
    features reach the lattice through a Nadaraya--Watson kernel of learnable
    width. Proposition~\\ref{prop:splat} already establishes that splatting
    contributes its own diffusive smoothing; if the operator merely inherits
    that width, the "diffusion length" is measuring the encoder, not the tissue.

Three controls separate these from a genuine tissue scale:

``shuffled``     expression is permuted across positions, destroying all spatial
                 structure while preserving every marginal. A length scale that
                 survives this is not coming from the data.
``sigma``        the splat bandwidth is fixed at several values spanning the
                 range the free model reaches.
``grid``         the lattice resolution is varied, which changes the pitch in
                 microns while leaving the tissue untouched.

The diagnostic is the length expressed in **lattice cells**. If the micron value
tracks the tissue it should stay fixed as the pitch changes; if it tracks the
discretization the cell value stays fixed instead.

    python experiments/exp11_difflen_null.py --seeds 0 1
"""

from __future__ import annotations

import argparse
import copy
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
from src.models.nmo import build_nmo
from src.training.dataset import SpatialSection, load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed


def shuffle_coords(sec: SpatialSection, seed: int) -> SpatialSection:
    """Permute expression across locations, preserving every gene's marginal.

    The coordinates, the graph and the splits are untouched, so the model sees a
    section of exactly the same geometry and sampling density carrying no
    spatial signal whatsoever.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sec.n_obs)
    return SpatialSection(**{**sec.__dict__,
                             "expr": sec.expr[torch.from_numpy(perm).to(sec.expr.device)]})


def conditions(base_grid: int, base_sigma: float) -> List[Dict]:
    """The control grid. ``baseline`` is the paper's configuration."""
    out = [dict(name="baseline", grid=base_grid, sigma=base_sigma, shuffled=False),
           dict(name="shuffled", grid=base_grid, sigma=base_sigma, shuffled=True)]
    for s in [0.5, 2.0, 4.0]:                     # spans the free-model range
        out.append(dict(name=f"sigma{s}", grid=base_grid, sigma=s, shuffled=False))
    for g in [32, 128]:                            # 64 is the baseline
        out.append(dict(name=f"grid{g}", grid=g, sigma=base_sigma, shuffled=False))
    # the decisive cross-cell: no signal AND a different pitch
    out.append(dict(name="shuffled_grid128", grid=128, sigma=base_sigma, shuffled=True))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--freeze-sigma", action="store_true", default=True,
                   help="hold the splat bandwidth at its configured value")
    p.add_argument("--out-dir", default="results/exp11")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "difflen_null.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["condition"], r["seed"]) for r in rows}

    base_grid = int(cfg.model.get("grid_size", 64))
    base_sigma = float(cfg.model.get("splat_sigma", 1.0))
    conds = conditions(base_grid, base_sigma)
    todo = [(c, s) for c in conds for s in a.seeds if (c["name"], s) not in done]
    print(f"{len(todo)} of {len(conds) * len(a.seeds)} runs remaining")

    for k, (cond, seed) in enumerate(todo, 1):
        try:
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad", device=device)
            if cond["shuffled"]:
                sec = shuffle_coords(sec, seed)

            mcfg = copy.deepcopy(cfg.model.to_dict())
            mcfg["grid_size"] = cond["grid"]
            mcfg["splat_sigma"] = cond["sigma"]
            model = build_nmo(mcfg, n_genes=sec.n_genes)
            if a.freeze_sigma:
                model.encoder.splat.log_sigma.requires_grad_(False)

            lg = ExperimentLogger(out / "runs" / f"{cond['name']}__s{seed}", cfg.to_dict())
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                    "epochs": a.epochs}),
                         LossWeights(**cfg.loss.to_dict()), lg, device, is_nmo=True)
            res = tr.fit()

            dl = np.asarray(model.diffusion_length_um(sec.coord_scale_um)).ravel()
            # one lattice cell, in microns: the domain is [-1, 1] so the pitch is
            # 2 / H in normalised units.
            pitch_um = 2.0 / cond["grid"] * sec.coord_scale_um
            sigma_um = float(model.encoder.splat.log_sigma.exp()) * pitch_um
            rows.append(dict(
                condition=cond["name"], seed=seed, shuffled=bool(cond["shuffled"]),
                grid=int(cond["grid"]), sigma_cells=float(cond["sigma"]),
                pitch_um=float(pitch_um), splat_sigma_um=sigma_um,
                difflen_median_um=float(np.median(dl)),
                difflen_p10_um=float(np.percentile(dl, 10)),
                difflen_p90_um=float(np.percentile(dl, 90)),
                difflen_median_cells=float(np.median(dl) / pitch_um),
                difflen_p10_cells=float(np.percentile(dl, 10) / pitch_um),
                difflen_p90_cells=float(np.percentile(dl, 90) / pitch_um),
                difflen_all_um=[float(x) for x in dl],
                pearson_mean=float(res["test"]["pearson_mean"]),
            ))
            print(f"[{k}/{len(todo)}] {cond['name']:<18} s{seed}  "
                  f"L={np.median(dl):7.1f} um = {np.median(dl)/pitch_um:5.2f} cells  "
                  f"(pitch {pitch_um:.0f} um)  r={res['test']['pearson_mean']:.4f}", flush=True)
        except Exception as exc:
            print(f"[{k}/{len(todo)}] FAIL {cond['name']} s{seed}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows.append(dict(condition=cond["name"], seed=seed, failed=True,
                             error=str(exc)[:160]))
        out_path.write_text(json.dumps(rows, indent=2, default=float))

    print(f"\nwrote {out_path} ({len(rows)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

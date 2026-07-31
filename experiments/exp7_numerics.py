"""Experiment 7 -- does the exact spectral solve actually matter?

The method section claims the Strang-split exponential integrator removes the
CFL restriction. This experiment tests that claim directly, and prices the
alternative fairly.

Three questions, one per part:

(a) **Stability.** Over a grid of step sizes, does each scheme remain bounded?
    The exponential scheme is predicted stable for every ``dt``
    (Proposition 5); explicit Euler on the same right-hand side is predicted
    stable only for ``dt < h^2 / (4 lambda_max)``.

(b) **Cost at matched accuracy.** A reviewer will object that Euler can simply
    take smaller steps. It can. This part measures how many steps each scheme
    needs to integrate a fixed horizon to a fixed accuracy, so the comparison is
    the honest one: not "Euler diverges" but "Euler costs N times more".

(c) **Downstream effect.** Training the full model with each scheme, to check
    that the numerical difference translates into a difference in held-out
    reconstruction rather than being invisible after fitting.

    python experiments/exp7_numerics.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.objectives import LossWeights
from src.models.dynamics import DynamicsConfig, ReactionDiffusionOperator
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

SCHEMES = {
    "strang-spectral": dict(integrator="strang", laplacian="spectral"),
    "euler-spectral":  dict(integrator="euler",  laplacian="spectral"),
    "strang-fd5":      dict(integrator="strang", laplacian="fd5"),
    "euler-fd5":       dict(integrator="euler",  laplacian="fd5"),
}


def cfl_limit(op: ReactionDiffusionOperator, H: int) -> float:
    """dt < h^2 / (4 lambda_max(D)) for the explicit 5-point Laplacian in 2-D."""
    lam = float(torch.linalg.eigvalsh(op.diffusion.tensor().detach()).max())
    h = 2.0 / H
    return h * h / (4.0 * max(lam, 1e-12))


# --------------------------------------------------------------------------- #
# (a) stability envelope
# --------------------------------------------------------------------------- #


def stability_sweep(C: int = 16, H: int = 64, n_steps: int = 300,
                    seed: int = 0) -> List[Dict]:
    rows = []
    dts = np.logspace(-4, -0.3, 16)
    for name, kw in SCHEMES.items():
        set_seed(seed)
        op = ReactionDiffusionOperator(DynamicsConfig(channels=C, dt=0.05, n_steps=1, **kw))
        with torch.no_grad():
            for p in op.reaction.parameters():
                p.add_(0.3 * torch.randn_like(p))
        limit = cfl_limit(op, H)
        z0 = torch.randn(1, C, H, H)
        for dt in dts:
            z = z0.clone()
            with torch.no_grad():
                for _ in range(n_steps):
                    z = op.step(z, float(dt))
                    if not torch.isfinite(z).all():
                        break
            ok = bool(torch.isfinite(z).all()) and float(z.abs().max()) < 1e3
            rows.append(dict(scheme=name, dt=float(dt), stable=ok,
                             cfl_limit=limit,
                             max_abs=float(z.abs().max()) if torch.isfinite(z).all() else float("inf")))
    return rows


# --------------------------------------------------------------------------- #
# (b) cost at matched accuracy
# --------------------------------------------------------------------------- #


def cost_at_accuracy(C: int = 16, H: int = 64, T: float = 0.4, tol: float = 1e-2,
                     seed: int = 0) -> List[Dict]:
    """Smallest step count each scheme needs to integrate to T within ``tol``."""
    set_seed(seed)
    ref_op = ReactionDiffusionOperator(
        DynamicsConfig(channels=C, dt=0.05, n_steps=1, integrator="strang", laplacian="spectral"))
    with torch.no_grad():
        for p in ref_op.reaction.parameters():
            p.add_(0.3 * torch.randn_like(p))
    z0 = torch.randn(1, C, H, H) * 0.5
    ref = z0.clone()
    with torch.no_grad():
        for _ in range(8192):
            ref = ref_op.step(ref, T / 8192)

    rows = []
    for name, kw in SCHEMES.items():
        set_seed(seed)
        op = ReactionDiffusionOperator(DynamicsConfig(channels=C, dt=0.05, n_steps=1, **kw))
        op.load_state_dict(ref_op.state_dict())        # identical parameters
        found = None
        for n in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            z = z0.clone()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(n):
                    z = op.step(z, T / n)
                    if not torch.isfinite(z).all():
                        break
            if not torch.isfinite(z).all():
                continue
            err = float((z - ref).norm() / ref.norm())
            if err < tol:
                found = dict(scheme=name, n_steps=n, rel_error=err,
                             wall_s=time.time() - t0)
                break
        rows.append(found or dict(scheme=name, n_steps=None, rel_error=None, wall_s=None))
    return rows


# --------------------------------------------------------------------------- #
# (c) effect on held-out reconstruction
# --------------------------------------------------------------------------- #


def downstream(cfg: Config, section: str, seeds: List[int], epochs: int,
               out_dir: Path) -> List[Dict]:
    device = get_device(cfg.experiment.get("device", "auto"))
    out_path = out_dir / f"downstream_{section}.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["scheme"], r["seed"]) for r in rows}
    for name, kw in SCHEMES.items():
        for seed in seeds:
            if (name, seed) in done:
                continue
            set_seed(seed)
            sec = load_section(Path(cfg.data.processed_dir) / f"{section}.h5ad", device=device)
            mcfg = cfg.model.to_dict()
            mcfg["dynamics"] = {**mcfg.get("dynamics", {}), **kw}
            model = build_nmo(mcfg, n_genes=sec.n_genes)
            lg = ExperimentLogger(out_dir / "runs" / f"{name}__seed{seed}", mcfg)
            tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                   "epochs": epochs}),
                         LossWeights(**cfg.loss.to_dict()), lg, device, is_nmo=True)
            try:
                r = tr.fit()
                rows.append(dict(scheme=name, seed=seed, diverged=False, **r["test"]))
                print(f"[ok] {name:<16} seed{seed}  r={r['test']['pearson_mean']:.4f}", flush=True)
            except Exception as exc:                     # a diverging scheme is a result
                rows.append(dict(scheme=name, seed=seed, diverged=True, error=str(exc)[:120]))
                print(f"[diverged] {name} seed{seed}: {exc}", flush=True)
            out_path.write_text(json.dumps(rows, indent=2, default=float))
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--out-dir", default="results/exp7")
    p.add_argument("--skip-downstream", action="store_true")
    a = p.parse_args()

    cfg = Config.load(a.config)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("== (a) stability envelope ==")
    stab = stability_sweep()
    (out / "stability.json").write_text(json.dumps(stab, indent=2))
    for name in SCHEMES:
        s = [r for r in stab if r["scheme"] == name]
        top = max((r["dt"] for r in s if r["stable"]), default=0.0)
        print(f"   {name:<16} largest stable dt = {top:.4g}   (CFL bound {s[0]['cfl_limit']:.4g})")

    print("\n== (b) cost at matched accuracy (rel. err < 1e-2) ==")
    cost = cost_at_accuracy()
    (out / "cost.json").write_text(json.dumps(cost, indent=2))
    base = next((r["n_steps"] for r in cost if r["scheme"] == "strang-spectral"), None)
    for r in cost:
        if r["n_steps"] is None:
            print(f"   {r['scheme']:<16} did not reach tolerance at any tested step count")
        else:
            ratio = r["n_steps"] / base if base else float("nan")
            print(f"   {r['scheme']:<16} {r['n_steps']:>5} steps  ({ratio:.0f}x)  "
                  f"err={r['rel_error']:.2e}")

    if not a.skip_downstream:
        print("\n== (c) effect on held-out reconstruction ==")
        downstream(cfg, a.section, a.seeds, a.epochs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

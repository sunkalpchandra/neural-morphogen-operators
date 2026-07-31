"""Regenerate every figure and table in the paper from run artifacts.

    python experiments/make_figures.py

Figures that need a fitted model reload the best checkpoint written by
Experiment 1; nothing is retrained here. Missing artifacts are skipped with a
warning rather than fabricated, so a partial run produces a partial paper rather
than a wrong one.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

from src.evaluation import numbers as numbers_mod
from src.evaluation.tables import build_all, dedupe, load_json_glob
from src.models.baselines import build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.utils.common import Config, get_device
from src.visualization import figures as F
from src.visualization.style import savefig


def _find_ckpt(results: Path, section: str, model: str, seed: int = 0) -> Optional[Path]:
    hits = sorted(results.glob(f"exp1/*/runs/{section}__{model}__seed{seed}/best.pt"))
    return hits[0] if hits else None


def _load_model(ckpt: Path, cfg: Config, n_genes: int, model_type: str, device):
    m = (build_nmo(cfg.model.to_dict(), n_genes=n_genes) if model_type == "nmo"
         else build_baseline(model_type, n_genes=n_genes, hidden=128,
                             latent=cfg.model.get("latent_channels", 32)))
    state = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(state["model"])
    return m.to(device).eval()


@torch.no_grad()
def _predict(model, sec, visible):
    out = model(sec.coords, sec.expr * visible.view(-1, 1), query_coords=sec.coords,
                edge_index=sec.edge_index, point_mask=visible)
    return sec.denormalise(out["pred"]).cpu().numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--results", default="results")
    p.add_argument("--section", default="visium_mouse_brain")
    p.add_argument("--fig-dir", default="paper/figures")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = Config.load(a.config)
    results, figdir = Path(a.results), Path(a.fig_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.experiment.get("device", "auto"))
    made: List[str] = []

    # ---- Figure 1: schematic (no data needed) ---------------------------- #
    savefig(F.figure1_overview(), figdir / "fig1_overview")
    made.append("fig1_overview")

    # ---- Model-dependent figures ----------------------------------------- #
    ck = _find_ckpt(results, a.section, "nmo", a.seed)
    if ck is None:
        print(f"[warn] no NMO checkpoint for {a.section}; skipping figures 2/3/5")
    else:
        sec = load_section(Path(cfg.data.processed_dir) / f"{a.section}.h5ad", device=device)
        nmo = _load_model(ck, cfg, sec.n_genes, "nmo", device)
        visible = sec.mask("train")

        pred_nmo = _predict(nmo, sec, visible)
        others: Dict[str, np.ndarray] = {}
        for other in ("gp", "stagate", "gnn"):
            ck2 = _find_ckpt(results, a.section, other, a.seed)
            if ck2 is not None:
                try:
                    m2 = _load_model(ck2, cfg, sec.n_genes, other, device)
                    others[other] = _predict(m2, sec, visible)
                except Exception as e:
                    print(f"[warn] could not load {other}: {e}")
            if len(others) >= 2:
                break

        savefig(F.figure2_reconstruction(sec, pred_nmo, others,
                                         visible.cpu().numpy()), figdir / "fig2_reconstruction")
        made.append("fig2_reconstruction")
        savefig(F.figure3_latent_fields(nmo, sec), figdir / "fig3_latent_fields")
        made.append("fig3_latent_fields")
        savefig(F.figure5_dynamics(nmo, sec), figdir / "fig5_dynamics")
        made.append("fig5_dynamics")

        # Physics diagnostics across every available NMO checkpoint, so the
        # numbers quoted in the paper (diffusion lengths, Turing fraction) are
        # aggregated over seeds rather than read off a single run.
        phys = []
        for ck_i in sorted(results.glob(f"exp1/*/runs/{a.section}__nmo__seed*/best.pt")):
            try:
                m = _load_model(ck_i, cfg, sec.n_genes, "nmo", device)
                with torch.no_grad():
                    z0, _ = m.encode(sec.coords, sec.expr * visible.view(-1, 1),
                                     sec.edge_index, visible)
                    zT = m.evolve(z0)
                rep = m.stability_report(zT, coord_scale_um=sec.coord_scale_um)
                phys.append({
                    "checkpoint": str(ck_i),
                    "turing_unstable": bool(rep["turing_unstable"]),
                    "k_max": float(rep["k_max"]),
                    "growth_max": float(rep["growth_max"]),
                    "growth_at_zero": float(rep["growth_rate"][0]),
                    "pattern_wavelength_um": (
                        float(2 * np.pi / rep["k_max"] * sec.coord_scale_um)
                        if rep["k_max"] > 1e-6 else None),
                    "diffusion_length_um": m.diffusion_length_um(sec.coord_scale_um).tolist(),
                })
            except Exception as e:
                print(f"[warn] physics for {ck_i.parent.name}: {e}")
        if phys:
            (results / "physics.json").write_text(json.dumps(phys, indent=2))
            print(f"physics: {len(phys)} NMO checkpoint(s) -> results/physics.json")

    # ---- Results-dependent figures --------------------------------------- #
    exp1 = [r for r in load_json_glob("exp1/**/*.json", results)
            if "model" in r and "pearson_mean" in r]
    if exp1:
        savefig(F.figure7_benchmark(exp1), figdir / "fig7_benchmark")
        made.append("fig7_benchmark")

    exp2 = [r for r in load_json_glob("exp2/**/*.json", results) if "setting" in r]
    exp3 = [r for r in load_json_glob("exp3/**/*.json", results) if "setting" in r]
    exp2 = dedupe(exp2, ["model", "seed", "setting", "target"])
    exp3 = dedupe(exp3, ["model", "seed", "setting", "target"])
    if exp2:
        savefig(F.figure4_transfer(exp2, exp3 or None), figdir / "fig4_transfer")
        made.append("fig4_transfer")

    exp5 = dedupe([r for r in load_json_glob("exp5/**/*.json", results) if "variant" in r],
                  ["variant", "seed", "section"])
    if exp5:
        savefig(F.figure6_ablations(exp5), figdir / "fig6_ablations")
        made.append("fig6_ablations")

    # ---- Tables + inline numbers ----------------------------------------- #
    tabs = build_all(results, "paper/tables")
    numbers_mod.build(results, "paper/numbers.tex")

    print(f"figures: {made}")
    print(f"tables:  {sorted(tabs)}")
    print("numbers: paper/numbers.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Experiment 4 -- counterfactual perturbation of the learned operator.

Two independent analyses, with deliberately different epistemic status.

4a. In-silico morphogen source ("bead implant")
-----------------------------------------------
The classical developmental-biology experiment is to implant a bead soaked in
SHH/WNT/BMP and observe the spatial response. We do the computational analogue:
identify the latent direction that maximally increases a pathway's decoded
activity, inject it in a small disc, integrate the learned operator forward, and
read out the predicted spatial response.

The test is made falsifiable by a **split-half design**: the perturbation
direction is defined using a random half of the pathway's genes, and enrichment
is scored on the *held-out* half. If the operator had merely memorised a
gene-to-latent projection, held-out pathway members would show no preferential
response. Significance is assessed against a permutation null over gene sets.

4b. Perturb-seq consistency of the reaction Jacobian
----------------------------------------------------
The composition decoder o f o encoder induces an effective gene-gene coupling
matrix ``J_eff``. Its column for gene *i* predicts the transcriptome-wide
response to increasing gene *i*. We compare those predictions against measured
CRISPRa responses from Norman et al. (2019).

**What this can and cannot establish.** Norman et al. profiled K562 cells, a
human erythroleukemia line, with no spatial component. Our operator is fitted to
spatial tissue. Any agreement therefore speaks to whether the reaction module
has captured *generic transcriptional coupling*, not to whether it recovered
tissue-specific signalling. We therefore (i) fit on the human Visium breast
section so at least the species and a proliferative context are shared, (ii)
report a permutation null, and (iii) report the same quantity for every baseline
so the comparison is relative rather than absolute. We do not claim this
validates the spatial dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import MORPHOGEN_PATHWAYS, pathway_gene_mask
from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed


# --------------------------------------------------------------------------- #
# 4a -- in-silico morphogen source
# --------------------------------------------------------------------------- #


def pathway_latent_direction(model, sec, gene_idx: np.ndarray, z_ref: torch.Tensor) -> torch.Tensor:
    """Latent direction that maximally increases the mean decoded expression of
    ``gene_idx``: the gradient of that score w.r.t. a spatially uniform latent
    perturbation, evaluated at the relaxed field.
    """
    z = z_ref.clone().detach().requires_grad_(True)
    pred = model.decode(z, sec.coords)                      # (N, G)
    score = pred[:, gene_idx].mean()
    g = torch.autograd.grad(score, z)[0]                    # (1, C, H, W)
    v = g.mean(dim=(2, 3), keepdim=True)                    # uniform component
    return v / v.norm().clamp_min(1e-8)


def inject_disc(z: torch.Tensor, v: torch.Tensor, centre=(0.0, 0.0), radius=0.15,
                amplitude=3.0) -> torch.Tensor:
    """Add ``amplitude * v`` inside a disc -- the computational bead implant."""
    B, C, H, W = z.shape
    ys = torch.linspace(-1, 1, H, device=z.device).view(-1, 1)
    xs = torch.linspace(-1, 1, W, device=z.device).view(1, -1)
    d = ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2).sqrt()
    disc = (d < radius).to(z.dtype).view(1, 1, H, W)
    return z + amplitude * v * disc


def run_bead_experiment(model, sec, pathway: str, seed: int = 0, amplitude: float = 3.0,
                        n_perm: int = 200) -> Optional[Dict]:
    """Split-half in-silico perturbation for one morphogen pathway."""
    rng = np.random.default_rng(seed)
    mask = pathway_gene_mask(sec.gene_names, pathway)
    idx = np.where(mask)[0]
    if len(idx) < 6:
        return None
    rng.shuffle(idx)
    half = len(idx) // 2
    define_idx, test_idx = idx[:half], idx[half:]

    visible = sec.mask(["train", "val", "test"])
    with torch.no_grad():
        z0, _ = model.encode(sec.coords, sec.expr * visible.view(-1, 1), sec.edge_index, visible)
        zT = model.evolve(z0)
        base = model.decode(zT, sec.coords)

    v = pathway_latent_direction(model, sec, define_idx, zT)

    with torch.no_grad():
        z_pert = inject_disc(zT, v, amplitude=amplitude)
        z_evolved = model.evolve(z_pert)
        pert = model.decode(z_evolved, sec.coords)
        delta = (pert - base).detach().cpu().numpy()        # (N, G)

    # Response magnitude per gene, restricted to the neighbourhood of the source
    xy = sec.coords.detach().cpu().numpy()
    near = np.linalg.norm(xy, axis=1) < 0.35
    resp = np.abs(delta[near]).mean(0)                       # (G,)

    rank = np.argsort(-resp)
    ranks = np.empty_like(rank); ranks[rank] = np.arange(len(rank))
    held_rank = float(np.mean(ranks[test_idx])) / len(resp)  # 0 = top responders

    # permutation null over random gene sets of the same size
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = rng.choice(len(resp), len(test_idx), replace=False)
        null[i] = float(np.mean(ranks[s])) / len(resp)
    p = float((null <= held_rank).mean())

    # spatial spread of the response (a direct read of the diffusion length)
    r = np.linalg.norm(xy, axis=1)
    mag = np.abs(delta).mean(1)
    order = np.argsort(r)
    r_s, m_s = r[order], mag[order]
    peak = m_s[: max(int(0.05 * len(m_s)), 5)].mean()
    half_r = float(r_s[np.argmax(m_s < 0.5 * peak)]) if (m_s < 0.5 * peak).any() else float(r_s[-1])

    return {
        "pathway": pathway,
        "n_pathway_genes": int(len(idx)),
        "n_held_out": int(len(test_idx)),
        "held_out_mean_rank_pct": held_rank * 100,
        "null_mean_rank_pct": float(null.mean()) * 100,
        "p_value": p,
        "response_halfwidth_um": half_r * sec.coord_scale_um,
        "mean_abs_delta": float(np.abs(delta).mean()),
        "top_responding_genes": [sec.gene_names[i] for i in rank[:15]],
    }


# --------------------------------------------------------------------------- #
# 4b -- effective gene-gene Jacobian vs Perturb-seq
# --------------------------------------------------------------------------- #


def effective_gene_jacobian(model, sec, gene_subset: np.ndarray, n_probe: int = 64) -> np.ndarray:
    """Columns of d g_hat / d g_in for the requested genes, via forward-mode JVPs.

    We perturb the *input* expression of one gene uniformly, propagate through
    encoder -> operator -> decoder, and record the change in every output gene.
    This is the model's counterfactual prediction for over-expressing that gene.
    """
    visible = sec.mask(["train", "val", "test"])
    with torch.no_grad():
        base_out = model(sec.coords, sec.expr * visible.view(-1, 1),
                         query_coords=sec.coords, edge_index=sec.edge_index, point_mask=visible)
        base = base_out["pred"]

    cols = []
    eps = 0.5
    for gi in gene_subset:
        expr_p = sec.expr.clone()
        expr_p[:, gi] += eps
        with torch.no_grad():
            out = model(sec.coords, expr_p * visible.view(-1, 1),
                        query_coords=sec.coords, edge_index=sec.edge_index, point_mask=visible)
        cols.append(((out["pred"] - base).mean(0) / eps).cpu().numpy())
    return np.stack(cols, axis=1)       # (G_out, n_perturbed)


def load_norman_lfc(processed_dir: Path, min_cells: int = 20) -> Optional[Tuple[Dict[str, np.ndarray], List[str]]]:
    """Mean log-fold-change per single-gene CRISPRa perturbation vs control."""
    import anndata as ad
    import scanpy as sc

    path = processed_dir / "perturb_norman.h5ad"
    if not path.exists():
        return None
    a = ad.read_h5ad(path)
    pert = a.obs["perturbation"].astype(str).to_numpy()
    ctrl = pert == "control"
    if ctrl.sum() < min_cells:
        return None

    X = np.asarray(a.X)
    base = X[ctrl].mean(0)
    genes = [str(g).upper() for g in a.var_names]

    lfc: Dict[str, np.ndarray] = {}
    singles = [p for p in np.unique(pert) if p not in ("control", "unassigned") and "+" not in p]
    for pgene in singles:
        sel = pert == pgene
        if sel.sum() < min_cells:
            continue
        lfc[pgene] = X[sel].mean(0) - base
    return lfc, genes


def perturbseq_consistency(model, sec, lfc: Dict[str, np.ndarray], norman_genes: List[str],
                           n_perturb: int = 40, seed: int = 0) -> Dict:
    """Correlate predicted counterfactual responses with measured CRISPRa LFCs."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    model_genes = [g.upper() for g in sec.gene_names]
    m_index = {g: i for i, g in enumerate(model_genes)}
    n_index = {g: i for i, g in enumerate(norman_genes)}
    shared = sorted(set(model_genes) & set(norman_genes))
    if len(shared) < 50:
        return {"error": f"only {len(shared)} shared genes"}

    testable = [g for g in lfc if g in m_index and g in n_index]
    if not testable:
        return {"error": "no perturbed gene is present in the spatial model's vocabulary"}
    rng.shuffle(testable)
    testable = testable[:n_perturb]

    cols = np.array([m_index[g] for g in testable])
    J = effective_gene_jacobian(model, sec, cols)            # (G_model, n_pert)

    s_m = np.array([m_index[g] for g in shared])
    s_n = np.array([n_index[g] for g in shared])

    rhos, nulls = [], []
    for j, g in enumerate(testable):
        pred = J[s_m, j]
        obs = lfc[g][s_n]
        if np.std(pred) < 1e-9 or np.std(obs) < 1e-9:
            continue
        rho = spearmanr(pred, obs).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
            perm = obs.copy(); rng.shuffle(perm)
            nulls.append(float(spearmanr(pred, perm).statistic))

    if not rhos:
        return {"error": "no evaluable perturbations"}
    rhos, nulls = np.array(rhos), np.array(nulls)
    from scipy.stats import wilcoxon

    try:
        stat = wilcoxon(rhos, nulls)
        pval = float(stat.pvalue)
    except Exception:
        pval = float("nan")
    return {
        "n_perturbations": len(rhos),
        "n_shared_genes": len(shared),
        "mean_spearman": float(rhos.mean()),
        "median_spearman": float(np.median(rhos)),
        "mean_null_spearman": float(nulls.mean()),
        "frac_positive": float((rhos > 0).mean()),
        "wilcoxon_p": pval,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _build(model_type: str, cfg: Config, n_genes: int):
    if model_type == "nmo":
        return build_nmo(cfg.model.to_dict(), n_genes=n_genes)
    return build_baseline(model_type, n_genes=n_genes, hidden=128,
                          latent=cfg.model.get("latent_channels", 32))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--bead-section", default="visium_mouse_brain")
    p.add_argument("--perturbseq-section", default="visium_human_breast")
    p.add_argument("--models", nargs="+", default=["nmo", "gnn", "stagate"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--pathways", nargs="+", default=["SHH", "WNT", "BMP", "FGF"])
    p.add_argument("--out-dir", default="results/exp4")
    p.add_argument("overrides", nargs="*", default=[])
    a = p.parse_args()

    cfg = Config.load(a.config).override(a.overrides)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    bead_results: List[Dict] = []
    ps_results: List[Dict] = []

    norman = load_norman_lfc(Path(cfg.data.processed_dir))
    if norman is None:
        print("[warn] Perturb-seq object not available; running 4a only")

    for seed in a.seeds:
        # ---- 4a : bead implant on the mouse brain (NMO only: needs an operator)
        set_seed(seed)
        sec = load_section(Path(cfg.data.processed_dir) / f"{a.bead_section}.h5ad", device=device)
        logger = ExperimentLogger(out / "runs" / f"bead__seed{seed}", cfg.to_dict())
        model = build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes)
        tcfg = TrainConfig(**{**cfg.train.to_dict(), "seed": seed, "epochs": a.epochs})
        Trainer(model, sec, tcfg, LossWeights(**cfg.loss.to_dict()), logger, device, True).fit()

        for pw in a.pathways:
            r = run_bead_experiment(model, sec, pw, seed=seed)
            if r:
                r.update({"seed": seed, "section": a.bead_section, "model": "nmo"})
                bead_results.append(r)
                print(f"[bead] seed{seed} {pw}: held-out rank {r['held_out_mean_rank_pct']:.1f}% "
                      f"(null {r['null_mean_rank_pct']:.1f}%), p={r['p_value']:.3f}", flush=True)
        (out / "bead_implant.json").write_text(json.dumps(bead_results, indent=2, default=float))

        # ---- 4b : Perturb-seq consistency on the human section
        if norman is not None:
            lfc, ngenes = norman
            sec_h = load_section(Path(cfg.data.processed_dir) / f"{a.perturbseq_section}.h5ad",
                                 device=device)
            for model_type in a.models:
                set_seed(seed)
                lg = ExperimentLogger(out / "runs" / f"ps__{model_type}__seed{seed}", cfg.to_dict())
                m = _build(model_type, cfg, sec_h.n_genes)
                Trainer(m, sec_h, tcfg, LossWeights(**cfg.loss.to_dict()), lg, device,
                        is_nmo=(model_type == "nmo")).fit()
                r = perturbseq_consistency(m, sec_h, lfc, ngenes, seed=seed)
                r.update({"seed": seed, "model": model_type,
                          "display": DISPLAY_NAMES.get(model_type, model_type)})
                ps_results.append(r)
                print(f"[perturb-seq] seed{seed} {model_type}: {r}", flush=True)
            (out / "perturbseq_consistency.json").write_text(
                json.dumps(ps_results, indent=2, default=float))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

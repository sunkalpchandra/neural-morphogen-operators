"""Experiment 9 -- does the reconstruction preserve tissue biology?

Held-out correlation says whether the numbers are right; it does not say whether
the reconstructed tissue is still usable for the analyses a biologist would run
on it. This experiment asks that question directly, using annotations that ship
with the public data rather than anything derived from our model:

1. **Spatial domain preservation.** Cluster the *predicted* expression at
   held-out locations and score the clustering against the reference
   annotation (Space Ranger graph clusters for Visium, the vendor clustering for
   Xenium, curated anatomical regions for the Stereo-seq embryo) with adjusted
   Rand index and normalized mutual information. A reconstruction that scores
   well on correlation but scrambles domain identity would be revealed here.

2. **Region-wise reconstruction quality.** Per-region held-out error, which
   exposes whether a model is uniformly decent or good on large homogeneous
   compartments and poor on small structured ones.

3. **Spatial autocorrelation of the reconstruction.** Moran's I and Geary's C,
   reported against the measured values, since a model may match the mean level
   while destroying spatial structure.

4. **Marker localization.** For genes that are differentially expressed in a
   reference region, whether the *predicted* field still discriminates that
   region (AUROC), which is the operational question for marker discovery.

5. **Neighborhood preservation.** Fraction of each location's k nearest
   neighbors in measured expression space that remain neighbors in the
   predicted space, a direct measure of whether local cellular context survives.

    python experiments/exp9_biology.py --sections visium_mouse_brain mosta_embryo_E9.5
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

import anndata as ad
from scipy import sparse as sp
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from src.evaluation.metrics import morans_i, spatial_weights
from src.losses.objectives import LossWeights
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section, subsample_section
from src.training.trainer import Trainer, TrainConfig
from src.utils.common import Config, ExperimentLogger, get_device, set_seed

#: Reference annotation column per section family, and a readable name for it.
REFERENCE_LABELS = {
    "visium_mouse_brain": ("sr_cluster", "Space Ranger graph clusters"),
    "visium_human_breast": ("sr_cluster", "Space Ranger graph clusters"),
    # The three specimens added to raise the independent-specimen count all ship
    # Space Ranger clusters, so they extend the biological evaluation too --
    # from four annotated sections to seven, on the same annotation type.
    "visium_mouse_kidney": ("sr_cluster", "Space Ranger graph clusters"),
    "visium_human_lymph_node": ("sr_cluster", "Space Ranger graph clusters"),
    "visium_mouse_brain_coronal": ("sr_cluster", "Space Ranger graph clusters"),
    "xenium_mouse_brain": ("xenium_cluster", "Xenium graph clusters"),
    "mosta_embryo_E9.5": ("region", "curated anatomical regions"),
    "mosta_embryo_E10.5": ("region", "curated anatomical regions"),
}


def gearys_c(values: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """Geary's C per column. C < 1 indicates positive spatial autocorrelation.

    Complements Moran's I: Geary's C is more sensitive to local differences,
    Moran's I to global structure, so reporting both distinguishes a model that
    is globally smooth from one that is locally smooth.
    """
    n = values.shape[0]
    z = values - values.mean(0, keepdims=True)
    denom = 2.0 * W.sum() * np.einsum("ng,ng->g", z, z) / (n - 1)
    Wd = W.toarray() if sp.issparse(W) else W
    num = np.zeros(values.shape[1])
    rows, cols = np.nonzero(Wd)
    wv = Wd[rows, cols]
    diff = values[rows] - values[cols]
    num = np.einsum("e,eg->g", wv, diff ** 2)
    return num / np.maximum(denom, 1e-12)


def neighborhood_preservation(true: np.ndarray, pred: np.ndarray, k: int = 15) -> float:
    """Mean overlap of k-NN sets in measured vs predicted expression space."""
    n = min(true.shape[0], 4000)
    idx = np.random.default_rng(0).choice(true.shape[0], n, replace=False)
    A, B = true[idx], pred[idx]
    ta = cKDTree(A).query(A, k=k + 1)[1][:, 1:]
    tb = cKDTree(B).query(B, k=k + 1)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ta, tb)]))


def domain_scores(true: np.ndarray, pred: np.ndarray, labels: np.ndarray,
                  n_pca: int = 30, seed: int = 0) -> Dict[str, float]:
    """Cluster measured and predicted expression; score both against reference."""
    n_clusters = len(np.unique(labels))
    out: Dict[str, float] = {"n_reference_domains": int(n_clusters)}
    for name, X in (("measured", true), ("predicted", pred)):
        d = min(n_pca, X.shape[1] - 1, X.shape[0] - 1)
        Z = PCA(n_components=d, random_state=seed).fit_transform(X)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Z)
        out[f"ari_{name}"] = float(adjusted_rand_score(labels, km))
        out[f"nmi_{name}"] = float(normalized_mutual_info_score(labels, km))
    # how much of the achievable domain structure survives reconstruction
    out["ari_retention"] = (out["ari_predicted"] / out["ari_measured"]
                            if out["ari_measured"] > 1e-6 else float("nan"))
    return out


def marker_auroc(true: np.ndarray, pred: np.ndarray, labels: np.ndarray,
                 n_markers: int = 5, min_region: int = 30) -> Dict[str, float]:
    """For each region, take its top markers by measured effect size and ask
    whether the *predicted* field still separates that region."""
    aur_t, aur_p = [], []
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < min_region or (~m).sum() < min_region:
            continue
        eff = true[m].mean(0) - true[~m].mean(0)
        top = np.argsort(-eff)[:n_markers]
        for g in top:
            if np.std(true[:, g]) < 1e-9:
                continue
            aur_t.append(roc_auc_score(m, true[:, g]))
            if np.std(pred[:, g]) > 1e-9:
                aur_p.append(roc_auc_score(m, pred[:, g]))
    return {"marker_auroc_measured": float(np.mean(aur_t)) if aur_t else float("nan"),
            "marker_auroc_predicted": float(np.mean(aur_p)) if aur_p else float("nan"),
            "n_marker_tests": len(aur_p)}


def region_wise_error(true: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> List[Dict]:
    rows = []
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() < 10:
            continue
        r = np.corrcoef(pred[m].ravel(), true[m].ravel())[0, 1]
        rows.append(dict(region=str(lab), n=int(m.sum()),
                         rmse=float(np.sqrt(((pred[m] - true[m]) ** 2).mean())),
                         pearson_flat=float(r) if np.isfinite(r) else float("nan")))
    return rows


def analyze(model, sec, labels: np.ndarray, visible: torch.Tensor) -> Dict:
    with torch.no_grad():
        out = model(sec.coords, sec.expr * visible.view(-1, 1), query_coords=sec.coords,
                    edge_index=sec.edge_index, point_mask=visible)
        pred = sec.denormalise(out["pred"]).cpu().numpy()
    true = sec.numpy_expr(denorm=True)
    coords = sec.coords.cpu().numpy()
    held = visible.cpu().numpy() == 0
    if held.sum() < 50:
        held = np.ones(len(true), bool)

    t, p, lab, xy = true[held], pred[held], labels[held], coords[held]
    # drop locations with missing reference labels and any non-finite predictions
    ok = ~np.isin(lab, ["nan", "NaN", "None", "<NA>", ""])
    ok &= np.isfinite(p).all(1) & np.isfinite(t).all(1)
    if ok.sum() < 50:
        return {"error": "too few usable locations after filtering"}
    t, p, lab, xy = t[ok], p[ok], lab[ok], xy[ok]
    W = spatial_weights(xy, k=6)
    res: Dict = {}
    res.update(domain_scores(t, p, lab))
    res.update(marker_auroc(t, p, lab))
    res["neighborhood_preservation"] = neighborhood_preservation(t, p)
    mi_t, mi_p = morans_i(t, W), morans_i(p, W)
    gc_t, gc_p = gearys_c(t, W), gearys_c(p, W)
    res["morans_i_measured"] = float(np.nanmean(mi_t))
    res["morans_i_predicted"] = float(np.nanmean(mi_p))
    res["gearys_c_measured"] = float(np.nanmean(gc_t))
    res["gearys_c_predicted"] = float(np.nanmean(gc_p))
    res["gearys_c_abs_error"] = float(np.nanmean(np.abs(gc_p - gc_t)))
    res["regions"] = region_wise_error(t, p, lab)
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--sections", nargs="+", default=list(REFERENCE_LABELS))
    p.add_argument("--models", nargs="+",
                   default=["nmo", "stagate", "gnn", "gp_multiscale", "neural_field"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--max-locations", type=int, default=4000)
    p.add_argument("--out-dir", default="results/exp9")
    a = p.parse_args()

    cfg = Config.load(a.config)
    device = get_device(cfg.experiment.get("device", "auto"))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    out_path = out / "biology.json"
    rows: List[Dict] = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["section"], r["model"], r["seed"]) for r in rows}
    processed = Path(cfg.data.processed_dir)

    for section in a.sections:
        f = processed / f"{section}.h5ad"
        if not f.exists() or section not in REFERENCE_LABELS:
            print(f"[skip] {section}"); continue
        col, ref_name = REFERENCE_LABELS[section]
        adata = ad.read_h5ad(f, backed="r")
        if col not in adata.obs.columns:
            print(f"[skip] {section}: no '{col}'"); adata.file.close(); continue
        labels_all = adata.obs[col].astype(str).to_numpy()
        adata.file.close()
        # A join that silently failed upstream leaves an all-NaN column; scoring
        # against it is meaningless, so the section is skipped with a message
        # rather than crashing the sweep several sections later.
        bad = np.isin(labels_all, ["nan", "NaN", "None", "<NA>", ""])
        if bad.all() or len(np.unique(labels_all[~bad])) < 2:
            print(f"[skip] {section}: '{col}' has no usable labels "
                  f"({100*bad.mean():.0f}% missing)")
            continue
        if bad.any():
            print(f"[note] {section}: {100*bad.mean():.1f}% of labels missing, excluded")

        for model_type in a.models:
            for seed in a.seeds:
                if (section, model_type, seed) in done:
                    print(f"[skip] {section} {model_type} s{seed}"); continue
                set_seed(seed)
                sec = load_section(f, device=device)
                labels = labels_all
                if a.max_locations and sec.n_obs > a.max_locations:
                    rng = np.random.default_rng(seed)
                    idx = np.sort(rng.choice(sec.n_obs, a.max_locations, replace=False))
                    labels = labels_all[idx]
                    sec = subsample_section(sec, a.max_locations, seed=seed).to(device)
                model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes) if model_type == "nmo"
                         else build_baseline(model_type, n_genes=sec.n_genes, hidden=128,
                                             latent=cfg.model.get("latent_channels", 32)))
                lg = ExperimentLogger(out / "runs" / f"{section}__{model_type}__s{seed}", cfg.to_dict())
                tr = Trainer(model, sec, TrainConfig(**{**cfg.train.to_dict(), "seed": seed,
                                                       "epochs": a.epochs}),
                             LossWeights(**cfg.loss.to_dict()), lg, device,
                             is_nmo=(model_type == "nmo"))
                tr.fit()
                model.eval()
                try:
                    res = analyze(model, sec, labels, tr.train_visible)
                except Exception as exc:
                    print(f"[FAIL] {section} {model_type} s{seed}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    rows.append(dict(section=section, model=model_type, seed=seed,
                                     failed=True, error=str(exc)[:140]))
                    out_path.write_text(json.dumps(rows, indent=2, default=float))
                    continue
                if "error" in res:
                    print(f"[skip] {section} {model_type} s{seed}: {res['error']}")
                    continue
                rows.append(dict(section=section, model=model_type,
                                 display=DISPLAY_NAMES.get(model_type, model_type),
                                 seed=seed, reference=ref_name, **res))
                print(f"[ok] {section:<20} {model_type:<14} s{seed}  "
                      f"ARI {res['ari_predicted']:.3f}/{res['ari_measured']:.3f} "
                      f"(ret {res['ari_retention']:.2f})  "
                      f"nbr {res['neighborhood_preservation']:.3f}  "
                      f"markerAUC {res['marker_auroc_predicted']:.3f}", flush=True)
                out_path.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n{len(rows)} records -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

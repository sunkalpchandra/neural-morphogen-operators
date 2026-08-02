"""Evaluation metrics for spatial expression prediction.

Four complementary axes, because no single number is sufficient here:

* **Pearson / Spearman (per gene, across locations)** -- is the spatial pattern
  of each gene correct?
* **RMSE / MAE** -- is the magnitude correct?
* **SSIM** -- is the *image structure* of the rasterised expression map correct?
  Correlation is invariant to local contrast and can be high for a blurred
  prediction; SSIM is not.
* **Moran's I** -- does the prediction reproduce the observed degree of spatial
  autocorrelation? A model that regresses towards the mean scores well on RMSE
  while destroying spatial structure, and this metric exposes that failure.

All functions accept numpy arrays of shape (N_locations, N_genes).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import rankdata


# --------------------------------------------------------------------------- #
# Correlation / error
# --------------------------------------------------------------------------- #


def _center(a: np.ndarray) -> np.ndarray:
    return a - a.mean(0, keepdims=True)


def pearson_per_gene(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pearson r per column of ``(N, G)`` arrays.

    Genes with no variance across the held-out locations return NaN rather than
    0: a constant gene has no correlation to measure, and scoring it 0 would
    silently drag the mean toward zero in proportion to how many such genes a
    section happens to contain. ``evaluate_prediction`` aggregates with
    ``_nanmean`` accordingly.

    Note that the number of genes dropped this way is not currently recorded:
    ``n_genes`` in the result dict is the total, not the number scored. On the
    primary section 78 of 2079 genes are undefined, so roughly 4% of the panel
    silently leaves every reported mean.
    """
    p, t = _center(pred), _center(true)
    num = (p * t).sum(0)
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    r = num / np.maximum(den, eps)
    # genes with zero variance in the held-out set are undefined, not zero
    flat = (np.linalg.norm(t, axis=0) < eps) | (np.linalg.norm(p, axis=0) < eps)
    r[flat] = np.nan
    return r


def spearman_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Spearman rho per gene: Pearson on column-wise ranks.

    Ranks are taken over locations, not over genes, so this measures whether a
    model orders locations correctly for each gene independently.
    """
    pr = np.apply_along_axis(rankdata, 0, pred)
    tr = np.apply_along_axis(rankdata, 0, true)
    return pearson_per_gene(pr, tr)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


# --------------------------------------------------------------------------- #
# Rasterisation + SSIM
# --------------------------------------------------------------------------- #


def rasterise(
    coords: np.ndarray, values: np.ndarray, grid: int = 64, agg: str = "mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin irregular points onto a ``grid x grid`` image.

    Returns (image stack (G, grid, grid), occupancy (grid, grid)).
    Empty bins are filled with the global mean of that gene so SSIM is not
    dominated by background; the occupancy map is returned so callers can
    restrict comparisons to occupied pixels.

    Caveat worth knowing when reading SSIM numbers
    ----------------------------------------------
    When a held-out split is rasterised, many bins receive no point and are
    filled with the gene mean in *both* the prediction and the reference. Those
    bins match trivially, so absolute SSIM values are inflated relative to a
    fully-occupied image. The inflation is identical for every model, so SSIM
    remains valid for *ranking* methods; it should not be read as an absolute
    measure of image fidelity. ``evaluate_prediction`` picks the grid size from
    the number of locations to keep occupancy reasonable.
    """
    values = np.atleast_2d(values.T).T if values.ndim == 1 else values
    lo, hi = coords.min(0), coords.max(0)
    frac = (coords - lo) / np.maximum(hi - lo, 1e-9)
    ix = np.clip((frac[:, 0] * (grid - 1)).round().astype(int), 0, grid - 1)
    iy = np.clip((frac[:, 1] * (grid - 1)).round().astype(int), 0, grid - 1)
    flat = iy * grid + ix

    G = values.shape[1]
    acc = np.zeros((grid * grid, G), dtype=np.float64)
    cnt = np.zeros(grid * grid, dtype=np.float64)
    np.add.at(acc, flat, values)
    np.add.at(cnt, flat, 1.0)

    occ = cnt.reshape(grid, grid) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        img = acc / np.maximum(cnt[:, None], 1e-9)
    empty = cnt == 0
    if empty.any():
        img[empty] = values.mean(0)
    return img.T.reshape(G, grid, grid), occ


def ssim_images(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-image SSIM for a stack (G, H, W). Uses skimage when available."""
    try:
        from skimage.metrics import structural_similarity as _ssim
    except Exception:  # pragma: no cover
        return np.full(a.shape[0], np.nan)

    out = np.empty(a.shape[0])
    for g in range(a.shape[0]):
        x, y = a[g], b[g]
        rng = float(max(x.max() - x.min(), y.max() - y.min(), 1e-6))
        out[g] = _ssim(x, y, data_range=rng)
    return out


# --------------------------------------------------------------------------- #
# Spatial autocorrelation
# --------------------------------------------------------------------------- #


def spatial_weights(coords: np.ndarray, k: int = 6, row_standardise: bool = True) -> sp.csr_matrix:
    """Row-standardised k-nearest-neighbour spatial weights matrix."""
    n = coords.shape[0]
    k = min(k + 1, n)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idx[:, 1:].reshape(-1)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    if row_standardise:
        d = np.asarray(W.sum(1)).ravel()
        W = sp.diags(1.0 / np.maximum(d, 1e-9)) @ W
    return W.tocsr()


def morans_i(values: np.ndarray, W: sp.csr_matrix) -> np.ndarray:
    """Moran's I per column of ``values`` (N, G).

    I = (N / S0) * (z' W z) / (z' z), with row-standardised W so S0 = N.
    """
    z = values - values.mean(0, keepdims=True)
    num = np.einsum("ng,ng->g", z, W @ z)
    den = np.einsum("ng,ng->g", z, z)
    return num / np.maximum(den, 1e-12)


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def gearys_c(values: np.ndarray, W) -> np.ndarray:
    """Geary's C per gene. ``C < 1`` indicates positive spatial autocorrelation.

    Reported alongside Moran's I because the two are sensitive to different
    things: Moran's I to global structure, Geary's C to local differences. A
    model that is globally smooth but locally correct scores differently on the
    two, and a dissipative operator is exactly the case where that matters.
    """
    n = values.shape[0]
    z = values - values.mean(0, keepdims=True)
    denom = 2.0 * W.sum() * np.einsum("ng,ng->g", z, z) / max(n - 1, 1)
    Wc = W.tocoo() if hasattr(W, "tocoo") else None
    if Wc is None:
        rows, cols = np.nonzero(W)
        wv = np.asarray(W)[rows, cols]
    else:
        rows, cols, wv = Wc.row, Wc.col, Wc.data
    diff = values[rows] - values[cols]
    num = np.einsum("e,eg->g", wv, diff ** 2)
    return num / np.maximum(denom, 1e-12)


def evaluate_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    coords: np.ndarray,
    grid: int = 64,
    knn: int = 6,
    gene_names: Optional[Sequence[str]] = None,
    compute_ssim: bool = True,
) -> Dict[str, float]:
    """Full metric bundle for one held-out prediction."""
    assert pred.shape == true.shape, f"{pred.shape} vs {true.shape}"

    r = pearson_per_gene(pred, true)
    rho = spearman_per_gene(pred, true)
    res: Dict[str, float] = {
        "pearson_mean": _nanmean(r),
        "pearson_median": float(np.nanmedian(r)) if np.isfinite(r).any() else float("nan"),
        "spearman_mean": _nanmean(rho),
        "rmse": rmse(pred, true),
        "mae": mae(pred, true),
        "n_locations": int(pred.shape[0]),
        "n_genes": int(pred.shape[1]),
    }

    # Location-wise correlation (profile similarity at each spot)
    rp = pearson_per_gene(pred.T, true.T)
    res["pearson_per_location"] = _nanmean(rp)

    if compute_ssim:
        # Choose the lattice from the number of held-out locations so that bins
        # are roughly singly occupied; a fixed fine grid would leave most bins
        # empty and inflate SSIM (see rasterise docstring).
        g = int(np.clip(np.sqrt(max(pred.shape[0], 1) / 1.5), 12, grid))
        ip, occ = rasterise(coords, pred, g)
        it, _ = rasterise(coords, true, g)
        s = ssim_images(ip, it)
        res["ssim_mean"] = _nanmean(s)
        res["ssim_grid"] = int(g)
        res["ssim_occupancy"] = float(occ.mean())

    W = spatial_weights(coords, knn)
    mi_p, mi_t = morans_i(pred, W), morans_i(true, W)
    res["morans_i_pred"] = _nanmean(mi_p)
    res["morans_i_true"] = _nanmean(mi_t)
    # How faithfully is spatial structure reproduced, gene by gene?
    res["morans_i_abs_error"] = _nanmean(np.abs(mi_p - mi_t))
    finite = np.isfinite(mi_p) & np.isfinite(mi_t)
    res["morans_i_corr"] = (
        float(np.corrcoef(mi_p[finite], mi_t[finite])[0, 1]) if finite.sum() > 2 else float("nan")
    )

    gc_p, gc_t = gearys_c(pred, W), gearys_c(true, W)
    res["gearys_c_pred"] = _nanmean(gc_p)
    res["gearys_c_true"] = _nanmean(gc_t)
    res["gearys_c_abs_error"] = _nanmean(np.abs(gc_p - gc_t))
    return res


def per_gene_table(
    pred: np.ndarray, true: np.ndarray, coords: np.ndarray, gene_names: Sequence[str]
) -> "object":
    """Per-gene metric table (returned as a pandas DataFrame)."""
    import pandas as pd

    W = spatial_weights(coords)
    return pd.DataFrame(
        {
            "gene": list(gene_names),
            "pearson": pearson_per_gene(pred, true),
            "spearman": spearman_per_gene(pred, true),
            "rmse": rmse_per_gene(pred, true),
            "morans_i_true": morans_i(true, W),
            "morans_i_pred": morans_i(pred, W),
        }
    )


def bootstrap_ci(
    pred: np.ndarray, true: np.ndarray, coords: np.ndarray, metric: str = "pearson_mean",
    n_boot: int = 200, seed: int = 0, alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Bootstrap CI over *locations* for one scalar metric."""
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = evaluate_prediction(pred[idx], true[idx], coords[idx], compute_ssim=False)[metric]
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.array(vals)
    return float(v.mean()), float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))

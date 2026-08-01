"""Platform-agnostic QC, normalisation, gene selection and splitting.

The same QC policy is applied to every platform so that cross-platform
comparisons are not confounded by preprocessing choices. Platform-specific
thresholds live in ``QC_PRESETS`` and are chosen from the published QC
characteristics of each assay (e.g. a Visium spot pools ~1-10 cells and so has
a far higher count floor than a Xenium cell).

A subtlety that matters for this project specifically
-----------------------------------------------------
Spatial coordinates are normalised **isotropically**: x and y are divided by the
*same* scale factor. Anisotropic scaling would silently distort the Laplacian
and make a learned isotropic diffusion coefficient un-interpretable. The scale
factor is stored in ``uns['nmo']['coord_scale_um']`` so that diffusion
coefficients learned in normalised units can be converted back to um^2 per unit
pseudo-time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from ..utils.common import get_logger

log = get_logger("nmo.data")


# --------------------------------------------------------------------------- #
# QC policy
# --------------------------------------------------------------------------- #


@dataclass
class QCConfig:
    min_counts: int = 500          # per spot/cell
    min_genes: int = 200           # per spot/cell
    min_cells_per_gene: int = 3    # per gene
    max_mito_frac: float = 0.30    # drop dying cells
    target_sum: float = 1e4        # counts-per-10k normalisation
    n_hvg: int = 2000
    hvg_flavor: str = "seurat_v3"  # operates on raw counts
    scale_clip: float = 10.0


#: Per-platform overrides. Panel-based assays (Xenium 248 genes, MERFISH 500
#: genes) cannot satisfy a 200-gene floor, and their per-cell counts are ~100x
#: lower than a Visium spot, so their thresholds are set from assay physics.
QC_PRESETS: Dict[str, Dict] = {
    "visium_mouse_brain":  dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_human_breast": dict(min_counts=500, min_genes=200, n_hvg=2000),
    # Same assay, same thresholds: applying one QC policy across Visium is what
    # keeps these specimens comparable to the ones already in the benchmark.
    "visium_mouse_kidney": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_human_lymph_node": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_mouse_brain_coronal": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_human_heart": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_ffpe_human_prostate": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "visium_ffpe_mouse_brain": dict(min_counts=500, min_genes=200, n_hvg=2000),
    "xenium_mouse_brain":  dict(min_counts=20,  min_genes=10,  n_hvg=248,  max_mito_frac=1.0),
    "merfish_allen":       dict(min_counts=0,   min_genes=5,   n_hvg=500,  max_mito_frac=1.0),
    "mosta_embryo":        dict(min_counts=100, min_genes=50,  n_hvg=2000),
    "perturb_norman":      dict(min_counts=500, min_genes=200, n_hvg=2000),
}


# --------------------------------------------------------------------------- #
# Morphogen pathway gene sets (used for interpretability, not for training)
# --------------------------------------------------------------------------- #
#
# Curated from canonical developmental-signalling literature. These are used
# ONLY post hoc, to ask whether learned latent channels align with known
# signalling programmes. They never enter the loss.

MORPHOGEN_PATHWAYS: Dict[str, List[str]] = {
    "SHH": [
        "SHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3", "SUFU", "HHIP",
        "BOC", "CDON", "GAS1", "FOXA2", "NKX2-2", "NKX6-1", "OLIG2",
    ],
    "WNT": [
        "WNT1", "WNT2", "WNT2B", "WNT3", "WNT3A", "WNT4", "WNT5A", "WNT7A", "WNT7B",
        "WNT8B", "WNT10B", "CTNNB1", "AXIN2", "LEF1", "TCF7", "TCF7L2", "DKK1", "DKK3",
        "SFRP1", "SFRP2", "FZD1", "FZD3", "FZD5", "LRP5", "LRP6", "APC", "NKD1", "RSPO2",
    ],
    "BMP": [
        "BMP1", "BMP2", "BMP4", "BMP5", "BMP6", "BMP7", "BMPR1A", "BMPR1B", "BMPR2",
        "SMAD1", "SMAD5", "SMAD9", "ID1", "ID2", "ID3", "ID4", "NOG", "CHRD", "GREM1",
        "GREM2", "MSX1", "MSX2",
    ],
    "FGF": [
        "FGF1", "FGF2", "FGF3", "FGF8", "FGF9", "FGF10", "FGF13", "FGF15", "FGF17",
        "FGFR1", "FGFR2", "FGFR3", "FGFR4", "SPRY1", "SPRY2", "SPRY4", "ETV4", "ETV5",
        "DUSP6", "SEF", "IL17RD",
    ],
    "NOTCH": [
        "NOTCH1", "NOTCH2", "NOTCH3", "DLL1", "DLL3", "DLL4", "JAG1", "JAG2",
        "HES1", "HES5", "HEY1", "HEY2", "RBPJ", "DTX1",
    ],
    "RA": [
        "RARA", "RARB", "RARG", "RXRA", "ALDH1A1", "ALDH1A2", "ALDH1A3", "CYP26A1",
        "CYP26B1", "CRABP1", "CRABP2", "RBP1",
    ],
}


def pathway_gene_mask(var_names: Sequence[str], pathway: str) -> np.ndarray:
    """Boolean mask over ``var_names`` for a morphogen pathway (case-insensitive)."""
    if pathway not in MORPHOGEN_PATHWAYS:
        raise KeyError(f"unknown pathway {pathway!r}; have {sorted(MORPHOGEN_PATHWAYS)}")
    want = {g.upper().replace("-", "") for g in MORPHOGEN_PATHWAYS[pathway]}
    up = np.array([str(v).upper().replace("-", "") for v in var_names])
    return np.isin(up, list(want))


# --------------------------------------------------------------------------- #
# QC + normalisation
# --------------------------------------------------------------------------- #


def _mito_prefix(organism: str) -> str:
    return "MT-" if "sapiens" in organism.lower() else "mt-"


def quality_control(adata: ad.AnnData, qc: QCConfig, organism: str = "") -> ad.AnnData:
    """Standard count/gene/mitochondrial filtering with reporting."""
    n0, g0 = adata.n_obs, adata.n_vars

    mito = adata.var_names.str.startswith(_mito_prefix(organism))
    counts = np.asarray(adata.X.sum(1)).ravel()
    adata.obs["total_counts"] = counts
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(1)).ravel()
    adata.obs["pct_mito"] = (
        np.asarray(adata[:, mito].X.sum(1)).ravel() / np.maximum(counts, 1) if mito.sum() else 0.0
    )

    keep = (
        (adata.obs["total_counts"].to_numpy() >= qc.min_counts)
        & (adata.obs["n_genes"].to_numpy() >= qc.min_genes)
        & (adata.obs["pct_mito"].to_numpy() <= qc.max_mito_frac)
    )
    adata = adata[keep].copy()

    gene_cells = np.asarray((adata.X > 0).sum(0)).ravel()
    adata = adata[:, gene_cells >= qc.min_cells_per_gene].copy()

    log.info(
        f"  QC: {n0} -> {adata.n_obs} obs ({100 * adata.n_obs / max(n0,1):.1f}% kept), "
        f"{g0} -> {adata.n_vars} genes"
    )
    adata.uns.setdefault("nmo", {})["qc"] = {
        "n_obs_before": int(n0), "n_obs_after": int(adata.n_obs),
        "n_vars_before": int(g0), "n_vars_after": int(adata.n_vars),
        **{k: v for k, v in vars(qc).items()},
    }
    return adata


def normalise(adata: ad.AnnData, qc: QCConfig, already_log: bool = False) -> ad.AnnData:
    """Library-size normalise + log1p, storing raw counts in ``layers['counts']``."""
    if already_log:
        # MERFISH log2(CPV+1) / MOSTA normalised values: already on a log scale.
        adata.layers["counts"] = adata.X.copy()
        log.info("  normalise: input already log-scaled; skipping CPM+log1p")
        return adata
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=qc.target_sum)
    sc.pp.log1p(adata)
    return adata


def select_genes(
    adata: ad.AnnData,
    qc: QCConfig,
    already_log: bool = False,
    keep_genes: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """Highly-variable gene selection, force-including morphogen pathway genes.

    We always retain every detected morphogen-pathway gene regardless of its
    variance rank. Otherwise the interpretability analysis would be evaluated
    on a gene set that HVG selection had already biased.
    """
    n_hvg = min(qc.n_hvg, adata.n_vars)
    if n_hvg >= adata.n_vars:
        adata.var["highly_variable"] = True
    else:
        try:
            if already_log:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
            else:
                sc.pp.highly_variable_genes(
                    adata, n_top_genes=n_hvg, flavor=qc.hvg_flavor, layer="counts"
                )
        except Exception as exc:
            log.info(f"  HVG flavor fallback ({exc}); using dispersion ranking")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")

    hv = adata.var["highly_variable"].to_numpy().copy()

    # Force-include morphogen pathway genes and any explicitly requested genes.
    forced = np.zeros(adata.n_vars, dtype=bool)
    for pw in MORPHOGEN_PATHWAYS:
        forced |= pathway_gene_mask(adata.var_names, pw)
    if keep_genes:
        want = {str(g).upper() for g in keep_genes}
        forced |= np.isin(np.array([v.upper() for v in adata.var_names]), list(want))

    adata.var["morphogen_gene"] = forced
    sel = hv | forced
    log.info(
        f"  gene selection: {int(hv.sum())} HVG + {int((forced & ~hv).sum())} "
        f"forced morphogen genes -> {int(sel.sum())}"
    )
    adata = adata[:, sel].copy()
    return adata


def normalise_coordinates(adata: ad.AnnData) -> ad.AnnData:
    """Centre coordinates and scale **isotropically** into roughly [-1, 1]^2.

    Returns the scale in ``uns['nmo']['coord_scale_um']`` (microns per
    normalised unit) so physical quantities remain recoverable.
    """
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    centre = xy.mean(0)
    centred = xy - centre
    # Single shared scale => isotropic; half the larger extent maps to 1.0.
    scale = float(np.abs(centred).max())
    if scale <= 0:
        raise ValueError("degenerate spatial extent")
    norm = (centred / scale).astype(np.float32)

    adata.obsm["spatial_um"] = xy.astype(np.float32)
    adata.obsm["spatial"] = norm
    adata.obs["x"] = norm[:, 0]
    adata.obs["y"] = norm[:, 1]
    adata.uns.setdefault("nmo", {}).update(
        {
            "coord_centre_um": centre.tolist(),
            "coord_scale_um": scale,
            "coord_note": "spatial is isotropically normalised; spatial_um is original microns",
        }
    )
    log.info(f"  coords: centre={centre.round(1).tolist()} um, scale={scale:.1f} um/unit")
    return adata


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def spatial_block_split(
    adata: ad.AnnData,
    n_blocks: int = 8,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
    seed: int = 0,
) -> np.ndarray:
    """Contiguous spatial-block split (the honest split for spatial models).

    A random spot-level split leaks information: neighbouring spots are highly
    correlated, so a random held-out spot is trivially predicted from its
    immediate neighbours. We therefore partition the tissue into contiguous
    blocks on a coarse grid and hold out whole blocks. This is the setting
    reported in the paper; ``random_split`` is provided only for ablation.

    Returns an array of {'train','val','test'} of length n_obs.
    """
    rng = np.random.default_rng(seed)
    xy = np.asarray(adata.obsm["spatial"])
    # Assign each observation to a cell of an n_blocks x n_blocks lattice.
    lo, hi = xy.min(0), xy.max(0)
    frac = (xy - lo) / np.maximum(hi - lo, 1e-9)
    idx = np.clip((frac * n_blocks).astype(int), 0, n_blocks - 1)
    block = idx[:, 0] * n_blocks + idx[:, 1]

    occupied = np.array(sorted(set(block.tolist())))
    rng.shuffle(occupied)
    n = len(occupied)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    test_b = set(occupied[:n_test].tolist())
    val_b = set(occupied[n_test : n_test + n_val].tolist())

    split = np.where(
        np.isin(block, list(test_b)), "test",
        np.where(np.isin(block, list(val_b)), "val", "train"),
    )
    adata.obs["spatial_block"] = block
    counts = {s: int((split == s).sum()) for s in ("train", "val", "test")}
    log.info(f"  split (contiguous blocks, {n_blocks}x{n_blocks}): {counts}")
    return split.astype(object)


def random_split(
    adata: ad.AnnData, val_frac: float = 0.15, test_frac: float = 0.20, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    perm = rng.permutation(n)
    n_test, n_val = int(test_frac * n), int(val_frac * n)
    split = np.array(["train"] * n, dtype=object)
    split[perm[:n_test]] = "test"
    split[perm[n_test : n_test + n_val]] = "val"
    return split


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #


def preprocess(
    adata: ad.AnnData,
    dataset_key: str,
    qc: Optional[QCConfig] = None,
    split_mode: str = "block",
    n_blocks: int = 8,
    seed: int = 0,
    keep_genes: Optional[Sequence[str]] = None,
    spatial: bool = True,
) -> ad.AnnData:
    """Run the full standardisation pipeline on a raw AnnData."""
    qc = qc or QCConfig(**QC_PRESETS.get(dataset_key, {}))
    prov = dict(adata.uns.get("nmo", {}))
    organism = prov.get("organism", "")
    already_log = bool(prov.get("already_log", False))

    log.info(f"preprocess[{dataset_key}] {adata.n_obs} x {adata.n_vars}")
    adata = quality_control(adata, qc, organism)
    adata = normalise(adata, qc, already_log)
    adata = select_genes(adata, qc, already_log, keep_genes)

    if spatial:
        adata = normalise_coordinates(adata)
        split = spatial_block_split(adata, n_blocks, seed=seed) if split_mode == "block" \
            else random_split(adata, seed=seed)
        adata.obs["split"] = pd.Categorical(split, categories=["train", "val", "test"])

    # Dense float32 matrix: downstream operators need dense math and these
    # matrices are at most ~50k x 2k, which is comfortably in memory.
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    adata.X = np.ascontiguousarray(X, dtype=np.float32)

    adata.uns.setdefault("nmo", {}).update(prov)
    adata.uns["nmo"]["preprocessed"] = True
    adata.uns["nmo"]["seed"] = int(seed)
    adata.uns["nmo"]["split_mode"] = split_mode if spatial else "none"
    return adata


def harmonise_genes(
    a: ad.AnnData, b: ad.AnnData, how: str = "upper"
) -> Tuple[ad.AnnData, ad.AnnData, List[str]]:
    """Align two datasets onto a shared gene vocabulary.

    ``how='upper'`` matches on upper-cased symbols, which is the standard
    first-order mouse/human ortholog heuristic (e.g. mouse *Sox2* -> human
    *SOX2*). It is imperfect -- it misses paralog renaming and family
    expansions -- and the paper says so explicitly rather than implying a
    curated ortholog map was used.
    """
    ua = pd.Index([str(v).upper() for v in a.var_names])
    ub = pd.Index([str(v).upper() for v in b.var_names])
    # Drop duplicates created by case-folding before intersecting.
    ka = ~ua.duplicated()
    kb = ~ub.duplicated()
    a2 = a[:, ka].copy(); a2.var_names = ua[ka]
    b2 = b[:, kb].copy(); b2.var_names = ub[kb]
    shared = sorted(set(a2.var_names) & set(b2.var_names))
    if not shared:
        raise RuntimeError("no shared genes after harmonisation")
    log.info(f"  harmonised vocabulary: {len(shared)} shared genes")
    return a2[:, shared].copy(), b2[:, shared].copy(), shared

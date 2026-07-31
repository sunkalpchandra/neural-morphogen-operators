"""Format-specific readers that turn each raw dataset into a *raw* AnnData.

Every loader returns an ``AnnData`` obeying one internal contract:

* ``adata.X``            -- raw counts (or the platform's native quantification;
                            see ``adata.uns['nmo']['count_type']``)
* ``adata.obsm['spatial']`` -- (n_obs, 2) float32 coordinates in *microns*
* ``adata.obs['x'] / ['y']`` -- the same coordinates as columns
* ``adata.var_names``    -- gene symbols, upper-cased for cross-species joins
                            in ``adata.var['symbol_upper']``
* ``adata.uns['nmo']``   -- provenance block (dataset key, technology, units,
                            physical spot spacing, citation)

Normalisation, QC and gene selection happen later, in ``preprocess.py``, so that
the QC choices are applied identically across platforms.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from ..utils.common import get_logger
from . import sources

log = get_logger("nmo.data")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _provenance(spec: sources.DatasetSpec, **extra) -> Dict:
    d = {
        "dataset": spec.key,
        "title": spec.title,
        "technology": spec.technology,
        "organism": spec.organism,
        "tissue": spec.tissue,
        "citation": spec.citation,
        "accession": spec.accession,
        "license": spec.license,
        "coord_units": "micron",
    }
    d.update(extra)
    return d


def _finalise(adata: ad.AnnData, xy: np.ndarray, spec: sources.DatasetSpec, **prov) -> ad.AnnData:
    """Attach coordinates + provenance and enforce the internal contract."""
    xy = np.asarray(xy, dtype=np.float32)
    if xy.shape != (adata.n_obs, 2):
        raise ValueError(f"coordinate shape {xy.shape} != ({adata.n_obs}, 2)")
    if not np.isfinite(xy).all():
        raise ValueError("non-finite spatial coordinates")

    adata.obsm["spatial"] = xy
    adata.obs["x"] = xy[:, 0]
    adata.obs["y"] = xy[:, 1]
    adata.var_names_make_unique()
    adata.var["symbol"] = adata.var_names.astype(str)
    adata.var["symbol_upper"] = adata.var["symbol"].str.upper()
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    adata.X = adata.X.astype(np.float32)
    adata.uns["nmo"] = _provenance(spec, **prov)
    log.info(
        f"  loaded {spec.key}: {adata.n_obs} obs x {adata.n_vars} vars, "
        f"extent {np.ptp(xy[:, 0]):.0f} x {np.ptp(xy[:, 1]):.0f} um"
    )
    return adata


def _read_10x_h5(path: Path) -> ad.AnnData:
    import scanpy as sc

    return sc.read_10x_h5(path)


# --------------------------------------------------------------------------- #
# Visium
# --------------------------------------------------------------------------- #


def load_visium(raw_dir: Path, key: str) -> ad.AnnData:
    """10x Visium: Space Ranger filtered matrix + tissue_positions + scalefactors.

    Visium spots sit on a hexagonal lattice with 100 um centre-to-centre
    spacing and 55 um diameter. Space Ranger reports positions in
    full-resolution *image pixels*, so we convert to microns using the known
    physical spot pitch rather than trusting any image metadata.
    """
    spec = sources.get(key)
    d = raw_dir / key

    adata = _read_10x_h5(d / "filtered_feature_bc_matrix.h5")

    spatial_dir = d / "spatial"
    pos_file = None
    for cand in ("tissue_positions_list.csv", "tissue_positions.csv"):
        if (spatial_dir / cand).exists():
            pos_file = spatial_dir / cand
            break
    if pos_file is None:
        raise FileNotFoundError(f"no tissue positions file under {spatial_dir}")

    # Space Ranger <2.0 writes headerless CSV; >=2.0 writes a header.
    head = pd.read_csv(pos_file, nrows=1, header=None)
    has_header = str(head.iloc[0, 0]).startswith("barcode")
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    pos = pd.read_csv(pos_file, header=0 if has_header else None)
    pos.columns = cols[: pos.shape[1]]
    pos = pos.set_index("barcode")

    common = adata.obs_names.intersection(pos.index)
    if len(common) == 0:
        raise RuntimeError("no barcode overlap between matrix and tissue positions")
    adata = adata[common].copy()
    pos = pos.loc[adata.obs_names]

    adata.obs["array_row"] = pos["array_row"].to_numpy()
    adata.obs["array_col"] = pos["array_col"].to_numpy()
    adata.obs["in_tissue"] = pos["in_tissue"].to_numpy().astype(bool)

    # pixel coordinates: (col = x, row = y)
    px = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=np.float64)

    # Convert px -> um. The array_row/array_col lattice has a known geometry:
    # 100 um between adjacent spots in a row (array_col steps by 2 along a row).
    # Estimate um-per-pixel empirically from the median nearest-neighbour
    # pixel distance, which must equal 100 um.
    from scipy.spatial import cKDTree

    tree = cKDTree(px)
    dists, _ = tree.query(px, k=2)
    med_nn_px = float(np.median(dists[:, 1]))
    um_per_px = 100.0 / med_nn_px if med_nn_px > 0 else 1.0
    xy = px * um_per_px

    sf_path = spatial_dir / "scalefactors_json.json"
    scalefactors = json.loads(sf_path.read_text()) if sf_path.exists() else {}

    # Keep the low-resolution H&E for figures (small, always present).
    images = {}
    for nm in ("tissue_lowres_image.png", "tissue_hires_image.png"):
        p = spatial_dir / nm
        if p.exists():
            images[nm.replace("tissue_", "").replace("_image.png", "")] = str(p)

    # Space Ranger reference clustering, used only as an annotation for figures.
    clust = d / "analysis" / "clustering" / "graphclust" / "clusters.csv"
    if clust.exists():
        cl = pd.read_csv(clust).set_index("Barcode")
        adata.obs["sr_cluster"] = (
            cl.reindex(adata.obs_names)["Cluster"].astype("Int64").astype(str).values
        )

    adata = _finalise(
        adata, xy, spec,
        count_type="UMI",
        spot_diameter_um=55.0,
        spot_pitch_um=100.0,
        um_per_pixel_fullres=um_per_px,
        scalefactors=scalefactors,
        image_paths=images,
        resolution="spot (multi-cell)",
    )
    return adata


# --------------------------------------------------------------------------- #
# Xenium
# --------------------------------------------------------------------------- #


def load_xenium(raw_dir: Path, key: str = "xenium_mouse_brain") -> ad.AnnData:
    """10x Xenium: MatrixMarket cell x gene matrix + segmentation centroids.

    Xenium reports centroids directly in microns, so no rescaling is needed.
    Control probes (negative controls, blanks, deprecated codewords) are
    dropped but their totals are retained per cell as a QC covariate.
    """
    spec = sources.get(key)
    d = raw_dir / key

    mdir = d / "cell_feature_matrix"
    if not mdir.exists():
        cands = [p for p in d.rglob("matrix.mtx.gz")]
        if not cands:
            raise FileNotFoundError(f"cell_feature_matrix not found under {d}")
        mdir = cands[0].parent

    X = scipy.io.mmread(mdir / "matrix.mtx.gz").T.tocsr()  # -> cells x genes
    features = pd.read_csv(
        mdir / "features.tsv.gz", sep="\t", header=None,
        names=["gene_id", "gene_name", "feature_type"],
    )
    barcodes = pd.read_csv(mdir / "barcodes.tsv.gz", sep="\t", header=None, names=["cell_id"])

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = barcodes["cell_id"].astype(str).values
    adata.var_names = features["gene_name"].astype(str).values
    adata.var["gene_id"] = features["gene_id"].values
    adata.var["feature_type"] = features["feature_type"].values

    cells = pd.read_csv(d / "cells.csv.gz")
    # cell_id is written as an integer in this bundle while barcodes.tsv is read
    # as text; cast both to str so the join cannot silently produce NaN.
    cells[cells.columns[0]] = cells[cells.columns[0]].astype(str)
    cells = cells.set_index(cells.columns[0])
    cells = cells.reindex(adata.obs_names)
    if cells[["x_centroid", "y_centroid"]].isna().any().any():
        raise RuntimeError("cells.csv did not cover every barcode in the matrix")

    for c in (
        "x_centroid", "y_centroid", "transcript_counts", "total_counts",
        "cell_area", "nucleus_area", "control_probe_counts", "control_codeword_counts",
    ):
        if c in cells.columns:
            adata.obs[c] = cells[c].to_numpy()

    # Split real genes from control probes (Blank / Negative Control codewords
    # and probes). Their totals are a per-cell specificity QC covariate.
    is_gene = adata.var["feature_type"].astype(str).str.contains("Gene Expression", case=False)
    if is_gene.sum() == 0:  # older bundles omit feature_type
        is_gene = ~adata.var_names.str.contains("NegControl|BLANK|antisense|Deprecated", case=False)
    is_gene = np.asarray(is_gene, dtype=bool)
    if (~is_gene).sum():
        adata.obs["control_counts_matrix"] = np.asarray(adata[:, ~is_gene].X.sum(1)).ravel()
    adata = adata[:, is_gene].copy()

    # Xenium writes Barcode as an integer while barcodes.tsv is read as text, so
    # the index must be cast before joining -- otherwise reindex silently yields
    # an all-NaN column and the reference clustering is lost.
    clust = d / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
    if not clust.exists():
        kms = sorted((d / "analysis" / "clustering").glob("gene_expression_kmeans_*_clusters"),
                     key=lambda q: int(q.name.split("_")[-2]))
        clust = (kms[len(kms) // 2] / "clusters.csv") if kms else clust
    if clust.exists():
        cl = pd.read_csv(clust)
        cl[cl.columns[0]] = cl[cl.columns[0]].astype(str)
        cl = cl.set_index(cl.columns[0])
        joined = cl.reindex(adata.obs_names)["Cluster"]
        if joined.isna().all():
            log.info(f"  [warn] {clust.parent.name} join produced no matches; dropping")
        else:
            adata.obs["xenium_cluster"] = joined.astype("Int64").astype(str).values
            log.info(f"  reference clustering: {clust.parent.name}, "
                     f"{joined.nunique()} clusters, {100*joined.notna().mean():.0f}% assigned")

    xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    return _finalise(
        adata, xy, spec,
        count_type="transcript",
        resolution="single cell",
        panel_size=int(adata.n_vars),
    )


# --------------------------------------------------------------------------- #
# MERFISH (Allen Brain Cell Atlas)
# --------------------------------------------------------------------------- #


def load_merfish_section(
    raw_dir: Path, section: str = sources.MERFISH_PRIMARY_SECTION, key: str = "merfish_allen"
) -> ad.AnnData:
    """Allen ABC-Atlas MERFISH, one coronal section.

    The section h5ad holds expression indexed by ``cell_label``; coordinates
    and cell metadata live in the atlas-wide ``cell_metadata.csv``, which we
    stream-filter to the requested section to avoid loading 4M rows.
    """
    spec = sources.get(key)
    d = raw_dir / key
    h5 = d / f"{section}-log2.h5ad"
    if not h5.exists():
        raise FileNotFoundError(f"missing MERFISH section {h5}")

    adata = ad.read_h5ad(h5)

    meta = _read_merfish_metadata(d, section)
    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        raise RuntimeError(f"no cell_label overlap for section {section}")
    adata = adata[common].copy()
    meta = meta.loc[adata.obs_names]

    for c in ("brain_section_label", "donor_label", "donor_sex", "cluster_alias", "z"):
        if c in meta.columns:
            adata.obs[c] = meta[c].values

    # Join cell-type labels through the taxonomy tables when available.
    ann_path = d / "cluster_annotation.csv"
    clu_path = d / "cluster.csv"
    if ann_path.exists() and clu_path.exists() and "cluster_alias" in meta.columns:
        try:
            clu = pd.read_csv(clu_path)
            ann = pd.read_csv(ann_path)
            merged = clu.merge(ann, left_on="label", right_on="cluster_label", how="left")
            alias2 = merged.set_index("cluster_alias") if "cluster_alias" in merged else None
            if alias2 is not None:
                for lvl in ("class", "subclass", "supertype"):
                    if lvl in alias2.columns:
                        adata.obs[f"cell_{lvl}"] = (
                            alias2[lvl].reindex(adata.obs["cluster_alias"]).astype(str).values
                        )
        except Exception as exc:  # annotation is optional garnish
            log.info(f"  (merfish cell-type join skipped: {exc})")

    # Gene symbols: the ABC h5ad uses Ensembl IDs in var_names.
    gene_csv = d / "gene.csv"
    if gene_csv.exists():
        g = pd.read_csv(gene_csv)
        idcol = "gene_identifier" if "gene_identifier" in g.columns else g.columns[0]
        symcol = "gene_symbol" if "gene_symbol" in g.columns else g.columns[1]
        mapping = g.set_index(idcol)[symcol]
        sym = mapping.reindex(adata.var_names)
        adata.var["ensembl"] = adata.var_names.astype(str)
        adata.var_names = pd.Index(
            [s if isinstance(s, str) and s else e for s, e in zip(sym.values, adata.var_names)]
        ).astype(str)

    xy = meta[["x", "y"]].to_numpy(dtype=np.float32) * 1000.0  # atlas stores mm
    return _finalise(
        adata, xy, spec,
        count_type="log2(CPV+1)",
        already_log=True,
        resolution="single cell",
        section=section,
    )


def _read_merfish_metadata(d: Path, section: str) -> pd.DataFrame:
    """Stream ``cell_metadata.csv`` (564 MB) keeping only one section."""
    # Cached as gzipped CSV rather than parquet so the pipeline needs no
    # pyarrow/fastparquet dependency; one section is only ~50k rows.
    cache = d / f"_meta_{section}.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, low_memory=False).set_index("cell_label")

    path = d / "cell_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    # ``brain_section_label`` looks like 'C57BL6J-638850.40'.
    keep: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
        col = "brain_section_label" if "brain_section_label" in chunk.columns else None
        if col is None:
            raise RuntimeError("cell_metadata.csv lacks brain_section_label")
        sel = chunk[chunk[col].astype(str) == section]
        if len(sel):
            keep.append(sel)
    if not keep:
        raise RuntimeError(f"section {section} absent from cell_metadata.csv")
    meta = pd.concat(keep, ignore_index=True)
    idcol = "cell_label" if "cell_label" in meta.columns else meta.columns[0]
    meta = meta.rename(columns={idcol: "cell_label"})
    meta.to_csv(cache, index=False)
    return meta.set_index("cell_label")


# --------------------------------------------------------------------------- #
# Stereo-seq (MOSTA)
# --------------------------------------------------------------------------- #


def load_mosta(raw_dir: Path, stage: str = "E9.5_E1S1", key: str = "mosta_embryo") -> ad.AnnData:
    """MOSTA Stereo-seq embryo section.

    The published h5ad already carries ``obsm['spatial']`` in bin units. Bin50
    corresponds to 50 x 50 DNB, i.e. 25 um, so we rescale to microns.
    """
    spec = sources.get(key)
    path = raw_dir / key / f"{stage}.MOSTA.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)

    adata = ad.read_h5ad(path)

    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    elif {"x", "y"}.issubset(adata.obs.columns):
        xy = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
    else:
        raise RuntimeError(f"no spatial coordinates in {path}")

    # MOSTA bin50: each bin edge is 50 DNB * 500 nm = 25 um.
    bin_um = 25.0
    xy = xy * bin_um

    # The published object stores normalised values in .X and counts in .layers.
    already_log = True
    if "count" in adata.layers:
        adata.X = adata.layers["count"]
        already_log = False
    elif "counts" in adata.layers:
        adata.X = adata.layers["counts"]
        already_log = False
    adata.layers.clear()

    if "annotation" in adata.obs.columns:
        adata.obs["region"] = adata.obs["annotation"].astype(str)

    adata.obs["stage"] = stage.split("_")[0]
    adata.obs["section"] = stage

    return _finalise(
        adata, xy, spec,
        count_type="UMI" if not already_log else "normalised",
        already_log=already_log,
        resolution="bin50 (~25 um)",
        stage=stage.split("_")[0],
        bin_size_um=bin_um,
    )


# --------------------------------------------------------------------------- #
# Perturb-seq (Norman 2019)
# --------------------------------------------------------------------------- #


def load_perturb_norman(raw_dir: Path, key: str = "perturb_norman") -> ad.AnnData:
    """GSE133344 CRISPRa Perturb-seq. Non-spatial: no coordinates are attached.

    Returns an AnnData with ``obs['perturbation']`` giving the guide identity
    ('control' for NTC cells). Used exclusively as an out-of-domain probe of
    the learned reaction Jacobian.
    """
    spec = sources.get(key)
    d = raw_dir / key

    X = scipy.io.mmread(d / "matrix.mtx.gz").T.tocsr()
    genes = pd.read_csv(d / "genes.tsv.gz", sep="\t", header=None)
    barcodes = pd.read_csv(d / "barcodes.tsv.gz", sep="\t", header=None)

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.obs_names = barcodes.iloc[:, 0].astype(str).values
    # column 0 = ensembl id, column 1 = symbol
    adata.var["ensembl"] = genes.iloc[:, 0].astype(str).values
    adata.var_names = (
        genes.iloc[:, 1].astype(str).values if genes.shape[1] > 1 else genes.iloc[:, 0].astype(str).values
    )
    adata.var_names_make_unique()

    ident = pd.read_csv(d / "cell_identities.csv.gz")
    idcol = "cell_barcode" if "cell_barcode" in ident.columns else ident.columns[0]
    ident = ident.set_index(idcol)
    ident = ident.reindex(adata.obs_names)

    guide_col = next(
        (c for c in ("guide_identity", "guide_target", "perturbation", "gene") if c in ident.columns),
        ident.columns[0],
    )
    guides = ident[guide_col].astype(str).fillna("NA")
    adata.obs["guide_identity"] = guides.values
    # 'A_B_1' style identities -> perturbed gene set; NTC/control markers vary.
    import re as _re

    # Norman et al. name their non-targeting guides NegCtrl0, NegCtrl1,
    # NegCtrl10, ... and pair them with a real gene to encode a *single*
    # perturbation (e.g. "KLF1_NegCtrl0"). Matching only bare NEG/CTRL/NTC
    # therefore misclassifies every control cell as perturbed and leaves the
    # dataset with no control group at all.
    _CTRL = _re.compile(r"^(NEGCTRL|NEGATIVECTRL|NTC|CTRL|CONTROL|NONTARGET\w*)\d*$")

    def _is_control_token(tok: str) -> bool:
        return bool(_CTRL.match(tok.upper()))

    def _norm(g: str) -> str:
        if g in ("nan", "NA", ""):
            return "unassigned"
        # identities look like "A_B__A_B"; take one half then split on "_"
        g = g.split("__")[0]
        parts = [p for p in g.split("_") if p and p.lower() != "nan"]
        if not parts:
            return "unassigned"
        targets = [p for p in parts if not _is_control_token(p)]
        if not targets:
            return "control"
        return "+".join(sorted(set(targets)))

    adata.obs["perturbation"] = [_norm(g) for g in guides.values]
    adata.obs["n_perturbed"] = [
        0 if p in ("control", "unassigned") else len(p.split("+")) for p in adata.obs["perturbation"]
    ]
    for c in ("number_of_cells", "read_count", "UMI_count", "coverage"):
        if c in ident.columns:
            adata.obs[c] = ident[c].values

    adata.var["symbol"] = adata.var_names.astype(str)
    adata.var["symbol_upper"] = adata.var["symbol"].str.upper()
    adata.uns["nmo"] = _provenance(spec, count_type="UMI", spatial=False, resolution="single cell")
    log.info(
        f"  loaded {key}: {adata.n_obs} cells x {adata.n_vars} genes, "
        f"{adata.obs['perturbation'].nunique()} perturbation classes"
    )
    return adata


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

LOADERS = {
    "visium_mouse_brain": lambda raw, **kw: load_visium(raw, "visium_mouse_brain"),
    "visium_human_breast": lambda raw, **kw: load_visium(raw, "visium_human_breast"),
    "xenium_mouse_brain": lambda raw, **kw: load_xenium(raw),
    "merfish_allen": lambda raw, **kw: load_merfish_section(raw, kw.get("section", sources.MERFISH_PRIMARY_SECTION)),
    "mosta_embryo": lambda raw, **kw: load_mosta(raw, kw.get("stage", "E9.5_E1S1")),
    "perturb_norman": lambda raw, **kw: load_perturb_norman(raw),
}


def load_raw(key: str, raw_dir: str | Path = "data/raw", **kw) -> ad.AnnData:
    if key not in LOADERS:
        raise KeyError(f"no loader for {key!r}; have {sorted(LOADERS)}")
    return LOADERS[key](Path(raw_dir), **kw)

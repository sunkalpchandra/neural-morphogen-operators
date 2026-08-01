"""Canonical registry of every public data source used in this project.

Each entry is a fully-specified, verified remote artifact. Nothing in this file
is fetched implicitly -- ``download.py`` consumes this registry and is the only
module permitted to touch the network.

Design notes
------------
* We deliberately do NOT download raw sequencing reads (e.g. the Visium
  ``*_fastqs.tar`` bundles, which are hundreds of GB). This project operates on
  count matrices; re-running CellRanger/SpaceRanger is out of scope and would
  not change any result. The provenance of the count matrices is recorded here
  instead.
* Large optional artifacts (full-resolution microscopy TIFFs, Xenium transcript
  tables, the 7.6 GB whole-brain MERFISH matrix) are marked ``optional=True``
  and are skipped unless explicitly requested. The default download set is
  ~4 GB and is sufficient to reproduce every number in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Remote artifact description
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RemoteFile:
    """A single downloadable artifact."""

    url: str
    # Path relative to ``data/raw/`` where the file is stored.
    dest: str
    # Human-readable role of this file within its dataset.
    role: str = ""
    # Expected size in bytes (0 = unknown/unverified). Used only for reporting
    # and for detecting truncated downloads.
    size: int = 0
    # If True, only fetched when the corresponding --with-* flag is passed.
    optional: bool = False
    # If True, the archive is expanded in place after download.
    extract: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    """A logical dataset: a named collection of remote artifacts."""

    key: str
    title: str
    technology: str
    organism: str
    tissue: str
    citation: str
    accession: str
    license: str
    files: List[RemoteFile] = field(default_factory=list)
    notes: str = ""

    @property
    def required_files(self) -> List[RemoteFile]:
        return [f for f in self.files if not f.optional]

    @property
    def optional_files(self) -> List[RemoteFile]:
        return [f for f in self.files if f.optional]


# --------------------------------------------------------------------------- #
# Base URLs
# --------------------------------------------------------------------------- #

_TENX_SPATIAL = "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0"
_TENX_XENIUM = "https://cf.10xgenomics.com/samples/xenium/1.0.2"
_ABC = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
_MOSTA = "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics"
_GEO133344 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl"

# MERFISH coronal sections used for the high-resolution validation experiment.
# These span anterior -> posterior across the Allen whole-brain MERFISH volume.
# Section 40 is used as the primary evaluation section (mid-brain, contains
# cortex + hippocampus + thalamus, so it is anatomically comparable to the
# Visium adult mouse brain sagittal section).
MERFISH_SECTIONS: List[str] = ["C57BL6J-638850.31", "C57BL6J-638850.40", "C57BL6J-638850.50"]
MERFISH_PRIMARY_SECTION = "C57BL6J-638850.40"

# Stereo-seq developmental stages. These four stages give the project its only
# genuine temporal axis and drive the developmental forecasting experiment.
MOSTA_STAGES: List[str] = ["E9.5_E1S1", "E10.5_E1S1", "E11.5_E1S1", "E12.5_E1S1"]


# --------------------------------------------------------------------------- #
# Dataset 1 -- 10x Visium, adult mouse brain (primary training benchmark)
# --------------------------------------------------------------------------- #

VISIUM_MOUSE_BRAIN = DatasetSpec(
    key="visium_mouse_brain",
    title="10x Genomics Visium -- Adult Mouse Brain (sagittal, FFPE-adjacent fresh frozen)",
    technology="Visium (55 um spots, 100 um centre-to-centre, hexagonal lattice)",
    organism="Mus musculus",
    tissue="Whole adult brain, sagittal section",
    citation="10x Genomics (2021), Space Ranger 1.3.0 demonstration dataset",
    accession="10x Visium_Adult_Mouse_Brain",
    license="10x Genomics public dataset terms (free for research use)",
    notes=(
        "Primary training benchmark. The *_fastqs.tar bundle is intentionally "
        "not downloaded: we consume the Space Ranger filtered count matrix."
    ),
    files=[
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Adult_Mouse_Brain/Visium_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5",
            "visium_mouse_brain/filtered_feature_bc_matrix.h5",
            role="count matrix (spots x genes)",
            size=21_107_817,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Adult_Mouse_Brain/Visium_Adult_Mouse_Brain_spatial.tar.gz",
            "visium_mouse_brain/spatial.tar.gz",
            role="spot coordinates, scale factors, H&E lowres/hires images",
            size=10_460_398,
            extract=True,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Adult_Mouse_Brain/Visium_Adult_Mouse_Brain_analysis.tar.gz",
            "visium_mouse_brain/analysis.tar.gz",
            role="Space Ranger graph clustering (used only as a reference annotation)",
            size=29_721_292,
            extract=True,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Adult_Mouse_Brain/Visium_Adult_Mouse_Brain_image.tif",
            "visium_mouse_brain/image.tif",
            role="full-resolution H&E microscopy image",
            size=398_267_856,
            optional=True,
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Dataset 2 -- 10x Visium, human breast cancer (cross-tissue transfer)
# --------------------------------------------------------------------------- #

VISIUM_HUMAN_BREAST = DatasetSpec(
    key="visium_human_breast",
    title="10x Genomics Visium -- Human Breast Cancer (Block A, Section 1)",
    technology="Visium (55 um spots, hexagonal lattice)",
    organism="Homo sapiens",
    tissue="Invasive ductal carcinoma",
    citation="10x Genomics (2021), Space Ranger 1.3.0 demonstration dataset",
    accession="10x Visium_Human_Breast_Cancer",
    license="10x Genomics public dataset terms (free for research use)",
    notes=(
        "Cross-tissue / cross-species generalisation target. Shares no gene "
        "identifiers with mouse except through orthology, and has a completely "
        "different spatial architecture (tumour nests vs laminar brain)."
    ),
    files=[
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5",
            "visium_human_breast/filtered_feature_bc_matrix.h5",
            role="count matrix (spots x genes)",
            size=23_942_611,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_spatial.tar.gz",
            "visium_human_breast/spatial.tar.gz",
            role="spot coordinates, scale factors, H&E lowres/hires images",
            size=5_262_726,
            extract=True,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_analysis.tar.gz",
            "visium_human_breast/analysis.tar.gz",
            role="Space Ranger graph clustering (reference annotation)",
            size=31_881_602,
            extract=True,
        ),
        RemoteFile(
            f"{_TENX_SPATIAL}/Visium_Human_Breast_Cancer/Visium_Human_Breast_Cancer_image.tif",
            "visium_human_breast/image.tif",
            role="full-resolution H&E microscopy image (3.5 GB)",
            size=3_461_492_844,
            optional=True,
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Dataset 3 -- Allen Brain Cell Atlas MERFISH (high-resolution validation)
# --------------------------------------------------------------------------- #

def _merfish_files() -> List[RemoteFile]:
    files: List[RemoteFile] = [
        RemoteFile(
            f"{_ABC}/metadata/MERFISH-C57BL6J-638850/20230830/cell_metadata.csv",
            "merfish_allen/cell_metadata.csv",
            role="per-cell x/y/z coordinates, section label, donor metadata",
            size=564_200_000,
        ),
        RemoteFile(
            f"{_ABC}/metadata/MERFISH-C57BL6J-638850/20230830/gene.csv",
            "merfish_allen/gene.csv",
            role="gene panel annotation (500 genes)",
            size=20_000,
        ),
        RemoteFile(
            f"{_ABC}/metadata/WMB-taxonomy/20230830/cluster.csv",
            "merfish_allen/cluster.csv",
            role="cluster identifiers",
            size=130_000,
        ),
        RemoteFile(
            f"{_ABC}/metadata/WMB-taxonomy/20230830/views/cluster_to_cluster_annotation_membership_pivoted.csv",
            "merfish_allen/cluster_annotation.csv",
            role="cluster -> class/subclass/supertype cell-type labels",
            size=530_000,
        ),
    ]
    for sec in MERFISH_SECTIONS:
        files.append(
            RemoteFile(
                f"{_ABC}/expression_matrices/MERFISH-C57BL6J-638850-sections/20230630/{sec}-log2.h5ad",
                f"merfish_allen/{sec}-log2.h5ad",
                role=f"log2 CPV expression for coronal section {sec}",
                size=0,
            )
        )
    files.append(
        RemoteFile(
            f"{_ABC}/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-log2.h5ad",
            "merfish_allen/whole-brain-log2.h5ad",
            role="whole-brain MERFISH matrix, all 59 sections (7.6 GB)",
            size=7_627_600_000,
            optional=True,
        )
    )
    return files


MERFISH_ALLEN = DatasetSpec(
    key="merfish_allen",
    title="Allen Brain Cell Atlas -- Whole Mouse Brain MERFISH (C57BL6J-638850)",
    technology="MERFISH, 500-gene panel, single-cell resolution",
    organism="Mus musculus",
    tissue="Whole adult brain, serial coronal sections",
    citation="Yao et al., Nature 2023, 'A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain'",
    accession="ABC Atlas MERFISH-C57BL6J-638850 (release 20230830)",
    license="Allen Institute Terms of Use (CC BY 4.0 style, attribution required)",
    notes=(
        "We download three coronal sections rather than the 7.6 GB whole-brain "
        "matrix. Coordinates live in cell_metadata.csv and are joined onto the "
        "per-section expression matrices by cell_label."
    ),
    files=_merfish_files(),
)


# --------------------------------------------------------------------------- #
# Dataset 4 -- 10x Xenium mouse brain (single-cell spatial validation)
# --------------------------------------------------------------------------- #

_XEN = f"{_TENX_XENIUM}/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP/Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP"

XENIUM_MOUSE_BRAIN = DatasetSpec(
    key="xenium_mouse_brain",
    title="10x Genomics Xenium -- Fresh Frozen Mouse Brain, Coronal (CTX + HP subset)",
    technology="Xenium In Situ, 248-gene panel, sub-cellular resolution",
    organism="Mus musculus",
    tissue="Coronal brain section, cortex and hippocampus",
    citation="10x Genomics (2023), Xenium Analyzer 1.0.2 demonstration dataset",
    accession="10x Xenium_V1_FF_Mouse_Brain_Coronal_Subset_CTX_HP",
    license="10x Genomics public dataset terms (free for research use)",
    notes=(
        "We take the cell-level feature matrix and segmentation centroids "
        "rather than the 3.5 GB outs bundle. The 384 MB transcript table "
        "(molecule-level x/y) is optional and only used for the "
        "sub-spot-resolution supplementary analysis."
    ),
    files=[
        RemoteFile(
            f"{_XEN}_cell_feature_matrix.tar.gz",
            "xenium_mouse_brain/cell_feature_matrix.tar.gz",
            role="cell x gene count matrix (MatrixMarket)",
            size=12_005_536,
            extract=True,
        ),
        RemoteFile(
            f"{_XEN}_cells.csv.gz",
            "xenium_mouse_brain/cells.csv.gz",
            role="cell centroids, area, transcript counts (segmentation output)",
            size=1_741_073,
        ),
        RemoteFile(
            f"{_XEN}_analysis.tar.gz",
            "xenium_mouse_brain/analysis.tar.gz",
            role="Xenium graph clustering (reference annotation)",
            size=5_711_161,
            extract=True,
        ),
        RemoteFile(
            f"{_XEN}_cell_boundaries.csv.gz",
            "xenium_mouse_brain/cell_boundaries.csv.gz",
            role="cell segmentation polygons",
            size=4_854_037,
        ),
        RemoteFile(
            f"{_TENX_XENIUM}/Xenium_V1_FF_Mouse_Brain_Coronal_Input/Xenium_V1_FF_Mouse_Brain_Coronal_Input_gene_panel.json",
            "xenium_mouse_brain/gene_panel.json",
            role="gene panel definition",
            size=128_715,
        ),
        RemoteFile(
            f"{_XEN}_transcripts.csv.gz",
            "xenium_mouse_brain/transcripts.csv.gz",
            role="molecule-level transcript coordinates (384 MB)",
            size=384_204_555,
            optional=True,
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Dataset 5 -- Stereo-seq mouse embryo (developmental morphogen validation)
# --------------------------------------------------------------------------- #

MOSTA_EMBRYO = DatasetSpec(
    key="mosta_embryo",
    title="MOSTA -- Mouse Organogenesis Spatiotemporal Transcriptomic Atlas (Stereo-seq)",
    technology="Stereo-seq, DNA nanoball arrays, bin50 aggregation",
    organism="Mus musculus",
    tissue="Whole embryo, sagittal sections, E9.5 / E10.5 / E11.5 / E12.5",
    citation="Chen et al., Cell 2022, 'Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays'",
    accession="CNGB STDS0000058",
    license="CNGB / STOMICS open data terms (free for research use, cite source)",
    notes=(
        "This is the only dataset in the project with a genuine temporal axis. "
        "Four consecutive developmental stages let us evaluate the learned "
        "operator as a forward-time model rather than only as a steady-state "
        "relaxation operator."
    ),
    files=[
        RemoteFile(
            f"{_MOSTA}/{stage}.MOSTA.h5ad",
            f"mosta_embryo/{stage}.MOSTA.h5ad",
            role=f"binned expression + spatial coordinates, stage {stage.split('_')[0]}",
            size=0,
        )
        for stage in MOSTA_STAGES
    ],
)


# --------------------------------------------------------------------------- #
# Dataset 6 -- Norman et al. 2019 Perturb-seq (counterfactual validation)
# --------------------------------------------------------------------------- #

PERTURB_NORMAN = DatasetSpec(
    key="perturb_norman",
    title="Norman et al. 2019 -- CRISPRa Perturb-seq (single and combinatorial)",
    technology="Perturb-seq (CRISPRa), 10x 3' scRNA-seq",
    organism="Homo sapiens (K562 erythroleukemia)",
    tissue="Cell line (non-spatial)",
    citation="Norman et al., Science 2019, 'Exploring genetic interaction manifolds constructed from rich single-cell phenotypes'",
    accession="GEO GSE133344",
    license="GEO public",
    notes=(
        "Non-spatial and from a human cell line, so it is used strictly as an "
        "ORTHOGONAL, out-of-context test of whether the learned reaction "
        "Jacobian encodes real transcriptional coupling -- never as in-domain "
        "ground truth. This mismatch is stated explicitly in the paper."
    ),
    files=[
        RemoteFile(
            f"{_GEO133344}/GSE133344_filtered_matrix.mtx.gz",
            "perturb_norman/matrix.mtx.gz",
            role="cell x gene counts (MatrixMarket)",
            size=0,
        ),
        RemoteFile(
            f"{_GEO133344}/GSE133344_filtered_genes.tsv.gz",
            "perturb_norman/genes.tsv.gz",
            role="gene identifiers",
            size=0,
        ),
        RemoteFile(
            f"{_GEO133344}/GSE133344_filtered_barcodes.tsv.gz",
            "perturb_norman/barcodes.tsv.gz",
            role="cell barcodes",
            size=0,
        ),
        RemoteFile(
            f"{_GEO133344}/GSE133344_filtered_cell_identities.csv.gz",
            "perturb_norman/cell_identities.csv.gz",
            role="guide assignment per cell (the perturbation labels)",
            size=0,
        ),
    ],
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Datasets 7-9 -- additional INDEPENDENT Visium specimens.
#
# The benchmark's binding constraint is not the number of sections but the
# number of independent biological specimens: twelve of the seventeen sections
# are serial sections of one brain, which leaves five specimens and a smallest
# attainable Wilcoxon p of 0.0625. Against the graph baselines the separation
# does not resolve at that sample size, and no amount of additional sectioning
# of the same tissue would change it.
#
# These three are distinct organs from distinct donors, so each adds a genuinely
# independent unit of analysis. They are the cheapest available fix for the
# limitation the paper identifies as its most important open question. The
# full-resolution microscopy images are deliberately not fetched: the pipeline
# consumes count matrices, and the images are the bulk of the download.
# --------------------------------------------------------------------------- #


def _visium_11(key: str, sample: str, title: str, tissue: str,
               organism: str) -> DatasetSpec:
    """A Visium 1.1.0 public dataset, count matrix and spatial metadata only."""
    base = f"https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/{sample}"
    return DatasetSpec(
        key=key,
        title=title,
        technology="Visium (55 um spots, 100 um centre-to-centre, hexagonal lattice)",
        organism=organism,
        tissue=tissue,
        citation="10x Genomics, Space Ranger 1.1.0 demonstration dataset",
        accession=f"10x {sample}",
        license="10x Genomics public dataset terms (free for research use)",
        notes=("Added to increase the number of INDEPENDENT specimens in the "
               "benchmark, which is what limits its statistical resolution. "
               "Microscopy images are not downloaded."),
        files=[
            RemoteFile(f"{base}/{sample}_filtered_feature_bc_matrix.h5",
                       f"{key}/filtered_feature_bc_matrix.h5",
                       role="count matrix (spots x genes)"),
            RemoteFile(f"{base}/{sample}_spatial.tar.gz",
                       f"{key}/spatial.tar.gz",
                       role="spot coordinates and scale factors", extract=True),
            RemoteFile(f"{base}/{sample}_analysis.tar.gz",
                       f"{key}/analysis.tar.gz",
                       role="Space Ranger clustering (reference annotation only)",
                       extract=True),
        ],
    )


VISIUM_MOUSE_KIDNEY = _visium_11(
    "visium_mouse_kidney", "V1_Mouse_Kidney",
    "10x Genomics Visium -- Adult Mouse Kidney (coronal)",
    "Whole adult kidney, coronal section", "Mus musculus")

VISIUM_HUMAN_LYMPH = _visium_11(
    "visium_human_lymph_node", "V1_Human_Lymph_Node",
    "10x Genomics Visium -- Human Lymph Node",
    "Whole lymph node", "Homo sapiens")

VISIUM_MOUSE_BRAIN_CORONAL = _visium_11(
    "visium_mouse_brain_coronal", "V1_Adult_Mouse_Brain_Coronal_Section_1",
    "10x Genomics Visium -- Adult Mouse Brain (coronal, separate animal)",
    "Whole adult brain, coronal section", "Mus musculus")


VISIUM_HUMAN_HEART = _visium_11(
    "visium_human_heart", "V1_Human_Heart",
    "10x Genomics Visium -- Human Heart",
    "Left ventricle, transverse section", "Homo sapiens")



def _visium_13(key: str, sample: str, title: str, tissue: str,
               organism: str) -> DatasetSpec:
    """A Visium 1.3.0 FFPE public dataset, count matrix and spatial metadata."""
    base = f"{_TENX_SPATIAL}/{sample}"
    return DatasetSpec(
        key=key, title=title,
        technology="Visium FFPE (55 um spots, 100 um centre-to-centre)",
        organism=organism, tissue=tissue,
        citation="10x Genomics, Space Ranger 1.3.0 demonstration dataset",
        accession=f"10x {sample}",
        license="10x Genomics public dataset terms (free for research use)",
        notes=("Independent specimen. FFPE chemistry differs from the "
               "fresh-frozen sections, which adds assay diversity within Visium "
               "rather than confounding it: one QC policy is applied throughout."),
        files=[
            RemoteFile(f"{base}/{sample}_filtered_feature_bc_matrix.h5",
                       f"{key}/filtered_feature_bc_matrix.h5",
                       role="count matrix (spots x genes)"),
            RemoteFile(f"{base}/{sample}_spatial.tar.gz", f"{key}/spatial.tar.gz",
                       role="spot coordinates and scale factors", extract=True),
            RemoteFile(f"{base}/{sample}_analysis.tar.gz", f"{key}/analysis.tar.gz",
                       role="Space Ranger clustering (reference annotation only)",
                       extract=True),
        ],
    )


VISIUM_FFPE_PROSTATE = _visium_13(
    "visium_ffpe_human_prostate", "Visium_FFPE_Human_Prostate_Cancer",
    "10x Genomics Visium FFPE -- Human Prostate (adenocarcinoma)",
    "Prostate, acinar cell carcinoma", "Homo sapiens")

VISIUM_FFPE_MOUSE_BRAIN = _visium_13(
    "visium_ffpe_mouse_brain", "Visium_FFPE_Mouse_Brain",
    "10x Genomics Visium FFPE -- Adult Mouse Brain (separate animal)",
    "Whole adult brain, coronal section", "Mus musculus")



DATASETS: Dict[str, DatasetSpec] = {
    d.key: d
    for d in [
        VISIUM_MOUSE_BRAIN,
        VISIUM_HUMAN_BREAST,
        MERFISH_ALLEN,
        XENIUM_MOUSE_BRAIN,
        MOSTA_EMBRYO,
        PERTURB_NORMAN,
        VISIUM_MOUSE_KIDNEY,
        VISIUM_HUMAN_LYMPH,
        VISIUM_MOUSE_BRAIN_CORONAL,
        VISIUM_HUMAN_HEART,
        VISIUM_FFPE_PROSTATE,
        VISIUM_FFPE_MOUSE_BRAIN,
    ]
}

#: Datasets fetched by ``make data`` / ``python -m src.data.download --all``.
DEFAULT_DATASETS: List[str] = list(DATASETS.keys())


def get(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset '{key}'. Available: {sorted(DATASETS)}")
    return DATASETS[key]


def summary_table() -> str:
    """Human-readable inventory, printed by ``download.py --list``."""
    rows = [
        f"{'key':<22} {'technology':<34} {'organism':<16} {'#files':>6} {'req. GB':>8}",
        "-" * 92,
    ]
    for d in DATASETS.values():
        gb = sum(f.size for f in d.required_files) / 1e9
        rows.append(
            f"{d.key:<22} {d.technology[:33]:<34} {d.organism[:15]:<16} "
            f"{len(d.required_files):>6} {gb:>8.2f}"
        )
    return "\n".join(rows)

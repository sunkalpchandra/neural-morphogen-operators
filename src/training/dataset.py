"""Tensorised spatial sections and the masking strategies used by experiments.

A "section" is one tissue slice: coordinates, expression, a k-NN graph and a
split assignment. Sections are small enough (<= ~50k locations) to live entirely
on device, so we avoid a DataLoader and its shuffling overhead entirely -- the
model consumes the whole field at once, which is the correct granularity for an
operator that is defined on the field rather than on individual spots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import torch

from ..models.layers import knn_graph
from ..utils.common import get_logger

log = get_logger("nmo.data")


# --------------------------------------------------------------------------- #
# Section container
# --------------------------------------------------------------------------- #


@dataclass
class SpatialSection:
    name: str
    coords: torch.Tensor        # (N, 2) in [-1, 1]
    expr: torch.Tensor          # (N, G) standardised
    gene_names: List[str]
    split: np.ndarray           # (N,) of {'train','val','test'}
    edge_index: torch.Tensor    # (2, E)
    gene_mean: torch.Tensor     # (G,)
    gene_std: torch.Tensor      # (G,)
    coord_scale_um: float = 1.0
    meta: Dict = None

    # -- convenience -------------------------------------------------------- #

    @property
    def n_obs(self) -> int:
        return self.coords.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expr.shape[1]

    def mask(self, which: str | Sequence[str]) -> torch.Tensor:
        """Float mask over locations belonging to the given split(s)."""
        which = [which] if isinstance(which, str) else list(which)
        m = np.isin(self.split, which).astype(np.float32)
        return torch.from_numpy(m).to(self.coords.device)

    def to(self, device: torch.device) -> "SpatialSection":
        self.coords = self.coords.to(device)
        self.expr = self.expr.to(device)
        self.edge_index = self.edge_index.to(device)
        self.gene_mean = self.gene_mean.to(device)
        self.gene_std = self.gene_std.to(device)
        return self

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gene_std + self.gene_mean

    def numpy_expr(self, denorm: bool = True) -> np.ndarray:
        e = self.denormalise(self.expr) if denorm else self.expr
        return e.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_section(
    path: str | Path,
    device: torch.device | str = "cpu",
    knn_k: int = 8,
    gene_subset: Optional[Sequence[str]] = None,
    standardise: bool = True,
    stats_from: str = "train",
) -> SpatialSection:
    """Read a processed ``.h5ad`` into a ``SpatialSection``.

    Gene standardisation statistics are computed on the **training split only**
    (``stats_from='train'``) so that held-out regions cannot influence the
    normalisation -- a subtle leak that is easy to miss and would inflate every
    reported metric.
    """
    path = Path(path)
    a = ad.read_h5ad(path)

    if gene_subset is not None:
        keep = [g for g in gene_subset if g in set(a.var_names)]
        if not keep:
            raise ValueError(f"none of the requested genes are present in {path.name}")
        a = a[:, keep].copy()

    X = np.asarray(a.X, dtype=np.float32)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)
    split = (
        a.obs["split"].to_numpy().astype(str)
        if "split" in a.obs.columns
        else np.array(["train"] * a.n_obs)
    )

    if standardise:
        ref = X[split == stats_from] if (split == stats_from).any() else X
        mu = ref.mean(0)
        sd = ref.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Xs = (X - mu) / sd
    else:
        mu = np.zeros(X.shape[1], dtype=np.float32)
        sd = np.ones(X.shape[1], dtype=np.float32)
        Xs = X

    c = torch.from_numpy(coords)
    ei = knn_graph(c, knn_k)

    sec = SpatialSection(
        name=path.stem,
        coords=c,
        expr=torch.from_numpy(np.ascontiguousarray(Xs, dtype=np.float32)),
        gene_names=[str(v) for v in a.var_names],
        split=split,
        edge_index=ei,
        gene_mean=torch.from_numpy(mu.astype(np.float32)),
        gene_std=torch.from_numpy(sd.astype(np.float32)),
        coord_scale_um=float(a.uns.get("nmo", {}).get("coord_scale_um", 1.0)),
        meta=dict(a.uns.get("nmo", {})),
    )
    log.info(
        f"section {sec.name}: {sec.n_obs} locations x {sec.n_genes} genes, "
        f"{ei.shape[1]} edges, scale {sec.coord_scale_um:.0f} um/unit"
    )
    return sec.to(torch.device(device))


# --------------------------------------------------------------------------- #
# Masking strategies
# --------------------------------------------------------------------------- #


def block_mask(section: SpatialSection, visible: Sequence[str] = ("train",)) -> torch.Tensor:
    """Visibility mask taken from the precomputed contiguous block split."""
    return section.mask(list(visible))


def disk_mask(
    section: SpatialSection, n_disks: int = 6, radius: float = 0.18, seed: int = 0
) -> torch.Tensor:
    """Occlude ``n_disks`` circular regions -- a harder, geometry-controlled hole.

    Radius is in normalised coordinate units, so 0.18 with a 3.2 mm half-extent
    is roughly a 570 um hole: several times the expected morphogen decay length,
    meaning the model cannot succeed by local interpolation alone.
    """
    rng = np.random.default_rng(seed)
    xy = section.coords.detach().cpu().numpy()
    keep = np.ones(len(xy), dtype=np.float32)
    lo, hi = xy.min(0), xy.max(0)
    for _ in range(n_disks):
        c = rng.uniform(lo, hi)
        d = np.linalg.norm(xy - c, axis=1)
        keep[d < radius] = 0.0
    return torch.from_numpy(keep).to(section.coords.device)


def stripe_mask(section: SpatialSection, n_stripes: int = 3, width: float = 0.12, axis: int = 0) -> torch.Tensor:
    xy = section.coords.detach().cpu().numpy()
    keep = np.ones(len(xy), dtype=np.float32)
    lo, hi = xy[:, axis].min(), xy[:, axis].max()
    centres = np.linspace(lo, hi, n_stripes + 2)[1:-1]
    for c in centres:
        keep[np.abs(xy[:, axis] - c) < width / 2] = 0.0
    return torch.from_numpy(keep).to(section.coords.device)


MASK_FNS = {"block": block_mask, "disk": disk_mask, "stripe": stripe_mask}


def make_mask(section: SpatialSection, kind: str = "block", **kw) -> torch.Tensor:
    if kind not in MASK_FNS:
        raise KeyError(f"unknown mask kind {kind!r}; have {sorted(MASK_FNS)}")
    return MASK_FNS[kind](section, **kw)


# --------------------------------------------------------------------------- #
# Gene-space alignment across sections (cross-tissue / cross-platform)
# --------------------------------------------------------------------------- #


def align_sections(
    a: SpatialSection, b: SpatialSection, max_genes: Optional[int] = None
) -> Tuple[SpatialSection, SpatialSection, List[str]]:
    """Restrict two sections to their shared (upper-cased) gene vocabulary.

    Used for mouse->human transfer and Visium->MERFISH/Xenium transfer. Ordering
    is made identical so a single decoder head applies to both.
    """
    ua = {g.upper(): i for i, g in enumerate(a.gene_names)}
    ub = {g.upper(): i for i, g in enumerate(b.gene_names)}
    shared = sorted(set(ua) & set(ub))
    if not shared:
        raise RuntimeError(f"no shared genes between {a.name} and {b.name}")
    if max_genes is not None and len(shared) > max_genes:
        # keep the most variable shared genes, measured on the source section
        var = a.expr.var(0).detach().cpu().numpy()
        order = sorted(shared, key=lambda g: -var[ua[g]])
        shared = sorted(order[:max_genes])

    ia = torch.tensor([ua[g] for g in shared], dtype=torch.long)
    ib = torch.tensor([ub[g] for g in shared], dtype=torch.long)

    def _sub(sec: SpatialSection, idx: torch.Tensor) -> SpatialSection:
        idx = idx.to(sec.expr.device)
        return SpatialSection(
            name=sec.name, coords=sec.coords, expr=sec.expr[:, idx],
            gene_names=shared, split=sec.split, edge_index=sec.edge_index,
            gene_mean=sec.gene_mean[idx], gene_std=sec.gene_std[idx],
            coord_scale_um=sec.coord_scale_um, meta=sec.meta,
        )

    log.info(f"aligned {a.name} <-> {b.name}: {len(shared)} shared genes")
    return _sub(a, ia), _sub(b, ib), shared


def subsample_section(section: SpatialSection, n: int, seed: int = 0) -> SpatialSection:
    """Uniformly subsample locations (used to fit large Xenium/MERFISH sections)."""
    if section.n_obs <= n:
        return section
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(section.n_obs, n, replace=False))
    t = torch.from_numpy(idx).to(section.coords.device)
    coords = section.coords[t]
    return SpatialSection(
        name=section.name, coords=coords, expr=section.expr[t],
        gene_names=section.gene_names, split=section.split[idx],
        edge_index=knn_graph(coords, 8),
        gene_mean=section.gene_mean, gene_std=section.gene_std,
        coord_scale_um=section.coord_scale_um, meta=section.meta,
    )

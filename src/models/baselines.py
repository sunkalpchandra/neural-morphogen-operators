"""Baselines, implemented against the identical interface as the NMO.

Every baseline exposes ``forward(coords, expr, query_coords, edge_index,
point_mask) -> {'pred': ...}`` so the *same* training loop, masking strategy,
optimiser schedule and evaluation code drives all of them. Any difference in
reported numbers is therefore attributable to the model, not to the harness.

On faithfulness to the published methods
----------------------------------------
SpaGCN and STAGATE were designed for spatial-domain identification and
denoising, not for predicting expression at unobserved coordinates. Rather than
report a strawman, we implement the *architectural core* of each published
method -- SpaGCN's exponentially-weighted spatial adjacency with graph
convolution, STAGATE's spatially-regularised graph attention autoencoder -- and
attach the same read-out head used by every other baseline. These are marked
``-style`` in the paper and in the table captions. We do not claim they
reproduce the authors' published numbers on the authors' own tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP, FourierFeatures, knn_graph


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[1], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    cnt.index_add_(0, index, torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp(min=1.0)


def _interp_from_points(
    src_coords: torch.Tensor, src_vals: torch.Tensor, query: torch.Tensor, k: int = 8, p: float = 2.0
) -> torch.Tensor:
    """Inverse-distance-weighted read-out at arbitrary query coordinates.

    Discrete graph models have no notion of a value at an unmeasured location,
    so to evaluate them on held-out coordinates at all we must interpolate. We
    give every graph baseline this same IDW head, which is the standard and
    most favourable choice for them.
    """
    d = torch.cdist(query, src_coords)
    k = min(k, src_coords.shape[0])
    dist, idx = d.topk(k, largest=False)
    w = 1.0 / dist.clamp_min(1e-6).pow(p)
    w = w / w.sum(1, keepdim=True)
    return (src_vals[idx] * w.unsqueeze(-1)).sum(1)


# --------------------------------------------------------------------------- #
# 1. Simple autoencoder (no spatial information at all)
# --------------------------------------------------------------------------- #


class AutoencoderBaseline(nn.Module):
    """Non-spatial MLP autoencoder. Establishes the floor: what is achievable
    from expression statistics alone, ignoring geometry."""

    def __init__(self, n_genes: int, latent: int = 32, hidden: int = 256, **kw):
        super().__init__()
        self.enc = MLP(n_genes, hidden, latent, n_layers=3)
        self.dec = MLP(latent, hidden, n_genes, n_layers=3)

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        z = self.enc(expr)
        if point_mask is not None and q.shape[0] != 0:
            vis = point_mask > 0
            z_q = _interp_from_points(coords[vis], z[vis], q)
        else:
            z_q = z
        return {"pred": self.dec(z_q), "latent": z}


# --------------------------------------------------------------------------- #
# 2. Graph neural network (GCN)
# --------------------------------------------------------------------------- #


class GCNLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x, edge_index, edge_weight=None):
        src, dst = edge_index[0], edge_index[1]
        m = x[src] if edge_weight is None else x[src] * edge_weight.unsqueeze(1)
        agg = _scatter_mean(m, dst, x.shape[0])
        return self.norm(self.lin(x + agg))


class GNNBaseline(nn.Module):
    """Standard k-NN graph convolutional network over spots."""

    def __init__(self, n_genes: int, hidden: int = 128, latent: int = 32, n_layers: int = 3, **kw):
        super().__init__()
        self.inp = nn.Linear(n_genes, hidden)
        self.layers = nn.ModuleList([GCNLayer(hidden, hidden) for _ in range(n_layers)])
        self.to_latent = nn.Linear(hidden, latent)
        self.dec = MLP(latent, hidden, n_genes, n_layers=3)

    def _embed(self, expr, edge_index, point_mask=None):
        h = self.inp(expr)
        if point_mask is not None:
            h = h * point_mask.view(-1, 1)
            keep = point_mask[edge_index[0]] > 0
            edge_index = edge_index[:, keep]
        for l in self.layers:
            h = F.gelu(l(h, edge_index))
        return self.to_latent(h)

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        if edge_index is None:
            edge_index = knn_graph(coords, 8)
        z = self._embed(expr, edge_index, point_mask)
        vis = (point_mask > 0) if point_mask is not None else torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        z_q = _interp_from_points(coords[vis], z[vis], q)
        return {"pred": self.dec(z_q), "latent": z}


# --------------------------------------------------------------------------- #
# 3. SpaGCN-style
# --------------------------------------------------------------------------- #


class SpaGCNStyleBaseline(nn.Module):
    """SpaGCN's core idea: a dense adjacency weighted by ``exp(-d^2 / 2 l^2)``.

    The published method also fuses histology-derived colour features to modulate
    the adjacency; we omit that, since three of our five spatial datasets have no
    paired H&E image and including it for only some would confound the
    comparison. The bandwidth ``l`` is learnable here, matching SpaGCN's
    ``l``-search procedure in spirit.
    """

    def __init__(self, n_genes: int, hidden: int = 128, latent: int = 32, n_layers: int = 2,
                 n_neighbors: int = 24, **kw):
        super().__init__()
        self.inp = nn.Linear(n_genes, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.to_latent = nn.Linear(hidden, latent)
        self.dec = MLP(latent, hidden, n_genes, n_layers=3)
        self.log_l = nn.Parameter(torch.tensor(math.log(0.05)))
        self.k = n_neighbors

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        vis = (point_mask > 0) if point_mask is not None else torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        c_v, e_v = coords[vis], expr[vis]

        d = torch.cdist(c_v, c_v)
        k = min(self.k, c_v.shape[0])
        dist, idx = d.topk(k, largest=False)
        l = self.log_l.exp().clamp(1e-3, 2.0)
        w = torch.exp(-dist.pow(2) / (2 * l**2))
        w = w / w.sum(1, keepdim=True).clamp_min(1e-8)

        h = self.inp(e_v)
        for lin, nrm in zip(self.layers, self.norms):
            agg = (h[idx] * w.unsqueeze(-1)).sum(1)
            h = F.gelu(nrm(lin(agg) + h))
        z = self.to_latent(h)
        return {"pred": self.dec(_interp_from_points(c_v, z, q)), "latent": z}


# --------------------------------------------------------------------------- #
# 4. STAGATE-style
# --------------------------------------------------------------------------- #


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, heads: int = 4):
        super().__init__()
        self.h, self.dk = heads, d_out // heads
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)
        self.o = nn.Linear(d_out, d_out)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        Q = self.q(x).view(-1, self.h, self.dk)
        K = self.k(x).view(-1, self.h, self.dk)
        V = self.v(x).view(-1, self.h, self.dk)
        score = (Q[dst] * K[src]).sum(-1) / math.sqrt(self.dk)  # (E, H)
        score = score - score.max()
        w = score.exp()
        denom = torch.zeros(x.shape[0], self.h, device=x.device, dtype=x.dtype)
        denom.index_add_(0, dst, w)
        alpha = w / denom[dst].clamp_min(1e-8)
        msg = (V[src] * alpha.unsqueeze(-1)).reshape(-1, self.h * self.dk)
        out = torch.zeros(x.shape[0], self.h * self.dk, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        return self.o(out)


class STAGATEStyleBaseline(nn.Module):
    """STAGATE's core: a graph *attention* autoencoder on the spatial graph,
    with tied encoder/decoder depth and a reconstruction objective."""

    def __init__(self, n_genes: int, hidden: int = 128, latent: int = 32, heads: int = 4, **kw):
        super().__init__()
        self.inp = nn.Linear(n_genes, hidden)
        self.att1 = GraphAttentionLayer(hidden, hidden, heads)
        self.att2 = GraphAttentionLayer(hidden, hidden, heads)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.to_latent = nn.Linear(hidden, latent)
        self.dec = MLP(latent, hidden, n_genes, n_layers=3)

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        if edge_index is None:
            edge_index = knn_graph(coords, 8)
        if point_mask is not None:
            keep = point_mask[edge_index[0]] > 0
            edge_index = edge_index[:, keep]
            expr = expr * point_mask.view(-1, 1)
        h = self.inp(expr)
        h = F.gelu(self.norm1(self.att1(h, edge_index) + h))
        h = F.gelu(self.norm2(self.att2(h, edge_index) + h))
        z = self.to_latent(h)
        vis = (point_mask > 0) if point_mask is not None else torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        return {"pred": self.dec(_interp_from_points(coords[vis], z[vis], q)), "latent": z}


# --------------------------------------------------------------------------- #
# 5. Graph Transformer
# --------------------------------------------------------------------------- #


class GraphTransformerBaseline(nn.Module):
    """Transformer over spots with a learned spatial (relative-position) bias.

    Full attention is O(N^2); for the large single-cell sections we attend over
    a sampled neighbourhood of ``n_neighbors`` instead, which is the standard
    sparse-attention treatment.
    """

    def __init__(self, n_genes: int, hidden: int = 128, latent: int = 32, heads: int = 4,
                 n_layers: int = 2, n_neighbors: int = 24, **kw):
        super().__init__()
        self.k = n_neighbors
        self.inp = nn.Linear(n_genes, hidden)
        self.pos = FourierFeatures(2, 16, sigma=3.0)
        self.pos_proj = nn.Linear(self.pos.dim_out, hidden)
        self.blocks = nn.ModuleList(
            [GraphAttentionLayer(hidden, hidden, heads) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.ff = nn.ModuleList([MLP(hidden, hidden * 2, hidden, n_layers=2) for _ in range(n_layers)])
        self.to_latent = nn.Linear(hidden, latent)
        self.dec = MLP(latent, hidden, n_genes, n_layers=3)

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        if edge_index is None:
            edge_index = knn_graph(coords, self.k)
        if point_mask is not None:
            keep = point_mask[edge_index[0]] > 0
            edge_index = edge_index[:, keep]
            expr = expr * point_mask.view(-1, 1)
        h = self.inp(expr) + self.pos_proj(self.pos(coords))
        for att, nrm, ff in zip(self.blocks, self.norms, self.ff):
            h = nrm(att(h, edge_index) + h)
            h = h + ff(h)
        z = self.to_latent(h)
        vis = (point_mask > 0) if point_mask is not None else torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        return {"pred": self.dec(_interp_from_points(coords[vis], z[vis], q)), "latent": z}


# --------------------------------------------------------------------------- #
# 6. Gaussian-process spatial model
# --------------------------------------------------------------------------- #


class GPSpatialBaseline(nn.Module):
    """Exact GP regression per gene with a shared learnable RBF kernel.

    This is the classical geostatistical answer to "predict a field at unmeasured
    locations", and is a genuinely strong baseline for smooth genes -- it is the
    right comparison for asking whether the PDE structure buys anything beyond
    smooth spatial interpolation.

    Implemented in closed form (Cholesky of the training covariance) rather than
    variationally; the sections here are small enough for exact inference after
    inducing-point subsampling.
    """

    def __init__(self, n_genes: int, n_inducing: int = 1024, jitter: float = 1e-3, **kw):
        super().__init__()
        self.n_inducing, self.jitter = n_inducing, jitter
        self.log_ls = nn.Parameter(torch.tensor(math.log(0.08)))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_noise = nn.Parameter(torch.tensor(math.log(0.3)))

    def _k(self, a, b):
        d2 = torch.cdist(a, b).pow(2)
        ls = self.log_ls.exp().clamp(1e-3, 5.0)
        return self.log_amp.exp().clamp(1e-3, 100.0) * torch.exp(-0.5 * d2 / ls**2)

    def forward(self, coords, expr, query_coords=None, edge_index=None, point_mask=None, **kw):
        q = coords if query_coords is None else query_coords
        vis = (point_mask > 0) if point_mask is not None else torch.ones(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
        cv, ev = coords[vis], expr[vis]
        if cv.shape[0] > self.n_inducing:
            sel = torch.randperm(cv.shape[0], device=cv.device)[: self.n_inducing]
            cv, ev = cv[sel], ev[sel]

        K = self._k(cv, cv)
        noise = self.log_noise.exp().clamp(1e-3, 10.0) ** 2 + self.jitter
        K = K + noise * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(ev, L)              # (M, G)
        pred = self._k(q, cv) @ alpha
        return {"pred": pred, "latent": None}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

BASELINES = {
    "autoencoder": AutoencoderBaseline,
    "gnn": GNNBaseline,
    "spagcn": SpaGCNStyleBaseline,
    "stagate": STAGATEStyleBaseline,
    "graph_transformer": GraphTransformerBaseline,
    "gp": GPSpatialBaseline,
}

DISPLAY_NAMES = {
    "autoencoder": "Autoencoder (non-spatial)",
    "gnn": "GNN (GCN)",
    "spagcn": "SpaGCN-style",
    "stagate": "STAGATE-style",
    "graph_transformer": "Graph Transformer",
    "gp": "Gaussian Process",
    "nmo": "NMO (ours)",
}


def build_baseline(name: str, n_genes: int, **kw) -> nn.Module:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; have {sorted(BASELINES)}")
    return BASELINES[name](n_genes=n_genes, **kw)

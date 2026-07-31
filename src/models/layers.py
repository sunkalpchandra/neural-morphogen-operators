"""Reusable primitives: scatter/gather between irregular points and a grid,
Fourier features, spectral convolution, and a graph kernel-integral layer.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        n_layers: int = 2,
        act: str = "gelu",
        dropout: float = 0.0,
        final_act: bool = False,
    ):
        super().__init__()
        A = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[act]
        dims = [dim_in] + [dim_hidden] * max(n_layers - 1, 0) + [dim_out]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or final_act:
                layers.append(A())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Fourier features
# --------------------------------------------------------------------------- #


class FourierFeatures(nn.Module):
    """Random Fourier positional encoding gamma(p) = [sin(2pi B p), cos(2pi B p)].

    Coordinate MLPs are spectrally biased towards low frequencies; the random
    Fourier lift (Tancik et al. 2020) restores the ability to represent sharp
    tissue boundaries.
    """

    def __init__(self, dim_in: int = 2, n_features: int = 32, sigma: float = 4.0, learnable: bool = False):
        super().__init__()
        B = torch.randn(dim_in, n_features) * sigma
        self.B = nn.Parameter(B, requires_grad=learnable)
        self.dim_out = 2 * n_features

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (p @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# --------------------------------------------------------------------------- #
# Point <-> grid transfer
# --------------------------------------------------------------------------- #


def normalise_to_grid(coords: torch.Tensor) -> torch.Tensor:
    """Map coordinates in [-1, 1]^2 to continuous grid index space is handled by
    ``grid_sample``; this helper only guards the range."""
    return coords.clamp(-1.0, 1.0)


class GaussianSplat(nn.Module):
    """Scatter values at irregular points onto a regular grid (Nadaraya-Watson).

    For grid node u and points p_i:

        F(u) = sum_i w_iu v_i / (sum_i w_iu + eps),   w_iu = exp(-||u - p_i||^2 / 2 s^2)

    Implemented with a bounded neighbourhood: each point contributes only to
    grid nodes within ``radius`` cells, which keeps the cost O(N r^2) instead of
    O(N H W). The accompanying occupancy map (sum of weights) is returned so the
    dynamics module knows where the tissue actually has support -- essential for
    the masked-inpainting task, where whole regions carry no observations.
    """

    def __init__(self, grid_size: int = 64, sigma_cells: float = 1.0, radius: int = 2):
        super().__init__()
        self.grid_size = grid_size
        self.radius = radius
        # log-parameterised so sigma stays positive under gradient descent
        self.log_sigma = nn.Parameter(torch.tensor(float(math.log(sigma_cells))))

    def forward(
        self, coords: torch.Tensor, values: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        coords : (N, 2) in [-1, 1]
        values : (N, C)
        weights: (N,) optional per-point weight (e.g. a mask)
        returns (field (C, H, W), occupancy (1, H, W))
        """
        H = W = self.grid_size
        N, C = values.shape
        dev, dt = values.device, values.dtype
        sigma = self.log_sigma.exp().clamp(0.25, 4.0)

        # continuous grid coordinates in [0, H-1]
        gx = (coords[:, 0] * 0.5 + 0.5) * (W - 1)
        gy = (coords[:, 1] * 0.5 + 0.5) * (H - 1)

        num = torch.zeros(C, H * W, device=dev, dtype=dt)
        den = torch.zeros(1, H * W, device=dev, dtype=dt)

        r = self.radius
        base_x = gx.floor().long()
        base_y = gy.floor().long()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ix = (base_x + dx).clamp(0, W - 1)
                iy = (base_y + dy).clamp(0, H - 1)
                d2 = (gx - ix.to(dt)) ** 2 + (gy - iy.to(dt)) ** 2
                w = torch.exp(-d2 / (2 * sigma**2))
                if weights is not None:
                    w = w * weights
                flat = (iy * W + ix)  # (N,)
                num.index_add_(1, flat, (values * w.unsqueeze(1)).T)
                den.index_add_(1, flat, w.unsqueeze(0))

        field = num / (den + 1e-6)
        return field.view(C, H, W), den.view(1, H, W)


def grid_gather(field: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Bilinearly sample a (C, H, W) field at (N, 2) continuous coords in [-1,1].

    This is the operation that makes the model *continuous*: the latent field is
    defined everywhere, and predictions at arbitrary (unobserved) locations are
    obtained by interpolation rather than by graph message passing to a node
    that must already exist.
    """
    C = field.shape[0]
    g = coords.view(1, -1, 1, 2)  # (1, N, 1, 2), grid_sample expects (x, y)
    out = F.grid_sample(field.unsqueeze(0), g, mode=mode, align_corners=True, padding_mode="border")
    return out.view(C, -1).T.contiguous()  # (N, C)


# --------------------------------------------------------------------------- #
# Spectral convolution (FNO block)
# --------------------------------------------------------------------------- #


class SpectralConv2d(nn.Module):
    """Fourier-domain convolution with a learnable low-frequency multiplier.

    The operator is parameterised directly in the frequency domain and truncated
    to ``modes`` retained wavenumbers, which makes it (approximately) invariant
    to the discretisation of the underlying grid -- the property we rely on when
    transferring an operator trained on 100 um Visium spots to single-cell
    MERFISH sampling.
    """

    def __init__(self, in_ch: int, out_ch: int, modes: int = 12):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))

    @staticmethod
    def _cmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # a: (B, Cin, M, M) complex ; w: (Cin, Cout, M, M, 2)
        wc = torch.view_as_complex(w.contiguous())
        return torch.einsum("bixy,ioxy->boxy", a, wc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m = min(self.modes, H // 2, W // 2)
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=xf.dtype, device=x.device)
        out[:, :, :m, :m] = self._cmul(xf[:, :, :m, :m], self.w1[:, :, :m, :m])
        out[:, :, -m:, :m] = self._cmul(xf[:, :, -m:, :m], self.w2[:, :, :m, :m])
        return torch.fft.irfft2(out, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, ch: int, modes: int = 12, act: bool = True):
        super().__init__()
        self.spec = SpectralConv2d(ch, ch, modes)
        self.local = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spec(x) + self.local(x))) + x


# --------------------------------------------------------------------------- #
# Graph kernel-integral layer (Graph Neural Operator)
# --------------------------------------------------------------------------- #


class GraphKernelLayer(nn.Module):
    """One Nystrom-style kernel-integral layer on a k-NN graph.

        v_i <- act( W v_i + (1/|N(i)|) sum_{j in N(i)} kappa_theta(p_j - p_i) * v_j )

    ``kappa_theta`` maps a *relative* displacement to a per-channel gain, so the
    layer depends on geometry rather than on node indices. That is what allows a
    model fitted on one sampling lattice to be evaluated on another.
    """

    def __init__(self, ch: int, edge_hidden: int = 64, act: bool = True):
        super().__init__()
        self.lin = nn.Linear(ch, ch)
        # inputs: (dx, dy, r, log(1+r))
        self.kernel = MLP(4, edge_hidden, ch, n_layers=3, act="gelu")
        self.norm = nn.LayerNorm(ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, v: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """v: (N, C); coords: (N, 2); edge_index: (2, E) with row 0 = source j, row 1 = target i."""
        src, dst = edge_index[0], edge_index[1]
        d = coords[src] - coords[dst]
        r = d.norm(dim=-1, keepdim=True)
        feat = torch.cat([d, r, torch.log1p(r)], dim=-1)
        msg = self.kernel(feat) * v[src]

        agg = torch.zeros_like(v)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(v.shape[0], 1, device=v.device, dtype=v.dtype)
        deg.index_add_(0, dst, torch.ones_like(r))
        agg = agg / deg.clamp(min=1.0)

        return self.act(self.norm(self.lin(v) + agg))


def knn_graph(coords: torch.Tensor, k: int = 8, max_radius: Optional[float] = None) -> torch.Tensor:
    """Build a symmetric k-nearest-neighbour edge list. Returns (2, E)."""
    N = coords.shape[0]
    k = min(k + 1, N)
    # chunked cdist keeps memory bounded for the 36k-cell Xenium section
    chunk = max(1, int(2e7 // max(N, 1)))
    idx_all = []
    for s in range(0, N, chunk):
        d = torch.cdist(coords[s : s + chunk], coords)
        idx = d.topk(k, largest=False).indices
        idx_all.append(idx)
    idx = torch.cat(idx_all, 0)  # (N, k) includes self
    dst = torch.arange(N, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
    src = idx.reshape(-1)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if max_radius is not None:
        r = (coords[src] - coords[dst]).norm(dim=-1)
        keep = r <= max_radius
        src, dst = src[keep], dst[keep]
    # symmetrise
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
    return torch.unique(ei, dim=1)

"""Neural Morphogen Operator: encoder -> reaction-diffusion operator -> decoder.

    (p_i, g_i)  --encoder-->  z(x,y)  --operator-->  z(x,y,T)  --decoder-->  g_hat

The encoder maps irregularly sampled expression onto a *continuous* latent field
discretised on a regular grid; the operator evolves that field under a learned
reaction-diffusion PDE; the decoder reads the field out at arbitrary continuous
coordinates. Because read-out is by interpolation rather than by message passing
to an existing node, the model can be queried at locations where no measurement
exists -- which is what the masked-region and cross-resolution experiments need.

A deliberate architectural choice: **the decoder does not see coordinates.**
Everything the model knows about space must travel through the latent field and
therefore through the PDE. If we let the decoder condition on (x, y) directly it
could memorise a coordinate-to-expression map and leave the dynamics decorative,
and the ablations would no longer mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import DynamicsConfig, ReactionDiffusionOperator
from .layers import (
    FNOBlock,
    FourierFeatures,
    GaussianSplat,
    GraphKernelLayer,
    MLP,
    grid_gather,
    knn_graph,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class NMOConfig:
    n_genes: int = 2000
    latent_channels: int = 24
    grid_size: int = 64

    # encoder
    encoder: str = "gno"            # {'gno', 'fno', 'mlp'}
    gene_embed_dim: int = 128
    encoder_hidden: int = 128
    n_graph_layers: int = 3
    #: Neighbourhood size used ONLY when forward() is called without an
    #: edge_index. Every experiment in this repository supplies one, built by
    #: load_section(knn_k=...), so changing this field alone has no effect --
    #: which made an ablation of it silently test nothing. Vary the graph where
    #: it is built, not here.
    knn_k: int = 8
    n_fno_blocks: int = 2
    fno_modes: int = 12
    splat_sigma: float = 1.0
    splat_radius: int = 2

    # decoder
    decoder_hidden: int = 256
    decoder_layers: int = 3
    decoder_sees_coords: bool = False

    # dynamics
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)

    dropout: float = 0.0

    def __post_init__(self):
        if isinstance(self.dynamics, dict):
            self.dynamics = DynamicsConfig(**self.dynamics)
        # the operator must act on the same number of channels as the field
        self.dynamics.channels = self.latent_channels


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #


class SpatialEncoder(nn.Module):
    """Irregular (p_i, g_i) -> continuous latent field on an H x W grid.

    Pipeline: gene projection -> optional graph kernel-integral layers ->
    Gaussian splat onto the grid -> spectral (FNO) refinement.

    The occupancy channel produced by the splat is concatenated before the FNO
    stage so the network can distinguish "tissue with no signal" from "no
    measurement here", which is exactly the distinction the inpainting task
    hinges on.
    """

    def __init__(self, cfg: NMOConfig):
        super().__init__()
        self.cfg = cfg
        C, Ch = cfg.latent_channels, cfg.encoder_hidden

        self.gene_proj = MLP(cfg.n_genes, Ch, Ch, n_layers=2, dropout=cfg.dropout)
        self.coord_feat = FourierFeatures(2, n_features=16, sigma=3.0)
        self.mix = nn.Linear(Ch + self.coord_feat.dim_out, Ch)

        if cfg.encoder == "gno":
            self.graph_layers = nn.ModuleList(
                [GraphKernelLayer(Ch) for _ in range(cfg.n_graph_layers)]
            )
        else:
            self.graph_layers = nn.ModuleList()

        self.to_latent = nn.Linear(Ch, C)
        self.splat = GaussianSplat(cfg.grid_size, cfg.splat_sigma, cfg.splat_radius)

        # +1 for the occupancy channel
        self.grid_in = nn.Conv2d(C + 1, C, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock(C, cfg.fno_modes) for _ in range(cfg.n_fno_blocks)]
        ) if cfg.encoder in ("gno", "fno") else nn.ModuleList()
        self.grid_out = nn.Conv2d(C, C, 1)

    def forward(
        self,
        coords: torch.Tensor,          # (N, 2) in [-1, 1]
        expr: torch.Tensor,            # (N, G)
        edge_index: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,  # (N,) 1 = observed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.gene_proj(expr)
        h = self.mix(torch.cat([h, self.coord_feat(coords)], dim=-1))

        if len(self.graph_layers):
            if edge_index is None:
                edge_index = knn_graph(coords, self.cfg.knn_k)
            elif self.cfg.knn_k != 8 and not getattr(self, "_knn_warned", False):
                # Silently ignoring a non-default setting is how an ablation of
                # this field came to measure nothing at all.
                import logging
                logging.getLogger(__name__).warning(
                    "model.knn_k=%d is ignored: an edge_index was supplied. "
                    "Set knn_k on load_section() to change the graph.",
                    self.cfg.knn_k)
                self._knn_warned = True
            if point_mask is not None:
                # keep only edges whose source is observed, so masked points
                # cannot leak their own (held-out) expression into the field
                keep = point_mask[edge_index[0]] > 0
                edge_index = edge_index[:, keep]
            for gl in self.graph_layers:
                h = gl(h, coords, edge_index)

        v = self.to_latent(h)
        field, occ = self.splat(coords, v, weights=point_mask)

        x = torch.cat([field, torch.log1p(occ)], dim=0).unsqueeze(0)
        x = self.grid_in(x)
        for b in self.blocks:
            x = b(x)
        return self.grid_out(x), occ.unsqueeze(0)


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #


class FieldDecoder(nn.Module):
    """Continuous read-out: sample the latent field at coords, map to genes."""

    def __init__(self, cfg: NMOConfig):
        super().__init__()
        self.cfg = cfg
        d_in = cfg.latent_channels
        self.coord_feat = None
        if cfg.decoder_sees_coords:
            self.coord_feat = FourierFeatures(2, n_features=16, sigma=3.0)
            d_in += self.coord_feat.dim_out
        self.net = MLP(
            d_in, cfg.decoder_hidden, cfg.n_genes,
            n_layers=cfg.decoder_layers, dropout=cfg.dropout,
        )

    def forward(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        z = grid_gather(field[0], coords)  # (N, C)
        if self.coord_feat is not None:
            z = torch.cat([z, self.coord_feat(coords)], dim=-1)
        return self.net(z)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #


class NeuralMorphogenOperator(nn.Module):
    def __init__(self, cfg: NMOConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpatialEncoder(cfg)
        self.operator = ReactionDiffusionOperator(cfg.dynamics)
        self.decoder = FieldDecoder(cfg)

    # -- core ---------------------------------------------------------------- #

    def encode(
        self, coords: torch.Tensor, expr: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None, point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(coords, expr, edge_index, point_mask)

    def evolve(self, z: torch.Tensor, n_steps: Optional[int] = None, dt: Optional[float] = None,
               return_traj: bool = False):
        return self.operator(z, n_steps=n_steps, dt=dt, return_traj=return_traj)

    def decode(self, field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.decoder(field, coords)

    def forward(
        self,
        coords: torch.Tensor,
        expr: torch.Tensor,
        query_coords: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        return_fields: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Encode, relax under the learned PDE, and decode.

        ``point_mask`` marks which observations the encoder may see (0 = held
        out). ``query_coords`` defaults to all input coordinates, so held-out
        locations are still predicted -- the model must reconstruct them by
        propagating information through the field.
        """
        q = coords if query_coords is None else query_coords
        z0, occ = self.encode(coords, expr, edge_index, point_mask)
        zT = self.evolve(z0, n_steps=n_steps)

        out = {
            "pred": self.decode(zT, q),
            "pred_z0": self.decode(z0, q),
            "z0": z0,
            "zT": zT,
            "occupancy": occ,
        }
        if return_fields:
            _, traj = self.evolve(z0, n_steps=n_steps, return_traj=True)
            out["trajectory"] = traj
        return out

    # -- analysis ------------------------------------------------------------ #

    @torch.no_grad()
    def latent_fields(self, coords, expr, edge_index=None, point_mask=None) -> Dict[str, np.ndarray]:
        z0, occ = self.encode(coords, expr, edge_index, point_mask)
        zT, traj = self.evolve(z0, return_traj=True)
        return {
            "z0": z0.cpu().numpy()[0],
            "zT": zT.cpu().numpy()[0],
            "trajectory": np.stack([t.cpu().numpy()[0] for t in traj]),
            "occupancy": occ.cpu().numpy()[0],
        }

    def stability_report(self, z_ref: torch.Tensor, coord_scale_um: float = 1.0) -> Dict:
        return self.operator.linear_stability(z_ref, coord_scale_um=coord_scale_um)

    def diffusion_length_um(self, coord_scale_um: float = 1.0) -> np.ndarray:
        if self.operator.diffusion is None:
            return np.zeros(self.cfg.latent_channels)
        horizon = self.cfg.dynamics.dt * self.cfg.dynamics.n_steps
        return self.operator.diffusion.length_scales(coord_scale_um, horizon)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_nmo(cfg: Dict | NMOConfig, n_genes: Optional[int] = None) -> NeuralMorphogenOperator:
    if isinstance(cfg, dict):
        d = dict(cfg)
        dyn = d.pop("dynamics", {}) or {}
        known = set(NMOConfig.__dataclass_fields__) - {"dynamics"}
        d = {k: v for k, v in d.items() if k in known}
        cfg = NMOConfig(**d, dynamics=DynamicsConfig(**dyn))
    if n_genes is not None:
        cfg.n_genes = n_genes
    return NeuralMorphogenOperator(cfg)

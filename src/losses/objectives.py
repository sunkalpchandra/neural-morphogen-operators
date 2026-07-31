"""Training objectives for the Neural Morphogen Operator.

Total objective::

    L = L_recon + a1 * L_smooth + a2 * L_pde + a3 * L_mass + a4 * L_jac

A note on what the "physics-informed" term can and cannot mean here
-------------------------------------------------------------------
Because the operator integrates the PDE *exactly* (spectral diffusion + RK2
reaction under Strang splitting), the residual evaluated along the model's own
trajectory is zero up to splitting error. A naive ``||dz/dt - RHS||`` term is
therefore vacuous -- it penalises the integrator, not the model.

The constraint that actually carries content is that the **observed tissue
configuration should be a near-stationary state of the learned operator**:

    L_pde = || RHS(z_T) ||^2                                              (2)

This asks the operator to place the measured field at (or near) a fixed point,
which is the correct formalisation of the quasi-steady-state assumption for
adult tissue. It is a real constraint on ``D`` and ``f`` jointly and it is what
makes the relaxation dynamics do useful work at inference time.

We additionally include ``L_split``, which measures the disagreement between the
Strang step and an explicit Euler step of the same right-hand side. It is a
numerical-consistency guard: it keeps ``dt`` inside the regime where the
splitting error is small, rather than letting the optimiser hide model error
inside integration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.view(-1, 1).to(pred.dtype)
    denom = m.sum() * pred.shape[1]
    return ((pred - target) ** 2 * m).sum() / denom.clamp_min(1.0)


def pearson_per_gene(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> torch.Tensor:
    """Pearson r computed **per gene across locations** -> (G,).

    This is the axis that matters scientifically: we care whether the *spatial
    pattern* of each gene is right, not whether the profile at a single spot
    correlates across genes (which is dominated by overall expression level and
    is easy to get right for trivial reasons).
    """
    if mask is not None:
        sel = mask.bool()
        pred, target = pred[sel], target[sel]
    p = pred - pred.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    num = (p * t).sum(0)
    den = p.norm(dim=0) * t.norm(dim=0)
    return num / den.clamp_min(eps)


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return 1.0 - pearson_per_gene(pred, target, mask).mean()


# --------------------------------------------------------------------------- #
# Spatial regularity
# --------------------------------------------------------------------------- #


def dirichlet_energy(field: torch.Tensor) -> torch.Tensor:
    """Mean |grad z|^2 on the grid (finite differences, periodic).

    Encourages latent fields that vary smoothly in space -- the field-theoretic
    prior that morphogen concentrations are continuous, not spot-wise noise.
    Deliberately weak: too much and the model cannot express sharp anatomical
    boundaries such as the hippocampal pyramidal layer.
    """
    dx = field - torch.roll(field, 1, dims=-1)
    dy = field - torch.roll(field, 1, dims=-2)
    return (dx.pow(2) + dy.pow(2)).mean()


def total_variation(field: torch.Tensor) -> torch.Tensor:
    dx = (field - torch.roll(field, 1, dims=-1)).abs()
    dy = (field - torch.roll(field, 1, dims=-2)).abs()
    return (dx + dy).mean()


# --------------------------------------------------------------------------- #
# Physics constraints
# --------------------------------------------------------------------------- #


def steady_state_residual(operator, z: torch.Tensor, weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Equation (2): squared norm of dz/dt at the relaxed field."""
    rhs = operator.time_derivative(z)
    if weight_map is not None:
        w = weight_map / weight_map.mean().clamp_min(1e-8)
        return (rhs.pow(2) * w).mean()
    return rhs.pow(2).mean()


def splitting_consistency(operator, z: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
    """Disagreement between the Strang step and explicit Euler on the same RHS.

    Small values certify that ``dt`` sits in the regime where the second-order
    splitting is faithful to the continuous operator.
    """
    dt = operator.cfg.dt if dt is None else dt
    z_strang = operator.step(z, dt)
    z_euler = z + dt * operator.time_derivative(z)
    return (z_strang - z_euler).pow(2).mean()


def mass_conservation(operator, z: torch.Tensor) -> torch.Tensor:
    """Penalise net creation/destruction of latent 'morphogen mass'.

    Spectral diffusion conserves the spatial mean exactly (it leaves the k = 0
    Fourier mode untouched), so any drift in total mass is attributable to the
    reaction term. We penalise the per-channel spatial mean of ``f(z)``, which
    softly imposes a closed reaction budget without forbidding local
    production/consumption -- exactly the behaviour of a real signalling system
    with bounded source terms.
    """
    if operator.reaction is None:
        return torch.zeros((), device=z.device)
    f = operator.reaction(z)
    return f.mean(dim=(2, 3)).pow(2).mean()


def _rasterize(coords: torch.Tensor, values: torch.Tensor, grid: int,
               weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nadaraya--Watson rasterization of scattered values onto a ``grid`` lattice.

    Prediction and measurement are rasterized with the *same* accumulation, so
    the sampling geometry, the irregular support and the empty cells are common
    to both and cancel in any comparison between them. What remains is the model.
    """
    n, g = values.shape
    idx = ((coords.clamp(-1, 1) + 1) * 0.5 * (grid - 1)).round().long()
    flat = (idx[:, 1] * grid + idx[:, 0]).clamp(0, grid * grid - 1)

    w = torch.ones(n, device=values.device, dtype=values.dtype) if weight is None else weight
    num = torch.zeros(grid * grid, g, device=values.device, dtype=values.dtype)
    num.index_add_(0, flat, values * w.unsqueeze(1))
    den = torch.zeros(grid * grid, device=values.device, dtype=values.dtype)
    den.index_add_(0, flat, w)
    field = num / den.clamp_min(1e-8).unsqueeze(1)
    return field.T.reshape(g, grid, grid)


def _radial_spectrum(field: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
    """Radially averaged power spectrum of a ``(G, H, W)`` stack of maps.

    Returns ``(n_bins,)``: the mean power in each annulus of the 2-D spectrum,
    averaged over genes. Radial averaging is the right summary here because the
    quantity of interest -- how much energy the reconstruction retains at each
    spatial scale -- is isotropic, while the individual Fourier coefficients are
    not comparable between a prediction and a measurement.
    """
    g, h, w = field.shape
    f = torch.fft.rfft2(field - field.mean(dim=(1, 2), keepdim=True), norm="ortho")
    power = (f.real ** 2 + f.imag ** 2).mean(0)                    # (H, W//2+1)

    ky = torch.fft.fftfreq(h, device=field.device).abs().view(-1, 1)
    kx = torch.fft.rfftfreq(w, device=field.device).view(1, -1)
    k = torch.sqrt(ky ** 2 + kx ** 2)
    k = k / k.max().clamp_min(1e-12)

    edges = torch.linspace(0, 1, n_bins + 1, device=field.device)
    bins = torch.bucketize(k.reshape(-1), edges[1:-1].contiguous())
    out = torch.zeros(n_bins, device=field.device, dtype=power.dtype)
    cnt = torch.zeros(n_bins, device=field.device, dtype=power.dtype)
    out.index_add_(0, bins, power.reshape(-1))
    cnt.index_add_(0, bins, torch.ones_like(power.reshape(-1)))
    return out / cnt.clamp_min(1.0)


def spectral_match(pred: torch.Tensor, target: torch.Tensor, coords: torch.Tensor,
                   mask: Optional[torch.Tensor] = None, grid: int = 64,
                   n_bins: int = 16, n_genes: int = 128,
                   generator: Optional[torch.Generator] = None,
                   mode: str = "full") -> torch.Tensor:
    """Penalize the *missing high-frequency energy* of the reconstruction.

    A dissipative operator attenuates high spatial frequencies, which is exactly
    what Moran's $I$ detects and what correlation does not: a prediction can
    track the smooth part of a gene's pattern and still lose every sharp
    boundary. This term compares the radially averaged power spectrum of the
    predicted field with that of the measured field, in log space so that the
    high-$k$ bins -- orders of magnitude weaker than the low-$k$ ones, and the
    ones actually at issue -- are not swamped.

    Genes are subsampled per step; the estimate is stochastic but unbiased over
    training, and the full rasterization of two thousand genes every step is not
    affordable on CPU.
    """
    n, g = pred.shape
    if n_genes and n_genes < g:
        sel = torch.randperm(g, device=pred.device, generator=generator)[:n_genes]
        pred, target = pred[:, sel], target[:, sel]

    fp = _rasterize(coords, pred, grid, mask)
    ft = _rasterize(coords, target, grid, mask)
    sp = _radial_spectrum(fp, n_bins)
    st = _radial_spectrum(ft, n_bins)

    if mode == "shape":
        # Match the *distribution* of energy across scales rather than its
        # absolute level. Normalizing by total power removes the amplitude
        # degree of freedom, which the full-spectrum form can exploit to lower
        # the loss without redistributing anything, and restricting to the
        # upper half of the band puts the penalty where the defect is: a
        # dissipative operator loses high-frequency energy, not low.
        lo = n_bins // 2
        sp = sp / sp.sum().clamp_min(1e-12)
        st = st / st.sum().clamp_min(1e-12)
        return F.mse_loss(torch.log(sp[lo:] + 1e-8), torch.log(st[lo:] + 1e-8))
    return F.mse_loss(torch.log(sp + 1e-8), torch.log(st + 1e-8))


def jacobian_stability(operator, z: torch.Tensor, max_norm: float = 3.0, n_iter: int = 3) -> torch.Tensor:
    """Hinge penalty on ||df/dz||_2 above ``max_norm``.

    Bounds the local Lipschitz constant of the reaction field. Combined with the
    unconditionally stable diffusion solve, this keeps long rollouts (used for
    the counterfactual simulations) from diverging, while leaving the operator
    free to be genuinely nonlinear below the threshold.
    """
    sigma = operator.jacobian_spectral_norm(z, n_iter=n_iter)
    return F.relu(sigma - max_norm).pow(2)


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #


@dataclass
class LossWeights:
    recon_mse: float = 1.0
    recon_pearson: float = 0.5
    smooth: float = 1e-3
    pde: float = 1e-2
    split: float = 1e-3
    mass: float = 1e-3
    jacobian: float = 1e-3
    jacobian_max_norm: float = 3.0
    # weight on decoding the *pre-relaxation* field; a small value keeps the
    # encoder well conditioned without letting the model bypass the operator
    aux_z0: float = 0.1
    # spectral matching: penalize the high-frequency energy a dissipative
    # operator discards. Off by default; Section 5 sweeps it.
    spectral: float = 0.0
    spectral_grid: int = 64
    spectral_bins: int = 16
    spectral_genes: int = 128
    spectral_mode: str = "full"      # {'full', 'shape'}


class NMOLoss(nn.Module):
    def __init__(self, weights: LossWeights | Dict | None = None):
        super().__init__()
        if isinstance(weights, dict):
            weights = LossWeights(**weights)
        self.w = weights or LossWeights()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        target: torch.Tensor,
        operator,
        eval_mask: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        out         : dict from ``NeuralMorphogenOperator.forward``
        target      : (N, G) ground-truth expression
        eval_mask   : (N,) locations the reconstruction loss is computed on.
                      For the inpainting task this is the *held-out* set, so the
                      model is scored on genuinely unseen locations.
        train_mask  : (N,) locations the encoder was allowed to see.
        """
        w = self.w
        pred, z0, zT = out["pred"], out["z0"], out["zT"]

        l_mse = masked_mse(pred, target, eval_mask)
        l_pear = pearson_loss(pred, target, eval_mask)
        loss = w.recon_mse * l_mse + w.recon_pearson * l_pear
        terms = {"mse": l_mse.detach(), "pearson_loss": l_pear.detach()}

        if w.aux_z0 > 0 and "pred_z0" in out:
            # supervise the pre-relaxation read-out only where data was visible
            l_aux = masked_mse(out["pred_z0"], target, train_mask)
            loss = loss + w.aux_z0 * l_aux
            terms["aux_z0"] = l_aux.detach()

        if w.smooth > 0:
            l_sm = dirichlet_energy(zT)
            loss = loss + w.smooth * l_sm
            terms["smooth"] = l_sm.detach()

        if w.pde > 0:
            l_pde = steady_state_residual(operator, zT, out.get("occupancy"))
            loss = loss + w.pde * l_pde
            terms["pde"] = l_pde.detach()

        if w.split > 0:
            l_sp = splitting_consistency(operator, zT)
            loss = loss + w.split * l_sp
            terms["split_consistency"] = l_sp.detach()

        if w.mass > 0:
            l_mass = mass_conservation(operator, zT)
            loss = loss + w.mass * l_mass
            terms["mass"] = l_mass.detach()

        if w.jacobian > 0:
            l_jac = jacobian_stability(operator, zT, w.jacobian_max_norm)
            loss = loss + w.jacobian * l_jac
            terms["jacobian"] = l_jac.detach()

        if w.spectral > 0 and coords is not None:
            l_spec = spectral_match(pred, target, coords, eval_mask,
                                    grid=w.spectral_grid, n_bins=w.spectral_bins,
                                    n_genes=w.spectral_genes, mode=w.spectral_mode)
            loss = loss + w.spectral * l_spec
            terms["spectral"] = l_spec.detach()

        terms["total"] = loss
        return terms

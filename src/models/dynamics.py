"""The learned reaction-diffusion operator.

We model the latent tissue field ``z(x, y, t) in R^C`` as obeying

    dz/dt = div( D_theta(z) grad z ) + f_theta(z)                       (1)

and learn both the anisotropic diffusion tensor ``D_theta`` and the reaction
network ``f_theta``.

Numerical scheme
----------------
Equation (1) is stiff: the diffusion operator has eigenvalues that scale as
``-|k|^2``, so an explicit step requires ``dt < O(h^2 / |D|)`` and small
step-sizes dominate cost while injecting gradient noise. We instead use a
second-order **Strang-split exponential integrator**:

    z <- E(dt/2) z ;   z <- z + dt * RK2[f_theta](z) ;   z <- E(dt/2) z

where the diffusion half-steps are applied *exactly* in the Fourier domain,

    E(tau) = F^{-1} diag( exp( -(k^T D_c k) tau ) ) F ,

because the constant-coefficient anisotropic Laplacian is diagonal there. Two
consequences matter:

1. The diffusive half of the update is **unconditionally stable** for any
   positive-definite ``D`` and any ``dt`` -- there is no CFL constraint, and the
   scheme cannot blow up through the diffusion channel no matter what the
   optimiser does to the parameters.
2. ``D`` is *directly interpretable*. Its eigenvalues are diffusion lengths
   squared per unit pseudo-time, so ``sqrt(lambda_i * T)`` converts to a
   physical correlation length in microns via the stored coordinate scale.

Positive-definiteness of ``D`` is structural, not penalised: each channel holds
a lower-triangular ``L_c`` and we set ``D_c = L_c L_c^T + eps I``.

Interpretability
----------------
``linear_stability`` performs the Turing analysis of the learned operator:
linearising (1) about a homogeneous state ``z*`` gives, for spatial mode ``k``,

    d(dz_k)/dt = ( J(z*) - k^T D k ) dz_k ,      J = df_theta/dz |_{z*}

so the sign of ``max Re spec( J - |k|^2 D )`` as a function of ``|k|`` states
whether the *learned* operator supports spontaneous pattern formation, and at
which wavelength. This is a falsifiable, quantitative readout of what the model
has inferred -- not a post-hoc visualisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Diffusion
# --------------------------------------------------------------------------- #


class AnisotropicDiffusion(nn.Module):
    """Per-channel 2x2 SPD diffusion tensor, applied exactly in Fourier space.

    Parameters
    ----------
    channels : latent channel count C
    isotropic : if True, constrain D_c = d_c * I (used by the ablation that
        removes anisotropy)
    state_dependent : if True, modulate the tensor by a learned positive scalar
        field m(z) in (0, 2), giving div(m(z) D grad z). The modulation is
        applied via an explicit correction term so the spectral solve remains
        exact for the constant part.
    """

    def __init__(
        self,
        channels: int,
        init_scale: float = 0.02,
        isotropic: bool = False,
        state_dependent: bool = False,
        eps: float = 1e-4,
        hidden: int = 32,
    ):
        super().__init__()
        self.C, self.eps = channels, eps
        self.isotropic, self.state_dependent = isotropic, state_dependent

        s = math.sqrt(init_scale)
        if isotropic:
            self.log_d = nn.Parameter(torch.full((channels,), math.log(init_scale)))
        else:
            # L = [[l11, 0], [l21, l22]] with positive diagonal via softplus
            self.l_diag = nn.Parameter(torch.full((channels, 2), _inv_softplus(s)))
            self.l_off = nn.Parameter(torch.zeros(channels))

        if state_dependent:
            self.modulator = nn.Sequential(
                nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
            )
            nn.init.zeros_(self.modulator[-1].weight)
            nn.init.zeros_(self.modulator[-1].bias)

    # -- tensor construction ------------------------------------------------ #

    def tensor(self) -> torch.Tensor:
        """Return D of shape (C, 2, 2), symmetric positive definite."""
        if self.isotropic:
            d = self.log_d.exp()
            I = torch.eye(2, device=d.device, dtype=d.dtype).expand(self.C, 2, 2)
            return d.view(-1, 1, 1) * I + self.eps * I
        ld = F.softplus(self.l_diag)  # (C, 2) > 0
        L = torch.zeros(self.C, 2, 2, device=ld.device, dtype=ld.dtype)
        L[:, 0, 0] = ld[:, 0]
        L[:, 1, 1] = ld[:, 1]
        L[:, 1, 0] = self.l_off
        D = L @ L.transpose(1, 2)
        return D + self.eps * torch.eye(2, device=D.device, dtype=D.dtype)

    def eigen(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eigenvalues (C,2) ascending and eigenvectors (C,2,2) of D."""
        return torch.linalg.eigh(self.tensor())

    # -- spectral operators ------------------------------------------------- #

    @staticmethod
    def _wavenumbers(H: int, W: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Angular wavenumbers for a domain of physical side length 2 (i.e. [-1,1])."""
        ky = torch.fft.fftfreq(H, d=2.0 / H, device=device, dtype=dtype) * 2 * math.pi
        kx = torch.fft.rfftfreq(W, d=2.0 / W, device=device, dtype=dtype) * 2 * math.pi
        return ky.view(-1, 1), kx.view(1, -1)

    def quadratic_form(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """k^T D k evaluated on the rFFT grid -> (C, H, W//2+1), non-negative."""
        D = self.tensor().to(dtype)
        ky, kx = self._wavenumbers(H, W, device, dtype)
        kxx, kyy, kxy = kx**2, ky**2, kx * ky
        q = (
            D[:, 0, 0].view(-1, 1, 1) * kxx
            + D[:, 1, 1].view(-1, 1, 1) * kyy
            + 2 * D[:, 0, 1].view(-1, 1, 1) * kxy
        )
        return q.clamp(min=0.0)

    def exp_step(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Exact solution of dz/dt = div(D grad z) over ``dt`` (constant D).

        z: (B, C, H, W). Unconditionally stable because the multiplier
        ``exp(-q dt)`` lies in (0, 1] for q >= 0, dt > 0.
        """
        B, C, H, W = z.shape
        q = self.quadratic_form(H, W, z.device, z.dtype)  # (C, H, W//2+1)
        mult = torch.exp(-q * dt).unsqueeze(0)
        zf = torch.fft.rfft2(z, norm="ortho")
        return torch.fft.irfft2(zf * mult, s=(H, W), norm="ortho")

    def laplacian(self, z: torch.Tensor) -> torch.Tensor:
        """Spectral div(D grad z), used for the PDE-residual loss."""
        B, C, H, W = z.shape
        q = self.quadratic_form(H, W, z.device, z.dtype).unsqueeze(0)
        zf = torch.fft.rfft2(z, norm="ortho")
        return torch.fft.irfft2(-q * zf, s=(H, W), norm="ortho")

    def forward(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        out = self.exp_step(z, dt)
        if self.state_dependent:
            # explicit correction for the state-dependent part:
            # div( (m(z)) D grad z ) - div( D grad z ) ~ m(z) * lap - lap
            m = 1.0 + torch.tanh(self.modulator(z))  # in (0, 2)
            out = out + dt * (m - 1.0) * self.laplacian(z)
        return out

    def length_scales(self, coord_scale_um: float = 1.0, horizon: float = 1.0) -> np.ndarray:
        """Diffusion length sqrt(2 * lambda * T) per channel, in microns."""
        ev, _ = self.eigen()
        return (torch.sqrt(2.0 * ev.detach() * horizon) * coord_scale_um).cpu().numpy()


def _inv_softplus(y: float) -> float:
    return float(math.log(math.expm1(max(y, 1e-6))))


# --------------------------------------------------------------------------- #
# Reaction
# --------------------------------------------------------------------------- #


class ReactionNetwork(nn.Module):
    """Pointwise nonlinear reaction f_theta : R^C -> R^C.

    Implemented as 1x1 convolutions so it acts identically at every spatial
    location -- the field-theoretic assumption that the local chemistry is
    translation invariant while the *state* is not.

    The output is passed through ``tanh`` and multiplied by a learnable
    per-channel gain. This bounds ``||f||`` and therefore bounds the reaction
    half-step of the integrator, which together with the unconditionally stable
    diffusion half-step means the full scheme cannot diverge.
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 64,
        n_layers: int = 3,
        gain_init: float = 0.5,
        include_linear: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        d = channels
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(d, hidden, 1), nn.GELU()]
            d = hidden
        layers += [nn.Conv2d(d, channels, 1)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # An explicit linear term makes the Jacobian at the operating point
        # non-degenerate at initialisation and is what the Turing analysis reads.
        self.linear = nn.Conv2d(channels, channels, 1, bias=True) if include_linear else None
        if self.linear is not None:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
        self.log_gain = nn.Parameter(torch.full((channels,), math.log(gain_init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.net(z)
        if self.linear is not None:
            h = h + self.linear(z)
        gain = self.log_gain.exp().view(1, -1, 1, 1)
        return gain * torch.tanh(h)

    # -- analysis ----------------------------------------------------------- #

    def jacobian_at(self, z_star: torch.Tensor) -> torch.Tensor:
        """Exact C x C Jacobian df/dz at a single homogeneous state.

        ``z_star`` is (C,). Cheap because C is small (16-32) and the network is
        pointwise, so we evaluate on a 1x1 "image".
        """
        z = z_star.view(1, -1, 1, 1).clone().requires_grad_(True)
        C = z_star.numel()
        rows = []
        out = self.forward(z).view(-1)
        for c in range(C):
            g = torch.autograd.grad(out[c], z, retain_graph=(c < C - 1), create_graph=False)[0]
            rows.append(g.view(-1))
        return torch.stack(rows, 0)  # J[c, c'] = d f_c / d z_c'


# --------------------------------------------------------------------------- #
# Full operator
# --------------------------------------------------------------------------- #


@dataclass
class DynamicsConfig:
    channels: int = 24
    dt: float = 0.05
    n_steps: int = 8
    reaction_hidden: int = 64
    reaction_layers: int = 3
    diffusion_init: float = 0.02
    isotropic: bool = False
    state_dependent_diffusion: bool = False
    use_diffusion: bool = True
    use_reaction: bool = True
    reaction_gain: float = 0.5


class ReactionDiffusionOperator(nn.Module):
    """Integrates dz/dt = div(D grad z) + f(z) with Strang splitting."""

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.diffusion = (
            AnisotropicDiffusion(
                cfg.channels,
                init_scale=cfg.diffusion_init,
                isotropic=cfg.isotropic,
                state_dependent=cfg.state_dependent_diffusion,
            )
            if cfg.use_diffusion
            else None
        )
        self.reaction = (
            ReactionNetwork(
                cfg.channels, cfg.reaction_hidden, cfg.reaction_layers, cfg.reaction_gain
            )
            if cfg.use_reaction
            else None
        )

    # -- single step -------------------------------------------------------- #

    def _reaction_half(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Midpoint (RK2) update for the reaction half-step."""
        if self.reaction is None:
            return z
        k1 = self.reaction(z)
        k2 = self.reaction(z + 0.5 * dt * k1)
        return z + dt * k2

    def step(self, z: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        dt = self.cfg.dt if dt is None else dt
        if self.diffusion is None:
            return self._reaction_half(z, dt)
        z = self.diffusion(z, dt / 2)
        z = self._reaction_half(z, dt)
        z = self.diffusion(z, dt / 2)
        return z

    def forward(
        self, z: torch.Tensor, n_steps: Optional[int] = None, dt: Optional[float] = None,
        return_traj: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        n = self.cfg.n_steps if n_steps is None else n_steps
        traj = [z]
        for _ in range(n):
            z = self.step(z, dt)
            if return_traj:
                traj.append(z)
        return (z, traj) if return_traj else z

    # -- physics diagnostics ------------------------------------------------ #

    def time_derivative(self, z: torch.Tensor) -> torch.Tensor:
        """Instantaneous dz/dt, i.e. the right-hand side of (1)."""
        out = torch.zeros_like(z)
        if self.diffusion is not None:
            out = out + self.diffusion.laplacian(z)
        if self.reaction is not None:
            out = out + self.reaction(z)
        return out

    def pde_residual(self, z: torch.Tensor, z_next: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """(z_next - z)/dt - RHS(z), the physics-informed residual."""
        dt = self.cfg.dt if dt is None else dt
        return (z_next - z) / dt - self.time_derivative(z)

    # -- interpretability --------------------------------------------------- #

    @torch.no_grad()
    def _homogeneous_state(self, z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=(0, 2, 3)) if z.dim() == 4 else z

    def linear_stability(
        self,
        z_ref: torch.Tensor,
        k_values: Optional[np.ndarray] = None,
        coord_scale_um: float = 1.0,
        n_dirs: int = 8,
    ) -> Dict[str, np.ndarray]:
        """Turing / dispersion analysis of the learned operator.

        Returns, for each wavenumber ``|k|``, the maximum real part of the
        spectrum of ``J - k^T D k`` maximised over ``n_dirs`` orientations of k
        (needed because D is anisotropic). A positive value at finite ``|k|``
        with a negative value at ``k = 0`` is the classical signature of a
        diffusion-driven (Turing) instability.
        """
        if k_values is None:
            k_values = np.linspace(0.0, 60.0, 121)

        z_star = self._homogeneous_state(z_ref).detach()
        C = z_star.numel()
        J = (
            self.reaction.jacobian_at(z_star).detach()
            if self.reaction is not None
            else torch.zeros(C, C, device=z_star.device)
        )
        D = (
            self.diffusion.tensor().detach()
            if self.diffusion is not None
            else torch.zeros(C, 2, 2, device=z_star.device)
        )

        thetas = np.linspace(0.0, math.pi, n_dirs, endpoint=False)
        growth = np.zeros(len(k_values))
        best_theta = np.zeros(len(k_values))
        for i, k in enumerate(k_values):
            best = -np.inf
            for th in thetas:
                kv = torch.tensor(
                    [k * math.cos(th), k * math.sin(th)], device=D.device, dtype=D.dtype
                )
                q = torch.einsum("i,cij,j->c", kv, D, kv)  # (C,)
                M = J - torch.diag(q)
                ev = torch.linalg.eigvals(M)
                m = float(ev.real.max())
                if m > best:
                    best, bt = m, th
            growth[i] = best
            best_theta[i] = bt

        # Convert to physical units: k is per normalised unit; a wavenumber k
        # corresponds to wavelength 2*pi/k normalised units = 2*pi/k * scale um.
        with np.errstate(divide="ignore"):
            wavelength_um = np.where(k_values > 0, 2 * math.pi / np.maximum(k_values, 1e-9) * coord_scale_um, np.inf)

        i_max = int(np.argmax(growth))
        turing = bool(growth[i_max] > 0 and k_values[i_max] > 1e-6 and growth[0] <= 0)
        return {
            "k": k_values,
            "growth_rate": growth,
            "direction": best_theta,
            "wavelength_um": wavelength_um,
            "k_max": float(k_values[i_max]),
            "growth_max": float(growth[i_max]),
            "turing_unstable": turing,
            "jacobian": J.cpu().numpy(),
            "diffusion_tensor": D.cpu().numpy(),
        }

    def jacobian_spectral_norm(self, z: torch.Tensor, n_iter: int = 4, n_points: int = 256) -> torch.Tensor:
        """Power-iteration estimate of ||df/dz||_2 on a random subset of pixels.

        Used as a differentiable stability regulariser. Kept cheap by sampling
        pixels rather than forming the full Jacobian.
        """
        if self.reaction is None:
            return torch.zeros((), device=z.device)
        B, C, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, C)
        idx = torch.randperm(flat.shape[0], device=z.device)[: min(n_points, flat.shape[0])]
        zs = flat[idx].detach().requires_grad_(True)
        zin = zs.T.reshape(1, C, -1, 1)

        v = torch.randn_like(zs)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sigma = torch.zeros((), device=z.device)
        for i in range(n_iter):
            with torch.enable_grad():
                out = self.reaction(zin).reshape(C, -1).T  # (P, C)
                # Jv via double-backward trick
                u = torch.autograd.grad(
                    out, zs, grad_outputs=v, create_graph=(i == n_iter - 1), retain_graph=True
                )[0]
            nrm = u.norm(dim=1, keepdim=True).clamp_min(1e-8)
            sigma = nrm.mean()
            v = (u / nrm).detach()
        return sigma

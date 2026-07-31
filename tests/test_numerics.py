"""Tests for the parts of the model whose correctness the paper's claims rest on.

These are not smoke tests. Each one checks a property that, if violated, would
invalidate something stated in the manuscript.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.losses.objectives import masked_mse, pearson_per_gene as t_pearson
from src.models.dynamics import (
    AnisotropicDiffusion, DynamicsConfig, ReactionDiffusionOperator, ReactionNetwork,
)
from src.models.layers import GaussianSplat, grid_gather, knn_graph
from src.models.nmo import NMOConfig, build_nmo


# --------------------------------------------------------------------------- #
# Diffusion operator
# --------------------------------------------------------------------------- #


def test_diffusion_tensor_is_positive_definite():
    """D = LL^T + eps*I must be SPD for arbitrary parameters, including hostile
    ones. The unconditional-stability argument depends on this."""
    d = AnisotropicDiffusion(channels=6, init_scale=0.05)
    with torch.no_grad():
        d.l_diag.uniform_(-50, 50)   # extreme values the optimiser could reach
        d.l_off.uniform_(-50, 50)
    D = d.tensor()
    ev = torch.linalg.eigvalsh(D)
    assert torch.isfinite(D).all()
    assert (ev > 0).all(), f"non-PD diffusion tensor, min eigenvalue {ev.min()}"
    assert torch.allclose(D, D.transpose(1, 2), atol=1e-6), "D must be symmetric"


def test_spectral_diffusion_is_unconditionally_stable():
    """No timestep may amplify the field. This is the CFL-free claim."""
    d = AnisotropicDiffusion(channels=4, init_scale=0.3)
    z = torch.randn(1, 4, 32, 32)
    n0 = z.norm()
    for dt in [1e-3, 0.1, 1.0, 10.0, 1e3]:
        out = d.exp_step(z, dt)
        assert torch.isfinite(out).all(), f"non-finite at dt={dt}"
        assert out.norm() <= n0 * (1 + 1e-4), f"diffusion amplified the field at dt={dt}"


def test_diffusion_conserves_spatial_mean():
    """Spectral diffusion leaves the k=0 mode untouched, which is what licenses
    attributing any mass drift to the reaction term."""
    d = AnisotropicDiffusion(channels=3, init_scale=0.1)
    z = torch.randn(1, 3, 24, 24)
    before = z.mean(dim=(2, 3))
    after = d.exp_step(z, dt=0.7).mean(dim=(2, 3))
    assert torch.allclose(before, after, atol=1e-5), "diffusion changed total mass"


def test_isotropic_mode_gives_isotropic_tensor():
    d = AnisotropicDiffusion(channels=4, init_scale=0.02, isotropic=True)
    D = d.tensor()
    off = D[:, 0, 1].abs().max()
    assert off < 1e-8, "isotropic mode produced off-diagonal diffusion"
    assert torch.allclose(D[:, 0, 0], D[:, 1, 1], atol=1e-8)


# --------------------------------------------------------------------------- #
# Full operator
# --------------------------------------------------------------------------- #


def test_long_rollout_stays_bounded():
    """The paper claims rollouts remain bounded far beyond the training horizon."""
    op = ReactionDiffusionOperator(DynamicsConfig(channels=8, n_steps=4, dt=0.05))
    z = torch.randn(1, 8, 32, 32)
    out = op(z, n_steps=1000)
    assert torch.isfinite(out).all(), "rollout diverged"
    assert out.abs().max() < 1e3


def test_reaction_jacobian_matches_finite_differences():
    """The Turing analysis reads J from autograd; verify it against FD."""
    torch.manual_seed(0)
    net = ReactionNetwork(channels=4, hidden=16, n_layers=2)
    with torch.no_grad():  # move off the zero-initialised output layer
        for p in net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    z = torch.randn(4)
    J = net.jacobian_at(z).detach()

    eps = 1e-4
    J_fd = torch.zeros(4, 4)
    with torch.no_grad():
        for j in range(4):
            dz = torch.zeros(4); dz[j] = eps
            f1 = net((z + dz).view(1, -1, 1, 1)).view(-1)
            f0 = net((z - dz).view(1, -1, 1, 1)).view(-1)
            J_fd[:, j] = (f1 - f0) / (2 * eps)
    assert torch.allclose(J, J_fd, atol=1e-3), f"max err {(J - J_fd).abs().max()}"


def test_dispersion_analysis_structure():
    op = ReactionDiffusionOperator(DynamicsConfig(channels=6, n_steps=3))
    z = torch.randn(1, 6, 16, 16)
    rep = op.linear_stability(z, k_values=np.linspace(0, 40, 25), coord_scale_um=3000.0)
    assert rep["k"].shape == rep["growth_rate"].shape
    assert np.isfinite(rep["growth_rate"]).all()
    assert rep["jacobian"].shape == (6, 6)
    assert rep["diffusion_tensor"].shape == (6, 2, 2)
    # with zero reaction the operator is purely dissipative: growth must decrease
    op_nr = ReactionDiffusionOperator(DynamicsConfig(channels=4, use_reaction=False))
    rep2 = op_nr.linear_stability(torch.randn(1, 4, 16, 16), k_values=np.linspace(0, 30, 20))
    g = rep2["growth_rate"]
    assert g[0] >= g[-1] - 1e-9, "pure diffusion must not grow with |k|"
    assert not rep2["turing_unstable"], "pure diffusion cannot be Turing unstable"


def test_ablation_flags_remove_components():
    op = ReactionDiffusionOperator(DynamicsConfig(channels=4, use_diffusion=False))
    assert op.diffusion is None
    op2 = ReactionDiffusionOperator(DynamicsConfig(channels=4, use_reaction=False))
    assert op2.reaction is None
    z = torch.randn(1, 4, 16, 16)
    # n_steps = 0 must be an exact identity: the 'no dynamics' control depends on it
    op3 = ReactionDiffusionOperator(DynamicsConfig(channels=4, n_steps=0))
    assert torch.equal(op3(z), z), "n_steps=0 was not an identity"


# --------------------------------------------------------------------------- #
# Point <-> grid transfer
# --------------------------------------------------------------------------- #


def test_grid_gather_recovers_known_field():
    """Interpolating a linear ramp must return the ramp."""
    H = W = 33
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    field = (2.0 * xs + 3.0 * ys).unsqueeze(0)          # (1, H, W)
    pts = torch.rand(200, 2) * 1.6 - 0.8
    got = grid_gather(field, pts).squeeze(1)
    want = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
    assert torch.allclose(got, want, atol=2e-2), f"max err {(got-want).abs().max()}"


def test_splat_is_mask_aware():
    """Masked points must not contribute; occupancy must reflect that."""
    splat = GaussianSplat(grid_size=16, sigma_cells=1.0, radius=2)
    coords = torch.rand(50, 2) * 2 - 1
    vals = torch.ones(50, 3)
    w = torch.ones(50); w[:25] = 0.0
    field, occ = splat(coords, vals, weights=w)
    assert field.shape == (3, 16, 16)
    assert occ.shape == (1, 16, 16)
    _, occ_full = splat(coords, vals, weights=torch.ones(50))
    assert occ.sum() < occ_full.sum(), "masking did not reduce occupancy"


def test_knn_graph_is_symmetric_and_selfloop_free():
    coords = torch.rand(80, 2)
    ei = knn_graph(coords, k=5)
    assert (ei[0] != ei[1]).all(), "self-loop present"
    edges = {(int(a), int(b)) for a, b in zip(*ei)}
    assert all((b, a) in edges for a, b in edges), "graph is not symmetric"


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def test_masked_mse_ignores_masked_entries():
    pred = torch.zeros(10, 4)
    target = torch.zeros(10, 4)
    target[5:] = 100.0                      # huge error, but masked out
    mask = torch.zeros(10); mask[:5] = 1.0
    assert masked_mse(pred, target, mask).item() == pytest.approx(0.0, abs=1e-8)


def test_pearson_matches_scipy():
    from scipy.stats import pearsonr

    torch.manual_seed(0)
    a = torch.randn(60, 3)
    b = 0.5 * a + torch.randn(60, 3)
    got = t_pearson(a, b).numpy()
    want = np.array([pearsonr(a[:, i].numpy(), b[:, i].numpy())[0] for i in range(3)])
    assert np.allclose(got, want, atol=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


def test_nmo_forward_backward_and_shapes():
    torch.manual_seed(0)
    N, G = 300, 40
    cfg = NMOConfig(n_genes=G, latent_channels=8, grid_size=32,
                    dynamics=DynamicsConfig(channels=8, n_steps=3))
    m = build_nmo(cfg)
    coords = torch.rand(N, 2) * 2 - 1
    expr = torch.randn(N, G)
    out = m(coords, expr)
    assert out["pred"].shape == (N, G)
    assert out["z0"].shape == (1, 8, 32, 32)
    out["pred"].pow(2).mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients reached the parameters"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_decoder_is_coordinate_free_by_default():
    """A coordinate-conditioned decoder would make every ablation meaningless."""
    cfg = NMOConfig(n_genes=10, latent_channels=4, grid_size=16)
    m = build_nmo(cfg)
    assert m.decoder.coord_feat is None, "decoder unexpectedly receives coordinates"


def test_prediction_at_unobserved_locations_is_defined():
    """The continuity claim: the model must answer at coordinates with no data."""
    torch.manual_seed(0)
    N, G = 200, 12
    cfg = NMOConfig(n_genes=G, latent_channels=6, grid_size=24,
                    dynamics=DynamicsConfig(channels=6, n_steps=2))
    m = build_nmo(cfg)
    coords = torch.rand(N, 2) * 2 - 1
    expr = torch.randn(N, G)
    mask = torch.ones(N); mask[100:] = 0.0
    query = torch.rand(37, 2) * 2 - 1          # arbitrary, not in `coords`
    out = m(coords, expr * mask.view(-1, 1), query_coords=query, point_mask=mask)
    assert out["pred"].shape == (37, G)
    assert torch.isfinite(out["pred"]).all()

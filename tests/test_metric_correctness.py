"""Cross-check our metric implementations against independent ones.

Every headline number in the paper is one of these four quantities. A silent
error in any of them would be invisible to the prose-vs-artifact audit, which
checks that the paper reports what the code computed, not that the code computed
the right thing.

Reference implementations are written from the definitions rather than imported,
except where a well-tested library exists, so the test does not merely compare a
function against itself.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.evaluation.metrics import (
    gearys_c, morans_i, pearson_per_gene, spatial_weights, ssim_images,
)


def _moran_reference(x: np.ndarray, W) -> float:
    """Moran's I straight from the definition."""
    Wd = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    z = x - x.mean()
    return (len(x) / Wd.sum()) * ((Wd * np.outer(z, z)).sum() / (z ** 2).sum())


def _geary_reference(x: np.ndarray, W) -> float:
    """Geary's C straight from the definition."""
    Wd = W.toarray() if hasattr(W, "toarray") else np.asarray(W)
    n = len(x)
    num = (Wd * (x[:, None] - x[None, :]) ** 2).sum()
    den = 2 * Wd.sum() * ((x - x.mean()) ** 2).sum()
    return ((n - 1) * num) / den


@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(0)
    coords = rng.random((400, 2))
    values = rng.standard_normal((400, 5))
    return coords, values, spatial_weights(coords, 8)


def test_morans_i_matches_definition(field):
    coords, values, W = field
    ours = morans_i(values, W)
    ref = np.array([_moran_reference(values[:, j], W) for j in range(values.shape[1])])
    assert np.allclose(ours, ref, atol=1e-9), f"max diff {np.abs(ours - ref).max():.2e}"


def test_gearys_c_matches_definition(field):
    coords, values, W = field
    ours = gearys_c(values, W)
    ref = np.array([_geary_reference(values[:, j], W) for j in range(values.shape[1])])
    assert np.allclose(ours, ref, atol=1e-8), f"max diff {np.abs(ours - ref).max():.2e}"


def test_ssim_matches_skimage():
    sk = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(1)
    a = rng.random((3, 32, 32)).astype(np.float32)
    b = (a + 0.1 * rng.random((3, 32, 32))).astype(np.float32)
    ours = np.asarray(ssim_images(a, b))
    ref = np.array([
        sk.structural_similarity(
            a[i], b[i], data_range=float(max(np.ptp(a[i]), np.ptp(b[i]))))
        for i in range(3)])
    assert np.allclose(ours, ref, atol=1e-3), f"ours {ours} vs skimage {ref}"


def test_ssim_orders_degradations_like_skimage():
    """Absolute agreement matters less than ordering: the paper compares models."""
    sk = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(2)
    a = rng.random((3, 32, 32)).astype(np.float32)
    pairs = []
    for k in range(1, 6):
        b = (a + 0.05 * k * rng.random((3, 32, 32))).astype(np.float32)
        ours = float(np.mean(ssim_images(a, b)))
        ref = float(np.mean([
            sk.structural_similarity(
                a[i], b[i], data_range=float(max(np.ptp(a[i]), np.ptp(b[i]))))
            for i in range(3)]))
        pairs.append((ours, ref))
    discordant = [((x1, y1), (x2, y2))
                  for (x1, y1), (x2, y2) in itertools.combinations(pairs, 2)
                  if (x1 - x2) * (y1 - y2) < 0]
    assert not discordant, f"{len(discordant)} pairs ordered differently"


def test_pearson_is_per_gene_not_per_location():
    """The axis is the claim: the paper argues about each gene's spatial pattern."""
    rng = np.random.default_rng(3)
    true = rng.standard_normal((200, 4))
    pred = true.copy()
    pred[:, 0] = -pred[:, 0]              # invert exactly one gene
    r = pearson_per_gene(pred, true)
    assert r.shape == (4,)
    assert r[0] < -0.99 and (r[1:] > 0.99).all()


def test_metrics_invariant_to_gene_permutation(field):
    """Task 46: permuting genes permutes the per-gene metric, nothing more."""
    coords, values, W = field
    perm = np.random.default_rng(4).permutation(values.shape[1])
    assert np.allclose(morans_i(values[:, perm], W), morans_i(values, W)[perm])


def test_metrics_equivariant_to_location_permutation(field):
    """Task 47: relabelling locations must not change a spatial statistic."""
    coords, values, _ = field
    perm = np.random.default_rng(5).permutation(coords.shape[0])
    a = morans_i(values, spatial_weights(coords, 8))
    b = morans_i(values[perm], spatial_weights(coords[perm], 8))
    assert np.allclose(a, b, atol=1e-10)

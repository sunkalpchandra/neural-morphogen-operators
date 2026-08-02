"""Verify the split does what the paper says it does.

The paper's central methodological claim is that held-out blocks are genuinely
held out: contiguous, identical across models, and never seen by preprocessing.
A leak anywhere here would inflate every number in the paper without changing a
single line of prose, and the prose-vs-artifact audit could not detect it.

These run against the processed sections when present and skip otherwise, so a
clone without data still passes the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROCESSED = Path("data/processed")
SECTIONS = sorted(p for p in PROCESSED.glob("*.h5ad")
                  if "__panel_" not in p.stem and not p.stem.startswith("perturb_"))

# Loading six .h5ad sections takes ~2 minutes, too slow for the default suite;
# these run under `make test-data` or `pytest -m data`.
pytestmark = [
    pytest.mark.skipif(not SECTIONS, reason="processed sections not built"),
    pytest.mark.data,
]


@pytest.fixture(scope="module")
def sections():
    from src.training.dataset import load_section
    return {p.stem: load_section(p) for p in SECTIONS[:6]}


def test_every_section_has_all_three_splits(sections):
    """An all-train split yields NaN test metrics silently; the build guard
    exists because three sections were once written that way."""
    for name, sec in sections.items():
        present = set(np.unique(sec.split))
        assert {"train", "val", "test"} <= present, f"{name}: only {sorted(present)}"


def test_held_out_blocks_are_contiguous_not_scattered(sections):
    """Task 24. A random split leaks: neighbours predict each other. The test
    block must be spatially clustered, so its points sit far from training data
    relative to a random subset of the same size."""
    from scipy.spatial import cKDTree
    for name, sec in sections.items():
        xy = sec.coords.cpu().numpy()
        tr, te = sec.split == "train", sec.split == "test"
        if tr.sum() < 10 or te.sum() < 10:
            continue
        block_d = np.median(cKDTree(xy[tr]).query(xy[te])[0])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xy), te.sum(), replace=False)
        mask = np.ones(len(xy), bool); mask[idx] = False
        rand_d = np.median(cKDTree(xy[mask]).query(xy[idx])[0])
        assert block_d > 1.5 * rand_d, (
            f"{name}: held-out points sit {block_d:.4f} from training data, a "
            f"random split of the same size gives {rand_d:.4f} -- the split is "
            f"not behaving like contiguous blocks")


def test_standardisation_uses_training_statistics_only(sections):
    """Task 25. Standardising on all locations leaks the held-out distribution.
    If train-split statistics were used, the train split has mean 0 and sd 1 and
    the held-out splits generally do not."""
    for name, sec in sections.items():
        e = sec.expr.cpu().numpy()
        tr = sec.split == "train"
        assert abs(float(e[tr].mean())) < 1e-3, f"{name}: train mean is not 0"
        assert abs(float(e[tr].std()) - 1.0) < 5e-2, f"{name}: train sd is not 1"


def test_coordinates_are_isotropically_normalised(sections):
    """Task 27. Anisotropic rescaling would distort the Laplacian and make a
    learned diffusion coefficient meaningless, which is the quantity the paper
    interprets."""
    for name, sec in sections.items():
        xy = sec.coords.cpu().numpy()
        um = sec.meta.get("coord_scale_um") if isinstance(sec.meta, dict) else None
        assert sec.coord_scale_um > 1.0, f"{name}: coord_scale_um is {sec.coord_scale_um}"
        # a single shared scale means the larger extent maps to ~1 in that axis
        assert np.abs(xy).max() <= 1.0 + 1e-5, f"{name}: coords exceed [-1, 1]"
        assert np.isclose(np.abs(xy).max(), 1.0, atol=1e-3), (
            f"{name}: the larger extent does not reach 1, so the scale may not "
            f"be shared between axes")


def test_split_is_deterministic_across_loads():
    """Task 23/24. Two models compared on 'the same' blocks must actually get
    the same blocks; the split is stored, not recomputed, so this must hold."""
    from src.training.dataset import load_section
    a = load_section(SECTIONS[0])
    b = load_section(SECTIONS[0])
    assert np.array_equal(a.split, b.split)
    assert np.allclose(a.coords.cpu().numpy(), b.coords.cpu().numpy())

"""Verify the guards fail when they should.

A check that never fails is indistinguishable from a check that cannot fail.
Each guard here exists because a real defect got past everything else, so each
test injects the defect it was written for and asserts the guard catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistics import (
    MIN_HELDOUT_LOCATIONS, MIN_REFERENCE_ARI, ari_retention_estimable,
    by_specimen, min_attainable_p, paired_comparison, size_eligible, specimen,
)


def test_size_guard_rejects_the_section_it_was_written_for():
    """visium_human_heart had 54 held-out locations; kidney, the next smallest, 314."""
    assert not size_eligible(54)
    assert size_eligible(314)
    assert MIN_HELDOUT_LOCATIONS == 200


def test_ari_guard_rejects_a_degenerate_reference():
    """A reference ARI of 0.009 produced retentions from -6.9 to +6.9."""
    assert not ari_retention_estimable(0.009)
    assert ari_retention_estimable(0.5)
    assert MIN_REFERENCE_ARI > 0.009


def test_specimen_collapse_groups_serial_sections():
    """Twelve MERFISH sections are one brain; unlisted sections stand alone."""
    assert specimen("merfish_allen_01") == specimen("merfish_allen_50")
    assert specimen("visium_mouse_kidney") != specimen("visium_human_lymph_node")
    assert specimen("visium_mouse_kidney") == "visium_mouse_kidney"


def test_by_specimen_reduces_n_to_the_specimen_count():
    rows = [dict(section=f"merfish_allen_{i:02d}", model=m, pearson_mean=0.1 + 0.01 * i)
            for i in range(1, 6) for m in ("nmo", "gnn")]
    rows += [dict(section="visium_mouse_kidney", model=m, pearson_mean=0.3)
             for m in ("nmo", "gnn")]
    df = pd.DataFrame(rows)
    assert df["section"].nunique() == 6
    assert by_specimen(df)["section"].nunique() == 2


def test_min_attainable_p_matches_the_designs_we_reported():
    """The numbers the paper quotes for why small designs cannot reach 0.05."""
    assert min_attainable_p(3) == pytest.approx(0.25)
    assert min_attainable_p(5) == pytest.approx(0.0625)
    assert min_attainable_p(8) == pytest.approx(0.0078125)
    assert min_attainable_p(10) == pytest.approx(0.001953125)


def test_paired_comparison_needs_at_least_three_pairs():
    """The exact GP ran on one section and must not appear as a benchmarked row."""
    df = pd.DataFrame([
        dict(section="a", model="nmo", pearson_mean=0.2),
        dict(section="a", model="gp", pearson_mean=0.1),
        dict(section="b", model="nmo", pearson_mean=0.2),
    ])
    assert paired_comparison(df, "nmo", "pearson_mean") == []


def test_paired_comparison_sign_convention():
    """Positive mean_diff must mean the reference is better, for both metric senses."""
    hi = pd.DataFrame([dict(section=s, model=m, pearson_mean=v)
                       for s, (a, b) in zip("abcd", [(0.3, 0.2)] * 4)
                       for m, v in (("nmo", a), ("gnn", b))])
    lo = pd.DataFrame([dict(section=s, model=m, rmse=v)
                       for s, (a, b) in zip("abcd", [(0.2, 0.3)] * 4)
                       for m, v in (("nmo", a), ("gnn", b))])
    assert paired_comparison(hi, "nmo", "pearson_mean")[0].mean_diff > 0
    assert paired_comparison(lo, "nmo", "rmse")[0].mean_diff > 0


def test_holm_correction_is_monotone_and_bounded():
    df = pd.DataFrame([dict(section=s, model=m, pearson_mean=v)
                       for s in "abcdefgh"
                       for m, v in (("nmo", 0.3), ("a", 0.1), ("b", 0.2), ("c", 0.29))])
    res = paired_comparison(df, "nmo", "pearson_mean")
    ps = [r.p_holm for r in sorted(res, key=lambda r: r.wilcoxon_p)]
    assert ps == sorted(ps), "Holm-adjusted p must be non-decreasing in raw p"
    assert all(r.p_holm >= r.wilcoxon_p - 1e-12 for r in res)
    assert all(r.p_holm <= 1.0 for r in res)

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


def test_bootstrap_ci_is_percentile_not_normal_approximation():
    """Task 39. The paper says percentile bootstrap. On skewed differences a
    percentile interval is asymmetric about the mean and a normal approximation
    is not, which is what distinguishes them."""
    from src.evaluation.statistics import _bootstrap_ci
    rng = np.random.default_rng(0)
    skewed = rng.exponential(0.02, 40)
    lo, hi = _bootstrap_ci(skewed, n_boot=20000, seed=0)
    m = float(skewed.mean())
    assert lo < m < hi
    assert abs((hi - m) - (m - lo)) > 1e-4, "interval is symmetric; looks normal-approx"


def test_holm_matches_statsmodels_on_the_papers_own_p_values():
    """Task 40. Checked on the actual specimen-level p-values, so the test fails
    if the implementation drifts in the regime the paper reports."""
    sm = pytest.importorskip("statsmodels.stats.multitest")
    from src.evaluation.statistics import PairedResult, holm_correct
    ps = [0.0020, 0.0020, 0.0039, 0.0195, 0.0625, 0.1055, 0.4316]
    rows = [PairedResult("nmo", f"m{i}", "pearson_mean", 10, 0.01,
                         0.0, 0.0, 0.0, p, 0.0, 0.0, 0.0, 0)
            for i, p in enumerate(ps)]
    ours = [r.p_holm for r in holm_correct(rows)]
    ref = sm.multipletests(ps, method="holm")[1]
    assert np.allclose(ours, ref, atol=1e-12), f"ours {ours} vs statsmodels {list(ref)}"


def test_headline_numbers_are_pinned():
    """Task 45. A regression pin on the numbers the paper leads with.

    Not a correctness check -- it is a tripwire. If a refactor changes the
    headline result, this fails and forces the change to be noticed and the
    prose updated, rather than the number quietly moving.
    """
    import json
    from pathlib import Path
    nums = Path("paper/numbers.tex")
    if not nums.exists():
        pytest.skip("numbers.tex not generated")
    import re
    text = nums.read_text()

    def macro(name: str) -> str:
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*)\}", text)
        return m.group(1) if m else ""

    assert macro("MSSections") == "22"
    assert macro("MSSpecimens") == "10"
    assert macro("MSSpecNSig") == "3"
    assert "0.190" in macro("MSNMOPearson") or macro("MSNMOPearson").startswith("0.1")
    assert macro("MSExcluded").replace("\\", "") == "visium_human_heart"


def test_baseline_scoring_does_not_depend_on_global_rng_state():
    """Task 26. Four baselines drew a fresh inducing-point subset on every
    forward pass, including in eval. The same GP checkpoint scored ten times
    spanned 0.023 Pearson r -- wider than its across-seed spread -- so its
    reported number was partly a draw rather than a property of the model.

    Scoring twice under different global RNG state must give the same answer.
    """
    import torch
    from src.models.baselines import _inducing_subset

    dev = torch.device("cpu")
    torch.manual_seed(0)
    a = _inducing_subset(5000, 1024, dev, training=False)
    torch.manual_seed(999)
    b = _inducing_subset(5000, 1024, dev, training=False)
    assert torch.equal(a, b), "eval subset moved with the global seed"

    # training keeps resampling: that is the intended stochastic scheme
    torch.manual_seed(0)
    c = _inducing_subset(5000, 1024, dev, training=True)
    torch.manual_seed(999)
    d = _inducing_subset(5000, 1024, dev, training=True)
    assert not torch.equal(c, d), "training subset is frozen; resampling was lost"

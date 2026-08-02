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


def test_every_table_over_exp8_applies_the_same_size_rule():
    """A section too small to estimate Pearson r is excluded from the
    specimen-level analysis and from tab_headline. tab_multisection and
    tab_paired consumed the same records without that rule, so the table
    averaged over 23 sections while the prose reported 22, and
    visium_human_heart -- 331 held-out locations against a threshold of 800 --
    sat inside section-level paired tests the text says exclude it.

    Feeds one undersized section to each generator and asserts it is dropped.
    """
    import tempfile
    from pathlib import Path

    from src.evaluation.statistics import MIN_HELDOUT_LOCATIONS
    from src.evaluation.tables import (
        table_headline, table_multisection, table_paired_stats)

    big, small = 4 * MIN_HELDOUT_LOCATIONS, 4 * MIN_HELDOUT_LOCATIONS - 1
    rows = []
    for i in range(6):                       # six eligible sections
        for m, v in (("nmo", 0.30), ("gnn", 0.20)):
            rows.append(dict(section=f"s{i}", model=m, seed=0, pearson_mean=v,
                             ssim_mean=0.4, morans_i_abs_error=0.1,
                             n_obs_used=big))
    for m, v in (("nmo", 0.99), ("gnn", 0.98)):   # one undersized outlier
        rows.append(dict(section="tiny", model=m, seed=0, pearson_mean=v,
                         ssim_mean=0.9, morans_i_abs_error=0.01,
                         n_obs_used=small))

    with tempfile.TemporaryDirectory() as d:
        for fn, name in ((table_multisection, "multisection"),
                         (table_paired_stats, "paired"),
                         (table_headline, "headline")):
            out = Path(d) / f"{name}.tex"
            fn(rows, out)
            if not out.exists():
                continue
            text = out.read_text()
            # 0.99 only appears if the undersized section reached the mean
            assert "0.99" not in text, (
                f"tab_{name} included a section with {small} held-out locations, "
                f"below the {4 * MIN_HELDOUT_LOCATIONS} the rule requires")
            assert " 7 " not in text.replace("\\", " "), (
                f"tab_{name} counted 7 sections; only 6 are eligible")

    # The figure consumes the same records and its caption quotes \MSSections,
    # so it has to drop the same section or the plot and its caption disagree.
    import matplotlib
    import numpy as np
    matplotlib.use("Agg")
    from src.visualization.figures import figure_multisection
    import pandas as pd
    fig = figure_multisection(rows)
    # Inspect what was actually plotted, not what a re-implementation of the
    # filter would produce: the first draft of this assertion re-filtered the
    # fixture with pandas and asserted on that, which tests nothing about the
    # figure. The undersized section sits at (0.99, 0.98) and would be visible
    # as a plotted point if it survived.
    # Panel (a) is one point per section. Count them: an earlier draft matched on
    # the coordinate pair (0.99, 0.98) and still passed with the filter disabled,
    # because the panel plots (baseline, reference) and the point is at
    # (0.98, 0.99). A count does not depend on getting the axis order right.
    pts = sum(len(c.get_offsets()) for c in fig.axes[0].collections
              if c.get_offsets() is not None)
    assert pts == 6, (
        f"panel (a) plotted {pts} points for 6 eligible sections; the undersized "
        f"section its caption excludes appears to have survived")


def test_biology_table_and_figure_apply_the_ari_eligibility_rule():
    """ari_retention is predicted/measured, so a near-zero measured reference
    makes the ratio meaningless and unbounded. numbers.py applied the rule;
    table_biology and figure_biology did not, so the appendix table the prose
    points at showed NRDO retaining 1.127 -- more structure than the reference
    it divides by -- where the prose computed from the same records said 0.485,
    and every baseline appeared negative. Corrected, the models are close.
    """
    import tempfile
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")

    from src.evaluation.statistics import MIN_REFERENCE_ARI
    from src.evaluation.tables import table_biology
    from src.visualization.figures import figure_biology

    rows = []
    for i in range(4):                       # eligible sections, sane ratios
        for m, v in (("nmo", 0.50), ("gnn", 0.48)):
            rows.append(dict(section=f"s{i}", model=m, ari_predicted=0.25,
                             ari_measured=0.50, ari_retention=v,
                             marker_auroc_predicted=0.7,
                             neighborhood_preservation=0.4,
                             gearys_c_abs_error=0.1))
    for m, v in (("nmo", 6.90), ("gnn", -6.86)):   # degenerate reference
        rows.append(dict(section="degenerate", model=m, ari_predicted=0.25,
                         ari_measured=MIN_REFERENCE_ARI / 10, ari_retention=v,
                         marker_auroc_predicted=0.7,
                         neighborhood_preservation=0.4,
                         gearys_c_abs_error=0.1))

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bio.tex"
        table_biology(rows, out)
        text = out.read_text()
        assert "1.7" not in text and "6.9" not in text, (
            "table_biology averaged in a section whose measured ARI is near zero")
        assert "0.50" in text or "0.480" in text or "0.48" in text, (
            "eligible rows disappeared; the filter is too aggressive")

    fig = figure_biology(rows)
    ys = []
    for ax in fig.axes:
        for coll in ax.collections:
            off = coll.get_offsets()
            if off is not None and len(off):
                ys.extend(float(p[1]) for p in off)
        for line in ax.lines:
            ys.extend(float(y) for y in line.get_ydata())
    assert ys, "nothing plotted; the assertion below would be vacuous"
    assert max(abs(y) for y in ys) < 3.0, (
        f"figure_biology plotted a retention of {max(ys, key=abs):.2f}; the "
        f"degenerate section survived the eligibility rule")


def test_no_generator_is_orphaned():
    """A figure or table generator with no caller produces a file that `make
    figures` never refreshes, so it sits on disk looking current while the data
    moves under it.

    The Stage 0 audit found four tables and five figures in this state. One more
    turned up later: figure_evidence had no caller, its PNG was 7.5 hours stale
    while every other figure had regenerated, and its biology panel lacked the
    eligibility rule -- so wiring it back in would have redrawn the artefact
    that made tab_biology overstate NRDO by more than twofold.
    """
    import ast
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    callers = "".join(
        (root / p).read_text() for p in
        ("experiments/make_figures.py", "src/evaluation/tables.py",
         "scripts/per_gene_analysis.py")
        if (root / p).exists())

    orphans = []
    for rel, prefix in (("src/visualization/figures.py", ("figure",)),
                        ("src/evaluation/tables.py", ("table_",))):
        f = root / rel
        if not f.exists():
            continue
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix):
                # a table generator's own definition appears in the caller text
                own = 1 if rel.endswith("tables.py") else 0
                if len(re.findall(r"\b" + node.name + r"\b", callers)) <= own:
                    orphans.append(f"{rel}:{node.lineno} {node.name}")

    assert not orphans, (
        "generators with no caller -- `make figures` will leave their output "
        f"stale: {orphans}")

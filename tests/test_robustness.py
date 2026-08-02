"""Degenerate inputs that the figure and table code will eventually meet.

Every figure in this paper is generated from a JSON artifact produced by a run
that may have partly failed, been interrupted, or covered fewer models than the
plotting code assumes. A figure that raises is annoying; a figure that silently
renders half its data and still looks plausible is the failure mode that already
occurred once here, when a `df.get("mode", "full")` returned a column instead of
a default and `groupby` dropped every pre-`--mode` record.

These feed the generators the inputs a partial run actually produces.
"""

from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def test_per_gene_figure_survives_an_empty_artifact():
    """Task 50. A run that produced no usable seed writes {"seeds": []}."""
    from src.visualization.figures import figure_per_gene
    fig = figure_per_gene({"seeds": []})
    assert fig is not None


def test_per_gene_figure_survives_a_single_seed():
    """The across-seed error bars must collapse rather than raise."""
    from src.visualization.figures import figure_per_gene
    one = {
        "seeds": [dict(seed=0, best_baseline="gnn", n_genes=3, corr=0.2,
                       bins=[dict(quartile=i + 1, lo=i * 0.1, hi=(i + 1) * 0.1,
                                  n=1, nrdo=0.2, rival=0.19, delta=0.01)
                             for i in range(4)],
                       mean_r={"nmo": 0.2, "gnn": 0.19})],
        "corr_lo": 0.2, "corr_hi": 0.2,
        "gene_structure": [0.1, 0.4, 0.7],
        "gene_advantage": [0.01, None, 0.03],
    }
    fig = figure_per_gene(one)
    assert fig is not None


def test_per_gene_figure_handles_all_advantages_undefined():
    """Every gene NaN: the scatter is empty and the running mean is skipped."""
    from src.visualization.figures import figure_per_gene
    d = {
        "seeds": [dict(seed=0, best_baseline="gnn", n_genes=0, corr=0.0,
                       bins=[dict(quartile=1, lo=0.0, hi=1.0, n=0,
                                  nrdo=0.0, rival=0.0, delta=0.0)],
                       mean_r={})],
        "corr_lo": 0.0, "corr_hi": 0.0,
        "gene_structure": [0.1, 0.2], "gene_advantage": [None, None],
    }
    assert figure_per_gene(d) is not None


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def test_architecture_table_with_only_the_reference_variant():
    """Task 50. A sweep killed after its first run has just `full`; the delta
    column has nothing to compare against and the s.d. is undefined."""
    from src.evaluation.tables import table_architecture
    import tempfile
    from pathlib import Path
    rows = [dict(variant="full", seed=0, pearson_mean=0.25,
                 learned_splat_sigma=0.96)]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.tex"
        try:
            table_architecture(rows, out)
        except Exception as exc:            # noqa: BLE001
            pytest.fail(f"single-variant input raised {type(exc).__name__}: {exc}")
        assert out.exists() and "tabular" in out.read_text()


def test_architecture_table_ignores_failed_runs():
    """Failed runs are recorded with `failed: True` and no metric; they must not
    reach the mean."""
    from src.evaluation.tables import table_architecture
    import tempfile
    from pathlib import Path
    rows = [dict(variant="full", seed=0, pearson_mean=0.25,
                 learned_splat_sigma=0.96),
            dict(variant="full", seed=1, pearson_mean=0.25,
                 learned_splat_sigma=0.96),
            dict(variant="knn4", seed=0, failed=True, error="boom")]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "t.tex"
        table_architecture(rows, out)
        text = out.read_text()
        assert "nan" not in text.lower(), "a failed run leaked into the table"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_permutation_p_on_degenerate_input():
    """All-zero differences: every sign flip gives the same mean, so p = 1."""
    import numpy as np
    from src.evaluation.statistics import exact_sign_permutation_p
    assert exact_sign_permutation_p(np.zeros(6)) == pytest.approx(1.0)
    assert np.isnan(exact_sign_permutation_p(np.array([])))


def test_permutation_p_refuses_an_intractable_enumeration():
    """2^n grows fast; the function must refuse rather than hang."""
    import numpy as np
    from src.evaluation.statistics import exact_sign_permutation_p
    with pytest.raises(ValueError, match="enumeration"):
        exact_sign_permutation_p(np.ones(40))


def test_permutation_p_matches_the_sign_test_on_a_clean_split():
    """With all differences equal and positive, the only flips that reach the
    observed mean are the all-positive one and its mirror: p = 2/2^n."""
    import numpy as np
    from src.evaluation.statistics import exact_sign_permutation_p
    for n in (4, 6, 8):
        p = exact_sign_permutation_p(np.ones(n))
        assert p == pytest.approx(2.0 / 2 ** n), f"n={n} gave {p}"


def test_specimens_needed_returns_the_cap_when_no_n_would_do():
    """A win rate at or below chance cannot reach power at any sample size."""
    import numpy as np
    from src.evaluation.statistics import specimens_needed
    assert specimens_needed(np.array([1.0, -1.0, 1.0, -1.0]), max_n=60) == 60
    assert specimens_needed(np.array([]), max_n=60) == 60
    # a strong effect resolves quickly
    assert specimens_needed(np.ones(10), max_n=60) < 15

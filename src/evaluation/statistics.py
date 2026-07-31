"""Statistical comparison of models across tissue sections.

A benchmark averaged over sections says nothing about whether an improvement is
reliable. With several sections evaluated under identical splits, the natural
unit of analysis is the *section*, and the natural test is a paired one: each
model is measured on exactly the same held-out blocks, so the pairing removes
between-section variance, which dominates the between-model differences.

Reported for every comparison against our method:

* the paired difference per section, and its bootstrap confidence interval;
* the Wilcoxon signed-rank statistic, which makes no normality assumption and is
  the appropriate default at these sample sizes;
* a paired *t*-test, reported alongside for readers who prefer it, with the
  Shapiro--Wilk *p* for the normality assumption it requires;
* Cohen's $d_z$ for paired designs, so the magnitude is interpretable
  independently of the number of sections;
* the number of sections on which our method wins, which is the statistic a
  reader can sanity-check against the per-section table.

Multiple comparisons are controlled with Holm--Bonferroni across the set of
baselines, since every baseline is compared against the same reference.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

HIGHER_IS_BETTER = {
    "pearson_mean": True, "spearman_mean": True, "ssim_mean": True,
    "pearson_median": True, "morans_i_corr": True,
    "rmse": False, "mae": False, "morans_i_abs_error": False,
}


@dataclass
class PairedResult:
    reference: str
    other: str
    metric: str
    n_sections: int
    mean_diff: float
    ci_lo: float
    ci_hi: float
    wilcoxon_stat: float
    wilcoxon_p: float
    ttest_p: float
    shapiro_p: float
    cohens_dz: float
    n_reference_wins: int
    p_holm: float = float("nan")

    def as_dict(self) -> Dict:
        return asdict(self)


def _bootstrap_ci(d: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                  seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    rng = np.random.default_rng(seed)
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_comparison(df: pd.DataFrame, reference: str = "nmo",
                      metric: str = "pearson_mean", seed: int = 0,
                      section_col: str = "section", model_col: str = "model",
                      ) -> List[PairedResult]:
    """Compare ``reference`` against every other model, paired by section.

    ``df`` must contain one row per (section, model, seed); seeds are averaged
    within a section first, so the pairing is over sections and the test is not
    inflated by treating seeds of the same section as independent.
    """
    if metric not in df.columns:
        return []
    per = (df.groupby([section_col, model_col])[metric].mean().unstack(model_col))
    if reference not in per.columns:
        return []
    higher = HIGHER_IS_BETTER.get(metric, True)
    out: List[PairedResult] = []
    for other in [c for c in per.columns if c != reference]:
        sub = per[[reference, other]].dropna()
        if len(sub) < 3:
            continue
        a = sub[reference].to_numpy(dtype=float)
        b = sub[other].to_numpy(dtype=float)
        d = (a - b) if higher else (b - a)      # positive => reference better
        lo, hi = _bootstrap_ci(d, seed=seed)
        try:
            w = stats.wilcoxon(a, b)
            wstat, wp = float(w.statistic), float(w.pvalue)
        except ValueError:                       # all differences zero
            wstat, wp = float("nan"), 1.0
        tp = float(stats.ttest_rel(a, b).pvalue)
        sp = float(stats.shapiro(d).pvalue) if 3 <= len(d) <= 5000 else float("nan")
        sd = d.std(ddof=1)
        out.append(PairedResult(
            reference=reference, other=other, metric=metric, n_sections=len(d),
            mean_diff=float(d.mean()), ci_lo=lo, ci_hi=hi,
            wilcoxon_stat=wstat, wilcoxon_p=wp, ttest_p=tp, shapiro_p=sp,
            cohens_dz=float(d.mean() / sd) if sd > 0 else float("inf"),
            n_reference_wins=int((d > 0).sum()),
        ))
    return holm_correct(out)


def holm_correct(results: List[PairedResult], field: str = "wilcoxon_p") -> List[PairedResult]:
    """Holm--Bonferroni step-down correction across the baseline family."""
    if not results:
        return results
    order = sorted(range(len(results)), key=lambda i: getattr(results[i], field))
    m = len(results)
    running = 0.0
    for rank, i in enumerate(order):
        p = getattr(results[i], field)
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)             # enforce monotonicity
        results[i].p_holm = running
    return results


def seed_variability(df: pd.DataFrame, metric: str = "pearson_mean",
                     section_col: str = "section", model_col: str = "model") -> pd.DataFrame:
    """Within-section across-seed s.d., which bounds how much of an observed
    difference could be optimisation noise."""
    g = df.groupby([section_col, model_col])[metric].agg(["mean", "std", "count"])
    return g.reset_index().rename(columns={"std": "seed_sd", "count": "n_seeds"})


def summarize(df: pd.DataFrame, reference: str = "nmo",
              metrics: Sequence[str] = ("pearson_mean", "rmse", "ssim_mean",
                                        "morans_i_abs_error"),
              seed: int = 0) -> pd.DataFrame:
    rows: List[Dict] = []
    for m in metrics:
        for r in paired_comparison(df, reference, m, seed=seed):
            rows.append(r.as_dict())
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    return "$^{***}$" if p < 1e-3 else "$^{**}$" if p < 1e-2 else "$^{*}$" if p < 0.05 else ""

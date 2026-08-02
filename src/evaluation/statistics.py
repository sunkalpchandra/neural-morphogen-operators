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


# --------------------------------------------------------------------------- #
# Specimen-level analysis (guards against pseudo-replication)
# --------------------------------------------------------------------------- #

#: Sections that come from the same biological specimen. Serial coronal sections
#: of one brain are not independent samples: anatomy, donor, batch and assay run
#: are all shared, so a test that treats them as independent overstates its
#: effective sample size. Any section not listed maps to itself.
SPECIMEN_OF = {
    "merfish_allen": "MERFISH C57BL6J-638850",   # 12 serial sections, one brain
    "mosta_embryo": "Stereo-seq embryo series",  # consecutive stages, one series
}


def specimen(section: str) -> str:
    for prefix, name in SPECIMEN_OF.items():
        if section.startswith(prefix):
            return name
    return section


def by_specimen(df: pd.DataFrame, metric: str = "pearson_mean",
                section_col: str = "section", model_col: str = "model") -> pd.DataFrame:
    """Collapse sections to specimens by averaging within specimen and model.

    This is the conservative unit of analysis. It costs statistical power --
    with a handful of specimens the Wilcoxon test cannot reach small p-values
    even under a perfect win record -- but it is the level at which
    'independent sample' is defensible.
    """
    out = df.copy()
    out[section_col] = out[section_col].map(specimen)
    return out.groupby([section_col, model_col], as_index=False)[metric].mean()


def two_level_report(df: pd.DataFrame, reference: str = "nmo",
                     metric: str = "pearson_mean") -> Dict[str, List[PairedResult]]:
    """Paired comparison at both the section and the specimen level.

    Reporting only the first would overstate significance; reporting only the
    second would discard the demonstration that the advantage is stable across
    anatomy. Both are returned so the paper can state each for what it is.
    """
    return {
        "section": paired_comparison(df, reference, metric),
        "specimen": paired_comparison(by_specimen(df, metric), reference, metric),
    }


def min_attainable_p(n: int) -> float:
    """Smallest two-sided Wilcoxon p attainable with ``n`` paired observations.

    Reported alongside specimen-level tests so a non-significant result is not
    misread as evidence of no effect when the design cannot produce a
    significant one.
    """
    return 2.0 ** (-(n - 1)) if n >= 1 else float("nan")

#: Minimum held-out locations for a section to enter the specimen-level
#: analysis.
#:
#: Per-gene Pearson r across n held-out locations has a standard error of
#: roughly 1/sqrt(n-3). At n = 54 that is 0.14, which is larger than every
#: effect this benchmark measures (the paired differences run 0.005 to 0.049).
#: A section that small cannot estimate the quantity being compared, so
#: including it would add noise, not evidence -- and including it *because* it
#: moves a p-value would be choosing the sample to fit the answer.
#:
#: The threshold is set at 200, which separates the one section below it (54)
#: from the next smallest (314) by a wide margin, and it is recorded here
#: rather than applied ad hoc so that it can be checked against the outcome it
#: was not chosen from.
MIN_HELDOUT_LOCATIONS = 200


def size_eligible(n_test: int) -> bool:
    """Whether a section's held-out set is large enough to estimate Pearson r."""
    return int(n_test) >= MIN_HELDOUT_LOCATIONS


#: Smallest reference ARI for which a retention *ratio* is meaningful.
#:
#: ARI retention is predicted/measured. When the measured ARI is near zero the
#: reference clustering carries almost no recoverable structure, the ratio is
#: dominated by its denominator, and it can exceed 1 or go negative -- neither
#: of which means what "retention" is supposed to mean. One section
#: (visium_human_heart, measured ARI 0.009) produced retentions from -6.9 to
#: +6.9 and pushed the reported mean above 1.0 before this guard existed.
MIN_REFERENCE_ARI = 0.05


def ari_retention_estimable(ari_measured: float) -> bool:
    """Whether a retention ratio can be formed from this reference ARI."""
    return float(ari_measured) >= MIN_REFERENCE_ARI


# --------------------------------------------------------------------------- #
# Exact sign-flip permutation test and power, for the specimen-level design
# --------------------------------------------------------------------------- #

def exact_sign_permutation_p(diffs: np.ndarray) -> float:
    """Two-sided p from enumerating all 2^n sign flips of the paired differences.

    The paper reports Wilcoxon signed-rank p-values on 10 specimens. Wilcoxon
    discards magnitude in favour of ranks, and at n=10 its null distribution is
    coarse. This enumerates the randomisation null directly under the sharp null
    that the sign of each specimen's difference is exchangeable, so it uses the
    magnitudes and is exact rather than asymptotic.

    It is a cross-check, not a replacement: if the two disagree materially, the
    conclusion rests on which test was chosen, and that is worth knowing.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    if n > 22:                                   # 2^22 ~ 4e6, still tractable
        raise ValueError(f"enumeration is 2^{n}; use a sampled permutation")
    obs = abs(float(d.mean()))
    signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    means = np.abs((signs * d).mean(axis=1))
    # >= : the observed assignment is itself one of the enumerated flips
    return float((means >= obs - 1e-15).mean())


def specimens_needed(diffs: np.ndarray, alpha: float = 0.05,
                     power: float = 0.80, max_n: int = 200) -> int:
    """Smallest n at which a sign test could reach `alpha` with `power`.

    Answers the question the STAGATE comparison raises: the paper reports it
    unresolved at 10 specimens, and a reader is entitled to ask how many would
    settle it. Uses the observed win rate as the effect size and the exact
    binomial sign test, which is the weakest of the tests in use here, so the
    answer is an upper bound on what a rank or permutation test would need.

    Returns ``max_n`` if the observed win rate cannot reach the target at any n
    below it -- with a win rate near 1/2 no achievable sample size helps.
    """
    from scipy.stats import binom
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return max_n
    p_win = float((d > 0).mean())
    if p_win <= 0.5:
        return max_n
    for n in range(3, max_n + 1):
        # critical count for a two-sided sign test at alpha under H0: p = 1/2
        k_crit = int(binom.ppf(1 - alpha / 2, n, 0.5)) + 1
        if k_crit > n:
            continue
        if 1.0 - binom.cdf(k_crit - 1, n, p_win) >= power:
            return n
    return max_n


# --------------------------------------------------------------------------- #
# Eligibility, applied at the point of loading rather than the point of use
# --------------------------------------------------------------------------- #

def size_eligible_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop sections whose held-out set is too small to estimate Pearson r.

    Four consumers of the exp8 records each decided separately whether to apply
    this rule, and four of them forgot: the benchmark table averaged over 23
    sections while the prose reported 22, the section-level Wilcoxon tests
    included a section the text names as excluded, and the benchmark figure
    plotted it under a caption saying otherwise.

    Call this immediately after loading. It is a no-op on frames with no
    ``n_obs_used`` column, so it is safe on older records.
    """
    if "n_obs_used" not in df.columns:
        return df
    return df[df["n_obs_used"] >= 4 * MIN_HELDOUT_LOCATIONS]


def ari_eligible_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop sections whose measured reference ARI is too small to divide by.

    ``ari_retention`` is predicted/measured; a near-zero denominator makes the
    ratio meaningless and unbounded. Omitting this rule put one section's
    retention at -6.86..+6.90 and made NRDO appear to retain 1.127 of a quantity
    bounded above by 1, with every baseline negative.
    """
    if "ari_measured" not in df.columns:
        return df
    return df[df["ari_measured"] >= MIN_REFERENCE_ARI]

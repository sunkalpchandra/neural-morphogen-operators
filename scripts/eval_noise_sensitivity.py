"""Could the eval-time RNG defect have changed any conclusion the paper draws?

Four baselines resampled their conditioning set on every forward pass, so each
recorded score is one draw from a distribution rather than a property of the
model (see AUDIT_OUTCOMES.md). The fix landed after those runs were recorded, and
regenerating them means retraining, since exp8 shards keep no checkpoints.

Before spending that compute it is worth knowing whether it could matter. This
perturbs the affected models' per-specimen scores by the measured noise and
re-runs the whole Holm-corrected comparison many times, counting how often any
verdict changes.

The noise scale is measured, not assumed: one gp checkpoint scored ten times
spanned 0.0227 Pearson r with s.d. 0.0059. That is the only affected model whose
checkpoint survives, so the same scale is applied to the other three; if
anything that overstates the noise for the non-GP models, which makes the
conclusion conservative.

    python scripts/eval_noise_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistics import (  # noqa: E402
    MIN_HELDOUT_LOCATIONS, by_specimen,
)

#: Models whose recorded scores predate the eval-determinism fix.
AFFECTED = {"gp", "gp_multiscale", "spage", "tangram"}

#: s.d. across ten scorings of one unchanged gp checkpoint.
MEASURED_SD = 0.0059


def _holm(ps: np.ndarray) -> np.ndarray:
    order = np.argsort(ps)
    out = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(running, 1.0)
    return out


def main() -> int:
    from scipy.stats import wilcoxon

    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--n-resamples", type=int, default=2000)
    p.add_argument("--sd", type=float, default=MEASURED_SD)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default="results/audit/eval_noise_sensitivity.json")
    a = p.parse_args()

    rows = [x for f in Path(a.results).glob("exp8/results_shard*.json")
            for x in json.loads(f.read_text())
            if "pearson_mean" in x and not x.get("failed")]
    rows = [x for x in rows
            if x.get("n_obs_used", 10 ** 9) >= 4 * MIN_HELDOUT_LOCATIONS]
    if not rows:
        print("no exp8 results found")
        return 1

    piv = by_specimen(pd.DataFrame(rows)).pivot_table(
        index="section", columns="model", values="pearson_mean")
    models = [c for c in piv.columns if c != "nmo"]

    def run(rng=None):
        ps, ms = [], []
        for m in models:
            sub = piv[["nmo", m]].dropna()
            if len(sub) < 3:
                continue
            pert = (rng.normal(0, a.sd, len(sub))
                    if rng is not None and m in AFFECTED else 0.0)
            d = (sub["nmo"] - (sub[m] + pert)).values
            ps.append(float(wilcoxon(d)[1]) if np.any(d != 0) else 1.0)
            ms.append(m)
        return ms, _holm(np.array(ps))

    names, base = run()
    rng = np.random.default_rng(0)
    flips = {m: 0 for m in names}
    for _ in range(a.n_resamples):
        _, pert = run(rng)
        for m, b, q in zip(names, base, pert):
            if (q < a.alpha) != (b < a.alpha):
                flips[m] += 1

    out = {}
    print(f"noise s.d. {a.sd}, {a.n_resamples} resamples\n")
    print(f"{'model':<16}{'p_holm':>9}{'sig':>6}{'flip rate':>11}")
    for m, b in zip(names, base):
        out[m] = dict(p_holm=float(b), significant=bool(b < a.alpha),
                      affected=m in AFFECTED,
                      flip_rate=flips[m] / a.n_resamples)
        print(f"{m:<16}{b:>9.4f}{'yes' if b < a.alpha else '--':>6}"
              f"{flips[m] / a.n_resamples:>10.2%}")

    risky = [m for m, v in out.items() if v["significant"] and v["flip_rate"] > 0]
    print()
    if risky:
        print(f"AT RISK: {risky} -- a reported result depends on the defect")
    else:
        print("No reported significant result changes under the noise.")
        could = [m for m, v in out.items()
                 if not v["significant"] and v["flip_rate"] > 0]
        if could:
            print(f"The defect may be masking a result for: {could}. "
                  f"Regenerating those runs can only add a finding, not remove "
                  f"one, so the current numbers are conservative.")

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

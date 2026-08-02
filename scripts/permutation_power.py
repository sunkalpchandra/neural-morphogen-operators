"""Cross-check the specimen-level p-values, and say what would resolve the rest.

Two questions a reviewer can reasonably ask about the headline comparison:

1. The paper reports Wilcoxon signed-rank p-values on 10 specimens. Wilcoxon
   throws away magnitude in favour of ranks and its null is coarse at n=10. Does
   the conclusion depend on that choice? This enumerates all 2^10 sign flips of
   the paired differences -- the exact randomisation null, using magnitudes --
   and reports both side by side.

2. STAGATE is reported unresolved. How many specimens would settle it? This uses
   the observed win rate with an exact binomial sign test, the weakest test in
   use here, so the answer is an upper bound.

Wilcoxon remains the reported test. The permutation p is a check on it, not a
substitute, and reporting whichever of the two is smaller would be test-shopping.

    python scripts/permutation_power.py
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
    MIN_HELDOUT_LOCATIONS, by_specimen, exact_sign_permutation_p,
    specimens_needed,
)


def main() -> int:
    from scipy.stats import wilcoxon

    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/audit/permutation_power.json")
    p.add_argument("--alpha", type=float, default=0.05)
    a = p.parse_args()

    rows = [x for f in Path(a.results).glob("exp8/results_shard*.json")
            for x in json.loads(f.read_text())
            if "pearson_mean" in x and not x.get("failed")]
    # Same size rule the manuscript applies, so the specimen count here matches
    # the count it reports rather than quietly including an excluded section.
    rows = [x for x in rows
            if x.get("n_obs_used", 10 ** 9) >= 4 * MIN_HELDOUT_LOCATIONS]
    if not rows:
        print("no exp8 results found")
        return 1

    piv = by_specimen(pd.DataFrame(rows)).pivot_table(
        index="section", columns="model", values="pearson_mean")

    out = {}
    print(f"{len(piv)} specimens")
    print(f"{'model':<16}{'wins':>9}{'wilcoxon':>10}{'perm':>9}{'n@80%':>8}")
    for m in sorted(c for c in piv.columns if c != "nmo"):
        sub = piv[["nmo", m]].dropna()
        d = (sub["nmo"] - sub[m]).values
        if len(d) < 3:
            continue
        w = float(wilcoxon(d)[1]) if np.any(d != 0) else 1.0
        perm = exact_sign_permutation_p(d)
        need = specimens_needed(d, alpha=a.alpha)
        out[m] = dict(n=len(d), wins=int((d > 0).sum()), wilcoxon=w,
                      permutation=perm, n_for_80pct_power=need,
                      agree=bool((w < a.alpha) == (perm < a.alpha)))
        print(f"{m:<16}{out[m]['wins']:>5}/{len(d):<3}{w:>10.4f}{perm:>9.4f}{need:>8}")

    n_dis = sum(1 for v in out.values() if not v["agree"])
    print(f"\n{len(out) - n_dis}/{len(out)} models: the two tests agree at "
          f"alpha={a.alpha}")
    if n_dis:
        print("  disagreeing: " +
              ", ".join(k for k, v in out.items() if not v["agree"]))

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

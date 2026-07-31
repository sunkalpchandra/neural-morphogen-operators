"""Experiment 12 -- how hard is each held-out set, geometrically?

Two protocols appear in this paper. Masked reconstruction hides *contiguous
tissue blocks*, which is the split the method sections describe and defend. The
transfer experiments instead reveal a *random half* of the target and score the
complement, because a zero-shot model still needs something to condition on.

Those are not the same problem. Under a random split the nearest visible
neighbour of a scored location is roughly one sampling pitch away, so a model can
succeed by local interpolation; under a block split it is a whole block away.
This script measures that distance directly, with no model and no training, so
the difficulty gap is a property of the design rather than an inference from
scores. It is what licenses the caveat in Section~\\ref{sec:results} that
zero-shot numbers may be compared with each other but not with the in-domain
oracle.

    python experiments/exp12_split_geometry.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.dataset import load_section, subsample_section
from src.utils.common import Config


def geometry(section: str, processed: Path, max_locations: int = 15000,
             visible_frac: float = 0.5, seed: int = 0) -> Dict:
    sec = load_section(processed / f"{section}.h5ad")
    if sec.n_obs > max_locations:
        sec = subsample_section(sec, max_locations, seed=seed)
    xy = sec.coords.cpu().numpy()
    um = float(sec.coord_scale_um)

    rng = np.random.default_rng(seed)
    vis = np.zeros(sec.n_obs, dtype=bool)
    vis[rng.choice(sec.n_obs, int(visible_frac * sec.n_obs), replace=False)] = True
    d_random = cKDTree(xy[vis]).query(xy[~vis])[0] * um

    train, test = sec.split == "train", sec.split == "test"
    d_block = (cKDTree(xy[train]).query(xy[test])[0] * um
               if train.any() and test.any() else np.array([np.nan]))

    return dict(
        section=section, n_obs=int(sec.n_obs), coord_scale_um=um,
        random_median_um=float(np.median(d_random)),
        random_p90_um=float(np.percentile(d_random, 90)),
        random_n_scored=int((~vis).sum()),
        block_median_um=float(np.median(d_block)),
        block_p90_um=float(np.percentile(d_block, 90)),
        block_n_scored=int(test.sum()),
        ratio=float(np.median(d_block) / max(np.median(d_random), 1e-9)),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--sections", nargs="+",
                   default=["xenium_mouse_brain", "visium_mouse_brain",
                            "visium_human_breast"])
    p.add_argument("--out-dir", default="results/exp12")
    a = p.parse_args()

    cfg = Config.load(a.config)
    processed = Path(cfg.data.processed_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for s in a.sections:
        if not (processed / f"{s}.h5ad").exists():
            print(f"[skip] {s}")
            continue
        r = geometry(s, processed)
        rows.append(r)
        print(f"{s:<22} random {r['random_median_um']:7.1f} um | "
              f"block {r['block_median_um']:7.1f} um | {r['ratio']:.1f}x")

    (out / "split_geometry.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out / 'split_geometry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

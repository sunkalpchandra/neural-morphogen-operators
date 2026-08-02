"""Does the error grow more slowly with distance for a field than for a graph?

The paper argues that a model defined on a continuous field should degrade more
gracefully where observations are sparse. The density sweep tests that at the
level of a whole section; this tests it point by point, which is the sharper
form of the same question: for each held-out location, how far is the nearest
observed location, and how large is the error there.

A graph model has no value between nodes and must extrapolate from its
neighbourhood; a field is defined everywhere. If the continuity argument means
anything, the gap between them should widen with distance.

    python scripts/error_vs_distance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import pearson_per_gene
from src.models.baselines import DISPLAY_NAMES, build_baseline
from src.models.nmo import build_nmo
from src.training.dataset import load_section
from src.utils.common import Config, get_device

MODELS = ["nmo", "stagate", "gnn", "autoencoder"]


def main() -> int:
    cfg = Config.load(ROOT / "configs" / "base.yaml")
    dev = get_device("cpu")
    section = "visium_mouse_brain"
    sec = load_section(ROOT / "data/processed" / f"{section}.h5ad", device=dev)
    vis = sec.mask("train")
    held = ~vis.cpu().numpy().astype(bool)

    xy = sec.coords.cpu().numpy()
    dist = cKDTree(xy[~held]).query(xy[held])[0] * sec.coord_scale_um
    true = sec.numpy_expr(denorm=False)[held]

    edges = np.quantile(dist, [0, 0.25, 0.5, 0.75, 1.0])
    rows: List[Dict] = []
    for name in MODELS:
        ck = sorted(ROOT.glob(f"results/exp1/*/runs/{section}__{name}__seed0/best.pt"))
        if not ck:
            print(f"  [skip] {name}: no checkpoint")
            continue
        try:
            model = (build_nmo(cfg.model.to_dict(), n_genes=sec.n_genes) if name == "nmo"
                     else build_baseline(name, n_genes=sec.n_genes, hidden=128,
                                         latent=cfg.model.get("latent_channels", 32)))
            state = torch.load(ck[0], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"] if "model" in state else state)
            model.eval()
            with torch.no_grad():
                pred = model(sec.coords, sec.expr * vis.view(-1, 1),
                             query_coords=sec.coords, edge_index=sec.edge_index,
                             point_mask=vis)["pred"].cpu().numpy()[held]
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            continue

        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            m = (dist >= lo) & (dist <= hi if i == 3 else dist < hi)
            if m.sum() < 30:
                continue
            r = float(np.nanmean(pearson_per_gene(pred[m], true[m])))
            rows.append(dict(model=name, display=DISPLAY_NAMES.get(name, name),
                             quartile=i + 1, lo_um=float(lo), hi_um=float(hi),
                             n=int(m.sum()), pearson=r))
        got = [x["pearson"] for x in rows if x["model"] == name]
        print(f"  {DISPLAY_NAMES.get(name, name):<26} " +
              "  ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(got)))

    out = ROOT / "results" / "error_vs_distance.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\ndistance quartiles (um): {np.round(edges, 0).tolist()}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

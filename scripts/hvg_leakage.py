"""Measure how much of the gene panel was chosen using held-out locations.

Gene selection runs before the split (``src/data/preprocess.py``: HVG at ~L197,
``spatial_block_split`` at L261), so the 2000-gene panel is computed from every
location in the section, including the blocks later held out. This is standard
practice in the spatial-transcriptomics literature and it is applied identically
to every model, so it cannot bias the comparison between them -- but it is
transductive, and the paper should say so rather than let a reader assume the
panel was derived from training data alone.

This quantifies it: recompute the panel on the training split only and report
the overlap with the panel actually used.

    python scripts/hvg_leakage.py --sections visium_mouse_brain
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAW = {
    "visium_mouse_brain": "data/raw/visium_mouse_brain/filtered_feature_bc_matrix.h5",
    "visium_human_breast": "data/raw/visium_human_breast/filtered_feature_bc_matrix.h5",
}


def panel(adata, n_top: int) -> set:
    import scanpy as sc
    b = adata.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, n_top_genes=n_top, flavor="seurat_v3")
    return set(b.var_names[b.var.highly_variable])


def main() -> int:
    import scanpy as sc

    p = argparse.ArgumentParser()
    p.add_argument("--sections", nargs="+", default=list(RAW))
    p.add_argument("--n-top", type=int, default=2000)
    p.add_argument("--out", default="results/audit/hvg_leakage.json")
    a = p.parse_args()

    rows: List[Dict] = []
    for name in a.sections:
        raw_path = Path(RAW.get(name, ""))
        proc_path = Path("data/processed") / f"{name}.h5ad"
        if not raw_path.exists() or not proc_path.exists():
            print(f"  {name}: raw or processed missing, skipped")
            continue
        proc = sc.read_h5ad(proc_path)
        raw = sc.read_10x_h5(raw_path)
        raw.var_names_make_unique()
        common = proc.obs_names.intersection(raw.obs_names)
        raw = raw[common].copy()
        train = (proc[common].obs["split"] == "train").values

        full, train_only = panel(raw, a.n_top), panel(raw[train].copy(), a.n_top)
        shared = len(full & train_only)
        rows.append(dict(
            section=name, n_top=a.n_top, n_locations=int(raw.shape[0]),
            train_fraction=float(train.mean()), overlap=shared,
            overlap_frac=shared / a.n_top, leaked=a.n_top - shared,
        ))
        print(f"  {name}: {shared}/{a.n_top} = {shared / a.n_top:.1%} overlap, "
              f"{a.n_top - shared} genes need held-out data")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

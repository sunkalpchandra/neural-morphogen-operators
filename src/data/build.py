"""Turn every raw download into a standardised processed ``.h5ad``.

    python -m src.data.build --all
    python -m src.data.build --dataset visium_mouse_brain --seed 0

Outputs land in ``data/processed/<key>.h5ad`` plus a ``SUMMARY.json`` giving the
shape, QC yield and spatial extent of every processed object -- the numbers
that populate the dataset table in the paper.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils.common import get_logger, set_seed
from . import sources
from .loaders import load_raw
from .preprocess import QCConfig, QC_PRESETS, preprocess

log = get_logger("nmo.build")

#: Datasets that carry spatial coordinates (Perturb-seq does not).
#: Datasets that carry spatial coordinates. Membership controls isotropic
#: coordinate normalisation and the contiguous-block split, so a spatial dataset
#: missing from this list is built silently as if it were dissociated cells:
#: every location lands in the train split and coord_scale_um stays 1. That
#: produces NaN test metrics rather than an error, which is how it was found.
SPATIAL_KEYS = [
    "visium_mouse_brain",
    "visium_human_breast",
    "visium_mouse_kidney",
    "visium_human_lymph_node",
    "visium_mouse_brain_coronal",
    "visium_human_heart",
    "merfish_allen",
    "xenium_mouse_brain",
    "mosta_embryo",
]


def _assert_spatial_built(adata, key: str) -> None:
    """Fail loudly if a spatial dataset was built without splits or a scale.

    Both are silent failures downstream: an all-train split yields NaN test
    metrics and a unit coordinate scale yields meaningless diffusion lengths.
    """
    import numpy as np
    sp = np.asarray(adata.obs["split"]) if "split" in adata.obs else np.array([])
    if len(set(sp.tolist())) < 2:
        raise RuntimeError(
            f"{key}: built with a single split value; it is probably missing "
            f"from SPATIAL_KEYS, so no contiguous-block split was applied")
    if float(adata.uns.get("nmo", {}).get("coord_scale_um", 1.0)) == 1.0:
        raise RuntimeError(
            f"{key}: coord_scale_um is 1.0, so coordinates were never "
            f"normalised; it is probably missing from SPATIAL_KEYS")


def build_one(
    key: str,
    raw_dir: Path,
    out_dir: Path,
    seed: int = 0,
    split_mode: str = "block",
    n_blocks: int = 8,
    overwrite: bool = False,
    **loader_kw,
) -> Optional[Path]:
    suffix = ""
    if key == "mosta_embryo" and "stage" in loader_kw:
        suffix = "_" + str(loader_kw["stage"]).split("_")[0]
    if key == "merfish_allen" and "section" in loader_kw:
        suffix = "_" + str(loader_kw["section"]).split(".")[-1]
    out = out_dir / f"{key}{suffix}.h5ad"

    if out.exists() and not overwrite:
        log.info(f"[skip] {out.name} exists")
        return out

    adata = load_raw(key, raw_dir, **loader_kw)
    adata = preprocess(
        adata,
        key,
        qc=QCConfig(**QC_PRESETS.get(key, {})),
        split_mode=split_mode,
        n_blocks=n_blocks,
        seed=seed,
        spatial=key in SPATIAL_KEYS,
    )
    if key in SPATIAL_KEYS:
        _assert_spatial_built(adata, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    # uns must be h5ad-serialisable
    adata.uns["nmo"] = {k: v for k, v in adata.uns["nmo"].items() if _serialisable(v)}
    adata.write_h5ad(out, compression="gzip")
    log.info(f"[ok]   wrote {out}  ({adata.n_obs} x {adata.n_vars}, {out.stat().st_size/1e6:.1f} MB)")
    return out


def _serialisable(v) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict, np.ndarray, type(None)))


def summarise(out_dir: Path) -> Dict:
    import anndata as ad

    summary = {}
    for p in sorted(out_dir.glob("*.h5ad")):
        a = ad.read_h5ad(p, backed="r")
        prov = a.uns.get("nmo", {})
        rec = {
            "file": p.name,
            "n_obs": int(a.n_obs),
            "n_vars": int(a.n_vars),
            "technology": prov.get("technology", ""),
            "organism": prov.get("organism", ""),
            "tissue": prov.get("tissue", ""),
            "resolution": prov.get("resolution", ""),
            "accession": prov.get("accession", ""),
            "size_mb": round(p.stat().st_size / 1e6, 1),
        }
        if "spatial_um" in a.obsm:
            xy = np.asarray(a.obsm["spatial_um"])
            rec["extent_um"] = [round(float(np.ptp(xy[:, 0]))), round(float(np.ptp(xy[:, 1])))]
            rec["coord_scale_um"] = round(float(prov.get("coord_scale_um", 0)), 1)
        if "split" in a.obs.columns:
            vc = a.obs["split"].value_counts().to_dict()
            rec["split"] = {str(k): int(v) for k, v in vc.items()}
        qc = prov.get("qc", {})
        if qc:
            rec["qc_kept_frac"] = round(qc["n_obs_after"] / max(qc["n_obs_before"], 1), 3)
        summary[p.stem] = rec
        a.file.close()
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--dataset", nargs="+", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-mode", default="block", choices=["block", "random"])
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args(argv)

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.summary_only:
        print(json.dumps(summarise(out_dir), indent=2))
        return 0

    keys = args.dataset or (list(sources.DATASETS) if args.all else None)
    if not keys:
        p.error("specify --all or --dataset KEY...")

    failures = []
    for key in keys:
        try:
            if key == "mosta_embryo":
                # Every developmental stage becomes its own processed object;
                # the temporal experiment consumes the ordered series.
                for stage in sources.MOSTA_STAGES:
                    if not (raw_dir / "mosta_embryo" / f"{stage}.MOSTA.h5ad").exists():
                        log.info(f"[skip] mosta stage {stage} not downloaded")
                        continue
                    build_one(key, raw_dir, out_dir, args.seed, args.split_mode,
                              args.n_blocks, args.overwrite, stage=stage)
            elif key == "merfish_allen":
                for sec in sources.MERFISH_SECTIONS:
                    if not (raw_dir / "merfish_allen" / f"{sec}-log2.h5ad").exists():
                        log.info(f"[skip] merfish section {sec} not downloaded")
                        continue
                    build_one(key, raw_dir, out_dir, args.seed, args.split_mode,
                              args.n_blocks, args.overwrite, section=sec)
            else:
                build_one(key, raw_dir, out_dir, args.seed, args.split_mode,
                          args.n_blocks, args.overwrite)
        except Exception as exc:
            log.info(f"[FAIL] {key}: {exc}")
            traceback.print_exc()
            failures.append(key)

    s = summarise(out_dir)
    log.info(f"processed objects: {len(s)}")
    for k, v in s.items():
        log.info(f"  {k:<28} {v['n_obs']:>7} x {v['n_vars']:>5}  {v.get('resolution','')}")
    if failures:
        log.info(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

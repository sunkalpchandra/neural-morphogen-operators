"""Stream additional MERFISH coronal sections into the processed pool.

The benchmark originally ran on a single tissue section, which is the main
limitation reviewers would raise against the empirical side of the paper. The
Allen ABC atlas provides 59 coronal sections of the same brain, which gives a
principled way to scale the evaluation: the anatomy varies systematically along
the anterior--posterior axis while assay, gene panel and preprocessing are held
fixed, so differences across sections reflect tissue architecture rather than
batch or platform.

Disk is the binding constraint on this machine (~6 GB free against 7.7 GB of
section files), so sections are processed one at a time and the raw download is
deleted immediately after the processed ``.h5ad`` is written. Peak usage stays
near the size of the shared cell-metadata table.

    python scripts/expand_merfish.py --n 14
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build import build_one
from src.utils.common import get_logger

log = get_logger("nmo.expand")
MANIFEST = ("https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
            "releases/20230830/manifest.json")
METADATA = ("https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
            "metadata/MERFISH-C57BL6J-638850/20230830/cell_metadata.csv")


def _download(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nmo/1.0 (academic)"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.rename(dest)
    log.info(f"  downloaded {desc or dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


def sections_from_manifest(n: int, exclude: List[str]) -> List[Dict]:
    m = json.load(urllib.request.urlopen(MANIFEST, timeout=120))
    node = m["file_listing"]["MERFISH-C57BL6J-638850-sections"]["expression_matrices"]
    rows = []
    for s in sorted(node):
        f = node[s].get("log2", {}).get("files", {}).get("h5ad")
        if f and s not in exclude:
            rows.append(dict(section=s, url=f["url"], bytes=int(f["size"])))
    step = max(1, len(rows) // n)          # spread along the A-P axis
    return rows[::step][:n]


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--min-free-gb", type=float, default=1.5,
                   help="abort before a download that would breach this margin")
    p.add_argument("--keep-raw", action="store_true")
    a = p.parse_args()

    raw = Path(a.raw_dir) / "merfish_allen"
    out = Path(a.out_dir)
    have = [f.stem.replace("merfish_allen_", "C57BL6J-638850.")
            for f in out.glob("merfish_allen_*.h5ad")]
    log.info(f"already processed: {sorted(have)}")

    sel = sections_from_manifest(a.n, exclude=have)
    log.info(f"selected {len(sel)} sections, {sum(r['bytes'] for r in sel)/1e9:.2f} GB to stream")

    # the coordinate table is shared across sections and must be present
    _download(METADATA, raw / "cell_metadata.csv", "cell_metadata.csv (538 MB)")

    built, failed = [], []
    for i, r in enumerate(sel, 1):
        sec = r["section"]
        if free_gb(out) < a.min_free_gb:
            log.info(f"[stop] free disk {free_gb(out):.1f} GB below margin; built {len(built)}")
            break
        log.info(f"[{i}/{len(sel)}] {sec}  ({r['bytes']/1e6:.0f} MB, free {free_gb(out):.1f} GB)")
        h5 = raw / f"{sec}-log2.h5ad"
        try:
            _download(r["url"], h5, sec)
            build_one("merfish_allen", Path(a.raw_dir), out, section=sec)
            built.append(sec)
        except Exception as exc:
            log.info(f"  [FAIL] {sec}: {exc}")
            failed.append(sec)
        finally:
            if not a.keep_raw and h5.exists():
                h5.unlink()            # reclaim immediately; re-downloadable
    log.info(f"built {len(built)} new sections; failures: {failed}")
    log.info(f"processed pool now: {len(list(out.glob('*.h5ad')))} .h5ad files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

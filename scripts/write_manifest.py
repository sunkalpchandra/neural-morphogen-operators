"""Record a SHA256 for every processed artifact and generated paper input.

The paper claims a manifest per artifact. That claim is only worth making if the
manifest covers what the paper is actually built from -- the processed sections,
the results the numbers come from, and the generated tables -- and if it can be
verified rather than taken on trust.

    python scripts/write_manifest.py            # write
    python scripts/write_manifest.py --verify   # check, non-zero exit on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    "data/processed/*.h5ad",
    "data/processed/SUMMARY.json",
    "results/**/*.json",
    "paper/tables/*.tex",
    "paper/numbers.tex",
]


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect() -> dict:
    out = {}
    for pat in PATTERNS:
        for f in sorted(ROOT.glob(pat)):
            if f.is_file():
                out[str(f.relative_to(ROOT))] = {"sha256": sha256(f),
                                                 "bytes": f.stat().st_size}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="MANIFEST.sha256.json")
    a = ap.parse_args()
    dest = ROOT / a.out
    now = collect()

    if a.verify:
        if not dest.exists():
            print(f"no manifest at {a.out}; run `make manifest` first")
            return 1
        was = json.loads(dest.read_text())["files"]
        added = sorted(set(now) - set(was))
        removed = sorted(set(was) - set(now))
        changed = sorted(k for k in set(now) & set(was)
                         if now[k]["sha256"] != was[k]["sha256"])
        for label, items in [("added", added), ("removed", removed),
                             ("changed", changed)]:
            if items:
                print(f"{label} ({len(items)}):")
                for k in items[:10]:
                    print(f"   {k}")
        if not (added or removed or changed):
            print(f"manifest verified: {len(now)} artifacts unchanged")
            return 0
        return 1

    total = sum(v["bytes"] for v in now.values())
    dest.write_text(json.dumps({"n_files": len(now), "total_bytes": total,
                                "files": now}, indent=2) + "\n")
    print(f"wrote {a.out}: {len(now)} artifacts, {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

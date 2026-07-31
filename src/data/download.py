"""Reproducible, resumable downloader for every dataset in the registry.

The only module in the project that performs network I/O.

Features
--------
* Resumable: uses HTTP Range requests to continue partial downloads.
* Idempotent: a file whose on-disk size matches the server's
  ``Content-Length`` is skipped.
* Verifiable: writes ``data/raw/MANIFEST.json`` recording URL, byte size and
  SHA256 of every artifact actually fetched, so a collaborator can confirm they
  have bit-identical inputs.
* Polite: sequential by default with a small retry/backoff loop.

Usage
-----
    python -m src.data.download --list
    python -m src.data.download --all
    python -m src.data.download --dataset visium_mouse_brain merfish_allen
    python -m src.data.download --all --with-images --with-transcripts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from . import sources
from .sources import DatasetSpec, RemoteFile

CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "neural-morphogen-operators/1.0 (academic research; contact via repo)"
MAX_RETRIES = 5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:6.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


def _remote_size(url: str, session: requests.Session) -> Optional[int]:
    """Content length via HEAD, falling back to a 1-byte Range GET.

    Some of our hosts (CNGB FTP-over-HTTP, NCBI) do not return a useful
    Content-Length on HEAD, so the Range fallback matters.
    """
    try:
        r = session.head(url, allow_redirects=True, timeout=60)
        if r.status_code < 400:
            cl = r.headers.get("Content-Length")
            if cl and int(cl) > 0:
                return int(cl)
    except requests.RequestException:
        pass
    try:
        r = session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
        cr = r.headers.get("Content-Range")
        r.close()
        if cr and "/" in cr:
            total = cr.rsplit("/", 1)[1]
            if total.isdigit():
                return int(total)
    except requests.RequestException:
        pass
    return None


def _sha256(path: Path, limit: Optional[int] = None) -> str:
    """SHA256 of a file. ``limit`` hashes only the first N bytes (fast mode)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            n = CHUNK if limit is None else min(CHUNK, limit - read)
            if n <= 0:
                break
            b = fh.read(n)
            if not b:
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()


def _download_one(rf: RemoteFile, raw_dir: Path, session: requests.Session) -> Path:
    """Fetch a single artifact with resume + retry. Returns the local path."""
    dest = raw_dir / rf.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    total = _remote_size(rf.url, session)

    if dest.exists():
        have = dest.stat().st_size
        if total is None or have == total:
            print(f"  [skip]  {rf.dest} ({_human(have)}) already complete")
            return dest
        print(f"  [warn]  {rf.dest} size {have} != remote {total}; re-fetching")
        dest.unlink()

    for attempt in range(1, MAX_RETRIES + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {}
        mode = "wb"
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
            mode = "ab"
        try:
            with session.get(rf.url, headers=headers, stream=True, timeout=(30, 300)) as r:
                if offset > 0 and r.status_code == 200:
                    # Server ignored our Range header -> restart cleanly.
                    offset, mode = 0, "wb"
                r.raise_for_status()
                t0, done = time.time(), offset
                with open(part, mode) as fh:
                    for chunk in r.iter_content(CHUNK):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100.0 * done / total
                            rate = (done - offset) / max(time.time() - t0, 1e-6)
                            sys.stdout.write(
                                f"\r  [get ]  {rf.dest:<52} {pct:5.1f}%  "
                                f"{_human(done)}/{_human(total)}  {_human(rate)}/s   "
                            )
                        else:
                            sys.stdout.write(f"\r  [get ]  {rf.dest:<52} {_human(done)}   ")
                        sys.stdout.flush()
            sys.stdout.write("\n")
            if total is not None and part.stat().st_size != total:
                raise IOError(f"truncated: got {part.stat().st_size}, expected {total}")
            part.rename(dest)
            return dest
        except (requests.RequestException, IOError) as exc:
            wait = min(2**attempt, 30)
            sys.stdout.write("\n")
            print(f"  [retry] {rf.dest} attempt {attempt}/{MAX_RETRIES}: {exc} (sleep {wait}s)")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _extract(path: Path) -> None:
    """Expand a tar.gz archive next to itself, once."""
    marker = path.with_suffix(path.suffix + ".extracted")
    if marker.exists():
        print(f"  [skip]  {path.name} already extracted")
        return
    print(f"  [tar ]  extracting {path.name}")
    with tarfile.open(path, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            # Guard against path traversal in untrusted archives.
            target = (path.parent / m.name).resolve()
            if not str(target).startswith(str(path.parent.resolve())):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        tf.extractall(path.parent)
    marker.write_text("ok\n")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def download_dataset(
    spec: DatasetSpec,
    raw_dir: Path,
    include_optional: Iterable[str] = (),
    hash_limit: Optional[int] = 64 << 20,
) -> List[Dict]:
    """Download one dataset; returns manifest records for the fetched files."""
    print(f"\n=== {spec.key} :: {spec.title}")
    print(f"    {spec.technology} | {spec.organism} | {spec.accession}")

    wanted = list(spec.required_files)
    for rf in spec.optional_files:
        tag = "images" if rf.dest.endswith((".tif", ".tiff")) else None
        if "transcripts" in rf.dest:
            tag = "transcripts"
        if "whole-brain" in rf.dest:
            tag = "wholebrain"
        if tag in include_optional:
            wanted.append(rf)

    records: List[Dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for rf in wanted:
        path = _download_one(rf, raw_dir, session)
        if rf.extract:
            _extract(path)
        size = path.stat().st_size
        records.append(
            {
                "dataset": spec.key,
                "url": rf.url,
                "path": str(path.relative_to(raw_dir)),
                "role": rf.role,
                "bytes": size,
                "sha256_prefix_bytes": min(size, hash_limit) if hash_limit else size,
                "sha256": _sha256(path, hash_limit),
            }
        )
    return records


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw", help="destination for raw artifacts")
    p.add_argument("--dataset", nargs="+", default=None, help="subset of dataset keys")
    p.add_argument("--all", action="store_true", help="download the full default set")
    p.add_argument("--list", action="store_true", help="print the dataset inventory and exit")
    p.add_argument("--with-images", action="store_true", help="also fetch full-resolution H&E TIFFs (3.9 GB)")
    p.add_argument("--with-transcripts", action="store_true", help="also fetch Xenium molecule table (384 MB)")
    p.add_argument("--with-wholebrain", action="store_true", help="also fetch the 7.6 GB MERFISH whole-brain matrix")
    p.add_argument("--full-hash", action="store_true", help="hash entire files (slow) rather than first 64 MB")
    args = p.parse_args(argv)

    if args.list:
        print(sources.summary_table())
        print("\nOptional extras (not downloaded by default):")
        for d in sources.DATASETS.values():
            for rf in d.optional_files:
                print(f"  {d.key:<22} {rf.dest:<44} {_human(rf.size)}  {rf.role}")
        return 0

    keys = args.dataset or (sources.DEFAULT_DATASETS if args.all else None)
    if not keys:
        p.error("specify --all, --dataset KEY..., or --list")

    optional = set()
    if args.with_images:
        optional.add("images")
    if args.with_transcripts:
        optional.add("transcripts")
    if args.with_wholebrain:
        optional.add("wholebrain")

    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(raw_dir).free
    print(f"Destination: {raw_dir}   (free: {_human(free)})")

    manifest_path = raw_dir / "MANIFEST.json"
    manifest: Dict[str, Dict] = {}
    if manifest_path.exists():
        try:
            manifest = {r["path"]: r for r in json.loads(manifest_path.read_text())["files"]}
        except Exception:
            manifest = {}

    t0 = time.time()
    failures = []
    for key in keys:
        spec = sources.get(key)
        try:
            for rec in download_dataset(
                spec, raw_dir, optional, None if args.full_hash else (64 << 20)
            ):
                manifest[rec["path"]] = rec
        except Exception as exc:  # keep going; report at the end
            print(f"  [FAIL]  {key}: {exc}")
            failures.append((key, str(exc)))
        # Persist incrementally so a crash does not lose provenance.
        manifest_path.write_text(
            json.dumps(
                {
                    "generated_by": "src.data.download",
                    "n_files": len(manifest),
                    "total_bytes": sum(r["bytes"] for r in manifest.values()),
                    "files": sorted(manifest.values(), key=lambda r: r["path"]),
                },
                indent=2,
            )
        )

    total = sum(r["bytes"] for r in manifest.values())
    print(f"\nDone in {time.time() - t0:.0f}s. {len(manifest)} artifacts, {_human(total)} total.")
    print(f"Manifest: {manifest_path}")
    if failures:
        print("\nFAILURES:")
        for k, e in failures:
            print(f"  {k}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

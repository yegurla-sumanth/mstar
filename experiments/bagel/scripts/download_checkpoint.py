#!/usr/bin/env python3
"""Download a pinned Hugging Face snapshot and write inexpensive provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = "ByteDance-Seed/BAGEL-7B-MoT"
DEFAULT_REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
SMALL_HASH_LIMIT = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    started = datetime.now(timezone.utc)
    snapshot = Path(
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            cache_dir=str(args.cache_dir),
            max_workers=args.max_workers,
            local_files_only=args.local_files_only,
        )
    ).resolve()
    files = []
    total_bytes = 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        resolved = path.resolve()
        entry = {
            "path": str(path.relative_to(snapshot)),
            "size_bytes": size,
            # Hugging Face cache blob names provide provenance without rereading
            # every multi-gigabyte weight file solely to hash it again.
            "cache_blob": resolved.name if resolved != path else None,
        }
        if path.suffix == ".safetensors":
            # Reading the key table is inexpensive and catches truncated files,
            # Git-LFS pointer payloads, and cache filesystems that cannot serve
            # safetensors correctly before an expensive model startup attempt.
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    entry["safetensors_key_count"] = len(handle.keys())
            except Exception as exc:
                raise RuntimeError(
                    f"invalid safetensors checkpoint file {path}: {exc}"
                ) from exc
        if size <= SMALL_HASH_LIMIT:
            entry["sha256"] = sha256(path)
        files.append(entry)
    finished = datetime.now(timezone.utc)
    payload = {
        "repo_id": args.repo,
        "requested_revision": args.revision,
        "resolved_snapshot_path": str(snapshot),
        "resolved_revision": snapshot.name,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    if snapshot.name != args.revision:
        raise RuntimeError(f"resolved snapshot {snapshot.name!r} does not match requested revision {args.revision!r}")
    atomic_json(args.manifest, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

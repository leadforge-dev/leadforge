#!/usr/bin/env python3
"""Verify SHA-256 hash determinism of the public release build.

Runs ``scripts/build_public_release.py`` twice into two temp directories with
the same seed/config and compares the SHA-256 digest of every generated file.

The architectural invariant is that generation is deterministic given
``(recipe, config, seed, version)``.  This script enforces that invariant on
the bundle layer: every file written under each bundle directory must hash
identically across runs.

Two practical exceptions are handled:

1. ``manifest.json`` contains a ``generation_timestamp`` field set to
   ``datetime.now(UTC)`` at write time, so the file bytes legitimately differ
   between runs.  The script strips that field and compares the remaining
   manifest payload (which already includes per-file SHA-256 digests for the
   relational and task Parquet files).

2. ``LICENSE`` is copied from the repo root and is identical by construction;
   it is hashed and compared like any other file.

Exit code: 0 on PASS (all hashes match), 1 on FAIL.

Usage:
    python scripts/verify_hash_determinism.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_public_release.py"

# Field stripped before comparing manifest payloads; differs by design between
# runs (set to wall-clock time in build_manifest()).
MANIFEST_TIMESTAMP_FIELD = "generation_timestamp"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files(root: Path) -> list[Path]:
    """Return all regular files under *root*, sorted by relative path."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def hash_tree(root: Path) -> dict[str, str]:
    """Map relative-path → SHA-256 for every file under *root*."""
    return {str(p.relative_to(root)): file_sha256(p) for p in walk_files(root)}


def manifest_payload_without_timestamp(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload.pop(MANIFEST_TIMESTAMP_FIELD, None)
    return payload


def run_build(out_dir: Path) -> None:
    cmd = [sys.executable, str(BUILD_SCRIPT), str(out_dir)]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)  # noqa: S603


def compare(run_a: Path, run_b: Path) -> list[str]:
    """Return a list of human-readable mismatch messages (empty == identical)."""
    tree_a = hash_tree(run_a)
    tree_b = hash_tree(run_b)

    mismatches: list[str] = []

    only_a = sorted(set(tree_a) - set(tree_b))
    only_b = sorted(set(tree_b) - set(tree_a))
    for rel in only_a:
        mismatches.append(f"only in run A: {rel}")
    for rel in only_b:
        mismatches.append(f"only in run B: {rel}")

    for rel in sorted(set(tree_a) & set(tree_b)):
        if tree_a[rel] == tree_b[rel]:
            continue
        # manifest.json carries a wall-clock timestamp; compare the rest.
        if Path(rel).name == "manifest.json":
            payload_a = manifest_payload_without_timestamp(run_a / rel)
            payload_b = manifest_payload_without_timestamp(run_b / rel)
            if payload_a == payload_b:
                continue
            mismatches.append(
                f"manifest payload mismatch (excluding {MANIFEST_TIMESTAMP_FIELD}): {rel}"
            )
            continue
        mismatches.append(f"hash mismatch: {rel}\n    A={tree_a[rel]}\n    B={tree_b[rel]}")

    return mismatches


def main() -> int:
    if not BUILD_SCRIPT.exists():
        print(f"FAIL: build script not found at {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="leadforge_determinism_") as tmp:
        run_a = Path(tmp) / "run_a"
        run_b = Path(tmp) / "run_b"

        print(f"Run A → {run_a}")
        run_build(run_a)
        print(f"Run B → {run_b}")
        run_build(run_b)

        files_a = len(walk_files(run_a))
        files_b = len(walk_files(run_b))
        print(f"\nRun A produced {files_a} files; run B produced {files_b} files.")

        mismatches = compare(run_a, run_b)

        if not mismatches:
            print(f"\nPASS: all {files_a} files hash identically across runs.")
            print(
                f"(manifest.json compared after stripping {MANIFEST_TIMESTAMP_FIELD}, "
                "which is wall-clock by design.)"
            )
            return 0

        print(f"\nFAIL: {len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  - {m}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

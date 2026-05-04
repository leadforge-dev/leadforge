#!/usr/bin/env python3
"""Verify SHA-256 hash determinism of the public release build.

Runs ``scripts/build_public_release.py`` twice into two output directories with
the same seed/config and a *pinned* manifest timestamp, then asserts every
generated file hashes identically across runs.

Pinning ``--generation-timestamp`` on the build script means the resulting
``manifest.json`` is also byte-identical — no special-cased manifest stripping
needed at compare time.  (For defence-in-depth, the underlying
:func:`leadforge.validation.invariants.compare_bundle_trees` still tolerates
a wall-clock-only manifest diff, but pinning is the supported workflow.)

The architectural invariant being enforced is
"generation is deterministic given (recipe, config, seed, version)".
The corresponding fast in-process check lives in
``tests/validation/test_invariants.py::TestDeterminism`` and runs in CI on
every PR; this script is the slower release-time check that exercises the
full ``build_public_release.py`` pipeline.

On failure, output directories are preserved (NOT auto-cleaned) so the
mismatching artifacts can be diffed directly.

Exit code: 0 on PASS, 1 on FAIL.

Usage:
    python scripts/verify_hash_determinism.py [--out DIR] [--keep-on-success]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from leadforge.core.hashing import file_sha256
from leadforge.validation.invariants import compare_bundle_trees

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_public_release.py"

# Pinned timestamp for both runs.  Any fixed ISO-8601 UTC string works; using
# the unix epoch makes it obvious that it's a sentinel, not a real run time.
PINNED_TIMESTAMP = "1970-01-01T00:00:00+00:00"

# Bundle subdirectories produced by build_public_release.py.  Hardcoded here
# because the script's BUNDLES list is not exposed as a public API.  If the
# build script grows new bundles, add them here.
BUNDLE_DIRS = ("intro", "intermediate", "advanced", "intermediate_instructor")


def run_build(out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        str(out_dir),
        "--generation-timestamp",
        PINNED_TIMESTAMP,
    ]
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)  # noqa: S603


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "release" / "_determinism",
        help="Base directory for both runs (will be wiped at start). "
        "Default: release/_determinism/",
    )
    parser.add_argument(
        "--keep-on-success",
        action="store_true",
        help="Keep output directories even on PASS (default: clean up on PASS, "
        "always preserve on FAIL).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not BUILD_SCRIPT.exists():
        print(f"FAIL: build script not found at {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    base = args.out
    run_a = base / "run_a"
    run_b = base / "run_b"

    # Wipe and recreate.
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    print(f"Run A → {run_a}")
    run_build(run_a)
    print(f"Run B → {run_b}")
    run_build(run_b)

    # Per-bundle comparison so error messages stay scoped to a single bundle.
    all_errors: list[tuple[str, list[str]]] = []
    total_files = 0
    for name in BUNDLE_DIRS:
        bundle_a = run_a / name
        bundle_b = run_b / name
        if not bundle_a.exists() or not bundle_b.exists():
            all_errors.append((name, [f"bundle directory missing: {name}"]))
            continue
        errors = compare_bundle_trees(bundle_a, bundle_b)
        bundle_files = sum(1 for p in bundle_a.rglob("*") if p.is_file())
        total_files += bundle_files
        if errors:
            all_errors.append((name, errors))

    # Top-level files (LICENSE, etc.) — compare via hash directly.
    top_a = {p.name for p in run_a.iterdir() if p.is_file()}
    top_b = {p.name for p in run_b.iterdir() if p.is_file()}
    top_errors: list[str] = []
    for name in sorted(top_a - top_b):
        top_errors.append(f"top-level file only in A: {name}")
    for name in sorted(top_b - top_a):
        top_errors.append(f"top-level file only in B: {name}")
    for name in sorted(top_a & top_b):
        if file_sha256(run_a / name) != file_sha256(run_b / name):
            top_errors.append(f"top-level hash mismatch: {name}")
    total_files += len(top_a)
    if top_errors:
        all_errors.append(("<top-level>", top_errors))

    if not all_errors:
        print(f"\nPASS: all {total_files} files hash identically across runs.")
        if not args.keep_on_success:
            shutil.rmtree(base)
            print(f"(cleaned up {base})")
        else:
            print(f"(kept artifacts at {base})")
        return 0

    print(f"\nFAIL: mismatches in {len(all_errors)} bundle(s):")
    for name, errors in all_errors:
        print(f"  [{name}]")
        for e in errors:
            print(f"    - {e}")
    print(f"\nArtifacts preserved for inspection:\n  A: {run_a}\n  B: {run_b}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

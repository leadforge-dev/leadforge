#!/usr/bin/env python3
"""Publish ``leadforge-lead-scoring-v1`` to Kaggle.

This is the final publish gate for the Kaggle side of the v1 release.
It wraps the two pre-publish steps that must already be complete
(packaging and linting) and then calls the Kaggle CLI to create or
update the dataset.

Three-stage runbook
-------------------
1. **Dry-run** (safe, no credentials needed)::

       python scripts/publish_kaggle.py --dry-run

   Re-packages ``release/kaggle/`` via ``package_kaggle_release.py``,
   lints the metadata via ``lint_platform_metadata.py``, and prints a
   summary.  Exits ``0`` only if every pre-flight check passes.

2. **Upload as private** (first publish)::

       python scripts/publish_kaggle.py

   Requires ``~/.kaggle/kaggle.json`` (or ``KAGGLE_USERNAME`` /
   ``KAGGLE_KEY`` env vars).  Calls ``kaggle datasets create`` without
   ``--public``, creating a private dataset.  Review the live Kaggle
   page; when satisfied, proceed to step 3.

3. **Flip to public** (manual step — no CLI flag for this)::

   There is no ``--go-public`` flag.  After reviewing the private dataset,
   flip visibility via the Kaggle web UI (Settings → Visibility → Public)
   or via the Kaggle API::

       kaggle datasets metadata {DATASET_ID}   # download current metadata.json
       # edit: set  isPrivate: false
       kaggle datasets update {DATASET_ID}     # push the change

   This script prints the exact commands with the real dataset ID after a
   successful private upload (step 2).

For a **new version** of an already-public dataset (future releases)::

    python scripts/publish_kaggle.py --update "Release notes for v1.1"

Options
-------
--release-dir PATH      Root of the release directory (default: release/).
--kaggle-dir PATH       Upload tree root (default: release/kaggle/).
--dry-run               Package + lint; no upload.
--public                Upload publicly in one step (skips private staging).
--update MESSAGE        Push a new version; MESSAGE is the version note.
--quiet                 Suppress Kaggle CLI progress output.
--dir-mode {zip,tar,skip}
                        How to handle subdirectories (default: zip).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

# Make ``scripts/`` importable regardless of invocation style.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lint_platform_metadata import LintOutcome, run_lint  # noqa: E402
from package_kaggle_release import (  # noqa: E402
    DEFAULT_DATASET_SLUG,
    DEFAULT_KAGGLE_DIR,
    DEFAULT_RELEASE_DIR,
    DEFAULT_USER_SLUG,
    run_packager,
)

DATASET_ID: Final[str] = f"{DEFAULT_USER_SLUG}/{DEFAULT_DATASET_SLUG}"
KAGGLE_DATASET_URL: Final[str] = f"https://www.kaggle.com/datasets/{DATASET_ID}"


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _repackage(release_dir: Path, kaggle_dir: Path) -> bool:
    """Re-run the packager to ensure the upload tree is current.

    Returns ``True`` on success, ``False`` on validation failure.
    Exits with rc=2 on pre-flight error (missing dirs).
    """
    print("[ 1/3 ] Packaging release/kaggle/ …", file=sys.stderr)
    try:
        outcome = run_packager(
            release_dir,
            kaggle_dir=kaggle_dir,
            dry_run=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"  error: {exc}", file=sys.stderr)
        sys.exit(2)

    if outcome.errors:
        print("  FAIL: packaging validation errors:", file=sys.stderr)
        for err in outcome.errors:
            print(f"    - {err.field}: {err.message}", file=sys.stderr)
        return False

    print(f"  OK   metadata → {outcome.metadata_path}", file=sys.stderr)
    if outcome.assembled:
        print(f"  OK   upload tree → {kaggle_dir}", file=sys.stderr)
    return True


def _lint(release_dir: Path) -> bool:
    """Run ``lint_platform_metadata`` against the packaged artifacts.

    Returns ``True`` on clean, ``False`` on lint failure.
    """
    print("[ 2/3 ] Linting platform metadata …", file=sys.stderr)
    try:
        outcome: LintOutcome = run_lint(release_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not outcome.ok:
        print("  FAIL: lint errors:", file=sys.stderr)
        for f in outcome.findings:
            print(f"    - {f.field}: {f.message}", file=sys.stderr)
        return False

    print("  OK   metadata passes all lint checks", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _kaggle_create(
    kaggle_dir: Path,
    *,
    public: bool,
    quiet: bool,
    dir_mode: str,
) -> int:
    """Call ``kaggle datasets create``; return the process exit code."""
    cmd = [
        "kaggle",
        "datasets",
        "create",
        "--path",
        str(kaggle_dir),
        "--dir-mode",
        dir_mode,
        "--keep-tabular",
    ]
    if public:
        cmd.append("--public")
    if quiet:
        cmd.append("--quiet")
    print(f"[ 3/3 ] Running: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd).returncode  # noqa: S603,S607


def _kaggle_version(
    kaggle_dir: Path,
    message: str,
    *,
    quiet: bool,
    dir_mode: str,
) -> int:
    """Call ``kaggle datasets version``; return the process exit code."""
    cmd = [
        "kaggle",
        "datasets",
        "version",
        "--path",
        str(kaggle_dir),
        "--message",
        message,
        "--dir-mode",
        dir_mode,
        "--keep-tabular",
    ]
    if quiet:
        cmd.append("--quiet")
    print(f"[ 3/3 ] Running: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd).returncode  # noqa: S603,S607


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        metavar="PATH",
        help="Root of the release directory (default: release/)",
    )
    parser.add_argument(
        "--kaggle-dir",
        type=Path,
        default=DEFAULT_KAGGLE_DIR,
        metavar="PATH",
        help="Upload tree root (default: release/kaggle/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Package + lint only; do not upload",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Upload publicly in one step (skip private staging)",
    )
    parser.add_argument(
        "--update",
        metavar="MESSAGE",
        default=None,
        help="Push a new dataset version with the given version note",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress Kaggle CLI progress output",
    )
    parser.add_argument(
        "--dir-mode",
        choices=["zip", "tar", "skip"],
        default="zip",
        help="How to handle subdirectories (default: zip)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    release_dir: Path = args.release_dir.resolve()
    kaggle_dir: Path = args.kaggle_dir.resolve()

    # --- Pre-flight ---------------------------------------------------------
    ok = _repackage(release_dir, kaggle_dir)
    ok = _lint(release_dir) and ok
    if not ok:
        print("\nPre-flight FAILED — fix errors above before uploading.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            "\nDry-run complete — all pre-flight checks passed.",
            file=sys.stderr,
        )
        print(f"Upload tree is ready at: {kaggle_dir}", file=sys.stderr)
        print(
            "\nTo upload (private staging):\n"
            "  python scripts/publish_kaggle.py\n"
            "\nTo upload publicly in one step:\n"
            "  python scripts/publish_kaggle.py --public",
            file=sys.stderr,
        )
        return 0

    # --- Upload -------------------------------------------------------------
    if args.update:
        rc = _kaggle_version(
            kaggle_dir,
            args.update,
            quiet=args.quiet,
            dir_mode=args.dir_mode,
        )
    else:
        rc = _kaggle_create(
            kaggle_dir,
            public=args.public,
            quiet=args.quiet,
            dir_mode=args.dir_mode,
        )

    if rc != 0:
        print(f"\nKaggle CLI exited with code {rc}.", file=sys.stderr)
        return rc

    if args.update:
        visibility_msg = "new version pushed"
    elif args.public:
        visibility_msg = "public"
    else:
        visibility_msg = "private"
    print(f"\nUpload succeeded ({visibility_msg}).", file=sys.stderr)
    print(f"Dataset URL: {KAGGLE_DATASET_URL}", file=sys.stderr)
    if not args.public and not args.update:
        print(
            "\nNext: review the private dataset at the URL above, then make it\n"
            "public via the Kaggle web UI (Settings → Visibility → Public) or:\n"
            f"  kaggle datasets metadata {DATASET_ID}  # download current metadata\n"
            f"  # edit isPrivate: false in the downloaded metadata.json\n"
            f"  kaggle datasets update {DATASET_ID}  # push the change",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

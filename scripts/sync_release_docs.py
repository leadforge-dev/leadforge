#!/usr/bin/env python3
"""Sync the agent-reviewable docs vendored under ``release/docs/``.

The Kaggle and HuggingFace mock pages link to documentation that lives
under ``docs/release/`` in the source repo.  An AI agent that lands on
the published bundle (or the mock preview) without web access cannot
follow those ``github.com/blob/main/...`` links, so the release-time
claims become unverifiable.

This script copies the canonical set of supporting docs into
``release/docs/`` so the published bundle is self-contained and the
mock previews render against the same files an agent would read on
Kaggle / HuggingFace.  The sync is idempotent: same inputs produce
byte-identical outputs.  CI runs ``--check`` to fail when the source
docs drift from the vendored copies.

Inputs (all under ``docs/release/``):

* ``generation_method.md`` — what is / isn't modelled by the DGP.
* ``channel_signal_audit.md`` — backing data for the "channel signal
  is weak" claim in the README.
* ``break_me_guide.md`` — nine adversarial patterns + how to detect
  them.
* ``feature_dictionary.md`` — long-form per-feature documentation.
* ``v1_acceptance_gates_bands.yaml`` — operational band thresholds.
* ``v2_decision_log.md`` — accepted-for-v2 findings register.

``release/docs/relational_table_schemas.csv`` is hand-authored (per
column docs for relational tables); validated against the live parquet
schemas, not copied from a source doc.

Exit codes: 0 success / 1 ``--check`` mode and copies are stale /
2 pre-flight error (source doc missing).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: ``(source, destination)`` pairs, both relative to the repo root.
#: Order is alphabetical by destination basename for deterministic
#: stderr output.
VENDORED_DOCS: Final[tuple[tuple[Path, Path], ...]] = (
    (
        Path("docs/release/break_me_guide.md"),
        Path("release/docs/break_me_guide.md"),
    ),
    (
        Path("docs/release/channel_signal_audit.md"),
        Path("release/docs/channel_signal_audit.md"),
    ),
    (
        Path("docs/release/feature_dictionary.md"),
        Path("release/docs/feature_dictionary.md"),
    ),
    (
        Path("docs/release/generation_method.md"),
        Path("release/docs/generation_method.md"),
    ),
    (
        Path("docs/release/v1_acceptance_gates_bands.yaml"),
        Path("release/docs/v1_acceptance_gates_bands.yaml"),
    ),
    (
        Path("docs/release/v2_decision_log.md"),
        Path("release/docs/v2_decision_log.md"),
    ),
)


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


@dataclass(frozen=True)
class _SyncResult:
    """Outcome of a sync run.

    * ``stale`` — destinations whose content differs from the source
      (overwritten unless ``check_only=True``).
    * ``missing_sources`` — sources declared in ``VENDORED_DOCS`` but
      absent on disk.
    * ``orphan_destinations`` — destinations whose content differs from
      the source AND whose mtime is newer than the source.  These look
      like local edits to the vendored copy; the sync refuses to clobber
      them unless ``force=True``, raising a clean error that points the
      reader at the source path.
    """

    stale: list[Path]
    missing_sources: list[Path]
    orphan_destinations: list[Path]


def sync_docs(repo_root: Path, *, check_only: bool, force: bool = False) -> _SyncResult:
    """Sync the vendored docs.

    Refuses to overwrite a destination that's newer than its source —
    that pattern means a contributor has edited the vendored copy
    (``release/docs/X.md``) rather than the canonical source
    (``docs/release/X.md``) and the sync would silently destroy their
    edit.  ``force=True`` bypasses the check (used by the
    ``--force`` CLI flag when the maintainer has confirmed the edits
    were intentional and is OK with discarding them).
    """

    stale: list[Path] = []
    missing_sources: list[Path] = []
    orphans: list[Path] = []

    for src_rel, dst_rel in VENDORED_DOCS:
        src = repo_root / src_rel
        dst = repo_root / dst_rel
        if not src.is_file():
            missing_sources.append(src_rel)
            continue
        src_bytes = _bytes(src)
        if dst.is_file() and _bytes(dst) == src_bytes:
            continue
        stale.append(dst_rel)
        if dst.is_file() and dst.stat().st_mtime > src.stat().st_mtime and not force:
            orphans.append(dst_rel)
            continue
        if not check_only:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    return _SyncResult(stale=stale, missing_sources=missing_sources, orphan_destinations=orphans)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sync_release_docs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale copies as an exit-code-1 failure without overwriting (CI use)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite destinations even when they appear to have been edited "
            "in place (mtime newer than source).  Default is to refuse and "
            "exit-code-1 so an accidental edit to release/docs/ is not silently "
            "discarded."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = sync_docs(REPO_ROOT, check_only=args.check, force=args.force)

    if result.missing_sources:
        print("error: source docs missing:", file=sys.stderr)
        for path in result.missing_sources:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if result.orphan_destinations and not args.force:
        print(
            "error: release/docs/ destinations look locally edited "
            "(mtime > source mtime).  Vendored docs are derived from "
            "docs/release/; edit the source there, then re-run this "
            "script.  Pass --force to discard the edits and overwrite "
            "from source:",
            file=sys.stderr,
        )
        for path in result.orphan_destinations:
            print(f"  - {path}", file=sys.stderr)
        return 1

    if args.check:
        if result.stale:
            print("error: release/docs/ is stale:", file=sys.stderr)
            for path in result.stale:
                print(f"  - {path}", file=sys.stderr)
            print(
                "run `python scripts/sync_release_docs.py` to refresh them.",
                file=sys.stderr,
            )
            return 1
        print("release/docs/ is up to date.", file=sys.stderr)
        return 0

    if result.stale:
        for path in result.stale:
            print(f"updated {path}", file=sys.stderr)
    else:
        print("release/docs/ is already up to date.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

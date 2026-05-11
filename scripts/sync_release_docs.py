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


def sync_docs(repo_root: Path, *, check_only: bool) -> tuple[list[Path], list[Path]]:
    """Sync the vendored docs.

    Returns ``(stale, missing_sources)``: ``stale`` is the list of
    destination paths whose content differs from the source (and were
    overwritten when ``check_only`` is False); ``missing_sources`` is
    the list of source paths the caller declared but that don't exist.
    """

    stale: list[Path] = []
    missing_sources: list[Path] = []
    for src_rel, dst_rel in VENDORED_DOCS:
        src = repo_root / src_rel
        dst = repo_root / dst_rel
        if not src.is_file():
            missing_sources.append(src_rel)
            continue
        src_bytes = _bytes(src)
        if not dst.is_file() or _bytes(dst) != src_bytes:
            stale.append(dst_rel)
            if not check_only:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    return stale, missing_sources


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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stale, missing = sync_docs(REPO_ROOT, check_only=args.check)

    if missing:
        print("error: source docs missing:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if args.check:
        if stale:
            print("error: release/docs/ is stale:", file=sys.stderr)
            for path in stale:
                print(f"  - {path}", file=sys.stderr)
            print(
                "run `python scripts/sync_release_docs.py` to refresh them.",
                file=sys.stderr,
            )
            return 1
        print("release/docs/ is up to date.", file=sys.stderr)
        return 0

    if stale:
        for path in stale:
            print(f"updated {path}", file=sys.stderr)
    else:
        print("release/docs/ is already up to date.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

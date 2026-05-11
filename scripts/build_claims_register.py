#!/usr/bin/env python3
"""Render the claims register from its YAML source.

``release/claims_register_source.yaml`` is the hand-edited source of
truth: every numerical / structural claim in the README plus a pointer
to the artifact and path that backs it.  This script renders two
machine-friendly outputs into the release tree:

* ``release/claims_register.json`` — structured payload an agent can
  parse without YAML support.  Includes the same claim metadata plus
  a top-level ``schema`` block describing the field semantics so a
  fresh agent doesn't have to infer them.
* ``release/claims_register.md`` — table-rendered version of the same
  data for humans skimming on GitHub or Kaggle.

Both files are deterministic: same source YAML → byte-identical
output.  ``--check`` mode reports drift as exit-code-1 without
overwriting (CI use).

Exit codes: 0 success / 1 ``--check`` mode and outputs are stale /
2 pre-flight error (source missing / malformed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE_DIR: Final[Path] = REPO_ROOT / "release"
DEFAULT_SOURCE: Final[Path] = DEFAULT_RELEASE_DIR / "claims_register_source.yaml"

#: Allowed category vocabulary; failing this is a build error.
VALID_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "composition",
        "calibration",
        "redaction",
        "difficulty",
        "limitations",
        "splits",
        "provenance",
        "out_of_scope",
        "intended_use",
    }
)

#: Required keys on every claim entry.
REQUIRED_CLAIM_KEYS: Final[tuple[str, ...]] = (
    "id",
    "text",
    "category",
    "backing_artifact",
    "backing_path",
    "verifier",
)

#: Schema description embedded in the JSON output so an agent landing
#: on ``claims_register.json`` without other context can interpret the
#: fields it sees.
SCHEMA_DOC: Final[dict[str, str]] = {
    "id": "Short stable identifier; quoted in CI failure messages.",
    "text": "The claim as it appears in the README (verbatim, where practical).",
    "category": (
        "One of: composition, calibration, redaction, difficulty, limitations, "
        "splits, provenance, out_of_scope, intended_use."
    ),
    "backing_artifact": (
        "Path within the published bundle (or repo) that carries the source of "
        "truth.  ``<tier>`` is a placeholder for intro / intermediate / "
        "advanced."
    ),
    "backing_path": (
        "JSON-path / YAML-path / column reference inside the backing artifact, "
        "or ``n/a`` for prose contracts and whole-file claims."
    ),
    "verifier": (
        "Free-form name of the script / probe / test that re-derives the "
        "claim end-to-end.  ``n/a`` means the claim is a prose contract that "
        "is not mechanically verifiable."
    ),
}


def _validate(claims: list[dict[str, Any]]) -> list[str]:
    """Return a list of human-readable validation errors (empty = OK)."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    for idx, claim in enumerate(claims):
        if not isinstance(claim, dict):
            errors.append(f"claims[{idx}] is not a mapping")
            continue
        for key in REQUIRED_CLAIM_KEYS:
            if key not in claim or claim.get(key) in (None, ""):
                errors.append(f"claims[{idx}] missing required key {key!r}")
        cid = claim.get("id")
        if isinstance(cid, str):
            if cid in seen_ids:
                errors.append(f"duplicate claim id {cid!r}")
            seen_ids.add(cid)
        category = claim.get("category")
        if isinstance(category, str) and category not in VALID_CATEGORIES:
            errors.append(f"claims[{idx}] category {category!r} not in {sorted(VALID_CATEGORIES)}")
    return errors


def load_claims(source_path: Path) -> list[dict[str, Any]]:
    """Load and validate the claims YAML."""

    if not source_path.is_file():
        raise FileNotFoundError(f"claims source not found at {source_path}")
    parsed = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or "claims" not in parsed:
        raise ValueError(f"{source_path}: expected top-level mapping with 'claims' key")
    claims = parsed["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"{source_path}: 'claims' must be a non-empty list")
    errors = _validate(claims)
    if errors:
        raise ValueError(f"{source_path} is invalid:\n  - " + "\n  - ".join(errors))
    return [dict(c) for c in claims]


def render_json(claims: list[dict[str, Any]]) -> str:
    """Deterministic JSON output with the schema embedded."""

    payload = {
        "schema": SCHEMA_DOC,
        "claims": [
            {
                "id": c["id"],
                "text": c["text"],
                "category": c["category"],
                "backing_artifact": c["backing_artifact"],
                "backing_path": c["backing_path"],
                "verifier": c["verifier"],
            }
            for c in claims
        ],
        "notes": (
            "This register is rendered from release/claims_register_source.yaml. "
            "Every claim in release/README.md should appear here.  Agents and CI "
            "can use the (backing_artifact, backing_path) tuple to locate the "
            "source-of-truth value without parsing prose."
        ),
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _escape_md(text: str) -> str:
    """Escape pipe characters so the cell doesn't break the table."""

    return text.replace("|", "\\|")


def render_markdown(claims: list[dict[str, Any]]) -> str:
    """Render a single GitHub-flavoured markdown table.

    Categories are grouped for readability; within a category, claim
    ids preserve source-file order.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        grouped.setdefault(claim["category"], []).append(claim)

    lines = [
        "# Claims register — `leadforge-lead-scoring-v1`",
        "",
        "Every numerical / structural claim made in `release/README.md` (and",
        "copied onto the Kaggle / HuggingFace dataset pages), paired with the",
        "artifact and path that backs it.  This file is auto-rendered from",
        "[`release/claims_register_source.yaml`](claims_register_source.yaml)",
        "by `scripts/build_claims_register.py`.  Edit the YAML, not this file.",
        "",
        "Tip for AI reviewers: `claims_register.json` is the machine-readable",
        "twin of this document with the same data plus a schema block.",
        "",
    ]

    for category in sorted(grouped):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| ID | Claim | Backing artifact | Path | Verifier |")
        lines.append("|---|---|---|---|---|")
        for claim in grouped[category]:
            row = (
                f"| `{claim['id']}` "
                f"| {_escape_md(claim['text'])} "
                f"| `{_escape_md(claim['backing_artifact'])}` "
                f"| `{_escape_md(claim['backing_path'])}` "
                f"| `{_escape_md(claim['verifier'])}` |"
            )
            lines.append(row)
        lines.append("")

    # Single trailing newline (no blank line at EOF) so the
    # ``end-of-file-fixer`` pre-commit hook is a no-op against the
    # rendered file.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def write_register(
    release_dir: Path,
    source_path: Path,
    *,
    check_only: bool,
) -> list[Path]:
    """Write (or check) the rendered files.  Returns the stale list."""

    claims = load_claims(source_path)
    json_path = release_dir / "claims_register.json"
    md_path = release_dir / "claims_register.md"

    stale: list[Path] = []

    def _write(path: Path, content: str) -> None:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        existing = path.read_text(encoding="utf-8") if path.is_file() else None
        if existing != content:
            stale.append(rel)
            if not check_only:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    _write(json_path, render_json(claims))
    _write(md_path, render_markdown(claims))
    return stale


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_claims_register",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release tree (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="path to claims_register_source.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale outputs as exit-code-1 without overwriting (CI use)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        stale = write_register(args.release_dir, args.source, check_only=args.check)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if stale:
            print("error: claims register is stale:", file=sys.stderr)
            for path in stale:
                print(f"  - {path}", file=sys.stderr)
            print(
                "run `python scripts/build_claims_register.py` to refresh.",
                file=sys.stderr,
            )
            return 1
        print("claims register is up to date.", file=sys.stderr)
        return 0

    if stale:
        for path in stale:
            print(f"wrote {path}", file=sys.stderr)
    else:
        print("claims register is already up to date.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

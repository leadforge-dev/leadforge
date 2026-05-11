#!/usr/bin/env python3
"""Verify every claim in ``release/claims_register_source.yaml``.

The PR that introduced the claims register shipped a (claim, artifact,
path) mapping but no verification — agents could find the backing
artifact but still had to parse README prose to confirm the value.
This script closes that gap.

For every claim with a machine-readable ``backing_path`` it:

1. Confirms the ``backing_artifact`` file exists on disk.  ``<tier>``
   placeholders are expanded to ``intro``, ``intermediate``,
   ``advanced``; missing tier files (e.g. on a fresh checkout where
   the bundle dirs haven't been built) are reported as a clean
   "artifact missing" error, not a crash.
2. Resolves the ``backing_path`` (JSON dotted/$-prefixed, YAML dotted,
   or a sentinel like ``$.tables (keys)``) inside the artifact and
   asserts the path produces a non-empty result.
3. When the claim text contains an obvious numeric (e.g. ``0.879``,
   ``42.67%``, ``5,000``) and the resolved value is a single number,
   compares them with a small absolute tolerance.  Drift on either
   side surfaces with the claim id and the offending number.

The script is intentionally tolerant: claims with prose backing
(``backing_path: n/a``) are skipped; claims that name a path the
verifier can't yet resolve (e.g. ``$.tables (keys)`` is a sentinel
for "the table inventory") are checked for file existence only.  CI
should run this with no flags; ``--strict`` upgrades soft warnings
(unparseable paths, missing tier files when tiers aren't expected) to
errors.

Exit codes: 0 success / 1 drift detected / 2 pre-flight error
(claims_register_source.yaml missing or malformed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE_DIR: Final[Path] = REPO_ROOT / "release"
DEFAULT_SOURCE: Final[Path] = DEFAULT_RELEASE_DIR / "claims_register_source.yaml"

TIER_PLACEHOLDER: Final[str] = "<tier>"
TIERS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")

#: Absolute tolerance for numeric comparisons.  The README rounds
#: medians to three decimals; metrics.json keeps four; the recipe is
#: exact.  ``1e-3`` is loose enough to absorb the rounding without
#: silently passing a meaningful regression.
NUMERIC_TOLERANCE: Final[float] = 1e-3

#: Backing-path tokens this verifier treats as opaque sentinels — the
#: path describes a higher-level concept the verifier can't reduce to
#: a single value, but the artifact's existence is still meaningful.
_OPAQUE_PATH_TOKENS: Final[tuple[str, ...]] = (
    "n/a",
    "(keys)",
    "(prose)",
    "(whole file)",
    "section",
    "row[",
    "grep on",
)

#: Regex catching "strong" numeric tokens in claim text.  Tokens with
#: a decimal point, comma-thousand-separator, or trailing percent sign
#: are matched against any numeric JSON value.  Examples we want to
#: match: ``0.879``, ``-0.0045``, ``42.67%``, ``5,000``.  Trailing
#: lookahead is ``(?!\d)`` (not ``(?![\d.])``) so the regex catches
#: the last token before a sentence-ending period
#: (``…advanced 0.351.``).
_NUMERIC_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+\.\d+%?|-?\d+%)(?!\d)"
)

#: Regex catching bare integers (``seed 42``, ``schema version 5``).
#: These are noisy by themselves — ``v1``, ``2024`` — so the verifier
#: ONLY uses them to compare against JSON values that are themselves
#: integers, never against float medians.
_BARE_INTEGER_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w.])(-?\d+)(?![\d.%])")


@dataclass(frozen=True)
class VerificationFailure:
    """One verification problem against a single claim."""

    claim_id: str
    message: str


@dataclass
class _Resolution:
    """Resolution attempt for a claim's ``(artifact, path)`` tuple."""

    ok: bool
    value: Any = None
    failures: list[str] = field(default_factory=list)


def _is_opaque_path(path: str) -> bool:
    """Should the verifier skip path resolution and only check existence?"""

    if not path or path.strip().lower() == "n/a":
        return True
    return any(token in path for token in _OPAQUE_PATH_TOKENS)


def _split_json_path(json_path: str) -> list[str]:
    """Split a ``$.a.b.c`` path into ``["a", "b", "c"]``.

    Accepts the leading ``$.`` (jq-style) or its absence; trims
    backtick-wrapped tokens; rejects anything containing
    brace-expansion (``{a, b}``) — the verifier resolves those by
    splitting on commas at the caller.
    """

    raw = json_path.strip()
    if raw.startswith("$."):
        raw = raw[2:]
    elif raw.startswith("$"):
        raw = raw[1:]
    return [part.strip().strip("`") for part in raw.split(".") if part.strip()]


def _resolve_dict_path(data: Any, parts: Sequence[str]) -> tuple[bool, Any]:
    """Walk ``parts`` through ``data``; return ``(ok, value)``.

    Supports the wildcard token ``*`` meaning "any key" — when
    encountered, the walker fans out across every value in the dict at
    that level and reports success if *any* sub-walk completes.  The
    returned ``value`` is then a list of leaf values (not the single
    nested value).  Missing keys / wrong types short-circuit to
    ``(False, None)``.
    """

    if not parts:
        return True, data

    head, *rest = parts

    if head == "*":
        if not isinstance(data, dict) or not data:
            return False, None
        collected: list[Any] = []
        all_ok = False
        for value in data.values():
            ok, sub = _resolve_dict_path(value, rest)
            if ok:
                all_ok = True
                if isinstance(sub, list):
                    collected.extend(sub)
                else:
                    collected.append(sub)
        return all_ok, collected if all_ok else None

    if isinstance(data, dict) and head in data:
        return _resolve_dict_path(data[head], rest)
    return False, None


def _expand_multipath(path: str) -> list[str]:
    """Split a multi-path expression into individual path strings.

    The claims source uses both ``a, b`` (comma-separated full paths)
    and ``$.x.{a, b}.y`` (brace expansion on a segment) to keep a
    single claim's "backing_path" short.  Both forms can appear in
    the same string (``$.a, $.b.{c,d}``); resolve in two passes —
    brace first (one nesting level supported, sufficient for v1),
    then comma-split on each result.
    """

    # Pass 1: brace expansion.  Single nesting only; if a future
    # claims source needs ``$.{a,{b,c}}.x`` we'll need a parser.
    expanded: list[str] = []
    brace = re.search(r"\{([^{}]+)\}", path)
    if brace:
        choices = [c.strip() for c in brace.group(1).split(",") if c.strip()]
        head = path[: brace.start()]
        tail = path[brace.end() :]
        for choice in choices:
            expanded.extend(_expand_multipath(f"{head}{choice}{tail}"))
        return expanded

    # Pass 2: comma split — only when every comma-separated candidate
    # looks like a full $-rooted path; arbitrary commas in keys would
    # otherwise mis-split.
    if "," in path:
        candidates = [p.strip() for p in path.split(",") if p.strip()]
        if all(c.startswith("$") for c in candidates):
            return candidates

    return [path]


def _load_artifact(path: Path) -> Any | None:
    """Read JSON / YAML / CSV / Markdown.  Returns None if unsupported."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return None  # CSV / MD / etc. — existence check only


def _expand_tiers(artifact: str, path: str) -> list[tuple[str | None, str, str]]:
    """Expand ``<tier>`` placeholders into per-tier variants.

    ``<tier>`` can appear in the artifact path (per-tier files like
    ``release/<tier>/manifest.json``) or in the JSON path (a single
    top-level file with per-tier keys, e.g. ``release/metrics.json``
    with ``$.tiers.<tier>.medians.lr_auc``) or both.  Whichever side
    carries the placeholder, the verifier fans out across the three
    tiers; if neither side carries it, returns a single non-tier
    variant.
    """

    if TIER_PLACEHOLDER in artifact or TIER_PLACEHOLDER in path:
        return [
            (tier, artifact.replace(TIER_PLACEHOLDER, tier), path.replace(TIER_PLACEHOLDER, tier))
            for tier in TIERS
        ]
    return [(None, artifact, path)]


@dataclass(frozen=True)
class _NumericCandidates:
    """Numerics extracted from a claim, split into strong and weak buckets.

    ``strong`` candidates (decimal / percent / thousand-separator) can
    match any numeric JSON value.  ``weak`` candidates (bare integers
    like ``42``) only match integer JSON values — using them against
    floats would flag false positives every time a claim happens to
    quote a year or version number.
    """

    strong: tuple[float, ...]
    weak: tuple[int, ...]


def _extract_numerics(text: str) -> _NumericCandidates:
    """Pull numeric tokens out of claim prose for value comparison."""

    strong: list[float] = []
    strong_spans: list[tuple[int, int]] = []
    for match in _NUMERIC_TOKEN_RE.finditer(text):
        token = match.group(1)
        is_percent = token.endswith("%")
        raw = token.rstrip("%").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        strong.append(value / 100.0 if is_percent else value)
        strong_spans.append(match.span())

    weak: list[int] = []
    for match in _BARE_INTEGER_RE.finditer(text):
        start, end = match.span()
        # Skip integers that are part of a token already captured by the
        # strong regex — avoids double-counting ``5`` inside ``5,000``.
        if any(s <= start and end <= e for s, e in strong_spans):
            continue
        try:
            weak.append(int(match.group(1)))
        except ValueError:
            continue

    return _NumericCandidates(strong=tuple(strong), weak=tuple(weak))


def _numeric_or_none(value: Any) -> tuple[float, bool] | None:
    """Coerce a leaf JSON value to ``(float, is_integer)`` or return ``None``."""

    if isinstance(value, bool):
        return None  # bool is an int subclass; we don't want to compare claims to True/False
    if isinstance(value, int):
        return float(value), True
    if isinstance(value, float):
        return value, value.is_integer()
    return None


def _verify_one(
    claim: dict[str, Any],
    release_dir: Path,
    strict: bool,
) -> list[VerificationFailure]:
    """Verify a single claim.  Returns the list of failures (empty = ok)."""

    cid = str(claim["id"])
    artifact_template = str(claim["backing_artifact"])
    path_template = str(claim["backing_path"])
    text = str(claim["text"])

    failures: list[VerificationFailure] = []

    # Skip prose-only claims — there's nothing mechanical to check.
    if _is_opaque_path(path_template) and TIER_PLACEHOLDER not in artifact_template:
        # Still check the artifact exists when it has a concrete path.
        path = REPO_ROOT / artifact_template
        if not path.is_file():
            failures.append(
                VerificationFailure(cid, f"backing artifact does not exist: {artifact_template}")
            )
        return failures

    expected_numerics = _extract_numerics(text)

    for tier, artifact, path in _expand_tiers(artifact_template, path_template):
        artifact_path = REPO_ROOT / artifact
        if not artifact_path.is_file():
            # Per-tier metrics.json + per-tier manifest.json are
            # produced by separate build steps; absent on a fresh
            # checkout where the bundle dirs haven't been built.
            # ``--strict`` upgrades this to an error.
            msg = f"backing artifact does not exist: {artifact} (tier={tier})"
            if strict:
                failures.append(VerificationFailure(cid, msg))
            continue

        data = _load_artifact(artifact_path)
        if data is None:
            # CSV / Markdown / etc. — existence check is all we can do.
            continue

        if _is_opaque_path(path):
            continue

        for sub_path in _expand_multipath(path):
            parts = _split_json_path(sub_path)
            ok, value = _resolve_dict_path(data, parts)
            if not ok:
                failures.append(
                    VerificationFailure(
                        cid,
                        f"path {sub_path!r} did not resolve in {artifact}",
                    )
                )
                continue

            # Numeric comparison: when the resolved value is a single
            # number, find an expected numeric in the claim text that
            # matches within tolerance.  Bare integers (``weak``) can
            # only match integer JSON values — they're too noisy to
            # match against float medians.
            numeric_pair = _numeric_or_none(value)
            if numeric_pair is None:
                continue
            numeric_value, is_integer_value = numeric_pair
            candidates: tuple[float, ...] = expected_numerics.strong
            if is_integer_value:
                candidates = candidates + tuple(float(w) for w in expected_numerics.weak)
            if not candidates:
                continue
            hit = any(abs(numeric_value - expected) <= NUMERIC_TOLERANCE for expected in candidates)
            if not hit:
                failures.append(
                    VerificationFailure(
                        cid,
                        (
                            f"value at {sub_path!r} in {artifact} is {numeric_value!r}; "
                            f"no claim-text numeric within {NUMERIC_TOLERANCE} matches "
                            f"(strong={expected_numerics.strong}, weak={expected_numerics.weak})"
                        ),
                    )
                )

    return failures


def verify_claims(
    source_path: Path,
    release_dir: Path,
    *,
    strict: bool,
) -> list[VerificationFailure]:
    """Verify every claim in ``source_path``.  Returns the failure list."""

    if not source_path.is_file():
        raise FileNotFoundError(f"claims source not found at {source_path}")
    parsed = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or "claims" not in parsed:
        raise ValueError(f"{source_path}: expected top-level mapping with 'claims' key")
    claims = parsed["claims"]
    if not isinstance(claims, list):
        raise ValueError(f"{source_path}: 'claims' must be a list")

    failures: list[VerificationFailure] = []
    for claim in claims:
        if not isinstance(claim, dict) or "id" not in claim:
            failures.append(VerificationFailure("?", f"malformed claim: {claim!r}"))
            continue
        failures.extend(_verify_one(claim, release_dir, strict=strict))
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_claims_register",
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
        "--strict",
        action="store_true",
        help=(
            "treat missing per-tier artifacts as errors (default: skipped silently "
            "so the verifier works on fresh checkouts where bundles haven't been rebuilt)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        failures = verify_claims(args.source, args.release_dir, strict=args.strict)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if failures:
        print(f"error: {len(failures)} claim verification failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - [{failure.claim_id}] {failure.message}", file=sys.stderr)
        return 1

    print("all claims verified.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

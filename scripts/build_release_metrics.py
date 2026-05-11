#!/usr/bin/env python3
"""Emit machine-readable metrics summaries for agent reviewers.

Headline metrics (LR AUC, AP, P@100, Brier, conversion rate, GBM-LR
delta, cohort-shift, cross-tier ordering) currently live only in the
README's markdown table.  An AI reviewer landing on the published
bundle would have to parse prose to verify any of them.

This script reads ``release/validation/validation_report.json`` (the
authoritative output of ``scripts/validate_release_candidate.py``) and
writes:

* ``release/metrics.json`` — top-level summary covering all three
  tiers + cross-tier ordering + cohort-shift, with explicit JSON-path
  back-references to the source-of-truth file.  Lives at the bundle
  root so the Kaggle and HuggingFace upload trees pick it up by
  default.
* ``release/<tier>/metrics.json`` (per tier, one of intro / intermediate
  / advanced) — the per-tier slice plus difficulty knobs from the
  recipe so each bundle is independently inspectable.

Both files are deterministic: same ``validation_report.json`` →
byte-identical output.  ``--check`` mode reports drift as exit-code-1
without overwriting (CI use).

Exit codes: 0 success / 1 ``--check`` mode and metrics are stale /
2 pre-flight error (validation_report.json missing / malformed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

DEFAULT_RELEASE_DIR: Final[Path] = REPO_ROOT / "release"
DEFAULT_REPORT_PATH: Final[Path] = DEFAULT_RELEASE_DIR / "validation" / "validation_report.json"

#: Per-tier "difficulty knobs" surfaced in the README.  Sourced once
#: here so the per-tier metrics file can include them inline; if the
#: recipe ever changes these, update both this constant and the
#: README's "Dataset summary" table.
DIFFICULTY_KNOBS: Final[dict[str, dict[str, float]]] = {
    "intro": {"signal_strength": 0.90, "noise_scale": 0.10, "missing_rate": 0.02},
    "intermediate": {"signal_strength": 0.70, "noise_scale": 0.30, "missing_rate": 0.08},
    "advanced": {"signal_strength": 0.50, "noise_scale": 0.55, "missing_rate": 0.18},
}

TIER_ORDER: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")

#: Subset of headline metrics we surface in the metrics files.  The
#: full per-seed payload stays in ``validation_report.json``; this is
#: the at-a-glance view an agent can verify without parsing every
#: nested key.
HEADLINE_KEYS: Final[tuple[str, ...]] = (
    "lr_auc",
    "gbm_auc",
    "gbm_minus_lr_auc",
    "lr_average_precision",
    "gbm_average_precision",
    "brier_score",
    "log_loss",
    "calibration_max_bin_error",
    "conversion_rate_test",
    "top_decile_rate",
)


def _round(value: Any, ndigits: int) -> Any:
    """Round a numeric value to ``ndigits``, leaving non-numerics alone.

    ``None`` and NaN are preserved as JSON ``null`` for downstream
    consumers (some metrics legitimately have no value in some seeds).
    """

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, int | float):
        return round(float(value), ndigits)
    return value


def _precision_at_100_median(per_seed: list[dict[str, Any]]) -> float | None:
    """Compute the cross-seed median of P@100.

    ``per_seed[*].precision_at_k`` is a dict ``{"50": 0.84, "100": 0.80}``
    in ``validation_report.json``; the median is not stored in
    ``medians`` and has to be computed here.
    """

    values = []
    for seed_block in per_seed:
        pk = seed_block.get("precision_at_k") or {}
        val = pk.get("100")
        if val is not None:
            values.append(float(val))
    if not values:
        return None
    values.sort()
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def _tier_summary(tier: str, tier_block: dict[str, Any]) -> dict[str, Any]:
    """Per-tier slice for the metrics files."""

    medians = tier_block.get("medians", {})
    spreads = tier_block.get("spreads", {})
    per_seed = tier_block.get("per_seed", []) or []

    p100 = _precision_at_100_median(per_seed)

    medians_out = {key: _round(medians.get(key), 4) for key in HEADLINE_KEYS}
    spreads_out = {key: _round(spreads.get(key), 4) for key in HEADLINE_KEYS}
    if p100 is not None:
        medians_out["precision_at_100"] = _round(p100, 4)

    n_seeds = len(per_seed)

    return {
        "tier": tier,
        "n_seeds": n_seeds,
        "seeds": list(tier_block.get("seeds", [])) or sorted(int(s.get("seed")) for s in per_seed),
        "difficulty_knobs": DIFFICULTY_KNOBS.get(tier, {}),
        "medians": medians_out,
        "spreads_max_minus_min": spreads_out,
        "source_of_truth": {
            "file": "release/validation/validation_report.json",
            "json_path": f"$.tiers.{tier}",
        },
        "acceptance_bands": {
            "file": "release/docs/v1_acceptance_gates_bands.yaml",
            "yaml_path": f"per_tier.{tier}",
        },
    }


def build_top_level_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Assemble the top-level ``release/validation/metrics.json`` payload."""

    tiers = report.get("tiers", {})
    cohort = report.get("cohort_shift", {})
    ordering = report.get("cross_tier_ordering", {})

    tier_summaries = {
        tier: _tier_summary(tier, tiers[tier]) for tier in TIER_ORDER if tier in tiers
    }

    cohort_out = {
        tier: {
            "random_split_auc": _round(cohort.get(tier, {}).get("random_split_auc"), 4),
            "cohort_split_auc": _round(cohort.get(tier, {}).get("cohort_split_auc"), 4),
            "auc_degradation": _round(cohort.get(tier, {}).get("auc_degradation"), 4),
            "seed": cohort.get(tier, {}).get("seed"),
        }
        for tier in TIER_ORDER
        if tier in cohort
    }

    return {
        "release_id": report.get("release_id"),
        "package_version": report.get("package_version"),
        "generation_timestamp": report.get("generation_timestamp"),
        "seeds": list(report.get("seeds", [])),
        "tiers": tier_summaries,
        "cross_tier_ordering": ordering,
        "cohort_shift": cohort_out,
        "source_of_truth": {
            "file": "release/validation/validation_report.json",
            "regenerated_by": "scripts/validate_release_candidate.py",
        },
        "acceptance_bands": {
            "file": "release/docs/v1_acceptance_gates_bands.yaml",
            "format": "yaml",
        },
        "notes": (
            "Headline metrics surfaced in the README are cross-seed medians over "
            "the canonical N=5 sweep (seeds 42-46). Per-seed values live under "
            "tiers.<tier>.per_seed in validation_report.json."
        ),
    }


def _render_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON renderer matching the project's conventions."""

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_metrics(
    release_dir: Path,
    report_path: Path,
    *,
    check_only: bool,
) -> tuple[list[Path], dict[str, Any]]:
    """Write (or check) the metrics files.  Returns ``(stale, top_level)``."""

    if not report_path.is_file():
        raise FileNotFoundError(f"validation report not found at {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"{report_path} is not a JSON object")

    top_level = build_top_level_metrics(report)
    stale: list[Path] = []

    def _write(path: Path, content: str) -> None:
        path_rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        existing = path.read_text(encoding="utf-8") if path.is_file() else None
        if existing != content:
            stale.append(path_rel)
            if not check_only:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    _write(release_dir / "metrics.json", _render_json(top_level))

    for tier, summary in top_level["tiers"].items():
        tier_dir = release_dir / tier
        # Per-tier bundle dirs are gitignored; skip when absent so the
        # script is safe to run on a fresh checkout that hasn't rebuilt
        # the bundles yet.  The release-day workflow always regenerates
        # bundles first, then this script, so the production path
        # populates them.
        if not tier_dir.is_dir():
            continue
        _write(tier_dir / "metrics.json", _render_json(summary))

    return stale, top_level


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_release_metrics",
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
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="path to validation_report.json (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale metrics as exit-code-1 without overwriting (CI use)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        stale, _ = write_metrics(args.release_dir, args.report_path, check_only=args.check)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        if stale:
            print("error: metrics files are stale:", file=sys.stderr)
            for path in stale:
                print(f"  - {path}", file=sys.stderr)
            print(
                "run `python scripts/build_release_metrics.py` to refresh them.",
                file=sys.stderr,
            )
            return 1
        print("metrics files are up to date.", file=sys.stderr)
        return 0

    if stale:
        for path in stale:
            print(f"wrote {path}", file=sys.stderr)
    else:
        print("metrics files are already up to date.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

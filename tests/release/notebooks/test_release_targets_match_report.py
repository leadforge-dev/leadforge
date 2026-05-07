"""Audit-sync gate: ``release/notebooks/_release_targets.json`` must
mirror the cross-seed-median values in
``release/validation/validation_report.json``.

Notebook 01 loads the targets file at runtime and pins its computed
metrics against it via ``assert_within_tolerance`` (G13.2).  Without
this audit, the targets file could silently drift from the validation
report — at which point the notebook's "reproduction" gate stops
reproducing anything in particular.  This test fails as soon as either
file moves out of step.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TARGETS_PATH = _REPO_ROOT / "release" / "notebooks" / "_release_targets.json"
_REPORT_PATH = _REPO_ROOT / "release" / "validation" / "validation_report.json"


def test_release_targets_match_validation_report() -> None:
    targets = json.loads(_TARGETS_PATH.read_text())
    report = json.loads(_REPORT_PATH.read_text())

    for tier_name, tier_targets in targets.items():
        if tier_name.startswith("_") or tier_name == "cohort_shift":
            # ``_doc`` and other meta keys; ``cohort_shift`` is checked
            # separately below against ``report["cohort_shift"]`` rather
            # than against ``report["tiers"][...]["medians"]``.
            continue
        assert tier_name in report["tiers"], (
            f"targets file mentions tier {tier_name!r} which is absent from "
            f"validation_report.json (known tiers: {list(report['tiers'])})"
        )
        report_medians = report["tiers"][tier_name]["medians"]
        for metric_name, target_value in tier_targets.items():
            assert metric_name in report_medians, (
                f"{tier_name}.{metric_name}: pinned in targets file but absent "
                f"from validation_report medians"
            )
            assert target_value == report_medians[metric_name], (
                f"{tier_name}.{metric_name}: targets file has {target_value} "
                f"but validation_report median is {report_medians[metric_name]} — "
                "regenerate the report or update _release_targets.json"
            )


def test_cohort_shift_targets_match_validation_report() -> None:
    """Audit-sync gate for the ``cohort_shift`` block.

    Notebook 04 reproduces the report's chronological-resplit AUCs and
    pins them via ``assert_within_tolerance``.  The report stores cohort-
    shift metrics under a top-level ``cohort_shift.<tier>`` key (single
    seed, not a cross-seed median), so the structure differs from the
    per-tier ``medians`` block above and warrants its own audit loop.

    The block is **required**: notebook 04's tolerance gate reads it
    directly, and silently allowing it to disappear would defeat the
    audit-sync invariant.  If notebook 04 ever stops needing this
    block, the test should be deleted, not bypassed.
    """
    targets = json.loads(_TARGETS_PATH.read_text())
    assert "cohort_shift" in targets, (
        "release_targets is missing the 'cohort_shift' block that notebook 04 "
        "reads at runtime — re-add it (sourced from "
        "validation_report.cohort_shift) or delete this test if the notebook "
        "no longer needs it"
    )
    cohort_targets = targets["cohort_shift"]

    report = json.loads(_REPORT_PATH.read_text())
    report_cohort = report["cohort_shift"]
    tiers_checked = 0
    for tier_name, tier_metrics in cohort_targets.items():
        if tier_name.startswith("_"):
            continue
        tiers_checked += 1
        assert tier_name in report_cohort, (
            f"targets cohort_shift mentions tier {tier_name!r} which is absent "
            f"from validation_report.json cohort_shift (known: {list(report_cohort)})"
        )
        report_block = report_cohort[tier_name]
        for metric_name, target_value in tier_metrics.items():
            assert metric_name in report_block, (
                f"cohort_shift.{tier_name}.{metric_name}: pinned in targets file "
                f"but absent from validation_report.cohort_shift.{tier_name}"
            )
            assert target_value == report_block[metric_name], (
                f"cohort_shift.{tier_name}.{metric_name}: targets file has "
                f"{target_value} but validation_report has {report_block[metric_name]} — "
                "regenerate the report or update _release_targets.json"
            )
    assert tiers_checked > 0, (
        "cohort_shift block contained only meta keys (none starting without an "
        "underscore) — at least one tier must be pinned"
    )

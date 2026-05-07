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
        if tier_name.startswith("_"):
            continue  # ``_doc`` and any other meta keys
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

"""Schema contract test for ``bundle_schema_version == "4"``.

The constants below are an *intentional* duplication of the column sets
the bundle writer produces.  The duplication is the point: any change to
``LEAD_SNAPSHOT_FEATURES``, ``LeadRow``, or the redaction policy that
also changes the published column set must update this contract.  A bare
"add a new feature" PR that touches the spec but not this file will fail
here, forcing the author to either update the contract (and bump
``BUNDLE_SCHEMA_VERSION``) or revisit the change.

v4 vs v3: the column SET is unchanged.  The bump records the semantic
shift introduced by the windowed snapshot — event-aggregate column
VALUES are now anchored at ``snapshot_day`` instead of the full horizon.
``manifest.snapshot_day`` makes the contract self-describing.

If you find yourself wondering "do I have to update this?": yes.  That
is the failure mode this test is designed to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator

# Pinned column lists for bundle schema v4.  Update *together* with
# ``BUNDLE_SCHEMA_VERSION`` and ``LEAD_SNAPSHOT_FEATURES``.

V4_TASK_COLUMNS_STUDENT_PUBLIC: frozenset[str] = frozenset(
    {
        "account_id",
        "industry",
        "region",
        "employee_band",
        "estimated_revenue_band",
        "process_maturity_band",
        "contact_id",
        "role_function",
        "seniority",
        "buyer_role",
        "lead_id",
        "lead_created_at",
        "lead_source",
        "first_touch_channel",
        "touch_count",
        "inbound_touch_count",
        "outbound_touch_count",
        "session_count",
        "pricing_page_views",
        "demo_page_views",
        "total_session_duration_seconds",
        "touches_week_1",
        "touches_last_7_days",
        "days_since_first_touch",
        "activity_count",
        "days_since_last_touch",
        "opportunity_created",
        "has_open_opportunity",
        "opportunity_estimated_acv",
        "expected_acv",
        "total_touches_all",
        "converted_within_90_days",
    }
)

V4_TASK_COLUMNS_RESEARCH_INSTRUCTOR: frozenset[str] = V4_TASK_COLUMNS_STUDENT_PUBLIC | {
    "current_stage",
    "is_sql",
}

V4_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC: frozenset[str] = frozenset(
    {
        "lead_id",
        "contact_id",
        "account_id",
        "lead_created_at",
        "lead_source",
        "first_touch_channel",
        # ``current_stage``, ``is_sql`` redacted in student_public
        "owner_rep_id",
        "converted_within_90_days",
        "conversion_timestamp",
    }
)

V4_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR: frozenset[str] = V4_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC | {
    "current_stage",
    "is_sql",
}

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


def _build(mode: str, out: Path, seed: int = 42) -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    gen.generate(**_SMALL).save(str(out))


@pytest.fixture(scope="module")
def student_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v4_student")
    _build("student_public", out)
    return out


@pytest.fixture(scope="module")
def instructor_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v4_instructor")
    _build("research_instructor", out)
    return out


def _task_cols(bundle: Path) -> frozenset[str]:
    return frozenset(pq.read_schema(bundle / "tasks/converted_within_90_days/train.parquet").names)


def _leads_cols(bundle: Path) -> frozenset[str]:
    return frozenset(pq.read_schema(bundle / "tables/leads.parquet").names)


def test_manifest_declares_v4(student_bundle: Path, instructor_bundle: Path) -> None:
    for b in (student_bundle, instructor_bundle):
        manifest = json.loads((b / "manifest.json").read_text())
        assert manifest["bundle_schema_version"] == "4", (
            f"{b.name}: bundle_schema_version is {manifest['bundle_schema_version']!r}, "
            "expected '4'"
        )


def test_manifest_records_snapshot_day(student_bundle: Path, instructor_bundle: Path) -> None:
    """v4 contract: manifest must surface ``snapshot_day`` so consumers can
    distinguish full-horizon (legacy) bundles from windowed bundles."""
    for b in (student_bundle, instructor_bundle):
        manifest = json.loads((b / "manifest.json").read_text())
        assert "snapshot_day" in manifest, f"{b.name}: manifest is missing 'snapshot_day' field"
        # The b2b_saas_procurement_v1 recipe pins snapshot_day=30.
        assert manifest["snapshot_day"] == 30, (
            f"{b.name}: snapshot_day is {manifest['snapshot_day']!r}, expected 30"
        )


def test_student_public_task_columns_match_v4_contract(student_bundle: Path) -> None:
    actual = _task_cols(student_bundle)
    assert actual == V4_TASK_COLUMNS_STUDENT_PUBLIC, (
        f"student_public task split columns drifted from v4 contract.\n"
        f"  unexpected: {sorted(actual - V4_TASK_COLUMNS_STUDENT_PUBLIC)}\n"
        f"  missing:    {sorted(V4_TASK_COLUMNS_STUDENT_PUBLIC - actual)}\n"
        "  → either update tests/render/test_bundle_schema_v4_contract.py and "
        "bump BUNDLE_SCHEMA_VERSION, or revert the schema change."
    )


def test_research_instructor_task_columns_match_v4_contract(instructor_bundle: Path) -> None:
    actual = _task_cols(instructor_bundle)
    assert actual == V4_TASK_COLUMNS_RESEARCH_INSTRUCTOR, (
        f"research_instructor task split columns drifted from v4 contract.\n"
        f"  unexpected: {sorted(actual - V4_TASK_COLUMNS_RESEARCH_INSTRUCTOR)}\n"
        f"  missing:    {sorted(V4_TASK_COLUMNS_RESEARCH_INSTRUCTOR - actual)}"
    )


def test_student_public_leads_table_columns_match_v4_contract(student_bundle: Path) -> None:
    actual = _leads_cols(student_bundle)
    assert actual == V4_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC, (
        f"student_public tables/leads.parquet columns drifted from v4 contract.\n"
        f"  unexpected: {sorted(actual - V4_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC)}\n"
        f"  missing:    {sorted(V4_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC - actual)}"
    )


def test_research_instructor_leads_table_columns_match_v4_contract(
    instructor_bundle: Path,
) -> None:
    actual = _leads_cols(instructor_bundle)
    assert actual == V4_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR, (
        f"research_instructor tables/leads.parquet columns drifted from v4 contract.\n"
        f"  unexpected: {sorted(actual - V4_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR)}\n"
        f"  missing:    {sorted(V4_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR - actual)}"
    )


def test_student_public_redacted_set_in_manifest(student_bundle: Path) -> None:
    manifest = json.loads((student_bundle / "manifest.json").read_text())
    assert set(manifest["redacted_columns"]) == {"current_stage", "is_sql"}


def test_research_instructor_redacted_set_in_manifest(instructor_bundle: Path) -> None:
    manifest = json.loads((instructor_bundle / "manifest.json").read_text())
    assert manifest["redacted_columns"] == []


def test_is_mql_absent_from_all_v4_artifacts(student_bundle: Path, instructor_bundle: Path) -> None:
    """``is_mql`` was removed entirely in v3 (issue #57) and remains absent in v4:
    not in the snapshot, not in the relational table, not in the feature
    dictionary, not anywhere."""
    for b in (student_bundle, instructor_bundle):
        assert "is_mql" not in _task_cols(b)
        assert "is_mql" not in _leads_cols(b)

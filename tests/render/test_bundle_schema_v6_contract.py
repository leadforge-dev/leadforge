"""Schema contract test for ``bundle_schema_version == "6"``.

The constants below are an *intentional* duplication of the column /
table sets the bundle writer produces.  The duplication is the point:
any change to ``LEAD_SNAPSHOT_FEATURES``, ``LeadRow``, the redaction
policy, or the snapshot-safe contract (``BANNED_LEAD_COLUMNS`` /
``BANNED_OPP_COLUMNS`` / ``BANNED_TABLES``) that also changes the
published shape must update this contract.  A bare "add a new feature"
PR that touches the spec but not this file will fail here, forcing the
author to either update the contract (and bump
``BUNDLE_SCHEMA_VERSION``) or revisit the change.

v5 vs v4: ``student_public`` bundles now route through the
snapshot-safe relational export.  Public ``leads.parquet`` drops
``converted_within_90_days`` and ``conversion_timestamp``; public
``opportunities.parquet`` drops ``close_outcome`` and ``closed_at``;
public bundles omit ``customers`` and ``subscriptions`` entirely; event
tables are filtered per-lead to ``lead_created_at + snapshot_day``.
``manifest.relational_snapshot_safe`` records the contract so the
bundle is self-describing.  ``research_instructor`` bundles keep the
full-horizon export.

v6 vs v5: the lead-scoring published *shape* (columns, tables,
snapshot-safe contract) is **unchanged**.  v6 adds a top-level
``manifest.generation_scheme`` field (``lead_scoring`` here) recording
which peer generation scheme produced the bundle.  The pinned column /
table sets below therefore carry over from v5 verbatim.

Task split column SET is unchanged from v4 — the structural fix lives
in ``tables/``, not the snapshot.

If you find yourself wondering "do I have to update this?": yes.  That
is the failure mode this test is designed to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator

# Pinned column / table sets for bundle schema v5.  Update *together*
# with ``BUNDLE_SCHEMA_VERSION``, ``LEAD_SNAPSHOT_FEATURES``, and the
# snapshot-safe contract constants in
# ``leadforge.validation.leakage_probes``.

V5_TASK_COLUMNS_STUDENT_PUBLIC: frozenset[str] = frozenset(
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
        # first_touch_channel removed: byte-identical to lead_source in v1
        "touch_count",
        "inbound_touch_count",
        "outbound_touch_count",
        "session_count",
        "pricing_page_views",
        "demo_page_views",
        "total_session_duration_seconds",
        "touches_days_0_7",  # renamed from touches_week_1
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

V5_TASK_COLUMNS_RESEARCH_INSTRUCTOR: frozenset[str] = V5_TASK_COLUMNS_STUDENT_PUBLIC | {
    "current_stage",
    "is_sql",
}

# v5: public ``leads.parquet`` drops ``converted_within_90_days`` and
# ``conversion_timestamp`` (BANNED_LEAD_COLUMNS).
V5_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC: frozenset[str] = frozenset(
    {
        "lead_id",
        "contact_id",
        "account_id",
        "lead_created_at",
        "lead_source",
        "first_touch_channel",  # still present in relational leads table (entity field)
        # ``current_stage``, ``is_sql`` redacted in student_public
        "owner_rep_id",
        # ``converted_within_90_days`` / ``conversion_timestamp`` dropped
        # by the snapshot-safe export (PR 2.2)
    }
)

V5_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR: frozenset[str] = frozenset(
    {
        "lead_id",
        "contact_id",
        "account_id",
        "lead_created_at",
        "lead_source",
        "first_touch_channel",  # still present in relational leads table (entity field)
        "current_stage",
        "is_sql",
        "owner_rep_id",
        "converted_within_90_days",
        "conversion_timestamp",
    }
)

# v5: public ``opportunities.parquet`` drops ``close_outcome`` and
# ``closed_at`` (BANNED_OPP_COLUMNS).
V5_OPP_TABLE_BANNED_COLUMNS_STUDENT_PUBLIC: frozenset[str] = frozenset(
    {"close_outcome", "closed_at"}
)

# v5: public bundles must NOT include these tables.
V5_PUBLIC_BANNED_TABLES: frozenset[str] = frozenset({"customers", "subscriptions"})

# v5: tables that ARE expected in public bundles.
V5_PUBLIC_EXPECTED_TABLES: frozenset[str] = frozenset(
    {
        "accounts",
        "contacts",
        "leads",
        "touches",
        "sessions",
        "sales_activities",
        "opportunities",
    }
)

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


def _build(mode: str, out: Path, seed: int = 42) -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    gen.generate(**_SMALL).save(str(out))


@pytest.fixture(scope="module")
def student_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v5_student")
    _build("student_public", out)
    return out


@pytest.fixture(scope="module")
def instructor_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("v5_instructor")
    _build("research_instructor", out)
    return out


def _task_cols(bundle: Path) -> frozenset[str]:
    return frozenset(pq.read_schema(bundle / "tasks/converted_within_90_days/train.parquet").names)


def _table_cols(bundle: Path, table_name: str) -> frozenset[str]:
    return frozenset(pq.read_schema(bundle / f"tables/{table_name}.parquet").names)


def _leads_cols(bundle: Path) -> frozenset[str]:
    return _table_cols(bundle, "leads")


def test_manifest_declares_v6(student_bundle: Path, instructor_bundle: Path) -> None:
    for b in (student_bundle, instructor_bundle):
        manifest = json.loads((b / "manifest.json").read_text())
        assert manifest["bundle_schema_version"] == "6", (
            f"{b.name}: bundle_schema_version is {manifest['bundle_schema_version']!r}, "
            "expected '6'"
        )


def test_manifest_records_generation_scheme(student_bundle: Path, instructor_bundle: Path) -> None:
    """v6 contract: every manifest records which peer generation scheme produced
    the bundle.  Both fixtures come from the lead-scoring recipe."""
    for b in (student_bundle, instructor_bundle):
        manifest = json.loads((b / "manifest.json").read_text())
        assert manifest["generation_scheme"] == "lead_scoring", (
            f"{b.name}: generation_scheme is {manifest.get('generation_scheme')!r}, "
            "expected 'lead_scoring'"
        )


def test_manifest_records_snapshot_day(student_bundle: Path, instructor_bundle: Path) -> None:
    for b in (student_bundle, instructor_bundle):
        manifest = json.loads((b / "manifest.json").read_text())
        assert "snapshot_day" in manifest, f"{b.name}: manifest is missing 'snapshot_day' field"
        assert manifest["snapshot_day"] == 30, (
            f"{b.name}: snapshot_day is {manifest['snapshot_day']!r}, expected 30"
        )


def test_manifest_records_relational_snapshot_safe(
    student_bundle: Path, instructor_bundle: Path
) -> None:
    """v5 contract: manifest must surface ``relational_snapshot_safe`` so a tool
    reading a bundle can tell from the manifest alone whether ``tables/`` is the
    snapshot-safe (public) shape or the full-horizon (instructor) shape."""
    student_manifest = json.loads((student_bundle / "manifest.json").read_text())
    assert student_manifest["relational_snapshot_safe"] is True, (
        "student_public bundle must declare relational_snapshot_safe = true"
    )

    instructor_manifest = json.loads((instructor_bundle / "manifest.json").read_text())
    assert instructor_manifest["relational_snapshot_safe"] is False, (
        "research_instructor bundle must declare relational_snapshot_safe = false"
    )


def test_manifest_records_structural_redactions(
    student_bundle: Path, instructor_bundle: Path
) -> None:
    """v5 contract: ``manifest.structural_redactions`` enumerates the table-level
    drops (columns + omitted tables) so the bundle is self-describing."""
    student = json.loads((student_bundle / "manifest.json").read_text())
    assert student["structural_redactions"] == {
        "columns": {
            "leads": ["conversion_timestamp", "converted_within_90_days"],
            "opportunities": ["close_outcome", "closed_at"],
        },
        "omitted_tables": ["customers", "subscriptions"],
    }, "student_public must record the snapshot-safe drops in manifest.structural_redactions"

    instructor = json.loads((instructor_bundle / "manifest.json").read_text())
    assert instructor["structural_redactions"] == {
        "columns": {},
        "omitted_tables": [],
    }, "research_instructor must record an empty structural_redactions"


def test_student_public_task_columns_match_v5_contract(student_bundle: Path) -> None:
    actual = _task_cols(student_bundle)
    assert actual == V5_TASK_COLUMNS_STUDENT_PUBLIC, (
        f"student_public task split columns drifted from v5 contract.\n"
        f"  unexpected: {sorted(actual - V5_TASK_COLUMNS_STUDENT_PUBLIC)}\n"
        f"  missing:    {sorted(V5_TASK_COLUMNS_STUDENT_PUBLIC - actual)}\n"
        "  → either update tests/render/test_bundle_schema_v5_contract.py and "
        "bump BUNDLE_SCHEMA_VERSION, or revert the schema change."
    )


def test_research_instructor_task_columns_match_v5_contract(instructor_bundle: Path) -> None:
    actual = _task_cols(instructor_bundle)
    assert actual == V5_TASK_COLUMNS_RESEARCH_INSTRUCTOR, (
        f"research_instructor task split columns drifted from v5 contract.\n"
        f"  unexpected: {sorted(actual - V5_TASK_COLUMNS_RESEARCH_INSTRUCTOR)}\n"
        f"  missing:    {sorted(V5_TASK_COLUMNS_RESEARCH_INSTRUCTOR - actual)}"
    )


def test_student_public_leads_table_columns_match_v5_contract(student_bundle: Path) -> None:
    actual = _leads_cols(student_bundle)
    assert actual == V5_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC, (
        f"student_public tables/leads.parquet columns drifted from v5 contract.\n"
        f"  unexpected: {sorted(actual - V5_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC)}\n"
        f"  missing:    {sorted(V5_LEAD_TABLE_COLUMNS_STUDENT_PUBLIC - actual)}"
    )


def test_research_instructor_leads_table_columns_match_v5_contract(
    instructor_bundle: Path,
) -> None:
    actual = _leads_cols(instructor_bundle)
    assert actual == V5_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR, (
        f"research_instructor tables/leads.parquet columns drifted from v5 contract.\n"
        f"  unexpected: {sorted(actual - V5_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR)}\n"
        f"  missing:    {sorted(V5_LEAD_TABLE_COLUMNS_RESEARCH_INSTRUCTOR - actual)}"
    )


def test_student_public_opportunities_drop_banned_columns(student_bundle: Path) -> None:
    actual = _table_cols(student_bundle, "opportunities")
    leak = actual & V5_OPP_TABLE_BANNED_COLUMNS_STUDENT_PUBLIC
    assert not leak, (
        f"student_public tables/opportunities.parquet retains banned columns: {sorted(leak)}.\n"
        "  → snapshot-safe export must drop close_outcome / closed_at."
    )


def test_research_instructor_opportunities_keep_banned_columns(instructor_bundle: Path) -> None:
    """Instructor bundles retain the full-horizon shape — those columns are
    *not* banned for them, they're load-bearing for hidden-truth analysis."""
    actual = _table_cols(instructor_bundle, "opportunities")
    assert "close_outcome" in actual, "research_instructor must retain close_outcome"
    assert "closed_at" in actual, "research_instructor must retain closed_at"


def test_student_public_omits_banned_tables(student_bundle: Path) -> None:
    tables_dir = student_bundle / "tables"
    for name in V5_PUBLIC_BANNED_TABLES:
        assert not (tables_dir / f"{name}.parquet").exists(), (
            f"student_public bundle must not include tables/{name}.parquet"
        )

    manifest = json.loads((student_bundle / "manifest.json").read_text())
    declared = set(manifest["tables"].keys())
    assert not (declared & V5_PUBLIC_BANNED_TABLES), (
        f"manifest.tables must not list banned tables: {sorted(declared & V5_PUBLIC_BANNED_TABLES)}"
    )


def test_student_public_includes_expected_tables(student_bundle: Path) -> None:
    tables_dir = student_bundle / "tables"
    for name in V5_PUBLIC_EXPECTED_TABLES:
        assert (tables_dir / f"{name}.parquet").exists(), (
            f"student_public bundle is missing required tables/{name}.parquet"
        )


def test_research_instructor_keeps_banned_tables(instructor_bundle: Path) -> None:
    """Instructor bundles retain ``customers`` / ``subscriptions``."""
    tables_dir = instructor_bundle / "tables"
    for name in V5_PUBLIC_BANNED_TABLES:
        assert (tables_dir / f"{name}.parquet").exists(), (
            f"research_instructor bundle must include tables/{name}.parquet"
        )


def test_student_public_redacted_set_in_manifest(student_bundle: Path) -> None:
    manifest = json.loads((student_bundle / "manifest.json").read_text())
    assert set(manifest["redacted_columns"]) == {"current_stage", "is_sql"}


def test_research_instructor_redacted_set_in_manifest(instructor_bundle: Path) -> None:
    manifest = json.loads((instructor_bundle / "manifest.json").read_text())
    assert manifest["redacted_columns"] == []


def test_is_mql_absent_from_all_v5_artifacts(student_bundle: Path, instructor_bundle: Path) -> None:
    """``is_mql`` was removed entirely in v3 (issue #57) and remains absent in v5:
    not in the snapshot, not in the relational table, not in the feature
    dictionary, not anywhere."""
    for b in (student_bundle, instructor_bundle):
        assert "is_mql" not in _task_cols(b)
        assert "is_mql" not in _leads_cols(b)

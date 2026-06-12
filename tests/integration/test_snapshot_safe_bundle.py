"""Integration tests for the snapshot-safe bundle write path (bundle schema v5).

Covers the contract turned on in PR 2.2: ``student_public`` bundles
route ``tables/`` through
:func:`leadforge.schemes.lead_scoring.render.relational_snapshot_safe.to_dataframes_snapshot_safe`
(the structural fix against the alpha-bundle reconstruction paths
A-E), ``research_instructor`` bundles keep the full-horizon export,
and the manifest is self-describing via ``relational_snapshot_safe``,
``structural_redactions``, and ``bundle_schema_version == "6"``.

Tests fall into three groups:

* **Round-trip** — both modes write, both validate clean, on-disk
  shape matches the contract (banned columns / banned tables drop in
  public; both retained in instructor).
* **Manifest contract** — row counts match disk; banned tables are
  not listed under ``manifest.tables``.
* **Negative** — a tampered "public" bundle (instructor copy moved
  into the public slot) trips the relational-leakage probes through
  ``validate_bundle``, with specific banned columns / tables / join
  paths named in the findings (not just channel-set membership).

Hash determinism is covered by
:class:`tests.validation.test_invariants.TestDeterminism` for both
modes — no need to duplicate it here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator
from leadforge.validation.bundle_checks import validate_bundle
from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    CHANNEL_BANNED_COLUMN,
    CHANNEL_BANNED_TABLE,
    CHANNEL_JOIN_RECONSTRUCTION,
    run_all_probes,
)

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


def _build(mode: str, out: Path, seed: int = 42) -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    gen.generate(**_SMALL).save(str(out))


@pytest.fixture(scope="module")
def public_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("snapshot_safe_public")
    _build("student_public", out)
    return out


@pytest.fixture(scope="module")
def instructor_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("snapshot_safe_instructor")
    _build("research_instructor", out)
    return out


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTripValidates:
    def test_public_bundle_validates(self, public_bundle: Path) -> None:
        errors = validate_bundle(public_bundle)
        assert errors == [], f"public bundle should validate clean, got: {errors}"

    def test_instructor_bundle_validates(self, instructor_bundle: Path) -> None:
        errors = validate_bundle(instructor_bundle)
        assert errors == [], f"instructor bundle should validate clean, got: {errors}"

    def test_public_bundle_omits_banned_tables_on_disk(self, public_bundle: Path) -> None:
        for name in BANNED_TABLES:
            assert not (public_bundle / "tables" / f"{name}.parquet").exists(), (
                f"public bundle must not write tables/{name}.parquet"
            )

    def test_instructor_bundle_keeps_banned_tables_on_disk(self, instructor_bundle: Path) -> None:
        for name in BANNED_TABLES:
            assert (instructor_bundle / "tables" / f"{name}.parquet").exists(), (
                f"instructor bundle must retain tables/{name}.parquet"
            )

    def test_public_leads_drops_banned_columns(self, public_bundle: Path) -> None:
        cols = set(pq.read_schema(public_bundle / "tables/leads.parquet").names)
        for c in BANNED_LEAD_COLUMNS:
            assert c not in cols, f"leads.{c} must be absent from public bundle"

    def test_public_opportunities_drops_banned_columns(self, public_bundle: Path) -> None:
        cols = set(pq.read_schema(public_bundle / "tables/opportunities.parquet").names)
        for c in BANNED_OPP_COLUMNS:
            assert c not in cols, f"opportunities.{c} must be absent from public bundle"

    def test_instructor_leads_keeps_banned_columns(self, instructor_bundle: Path) -> None:
        cols = set(pq.read_schema(instructor_bundle / "tables/leads.parquet").names)
        for c in BANNED_LEAD_COLUMNS:
            assert c in cols, (
                f"leads.{c} must be retained in research_instructor bundle (full-horizon export)"
            )


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------


class TestManifestContract:
    def test_public_manifest_table_row_counts_match_disk(self, public_bundle: Path) -> None:
        """Manifest row counts must come from the *post-redaction* dict so
        consumers reading the manifest see the truth on disk, not the
        pre-redaction full-horizon shape."""
        manifest = json.loads((public_bundle / "manifest.json").read_text())
        for name, info in manifest["tables"].items():
            actual = pq.read_metadata(public_bundle / f"tables/{name}.parquet").num_rows
            assert info["row_count"] == actual, (
                f"manifest row_count for {name} ({info['row_count']}) "
                f"disagrees with parquet ({actual})"
            )

    def test_public_manifest_does_not_list_banned_tables(self, public_bundle: Path) -> None:
        manifest = json.loads((public_bundle / "manifest.json").read_text())
        for name in BANNED_TABLES:
            assert name not in manifest["tables"], (
                f"manifest.tables must not list banned table {name!r}"
            )


# ---------------------------------------------------------------------------
# Negative — tampered public bundle
# ---------------------------------------------------------------------------


class TestTamperedPublicBundle:
    """Hand-craft a tampered public bundle and verify ``validate_bundle``
    surfaces the leakage findings.  Tampering = take an instructor bundle
    (full-horizon shape) and rewrite its manifest to claim
    ``student_public``.  This is the structural attack the contract
    defends against."""

    def _make_tampered(self, instructor_bundle: Path, dest: Path) -> Path:
        shutil.copytree(instructor_bundle, dest)
        manifest = json.loads((dest / "manifest.json").read_text())
        manifest["exposure_mode"] = "student_public"
        # ``redacted_columns`` would mismatch the public expectation
        # (instructor has []); align it so we test the relational
        # leakage path specifically and not the redaction-set check.
        manifest["redacted_columns"] = ["current_stage", "is_sql"]
        # Leave ``relational_snapshot_safe`` at False — that IS the
        # lie this test exercises (manifest claims public, tables are
        # full-horizon).
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return dest

    def test_validate_surfaces_leakage_findings(
        self, tmp_path: Path, instructor_bundle: Path
    ) -> None:
        tampered = self._make_tampered(instructor_bundle, tmp_path / "tampered")
        errors = validate_bundle(tampered, include_realism=False)
        leak_errors = [e for e in errors if e.startswith("Relational leakage")]
        assert leak_errors, (
            f"tampered public bundle must surface relational-leakage errors; got {errors}"
        )

    def test_findings_name_specific_banned_columns(
        self, tmp_path: Path, instructor_bundle: Path
    ) -> None:
        """Channel membership isn't enough — assert the *details* point at
        the actual banned columns / tables, so a future regression that
        leaves one finding behind doesn't slip through."""
        tampered = self._make_tampered(instructor_bundle, tmp_path / "tampered_details")
        manifest = json.loads((tampered / "manifest.json").read_text())
        report = run_all_probes(tampered, snapshot_day=manifest["snapshot_day"])

        details_by_channel: dict[str, set[str]] = {}
        for f in report.findings:
            details_by_channel.setdefault(f.channel, set()).add(f.detail)

        # Every banned column must be named in a banned-column finding.
        banned_col_details = details_by_channel.get(CHANNEL_BANNED_COLUMN, set())
        expected_banned_cols = {f"leads.{c}" for c in BANNED_LEAD_COLUMNS} | {
            f"opportunities.{c}" for c in BANNED_OPP_COLUMNS
        }
        assert expected_banned_cols <= banned_col_details, (
            f"banned-column findings missing entries: "
            f"{sorted(expected_banned_cols - banned_col_details)}"
        )

        # Every banned table must be named in a banned-table finding.
        banned_tbl_details = details_by_channel.get(CHANNEL_BANNED_TABLE, set())
        assert set(BANNED_TABLES) <= banned_tbl_details, (
            f"banned-table findings missing entries: "
            f"{sorted(set(BANNED_TABLES) - banned_tbl_details)}"
        )

        # Join-reconstruction must name at least path B (closed_won
        # opportunities are in the tampered bundle).  Paths C / D may
        # also fire when customers / subscriptions are present.
        join_details = details_by_channel.get(CHANNEL_JOIN_RECONSTRUCTION, set())
        assert "path_b_opportunity_won" in join_details, (
            f"join-reconstruction finding for path B missing; got {sorted(join_details)}"
        )

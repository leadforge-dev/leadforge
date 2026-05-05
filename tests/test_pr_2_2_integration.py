"""PR 2.2 integration tests — snapshot-safe routing through the writer.

Covers the contract turned on in PR 2.2: ``student_public`` bundles
route ``tables/`` through
:func:`leadforge.render.relational_snapshot_safe.to_dataframes_snapshot_safe`
(structural fix for the alpha-bundle reconstruction paths A-E),
``research_instructor`` bundles keep the full-horizon export, and the
manifest is self-describing via ``relational_snapshot_safe`` and
``bundle_schema_version == "5"``.

Tests fall into four groups:

* Round-trip on both modes — both bundles must validate cleanly.
* Manifest contract — version + ``relational_snapshot_safe`` flag.
* Negative — a tampered "public" bundle (instructor copy moved into
  the public slot) must trip the relational-leakage probes through
  ``validate_bundle``, with the expected channels populated.
* Hash determinism — two pinned-timestamp builds produce
  byte-identical bundle content (same in-process check the slower
  ``scripts/verify_hash_determinism.py`` runs).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator
from leadforge.core.hashing import file_sha256
from leadforge.validation.bundle_checks import validate_bundle
from leadforge.validation.relational_leakage import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    CHANNEL_BANNED_COLUMN,
    CHANNEL_BANNED_TABLE,
    CHANNEL_JOIN_RECONSTRUCTION,
    run_all_probes,
)

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}
_PINNED_TS = "1970-01-01T00:00:00+00:00"


def _build(
    mode: str, out: Path, seed: int = 42, *, generation_timestamp: str | None = None
) -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    gen.generate(**_SMALL).save(str(out), generation_timestamp=generation_timestamp)


@pytest.fixture(scope="module")
def public_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pr22_public")
    _build("student_public", out)
    return out


@pytest.fixture(scope="module")
def instructor_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pr22_instructor")
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
    def test_public_manifest_declares_v5(self, public_bundle: Path) -> None:
        manifest = json.loads((public_bundle / "manifest.json").read_text())
        assert manifest["bundle_schema_version"] == "5"

    def test_instructor_manifest_declares_v5(self, instructor_bundle: Path) -> None:
        manifest = json.loads((instructor_bundle / "manifest.json").read_text())
        assert manifest["bundle_schema_version"] == "5"

    def test_public_manifest_relational_snapshot_safe_true(self, public_bundle: Path) -> None:
        manifest = json.loads((public_bundle / "manifest.json").read_text())
        assert manifest["relational_snapshot_safe"] is True

    def test_instructor_manifest_relational_snapshot_safe_false(
        self, instructor_bundle: Path
    ) -> None:
        manifest = json.loads((instructor_bundle / "manifest.json").read_text())
        assert manifest["relational_snapshot_safe"] is False

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
        # (instructor has []); align it so we test the relational leakage
        # path specifically and not the redaction-set check.
        manifest["redacted_columns"] = ["current_stage", "is_sql"]
        # Same reason: leave ``relational_snapshot_safe`` at False — that
        # IS the lie this test exercises (manifest claims public, tables
        # are full-horizon).
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

    def test_findings_cover_expected_channels(
        self, tmp_path: Path, instructor_bundle: Path
    ) -> None:
        tampered = self._make_tampered(instructor_bundle, tmp_path / "tampered2")
        manifest = json.loads((tampered / "manifest.json").read_text())
        report = run_all_probes(tampered, snapshot_day=manifest["snapshot_day"])
        channels = {f.channel for f in report.findings}
        # An instructor bundle masquerading as public will trip:
        #   - banned columns (leads.converted_within_90_days,
        #     leads.conversion_timestamp, opportunities.close_outcome,
        #     opportunities.closed_at);
        #   - banned tables (customers, subscriptions);
        #   - join reconstruction (paths B/C/D fire because the
        #     conversion-conditional tables are present).
        assert CHANNEL_BANNED_COLUMN in channels
        assert CHANNEL_BANNED_TABLE in channels
        assert CHANNEL_JOIN_RECONSTRUCTION in channels


# ---------------------------------------------------------------------------
# Hash determinism
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    """Pinning ``generation_timestamp`` should produce byte-identical bundle
    content across runs.  Mirrors the contract of
    ``scripts/verify_hash_determinism.py`` for a single bundle."""

    def _hash_tree(self, root: Path) -> dict[str, str]:
        return {
            str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()
        }

    @pytest.mark.parametrize("mode", ["student_public", "research_instructor"])
    def test_pinned_timestamp_byte_identical(self, tmp_path: Path, mode: str) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        _build(mode, a, generation_timestamp=_PINNED_TS)
        _build(mode, b, generation_timestamp=_PINNED_TS)

        ha = self._hash_tree(a)
        hb = self._hash_tree(b)

        assert set(ha.keys()) == set(hb.keys()), (
            f"file set differs across runs ({mode}): "
            f"only_in_a={sorted(set(ha) - set(hb))} "
            f"only_in_b={sorted(set(hb) - set(ha))}"
        )
        diffs = [k for k in ha if ha[k] != hb[k]]
        assert not diffs, f"hash mismatches across runs ({mode}): {diffs}"

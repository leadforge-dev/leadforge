"""Tests for leadforge.validation.bundle_checks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.bundle_checks import validate_bundle

# ---------------------------------------------------------------------------
# Fixture — generate a small bundle once
# ---------------------------------------------------------------------------

_SMALL = {"n_leads": 20, "n_accounts": 10, "n_contacts": 30}


@pytest.fixture(scope="module")
def valid_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a valid bundle for reuse.  Do not mutate."""
    out = tmp_path_factory.mktemp("valid_bundle")
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=99, exposure_mode="student_public")
    gen.generate(**_SMALL).save(str(out))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidBundle:
    def test_passes(self, valid_bundle: Path) -> None:
        assert validate_bundle(valid_bundle) == []


class TestMetadataRowCounts:
    """Verify that Parquet metadata row counts match actual data."""

    def test_task_split_metadata_matches_data(self, valid_bundle: Path) -> None:
        import pandas as pd
        import pyarrow.parquet as pq

        manifest = json.loads((valid_bundle / "manifest.json").read_text())
        for task_id in manifest.get("tasks", {}):
            for split in ("train", "valid", "test"):
                path = valid_bundle / f"tasks/{task_id}/{split}.parquet"
                assert path.exists()
                meta_rows = pq.read_metadata(path).num_rows
                actual_rows = len(pd.read_parquet(path))
                assert meta_rows == actual_rows


class TestCorruptBundle:
    def test_row_count_mismatch(self, tmp_path: Path, valid_bundle: Path) -> None:
        corrupt = tmp_path / "bad"
        shutil.copytree(valid_bundle, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_table = next(iter(manifest["tables"]))
        manifest["tables"][first_table]["row_count"] = 999999
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = validate_bundle(corrupt)
        assert any("expected 999999 rows" in e for e in errors)

    def test_missing_table_reports_fk_skip(self, tmp_path: Path, valid_bundle: Path) -> None:
        corrupt = tmp_path / "missing"
        shutil.copytree(valid_bundle, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_table = next(iter(manifest["tables"]))
        (corrupt / f"tables/{first_table}.parquet").unlink()

        errors = validate_bundle(corrupt)
        assert any("Missing table file" in e for e in errors)
        assert any("FK check skipped" in e for e in errors)

    def test_sha256_mismatch(self, tmp_path: Path, valid_bundle: Path) -> None:
        corrupt = tmp_path / "sha"
        shutil.copytree(valid_bundle, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_table = next(iter(manifest["tables"]))
        manifest["tables"][first_table]["sha256"] = "0" * 64
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = validate_bundle(corrupt)
        assert any("SHA-256 mismatch" in e for e in errors)

    def test_task_split_row_count_mismatch(self, tmp_path: Path, valid_bundle: Path) -> None:
        corrupt = tmp_path / "bad_task"
        shutil.copytree(valid_bundle, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_task = next(iter(manifest["tasks"]))
        manifest["tasks"][first_task]["train_rows"] = 999999
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = validate_bundle(corrupt)
        assert any("expected 999999 rows" in e for e in errors)

    def test_leakage_detects_extra_columns(self, tmp_path: Path, valid_bundle: Path) -> None:
        import pandas as pd

        corrupt = tmp_path / "leak"
        shutil.copytree(valid_bundle, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_task = next(iter(manifest["tasks"]))
        train_path = corrupt / f"tasks/{first_task}/train.parquet"
        df = pd.read_parquet(train_path)
        df["__sneaky_leak__"] = 1
        df.to_parquet(train_path, index=False)

        errors = validate_bundle(corrupt, include_realism=False)
        assert any("__sneaky_leak__" in e for e in errors)

    def test_missing_required_file(self, tmp_path: Path, valid_bundle: Path) -> None:
        corrupt = tmp_path / "nocard"
        shutil.copytree(valid_bundle, corrupt)
        (corrupt / "dataset_card.md").unlink()

        errors = validate_bundle(corrupt)
        assert any("dataset_card.md" in e for e in errors)

"""Tests for leadforge.validation.realism."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from leadforge.api.generator import Generator
from leadforge.core.serialization import load_json
from leadforge.validation.realism import check_realism

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("realism")
    Generator.from_recipe(
        "b2b_saas_procurement_v1", seed=42, exposure_mode="student_public"
    ).generate(**_SMALL).save(str(out))
    return out


@pytest.fixture(scope="module")
def manifest(bundle_dir: Path) -> dict:
    return load_json(bundle_dir / "manifest.json")


class TestRealism:
    def test_valid_bundle_passes(self, bundle_dir: Path, manifest: dict) -> None:
        errors = check_realism(bundle_dir, manifest)
        assert errors == [], f"Unexpected realism errors: {errors}"

    def test_detects_zero_row_table(self, tmp_path: Path, bundle_dir: Path) -> None:
        """An empty accounts Parquet file should flag."""
        corrupt = tmp_path / "zero_rows"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        # Write an empty Parquet file (preserving columns).
        orig = pd.read_parquet(corrupt / "tables/accounts.parquet")
        orig.head(0).to_parquet(corrupt / "tables/accounts.parquet")

        errors = check_realism(corrupt, manifest)
        assert any("0 rows" in e for e in errors)

    def test_detects_low_conversion_rate(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "low_rate"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        train_path = corrupt / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["converted_within_90_days"] = False
        df.to_parquet(train_path)

        errors = check_realism(corrupt, manifest)
        assert any("suspiciously low" in e for e in errors)

    def test_detects_high_conversion_rate(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "high_rate"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        train_path = corrupt / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["converted_within_90_days"] = True
        df.to_parquet(train_path)

        errors = check_realism(corrupt, manifest)
        assert any("suspiciously high" in e for e in errors)

    def test_detects_negative_count_feature(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "neg_count"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        train_path = corrupt / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["touch_count"] = pd.array([-1] * len(df), dtype="Int64")
        df.to_parquet(train_path)

        errors = check_realism(corrupt, manifest)
        assert any("negative" in e and "touch_count" in e for e in errors)

    def test_detects_non_boolean_feature(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "bad_bool"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        train_path = corrupt / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        # Pick the first non-target boolean column at test time so this
        # test self-heals when feature names change.  Falls back gracefully
        # if the spec ever has zero non-target booleans (currently impossible).
        from leadforge.schemes.lead_scoring.features import LEAD_SNAPSHOT_FEATURES

        bool_col = next(
            f.name
            for f in LEAD_SNAPSHOT_FEATURES
            if f.dtype == "boolean" and not f.is_target and f.name in df.columns
        )
        # Replace boolean column with a string — clearly not boolean dtype.
        df[bool_col] = "yes"
        df.to_parquet(train_path)

        errors = check_realism(corrupt, manifest)
        assert any("non-boolean dtype" in e and bool_col in e for e in errors)

    def test_detects_single_stage(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "one_stage"
        shutil.copytree(bundle_dir, corrupt)
        manifest = load_json(corrupt / "manifest.json")
        leads_path = corrupt / "tables/leads.parquet"
        df = pd.read_parquet(leads_path)
        df["current_stage"] = "mql"
        df.to_parquet(leads_path)

        errors = check_realism(corrupt, manifest)
        assert any("single funnel stage" in e for e in errors)

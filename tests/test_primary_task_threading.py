"""Tests for primary_task threading through the generation pipeline.

Verifies that ``config.primary_task`` and ``config.label_window_days``
are respected by bundle writing, manifest, validation, and pipeline
rename functions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from leadforge.api.generator import Generator
from leadforge.core.serialization import load_json
from leadforge.pipelines.build_v5 import rename_and_select as v5_rename
from leadforge.pipelines.build_v6 import rename_and_select as v6_rename
from leadforge.schema.tasks import CONVERTED_WITHIN_90_DAYS, task_manifest_for_config
from leadforge.validation.drift import check_cross_seed_stability
from leadforge.validation.realism import check_realism

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}
_CUSTOM_TASK = "converted_within_60_days"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def default_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bundle with default primary_task."""
    out = tmp_path_factory.mktemp("default_task")
    Generator.from_recipe(
        "b2b_saas_procurement_v1", seed=42, exposure_mode="student_public"
    ).generate(**_SMALL).save(str(out))
    return out


@pytest.fixture(scope="module")
def custom_task_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bundle with a non-default primary_task."""
    out = tmp_path_factory.mktemp("custom_task")
    Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=42,
        exposure_mode="student_public",
        primary_task=_CUSTOM_TASK,
        label_window_days=60,
    ).generate(**_SMALL).save(str(out))
    return out


# ---------------------------------------------------------------------------
# task_manifest_for_config
# ---------------------------------------------------------------------------


class TestTaskManifestForConfig:
    def test_default_matches_constant(self) -> None:
        m = task_manifest_for_config()
        assert m.task_id == CONVERTED_WITHIN_90_DAYS.task_id
        assert m.label_column == CONVERTED_WITHIN_90_DAYS.label_column
        assert m.label_window_days == CONVERTED_WITHIN_90_DAYS.label_window_days

    def test_custom_task_id(self) -> None:
        m = task_manifest_for_config(primary_task="my_task", label_window_days=30)
        assert m.task_id == "my_task"
        assert m.label_window_days == 30
        assert m.label_column == "converted_within_90_days"

    def test_description_includes_window(self) -> None:
        m = task_manifest_for_config(label_window_days=45)
        assert "45 days" in m.description


# ---------------------------------------------------------------------------
# Bundle directory layout
# ---------------------------------------------------------------------------


class TestBundleLayout:
    def test_default_task_directory(self, default_bundle: Path) -> None:
        task_dir = default_bundle / "tasks/converted_within_90_days"
        assert task_dir.is_dir()
        for split in ("train", "valid", "test"):
            assert (task_dir / f"{split}.parquet").exists()

    def test_custom_task_directory(self, custom_task_bundle: Path) -> None:
        task_dir = custom_task_bundle / f"tasks/{_CUSTOM_TASK}"
        assert task_dir.is_dir()
        for split in ("train", "valid", "test"):
            assert (task_dir / f"{split}.parquet").exists()

    def test_custom_task_no_default_dir(self, custom_task_bundle: Path) -> None:
        assert not (custom_task_bundle / "tasks/converted_within_90_days").exists()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_default_manifest_task_key(self, default_bundle: Path) -> None:
        manifest = load_json(default_bundle / "manifest.json")
        assert "converted_within_90_days" in manifest["tasks"]
        assert manifest["primary_task"] == "converted_within_90_days"
        assert manifest["label_window_days"] == 90

    def test_custom_manifest_task_key(self, custom_task_bundle: Path) -> None:
        manifest = load_json(custom_task_bundle / "manifest.json")
        assert _CUSTOM_TASK in manifest["tasks"]
        assert "converted_within_90_days" not in manifest["tasks"]
        assert manifest["primary_task"] == _CUSTOM_TASK
        assert manifest["label_window_days"] == 60

    def test_task_manifest_json_uses_custom_id(self, custom_task_bundle: Path) -> None:
        tm = load_json(custom_task_bundle / f"tasks/{_CUSTOM_TASK}/task_manifest.json")
        assert tm["task_id"] == _CUSTOM_TASK
        assert tm["label_window_days"] == 60


# ---------------------------------------------------------------------------
# Validation respects manifest-driven paths
# ---------------------------------------------------------------------------


class TestValidation:
    def test_realism_passes_default(self, default_bundle: Path) -> None:
        manifest = load_json(default_bundle / "manifest.json")
        errors = check_realism(default_bundle, manifest)
        assert errors == []

    def test_realism_passes_custom_task(self, custom_task_bundle: Path) -> None:
        manifest = load_json(custom_task_bundle / "manifest.json")
        errors = check_realism(custom_task_bundle, manifest)
        assert errors == []

    def test_drift_finds_custom_task_path(self, tmp_path: Path, custom_task_bundle: Path) -> None:
        """Verify drift check reads from manifest-based task path, not hardcoded."""
        # Copy the same bundle as a second "seed" so rates match exactly.
        out2 = tmp_path / "seed2"
        shutil.copytree(custom_task_bundle, out2)

        bundles = {42: custom_task_bundle, 99: out2}
        errors = check_cross_seed_stability(bundles)
        # Should not produce "missing task train.parquet" errors.
        assert not any("missing" in e for e in errors)


# ---------------------------------------------------------------------------
# Pipeline rename functions
# ---------------------------------------------------------------------------


class TestPipelineRename:
    @pytest.fixture
    def sample_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "industry": ["tech"],
                "region": ["us"],
                "employee_band": ["200-499"],
                "estimated_revenue_band": ["$10M-$50M"],
                "role_function": ["finance"],
                "seniority": ["vp"],
                "lead_source": ["inbound_marketing"],
                "opportunity_created": [True],
                "demo_completed": [False],
                "expected_acv": [50000],
                "inbound_touch_count": [5],
                "outbound_touch_count": [3],
                "touches_week_1": [2],
                "days_since_first_touch": [10],
                "session_count": [4],
                "activity_count": [2],
                "days_since_last_touch": [1.0],
                "total_touches_all": [8],
                "demo_page_views": [0],
                "converted_within_90_days": [True],
            }
        )

    def test_v5_default_label_column(self, sample_snapshot: pd.DataFrame) -> None:
        from leadforge.pipelines.build_v5 import derive_binary_features

        df = derive_binary_features(sample_snapshot)
        result = v5_rename(df)
        assert "converted" in result.columns
        assert result["converted"].iloc[0] == 1

    def test_v5_custom_label_column(self, sample_snapshot: pd.DataFrame) -> None:
        from leadforge.pipelines.build_v5 import derive_binary_features

        df = sample_snapshot.rename(
            columns={"converted_within_90_days": "converted_within_60_days"}
        )
        df = derive_binary_features(df)
        result = v5_rename(df, label_column="converted_within_60_days")
        assert "converted" in result.columns
        assert result["converted"].iloc[0] == 1

    def test_v6_default_label_column(self, sample_snapshot: pd.DataFrame) -> None:
        from leadforge.pipelines.build_v6 import derive_features

        df = sample_snapshot.copy()
        df["touches_last_7_days"] = 1
        df["acquisition_wave"] = "A"
        df = derive_features(df)
        result = v6_rename(df)
        assert "converted" in result.columns

    def test_v6_custom_label_column(self, sample_snapshot: pd.DataFrame) -> None:
        from leadforge.pipelines.build_v6 import derive_features

        df = sample_snapshot.rename(
            columns={"converted_within_90_days": "converted_within_60_days"}
        )
        df["touches_last_7_days"] = 1
        df["acquisition_wave"] = "A"
        df = derive_features(df)
        result = v6_rename(df, label_column="converted_within_60_days")
        assert "converted" in result.columns
        assert result["converted"].iloc[0] == 1

"""End-to-end tests for exposure-mode column redaction.

These tests cover the post-v1 leakage fix: ``current_stage`` (and any other
``leakage_risk and not is_leakage_trap`` column) is stripped from
``student_public`` bundles, preserved in ``research_instructor`` bundles,
and the deliberately included pedagogical trap ``total_touches_all`` is
left in place in both modes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator
from leadforge.core.enums import ExposureMode
from leadforge.exposure.filters import FILTERS
from leadforge.schema.features import (
    LEAD_SNAPSHOT_FEATURES,
    STUDENT_PUBLIC_REDACTED_COLUMNS,
)
from leadforge.validation.bundle_checks import validate_bundle

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


def _build(mode: str, out: Path, seed: int = 42) -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    gen.generate(**_SMALL).save(str(out))


def _task_columns(bundle_root: Path, split: str) -> set[str]:
    path = bundle_root / "tasks" / "converted_within_90_days" / f"{split}.parquet"
    return set(pq.read_schema(path).names)


# ---------------------------------------------------------------------------
# Static / unit-level checks
# ---------------------------------------------------------------------------


def test_filter_redaction_set_matches_features() -> None:
    """``BundleFilter.redacted_columns`` for student_public must match the
    derived feature-spec set."""
    assert FILTERS[ExposureMode.student_public].redacted_columns == STUDENT_PUBLIC_REDACTED_COLUMNS


def test_research_instructor_redacts_nothing() -> None:
    assert FILTERS[ExposureMode.research_instructor].redacted_columns == frozenset()


def test_redaction_set_is_non_empty() -> None:
    """If this regresses to empty, the fix is not actually doing anything."""
    assert "current_stage" in STUDENT_PUBLIC_REDACTED_COLUMNS


def test_redaction_set_excludes_pedagogical_trap() -> None:
    assert "total_touches_all" not in STUDENT_PUBLIC_REDACTED_COLUMNS


# ---------------------------------------------------------------------------
# End-to-end: student_public has no redacted columns
# ---------------------------------------------------------------------------


class TestStudentPublicRedaction:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("student_public_redaction")
        _build("student_public", out)
        return out

    def test_current_stage_absent_from_all_splits(self, bundle: Path) -> None:
        for split in ("train", "valid", "test"):
            cols = _task_columns(bundle, split)
            assert "current_stage" not in cols, (
                f"current_stage leaked into student_public {split} split"
            )

    def test_total_touches_all_present_in_all_splits(self, bundle: Path) -> None:
        for split in ("train", "valid", "test"):
            cols = _task_columns(bundle, split)
            assert "total_touches_all" in cols, (
                f"pedagogical trap total_touches_all dropped from {split}"
            )

    def test_no_redacted_column_in_any_split(self, bundle: Path) -> None:
        for split in ("train", "valid", "test"):
            cols = _task_columns(bundle, split)
            leaked = cols & STUDENT_PUBLIC_REDACTED_COLUMNS
            assert not leaked, f"redacted columns present in student_public {split}: {leaked}"

    def test_target_column_still_present(self, bundle: Path) -> None:
        cols = _task_columns(bundle, "train")
        assert "converted_within_90_days" in cols

    def test_feature_dictionary_excludes_current_stage(self, bundle: Path) -> None:
        df = pd.read_csv(bundle / "feature_dictionary.csv")
        assert "current_stage" not in set(df["name"])

    def test_feature_dictionary_includes_pedagogical_trap(self, bundle: Path) -> None:
        df = pd.read_csv(bundle / "feature_dictionary.csv")
        assert "total_touches_all" in set(df["name"])

    def test_feature_dictionary_row_count_matches_visible_features(self, bundle: Path) -> None:
        df = pd.read_csv(bundle / "feature_dictionary.csv")
        expected = sum(
            1 for f in LEAD_SNAPSHOT_FEATURES if f.name not in STUDENT_PUBLIC_REDACTED_COLUMNS
        )
        assert len(df) == expected

    def test_validate_bundle_passes(self, bundle: Path) -> None:
        """The new exposure-redaction check must not flag a properly built bundle."""
        errors = validate_bundle(bundle)
        # Realism checks may emit warnings on tiny bundles, but exposure
        # redaction errors should not be among them.
        redaction_errors = [e for e in errors if "redacted columns" in e]
        assert redaction_errors == []


# ---------------------------------------------------------------------------
# End-to-end: research_instructor keeps everything
# ---------------------------------------------------------------------------


class TestResearchInstructorPreservesAll:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("research_instructor_full")
        _build("research_instructor", out)
        return out

    def test_current_stage_present_in_all_splits(self, bundle: Path) -> None:
        for split in ("train", "valid", "test"):
            cols = _task_columns(bundle, split)
            assert "current_stage" in cols, f"current_stage missing from instructor {split} split"

    def test_total_touches_all_present(self, bundle: Path) -> None:
        cols = _task_columns(bundle, "train")
        assert "total_touches_all" in cols

    def test_feature_dictionary_includes_all_features(self, bundle: Path) -> None:
        df = pd.read_csv(bundle / "feature_dictionary.csv")
        assert len(df) == len(LEAD_SNAPSHOT_FEATURES)
        assert "current_stage" in set(df["name"])
        assert "total_touches_all" in set(df["name"])


# ---------------------------------------------------------------------------
# Cross-mode invariant: shared columns have identical values
# ---------------------------------------------------------------------------


class TestCrossModeConsistency:
    @pytest.fixture(scope="class")
    def both(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
        student = tmp_path_factory.mktemp("xmode_student")
        instructor = tmp_path_factory.mktemp("xmode_instructor")
        _build("student_public", student, seed=99)
        _build("research_instructor", instructor, seed=99)
        return student, instructor

    def test_student_columns_are_subset_of_instructor(self, both: tuple[Path, Path]) -> None:
        student, instructor = both
        s_cols = _task_columns(student, "train")
        i_cols = _task_columns(instructor, "train")
        assert s_cols.issubset(i_cols)

    def test_instructor_extra_columns_are_exactly_redacted_set(
        self, both: tuple[Path, Path]
    ) -> None:
        student, instructor = both
        s_cols = _task_columns(student, "train")
        i_cols = _task_columns(instructor, "train")
        extra = i_cols - s_cols
        assert extra == set(STUDENT_PUBLIC_REDACTED_COLUMNS)

    def test_shared_column_values_match(self, both: tuple[Path, Path]) -> None:
        student, instructor = both
        s_df = pd.read_parquet(student / "tasks/converted_within_90_days/train.parquet")
        i_df = pd.read_parquet(instructor / "tasks/converted_within_90_days/train.parquet")
        shared = [c for c in s_df.columns if c in i_df.columns]
        assert s_df[shared].reset_index(drop=True).equals(i_df[shared].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Validation: enforce the invariant via validate_bundle
# ---------------------------------------------------------------------------


class TestValidateBundleEnforcesRedaction:
    def test_corrupted_student_bundle_fails_redaction_check(self, tmp_path: Path) -> None:
        """If a build pipeline regresses and re-introduces ``current_stage``,
        ``validate_bundle`` must catch it."""
        # Build an instructor bundle (which includes current_stage), then
        # rewrite the manifest to claim it is student_public.
        out = tmp_path / "tampered"
        _build("research_instructor", out)
        manifest_path = out / "manifest.json"
        import json

        manifest = json.loads(manifest_path.read_text())
        manifest["exposure_mode"] = "student_public"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        errors = validate_bundle(out, include_realism=False)
        redaction_errors = [e for e in errors if "redacted columns" in e]
        assert redaction_errors, (
            "validate_bundle must flag a student_public-claiming bundle that "
            "still contains current_stage"
        )

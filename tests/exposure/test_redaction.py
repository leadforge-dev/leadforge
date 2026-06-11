"""End-to-end tests for exposure-mode column redaction.

These tests cover the post-v1 leakage fix: any column whose FeatureSpec
has the current ``ExposureMode`` in its ``redact_in_modes`` set is
stripped from the published bundle for that mode.  ``current_stage`` is
redacted in ``student_public``; ``total_touches_all`` (the deliberately
included pedagogical trap) is preserved in all modes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from leadforge.api.generator import Generator
from leadforge.core.enums import ExposureMode
from leadforge.schemes.lead_scoring.features import LEAD_SNAPSHOT_FEATURES, redacted_columns_for
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


def test_redaction_set_for_student_public_is_non_empty() -> None:
    """If this regresses to empty, the fix is not actually doing anything."""
    assert "current_stage" in redacted_columns_for(ExposureMode.student_public)


def test_redaction_set_excludes_pedagogical_trap() -> None:
    assert "total_touches_all" not in redacted_columns_for(ExposureMode.student_public)


def test_redaction_set_for_research_instructor_is_empty() -> None:
    assert redacted_columns_for(ExposureMode.research_instructor) == frozenset()


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
            leaked = cols & redacted_columns_for(ExposureMode.student_public)
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
        redacted = redacted_columns_for(ExposureMode.student_public)
        expected = sum(1 for f in LEAD_SNAPSHOT_FEATURES if f.name not in redacted)
        assert len(df) == expected

    def test_manifest_records_redacted_columns(self, bundle: Path) -> None:
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert "redacted_columns" in manifest
        declared = set(manifest["redacted_columns"])
        expected = set(redacted_columns_for(ExposureMode.student_public))
        assert declared == expected

    def test_validate_bundle_passes(self, bundle: Path) -> None:
        """The new exposure-redaction check must not flag a properly built bundle."""
        errors = validate_bundle(bundle)
        # Realism checks may emit warnings on tiny bundles, but exposure
        # redaction errors should not be among them.
        redaction_errors = [e for e in errors if "redacted columns" in e]
        assert redaction_errors == []

    def test_no_zero_variance_features(self, bundle: Path) -> None:
        """Guard against constant or near-constant columns regressing into
        the bundle.

        ``is_mql`` was constant ``True`` and is caught by the strict
        ``nunique >= 2`` assertion that runs on every bundle size.

        For larger bundles (≥ 200 rows), additionally guard against the
        weaker case of a column whose rarest low-cardinality value
        appears in fewer than 1% of rows — practically zero-variance for
        modelling.  Skipped on the tiny test fixtures because the 1%
        threshold is below 2 rows there and the test would false-positive
        on legitimate small-sample sparsity.

        ID columns, the timestamp, and the target are exempt — IDs are
        unique by design, the timestamp varies trivially, and the target's
        variance is the dataset's purpose.
        """
        df = pd.read_parquet(bundle / "tasks/converted_within_90_days/train.parquet")
        exempt = {"account_id", "contact_id", "lead_id", "lead_created_at"}
        target = next(f.name for f in LEAD_SNAPSHOT_FEATURES if f.is_target)
        exempt.add(target)
        n = len(df)
        # 1% of rows, at least 2.  Only enforced on bundles large enough
        # for the threshold to be statistically meaningful.
        check_rare_class = n >= 200
        min_rare_count = max(2, n // 100)

        for col in df.columns:
            if col in exempt:
                continue
            counts = df[col].value_counts(dropna=False)
            assert len(counts) >= 2, (
                f"feature {col!r} has zero variance in the published "
                f"student_public bundle ({len(counts)} distinct value)"
            )
            if check_rare_class and len(counts) <= 5:
                rarest_count = int(counts.min())
                assert rarest_count >= min_rare_count, (
                    f"feature {col!r} is near-constant in the published "
                    f"student_public bundle: rarest value appears "
                    f"{rarest_count} times in {n} rows "
                    f"(threshold {min_rare_count})"
                )


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
        assert extra == set(redacted_columns_for(ExposureMode.student_public))

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
    def test_regression_re_inserted_redacted_column_is_caught(self, tmp_path: Path) -> None:
        """Real regression scenario: a future bug causes the writer to leave
        ``current_stage`` in a student_public task split.  We simulate this
        by writing a real student_public bundle, then re-injecting
        ``current_stage`` into one of its parquet files.  ``validate_bundle``
        must flag it independently of the writer's filter logic.
        """
        out = tmp_path / "regressed"
        _build("student_public", out)

        train_path = out / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["current_stage"] = "negotiation"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), train_path)

        errors = validate_bundle(out, include_realism=False)
        redaction_errors = [e for e in errors if "redacted columns" in e and "current_stage" in e]
        assert redaction_errors, (
            "validate_bundle must flag a student_public bundle whose task split "
            "contains current_stage, derived from the feature spec independently"
        )

    def test_manifest_disagreement_with_feature_spec_is_caught(self, tmp_path: Path) -> None:
        """The validator cross-checks ``manifest.redacted_columns`` against
        the feature-spec-derived expected set."""
        out = tmp_path / "manifest_mismatch"
        _build("student_public", out)

        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["redacted_columns"] = []  # claim nothing was redacted
        manifest_path.write_text(json.dumps(manifest, indent=2))

        errors = validate_bundle(out, include_realism=False)
        mismatch_errors = [e for e in errors if "manifest.redacted_columns" in e]
        assert mismatch_errors

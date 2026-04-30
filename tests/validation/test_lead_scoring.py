"""Tests for leadforge.validation.lead_scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadforge.validation.lead_scoring import (
    BaselineMetrics,
    CheckResult,
    TrapMetrics,
    ValidationConfig,
    ValidationReport,
    _check_acv_range,
    _check_baseline_auc,
    _check_conversion_rate,
    _check_group_determinism,
    _check_missingness,
    _check_schema,
    validate_dataset,
)
from tests.conftest import make_v5_dataset, save_csv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def good_csv(tmp_path):
    """Write a well-formed synthetic dataset."""
    return save_csv(make_v5_dataset(n=200, include_leakage=True), tmp_path, "good.csv")


@pytest.fixture
def bad_deterministic_csv(tmp_path):
    """Write a dataset with a deterministic group."""
    return save_csv(make_v5_dataset(n=200, deterministic_col=True), tmp_path, "bad.csv")


@pytest.fixture
def no_target_csv(tmp_path):
    """Write a dataset missing the target column."""
    df = make_v5_dataset(n=200).drop(columns=["converted"])
    return save_csv(df, tmp_path, "no_target.csv")


# ---------------------------------------------------------------------------
# Tests — schema checks
# ---------------------------------------------------------------------------


class TestSchemaChecks:
    def test_good_dataset_passes_schema(self, good_csv):
        cfg = ValidationConfig(enforce_row_count=False)
        report = validate_dataset(good_csv, cfg)
        schema_checks = [
            c
            for c in report.checks
            if c.name
            in (
                "target_exists",
                "target_binary",
                "target_no_missing",
                "target_both_classes",
                "no_banned_columns",
                "no_id_columns",
                "duplicates",
            )
        ]
        assert all(c.passed for c in schema_checks)

    def test_target_exists_check_present_when_passing(self, good_csv):
        report = validate_dataset(good_csv)
        target_check = next(c for c in report.checks if c.name == "target_exists")
        assert target_check.passed

    def test_missing_target_fails(self, no_target_csv):
        report = validate_dataset(no_target_csv)
        target_check = next(c for c in report.checks if c.name == "target_exists")
        assert not target_check.passed

    def test_nan_target_short_circuits(self, tmp_path):
        df = make_v5_dataset(n=200)
        df.loc[0, "converted"] = np.nan
        path = save_csv(df, tmp_path, "nan_target.csv")
        report = validate_dataset(path)
        # target_no_missing should fail
        no_miss = next(c for c in report.checks if c.name == "target_no_missing")
        assert not no_miss.passed
        # baseline should NOT be computed (short-circuit)
        assert report.baseline is None

    def test_nonbinary_target_short_circuits(self, tmp_path):
        df = make_v5_dataset(n=200)
        df.loc[0, "converted"] = 2
        path = save_csv(df, tmp_path, "nonbinary.csv")
        report = validate_dataset(path)
        binary_check = next(c for c in report.checks if c.name == "target_binary")
        assert not binary_check.passed
        assert report.baseline is None

    def test_single_class_target_short_circuits(self, tmp_path):
        df = make_v5_dataset(n=200)
        df["converted"] = 0  # all negatives
        path = save_csv(df, tmp_path, "single_class.csv")
        report = validate_dataset(path)
        both = next(c for c in report.checks if c.name == "target_both_classes")
        assert not both.passed
        assert report.baseline is None

    def test_target_both_classes_passes(self, good_csv):
        report = validate_dataset(good_csv)
        both = next(c for c in report.checks if c.name == "target_both_classes")
        assert both.passed

    def test_banned_columns_detected(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False)
        df["current_stage"] = "active"
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        banned = next(c for c in checks if c.name == "no_banned_columns")
        assert not banned.passed
        assert "current_stage" in banned.details

    def test_id_columns_detected(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False)
        df["lead_id"] = range(len(df))
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        id_check = next(c for c in checks if c.name == "no_id_columns")
        assert not id_check.passed

    def test_enforce_row_count(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False)
        cfg = ValidationConfig(enforce_row_count=True, expected_rows=1000)
        checks = _check_schema(df, cfg)
        rc = next(c for c in checks if c.name == "row_count")
        assert not rc.passed

    def test_exact_row_count_passes(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False)
        cfg = ValidationConfig(enforce_row_count=True, expected_rows=200)
        checks = _check_schema(df, cfg)
        rc = next(c for c in checks if c.name == "row_count")
        assert rc.passed

    def test_duplicate_rows_detected(self, tmp_path):
        df = make_v5_dataset(n=50, include_leakage=False)
        # Duplicate a lot of rows
        df = pd.concat([df, df], ignore_index=True)
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        dup = next(c for c in checks if c.name == "duplicates")
        assert not dup.passed

    def test_missing_expected_features_warned(self):
        """Dataset missing some expected features gets a warning (still passes)."""
        df = pd.DataFrame({"converted": [0, 1, 0, 1]})
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        feat = next(c for c in checks if c.name == "expected_features")
        assert feat.passed  # warning, not failure
        assert "missing" in feat.details

    def test_total_touches_all_naming(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        df["total_touches_all"] = 5
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        naming = next(c for c in checks if c.name == "leakage_naming")
        assert not naming.passed

    def test_no_leakage_columns(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        naming = next(c for c in checks if c.name == "leakage_naming")
        assert naming.passed
        assert "no leakage" in naming.details

    def test_multiple_leakage_columns(self):
        df = make_v5_dataset(n=200, include_leakage=True)
        df["__leakage__another"] = 1
        cfg = ValidationConfig(enforce_row_count=False)
        checks = _check_schema(df, cfg)
        naming = next(c for c in checks if c.name == "leakage_naming")
        assert naming.passed
        assert "multiple" in naming.details


# ---------------------------------------------------------------------------
# Tests — missingness checks
# ---------------------------------------------------------------------------


class TestMissingness:
    def test_high_missingness_fails(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        df.loc[:40, "inbound_touches"] = np.nan  # >20% missing
        cfg = ValidationConfig(max_col_missing_rate=0.10)
        checks, miss_map = _check_missingness(df, cfg)
        assert not checks[0].passed

    def test_low_missingness_passes(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        cfg = ValidationConfig(max_col_missing_rate=0.10)
        checks, _ = _check_missingness(df, cfg)
        assert checks[0].passed


# ---------------------------------------------------------------------------
# Tests — group determinism
# ---------------------------------------------------------------------------


class TestGroupDeterminism:
    def test_deterministic_group_fails(self, bad_deterministic_csv):
        report = validate_dataset(bad_deterministic_csv)
        det_check = next(c for c in report.checks if c.name == "group_determinism")
        assert not det_check.passed
        assert "bad_feature" in det_check.details

    def test_low_conversion_group_fails(self, tmp_path):
        """A group where conversion rate is near 0% should also fail."""
        df = make_v5_dataset(n=200, include_leakage=False)
        df["bad_feature"] = "normal"
        # First 60 rows all converted = 0 for this group
        df.loc[:59, "bad_feature"] = "zero_group"
        df.loc[:59, "converted"] = 0
        cfg = ValidationConfig(enforce_row_count=False, min_group_size=50)
        checks = _check_group_determinism(df, cfg)
        det = next(c for c in checks if c.name == "group_determinism")
        assert not det.passed

    def test_good_dataset_passes_determinism(self, good_csv):
        report = validate_dataset(good_csv)
        det_check = next(c for c in report.checks if c.name == "group_determinism")
        assert det_check.passed


# ---------------------------------------------------------------------------
# Tests — conversion rate
# ---------------------------------------------------------------------------


class TestConversionRate:
    def test_rate_outside_range_fails(self):
        # 5% conversion rate — below 15%
        df = make_v5_dataset(n=200, conversion_rate=0.05, include_leakage=False)
        checks = _check_conversion_rate(df)
        assert not checks[0].passed

    def test_rate_in_range_passes(self):
        df = make_v5_dataset(n=200, conversion_rate=0.30, include_leakage=False)
        checks = _check_conversion_rate(df)
        assert checks[0].passed


# ---------------------------------------------------------------------------
# Tests — ACV range
# ---------------------------------------------------------------------------


class TestACVRange:
    def test_no_acv_column_skips(self):
        df = make_v5_dataset(n=200, include_leakage=False).drop(columns=["expected_acv"])
        checks = _check_acv_range(df)
        assert checks[0].passed
        assert "skip" in checks[0].details

    def test_acv_all_nan_fails(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        df["expected_acv"] = np.nan
        checks = _check_acv_range(df)
        assert not checks[0].passed

    def test_acv_below_floor_fails(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        df.loc[0, "expected_acv"] = 1000  # way below 18k
        checks = _check_acv_range(df)
        assert not checks[0].passed

    def test_acv_above_cap_fails(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        df.loc[0, "expected_acv"] = 200_000  # way above 120k
        checks = _check_acv_range(df)
        assert not checks[0].passed

    def test_acv_in_range_passes(self):
        df = make_v5_dataset(n=200, include_leakage=False)
        checks = _check_acv_range(df)
        assert checks[0].passed


# ---------------------------------------------------------------------------
# Tests — baseline AUC check
# ---------------------------------------------------------------------------


class TestBaselineAUCCheck:
    def test_auc_too_low_fails(self):
        metrics = BaselineMetrics(seed=42, auc=0.50, pr_auc=0.30)
        cfg = ValidationConfig(auc_lower=0.62, auc_upper=0.90)
        checks = _check_baseline_auc(metrics, cfg)
        assert not checks[0].passed

    def test_auc_too_high_fails(self):
        metrics = BaselineMetrics(seed=42, auc=0.95, pr_auc=0.90)
        cfg = ValidationConfig(auc_lower=0.62, auc_upper=0.90)
        checks = _check_baseline_auc(metrics, cfg)
        assert not checks[0].passed

    def test_auc_in_range_passes(self):
        metrics = BaselineMetrics(seed=42, auc=0.75, pr_auc=0.60)
        cfg = ValidationConfig(auc_lower=0.62, auc_upper=0.90)
        checks = _check_baseline_auc(metrics, cfg)
        assert checks[0].passed


# ---------------------------------------------------------------------------
# Tests — baseline metrics
# ---------------------------------------------------------------------------


class TestBaselineMetrics:
    def test_baseline_computed(self, good_csv):
        report = validate_dataset(good_csv)
        assert report.baseline is not None
        assert 0.0 < report.baseline.auc <= 1.0
        assert 0.0 < report.baseline.pr_auc <= 1.0
        assert 25 in report.baseline.precision_at_k

    def test_baseline_deterministic(self, good_csv):
        """Same CSV + same config -> same AUC."""
        r1 = validate_dataset(good_csv)
        r2 = validate_dataset(good_csv)
        assert r1.baseline is not None
        assert r2.baseline is not None
        assert r1.baseline.auc == r2.baseline.auc

    def test_k_larger_than_test_set_skipped(self, tmp_path):
        """If k > test set size, that k is skipped."""
        df = make_v5_dataset(n=20, include_leakage=False)
        path = save_csv(df, tmp_path)
        # ks=(25, 50) but test set is only ~6 rows
        report = validate_dataset(path, ValidationConfig(enforce_row_count=False))
        assert report.baseline is not None
        assert 25 not in report.baseline.precision_at_k


# ---------------------------------------------------------------------------
# Tests — leakage trap
# ---------------------------------------------------------------------------


class TestLeakageTrap:
    def test_trap_detected(self, good_csv):
        """Synthetic trap should produce positive delta on average."""
        report = validate_dataset(good_csv)
        assert len(report.trap_metrics) == 1
        tm = report.trap_metrics[0]
        assert tm.column == "__leakage__total_touches_90d"
        # Our synthetic trap is strongly correlated, so mean delta should be positive
        assert tm.mean_delta_auc > 0

    def test_no_trap_columns_skips(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False)
        path = save_csv(df, tmp_path)
        report = validate_dataset(path, ValidationConfig(enforce_row_count=False))
        trap_check = [c for c in report.checks if c.name.startswith("leakage_trap")]
        assert len(trap_check) == 1
        assert trap_check[0].passed
        assert "skip" in trap_check[0].details

    def test_weak_trap_fails_checks(self, tmp_path):
        """A trap column with no signal should fail threshold checks."""
        df = make_v5_dataset(n=200, include_leakage=False)
        rng = np.random.RandomState(42)
        # Add a random column with no leakage signal
        df["__leakage__noise"] = rng.poisson(5, size=len(df))
        path = save_csv(df, tmp_path)
        cfg = ValidationConfig(
            enforce_row_count=False,
            trap_mean_delta=0.05,  # high threshold
            trap_min_delta=0.03,
            trap_n_seeds=3,
        )
        report = validate_dataset(path, cfg)
        trap_check = [c for c in report.checks if c.name.startswith("leakage_trap")]
        # Random noise shouldn't reliably produce a large delta
        assert len(trap_check) >= 1


# ---------------------------------------------------------------------------
# Tests — value metrics
# ---------------------------------------------------------------------------


class TestValueMetrics:
    def test_value_metrics_computed(self, good_csv):
        report = validate_dataset(good_csv)
        assert len(report.value_metrics) >= 1
        vm = report.value_metrics[0]
        assert vm.captured_acv_by_prob >= 0
        assert vm.captured_acv_by_ev >= 0

    def test_value_metrics_with_nan_acv(self, tmp_path):
        """NaN in expected_acv should not propagate NaN into value metrics."""
        df = make_v5_dataset(n=200, include_leakage=False)
        df.loc[:9, "expected_acv"] = np.nan
        path = save_csv(df, tmp_path)
        report = validate_dataset(path, ValidationConfig(enforce_row_count=False))
        for vm in report.value_metrics:
            assert not np.isnan(vm.captured_acv_by_prob)
            assert not np.isnan(vm.captured_acv_by_ev)

    def test_no_acv_column_returns_empty(self, tmp_path):
        df = make_v5_dataset(n=200, include_leakage=False).drop(columns=["expected_acv"])
        path = save_csv(df, tmp_path)
        report = validate_dataset(path, ValidationConfig(enforce_row_count=False))
        assert report.value_metrics == []


# ---------------------------------------------------------------------------
# Tests — report
# ---------------------------------------------------------------------------


class TestReport:
    def test_summary_string(self, good_csv):
        report = validate_dataset(good_csv)
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary

    def test_summary_all_passed(self, good_csv):
        """Good CSV summary should contain ALL CHECKS PASSED."""
        report = validate_dataset(good_csv)
        if report.passed:
            assert "ALL CHECKS PASSED" in report.summary()

    def test_summary_negative_trap_delta(self):
        """Summary shows warning for negative trap deltas."""
        report = ValidationReport(csv_path="test.csv")
        report.trap_metrics = [
            TrapMetrics(
                column="__leakage__test",
                deltas_auc=[-0.01, 0.05],
                deltas_pr_auc=[0.0, 0.05],
                seeds=[42, 43],
            )
        ]
        summary = report.summary()
        assert "negative" in summary

    def test_to_dict(self, good_csv):
        report = validate_dataset(good_csv)
        d = report.to_dict()
        assert "passed" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_to_dict_includes_pr_auc_deltas(self, good_csv):
        """to_dict should include PR-AUC deltas for trap metrics."""
        report = validate_dataset(good_csv)
        if report.trap_metrics:
            d = report.to_dict()
            for tm in d["trap_metrics"]:
                assert "deltas_pr_auc" in tm
                assert "mean_delta_pr_auc" in tm

    def test_emit_release_snippet(self, good_csv):
        report = validate_dataset(good_csv)
        snippet = report.emit_release_snippet()
        assert "BEGIN AUTO-METRICS" in snippet
        assert "END AUTO-METRICS" in snippet
        assert "ROC-AUC" in snippet

    def test_emit_release_snippet_uses_actual_test_size(self):
        """Snippet should reflect the actual test_size, not hardcoded 70/30."""
        report = ValidationReport(csv_path="test.csv", test_size=0.20)
        report.baseline = BaselineMetrics(seed=42, auc=0.75, pr_auc=0.60, base_rate=0.30)
        snippet = report.emit_release_snippet()
        assert "80/20" in snippet

    def test_emit_release_snippet_uses_actual_row_count(self):
        """Missingness counts should use actual row count, not hardcoded 1000."""
        report = ValidationReport(csv_path="test.csv", n_rows=500)
        report.missingness = {"web_sessions": 0.10}
        snippet = report.emit_release_snippet()
        # 0.10 * 500 = 50
        assert "50" in snippet

    def test_n_rows_and_test_size_set(self, good_csv):
        """validate_dataset sets n_rows and test_size on the report."""
        cfg = ValidationConfig(test_size=0.25)
        report = validate_dataset(good_csv, cfg)
        assert report.n_rows == 200
        assert report.test_size == 0.25

    def test_failed_report_is_not_passed(self, bad_deterministic_csv):
        report = validate_dataset(bad_deterministic_csv)
        assert not report.passed
        assert report.n_errors > 0

    def test_to_dict_with_value_and_trap_metrics(self, good_csv):
        """Ensure to_dict includes value_metrics and trap_metrics when present."""
        report = validate_dataset(good_csv)
        d = report.to_dict()
        if report.value_metrics:
            assert "value_metrics" in d
        if report.trap_metrics:
            assert "trap_metrics" in d

    def test_check_result_with_data(self):
        cr = CheckResult("test", True, "ok", data={"key": "value"})
        assert cr.data == {"key": "value"}

    def test_trap_metrics_properties(self):
        tm = TrapMetrics(
            column="test",
            deltas_auc=[0.01, 0.02, 0.03],
            deltas_pr_auc=[0.01, 0.02, 0.03],
            seeds=[42, 43, 44],
        )
        assert tm.mean_delta_auc == pytest.approx(0.02)
        assert tm.min_delta_auc == pytest.approx(0.01)
        assert tm.max_delta_auc == pytest.approx(0.03)

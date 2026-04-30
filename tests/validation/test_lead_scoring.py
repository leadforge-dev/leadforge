"""Tests for leadforge.validation.lead_scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadforge.validation.lead_scoring import (
    ValidationConfig,
    validate_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(99)


def _make_dataset(
    n: int = 200,
    conversion_rate: float = 0.30,
    include_leakage: bool = True,
    deterministic_col: bool = False,
    seed: int = 99,
) -> pd.DataFrame:
    """Build a small synthetic dataset that passes basic checks."""
    rng = np.random.RandomState(seed)
    n_pos = int(n * conversion_rate)
    n_neg = n - n_pos

    converted = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(converted)

    industries = rng.choice(["manufacturing", "logistics", "services", "healthcare"], size=n)
    regions = rng.choice(["US", "UK"], size=n)
    sizes = rng.choice(["200-499", "500-999", "1000-1999", "2000+"], size=n)
    revenues = rng.choice(["$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+"], size=n)
    roles = rng.choice(["finance", "ap_manager", "it_director", "procurement"], size=n)
    seniority = rng.choice(
        ["individual_contributor", "manager", "director", "vp", "c_suite"], size=n
    )
    sources = rng.choice(["inbound_marketing", "sdr_outbound", "partner_referral"], size=n)

    df = pd.DataFrame(
        {
            "industry": industries,
            "region": regions,
            "company_size": sizes,
            "company_revenue": revenues,
            "contact_role": roles,
            "seniority": seniority,
            "lead_source": sources,
            "opportunity_created": rng.randint(0, 2, size=n),
            "demo_completed": rng.randint(0, 2, size=n),
            "expected_acv": rng.uniform(18_000, 120_000, size=n).round(0),
            "inbound_touches": rng.poisson(3, size=n),
            "outbound_touches": rng.poisson(2, size=n),
            "touches_week_1": rng.poisson(2, size=n),
            "days_since_first_touch": rng.uniform(0, 14, size=n).round(1),
            "web_sessions": rng.poisson(4, size=n).astype(float),
            "sales_activities": rng.poisson(3, size=n),
            "days_since_last_touch": rng.uniform(0, 14, size=n).round(1),
            "converted": converted,
        }
    )

    # Inject some missingness
    miss_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[miss_idx, "web_sessions"] = np.nan

    if include_leakage:
        # Leakage: positively correlated with target
        noise = rng.poisson(3, size=n)
        df["__leakage__total_touches_90d"] = converted * rng.poisson(8, size=n) + noise

    if deterministic_col:
        # Make a column that perfectly predicts conversion for a large group
        df["bad_feature"] = "normal"
        # First 60 rows all converted = 1
        df.loc[:59, "bad_feature"] = "leaked"
        df.loc[:59, "converted"] = 1

    return df


@pytest.fixture
def good_csv(tmp_path):
    """Write a well-formed synthetic dataset."""
    path = tmp_path / "good.csv"
    df = _make_dataset(n=200, include_leakage=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def bad_deterministic_csv(tmp_path):
    """Write a dataset with a deterministic group."""
    path = tmp_path / "bad.csv"
    df = _make_dataset(n=200, deterministic_col=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def no_target_csv(tmp_path):
    """Write a dataset missing the target column."""
    path = tmp_path / "no_target.csv"
    df = _make_dataset(n=200)
    df = df.drop(columns=["converted"])
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests
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
                "no_banned_columns",
                "no_id_columns",
                "duplicates",
            )
        ]
        assert all(c.passed for c in schema_checks)

    def test_missing_target_fails(self, no_target_csv):
        report = validate_dataset(no_target_csv)
        target_check = next(c for c in report.checks if c.name == "target_exists")
        assert not target_check.passed


class TestGroupDeterminism:
    def test_deterministic_group_fails(self, bad_deterministic_csv):
        report = validate_dataset(bad_deterministic_csv)
        det_check = next(c for c in report.checks if c.name == "group_determinism")
        assert not det_check.passed
        assert "bad_feature" in det_check.details

    def test_good_dataset_passes_determinism(self, good_csv):
        report = validate_dataset(good_csv)
        det_check = next(c for c in report.checks if c.name == "group_determinism")
        assert det_check.passed


class TestBaselineMetrics:
    def test_baseline_computed(self, good_csv):
        report = validate_dataset(good_csv)
        assert report.baseline is not None
        assert 0.0 < report.baseline.auc <= 1.0
        assert 0.0 < report.baseline.pr_auc <= 1.0
        assert 25 in report.baseline.precision_at_k

    def test_baseline_deterministic(self, good_csv):
        """Same CSV + same config → same AUC."""
        r1 = validate_dataset(good_csv)
        r2 = validate_dataset(good_csv)
        assert r1.baseline is not None
        assert r2.baseline is not None
        assert r1.baseline.auc == r2.baseline.auc


class TestLeakageTrap:
    def test_trap_detected(self, good_csv):
        """Synthetic trap should produce positive delta on average."""
        report = validate_dataset(good_csv)
        assert len(report.trap_metrics) == 1
        tm = report.trap_metrics[0]
        assert tm.column == "__leakage__total_touches_90d"
        # Our synthetic trap is strongly correlated, so mean delta should be positive
        assert tm.mean_delta_auc > 0


class TestValueMetrics:
    def test_value_metrics_computed(self, good_csv):
        report = validate_dataset(good_csv)
        assert len(report.value_metrics) >= 1
        vm = report.value_metrics[0]
        assert vm.captured_acv_by_prob >= 0
        assert vm.captured_acv_by_ev >= 0


class TestReport:
    def test_summary_string(self, good_csv):
        report = validate_dataset(good_csv)
        summary = report.summary()
        assert "PASS" in summary or "FAIL" in summary

    def test_to_dict(self, good_csv):
        report = validate_dataset(good_csv)
        d = report.to_dict()
        assert "passed" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_emit_release_snippet(self, good_csv):
        report = validate_dataset(good_csv)
        snippet = report.emit_release_snippet()
        assert "BEGIN AUTO-METRICS" in snippet
        assert "END AUTO-METRICS" in snippet
        assert "ROC-AUC" in snippet

    def test_failed_report_is_not_passed(self, bad_deterministic_csv):
        report = validate_dataset(bad_deterministic_csv)
        assert not report.passed
        assert report.n_errors > 0

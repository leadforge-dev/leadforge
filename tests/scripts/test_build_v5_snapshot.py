"""Tests for scripts/build_v5_snapshot.py pipeline functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script module (not in a package, so use importlib)
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_v5_snapshot.py"

spec = importlib.util.spec_from_file_location("build_v5_snapshot", _SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
build_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_v5)

# Re-export for convenience
subsample = build_v5.subsample
inject_missingness = build_v5.inject_missingness
derive_binary_features = build_v5.derive_binary_features
cap_expected_acv = build_v5.cap_expected_acv
rename_and_select = build_v5.rename_and_select
boost_leakage_trap = build_v5.boost_leakage_trap
ACV_FLOOR = build_v5.ACV_FLOOR
ACV_CAP = build_v5.ACV_CAP
_FINAL_COLUMNS = build_v5._FINAL_COLUMNS
_RENAME_MAP = build_v5._RENAME_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    n: int = 500,
    conversion_rate: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal snapshot DataFrame that looks like build_snapshot() output.

    Contains the pre-rename column names expected by the pipeline steps.
    """
    rng = np.random.RandomState(seed)
    n_pos = int(n * conversion_rate)
    n_neg = n - n_pos
    converted = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(converted)

    return pd.DataFrame(
        {
            "industry": rng.choice(["manufacturing", "logistics", "services"], size=n),
            "region": rng.choice(["US", "UK", "EU"], size=n),
            "employee_band": rng.choice(["200-499", "500-999", "1000-1999"], size=n),
            "estimated_revenue_band": rng.choice(["$1M-$10M", "$10M-$50M", "$50M-$200M"], size=n),
            "role_function": rng.choice(["finance", "ap_manager", "it_director"], size=n),
            "seniority": rng.choice(
                ["individual_contributor", "manager", "director", "vp"], size=n
            ),
            "lead_source": rng.choice(
                ["inbound_marketing", "sdr_outbound", "partner_referral"], size=n
            ),
            "opportunity_created": rng.choice([True, False], size=n),
            "demo_page_views": rng.poisson(1, size=n),
            "expected_acv": rng.uniform(5_000, 200_000, size=n).round(0),
            "inbound_touch_count": rng.poisson(3, size=n),
            "outbound_touch_count": rng.poisson(2, size=n),
            "touches_week_1": rng.poisson(2, size=n),
            "days_since_first_touch": rng.uniform(0, 14, size=n).round(1),
            "session_count": rng.poisson(4, size=n).astype(float),
            "activity_count": rng.poisson(3, size=n),
            "days_since_last_touch": rng.uniform(0, 14, size=n).round(1),
            "total_touches_all": rng.poisson(8, size=n),
            "converted_within_90_days": converted,
        }
    )


def _make_v5_df(
    n: int = 500,
    conversion_rate: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a DataFrame in v5 format (post-rename, with all final columns)."""
    snapshot = _make_snapshot(n=n, conversion_rate=conversion_rate, seed=seed)
    df = derive_binary_features(snapshot)
    df = cap_expected_acv(df)
    return rename_and_select(df)


# ---------------------------------------------------------------------------
# Tests — derive_binary_features
# ---------------------------------------------------------------------------


class TestDeriveBinaryFeatures:
    def test_opportunity_created_is_int(self):
        snapshot = _make_snapshot()
        result = derive_binary_features(snapshot)
        assert result["opportunity_created"].dtype in (np.int64, np.int32, int)
        assert set(result["opportunity_created"].unique()).issubset({0, 1})

    def test_demo_completed_derived_from_page_views(self):
        snapshot = _make_snapshot()
        snapshot["demo_page_views"] = [0, 3, 0, 1, 0] * (len(snapshot) // 5)
        result = derive_binary_features(snapshot)
        expected = (snapshot["demo_page_views"] > 0).astype(int)
        pd.testing.assert_series_equal(result["demo_completed"], expected, check_names=False)

    def test_does_not_modify_input(self):
        snapshot = _make_snapshot()
        original = snapshot.copy()
        derive_binary_features(snapshot)
        pd.testing.assert_frame_equal(snapshot, original)


# ---------------------------------------------------------------------------
# Tests — cap_expected_acv
# ---------------------------------------------------------------------------


class TestCapExpectedACV:
    def test_values_clipped_to_range(self):
        snapshot = _make_snapshot()
        snapshot["expected_acv"] = [1_000, 50_000, 200_000, ACV_FLOOR, ACV_CAP] * (
            len(snapshot) // 5
        )
        result = cap_expected_acv(snapshot)
        assert result["expected_acv"].min() >= ACV_FLOOR
        assert result["expected_acv"].max() <= ACV_CAP

    def test_values_within_range_unchanged(self):
        snapshot = _make_snapshot()
        snapshot["expected_acv"] = 50_000.0
        result = cap_expected_acv(snapshot)
        assert (result["expected_acv"] == 50_000.0).all()

    def test_does_not_modify_input(self):
        snapshot = _make_snapshot()
        original = snapshot.copy()
        cap_expected_acv(snapshot)
        pd.testing.assert_frame_equal(snapshot, original)


# ---------------------------------------------------------------------------
# Tests — rename_and_select
# ---------------------------------------------------------------------------


class TestRenameAndSelect:
    def test_output_columns_match_final(self):
        df = _make_v5_df()
        assert list(df.columns) == _FINAL_COLUMNS

    def test_converted_is_int(self):
        df = _make_v5_df()
        assert df["converted"].dtype in (np.int64, np.int32, int)
        assert set(df["converted"].unique()).issubset({0, 1})

    def test_missing_column_raises(self):
        snapshot = _make_snapshot()
        snapshot = derive_binary_features(snapshot)
        snapshot = cap_expected_acv(snapshot)
        # Drop a required source column
        snapshot = snapshot.drop(columns=["industry"])
        with pytest.raises(ValueError, match="Missing required columns"):
            rename_and_select(snapshot)

    def test_rename_mapping_applied(self):
        snapshot = _make_snapshot()
        df = derive_binary_features(snapshot)
        df = cap_expected_acv(df)
        result = rename_and_select(df)
        # All renamed columns should exist in output
        for new_name in _RENAME_MAP.values():
            assert new_name in result.columns


# ---------------------------------------------------------------------------
# Tests — subsample
# ---------------------------------------------------------------------------


class TestSubsample:
    def test_output_size(self):
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        result = subsample(df, rng, n=100, target_rate=0.30)
        assert len(result) == 100

    def test_target_rate_approximate(self):
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        result = subsample(df, rng, n=200, target_rate=0.30)
        actual_rate = result["converted"].mean()
        assert actual_rate == pytest.approx(0.30, abs=0.01)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = subsample(df, np.random.RandomState(42), n=100, target_rate=0.30)
        r2 = subsample(df, np.random.RandomState(42), n=100, target_rate=0.30)
        pd.testing.assert_frame_equal(r1, r2)

    def test_insufficient_positives(self, capsys):
        """When fewer positives available than needed, warns and adjusts."""
        df = _make_v5_df(n=200, conversion_rate=0.05)  # only ~10 positives
        rng = np.random.RandomState(42)
        result = subsample(df, rng, n=100, target_rate=0.50)  # need 50 positives
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        # All available positives should be included
        assert result["converted"].sum() <= 10

    def test_insufficient_negatives(self, capsys):
        """When fewer negatives available than needed, warns and adjusts."""
        df = _make_v5_df(n=200, conversion_rate=0.95)  # only ~10 negatives
        rng = np.random.RandomState(42)
        subsample(df, rng, n=100, target_rate=0.10)  # need 90 negatives
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_index_is_reset(self):
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        result = subsample(df, rng, n=100, target_rate=0.30)
        assert list(result.index) == list(range(len(result)))

    def test_rows_come_from_input(self):
        """All subsampled rows should exist in the original."""
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        result = subsample(df, rng, n=100, target_rate=0.30)
        # Check a non-index column for membership
        for val in result["expected_acv"]:
            assert val in df["expected_acv"].values


# ---------------------------------------------------------------------------
# Tests — inject_missingness
# ---------------------------------------------------------------------------


class TestInjectMissingness:
    def test_web_sessions_has_missing(self):
        df = _make_v5_df(n=1000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        assert result["web_sessions"].isna().sum() > 0

    def test_seniority_has_missing(self):
        df = _make_v5_df(n=1000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        assert result["seniority"].isna().sum() > 0

    def test_days_since_last_touch_has_missing(self):
        df = _make_v5_df(n=1000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        assert result["days_since_last_touch"].isna().sum() > 0

    def test_days_since_first_touch_has_missing(self):
        df = _make_v5_df(n=1000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        assert result["days_since_first_touch"].isna().sum() > 0

    def test_missingness_rates_bounded(self):
        """Each column's missingness rate should stay under ~20% (well above contract <10%)."""
        df = _make_v5_df(n=2000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        for col in ["web_sessions", "seniority", "days_since_last_touch", "days_since_first_touch"]:
            rate = result[col].isna().mean()
            assert rate < 0.20, f"{col} missingness rate {rate:.2%} exceeds 20%"

    def test_other_columns_not_affected(self):
        """Columns not in the missingness spec should have no new NaN."""
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        no_miss_cols = [
            c
            for c in _FINAL_COLUMNS
            if c
            not in ("web_sessions", "seniority", "days_since_last_touch", "days_since_first_touch")
        ]
        for col in no_miss_cols:
            orig_nan = df[col].isna().sum()
            new_nan = result[col].isna().sum()
            assert new_nan == orig_nan, f"{col} gained unexpected NaN"

    def test_does_not_modify_input(self):
        df = _make_v5_df(n=500)
        original = df.copy()
        rng = np.random.RandomState(42)
        inject_missingness(df, rng)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = inject_missingness(df, np.random.RandomState(42))
        r2 = inject_missingness(df, np.random.RandomState(42))
        pd.testing.assert_frame_equal(r1, r2)

    def test_web_sessions_missingness_varies_by_source(self):
        """SDR outbound should have higher web_sessions missingness than inbound marketing."""
        df = _make_v5_df(n=3000)
        rng = np.random.RandomState(42)
        result = inject_missingness(df, rng)
        sdr_rate = result.loc[df["lead_source"] == "sdr_outbound", "web_sessions"].isna().mean()
        inbound_rate = (
            result.loc[df["lead_source"] == "inbound_marketing", "web_sessions"].isna().mean()
        )
        assert sdr_rate > inbound_rate


# ---------------------------------------------------------------------------
# Tests — boost_leakage_trap
# ---------------------------------------------------------------------------


class TestBoostLeakageTrap:
    def test_only_converted_leads_boosted(self):
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        trap_col = "__leakage__total_touches_90d"
        original_trap = df[trap_col].copy()
        result = boost_leakage_trap(df, rng)
        # Non-converted leads should be unchanged
        neg_mask = df["converted"] == 0
        pd.testing.assert_series_equal(
            result.loc[neg_mask, trap_col],
            original_trap[neg_mask],
            check_names=False,
        )

    def test_converted_leads_get_higher_or_equal(self):
        df = _make_v5_df(n=500)
        rng = np.random.RandomState(42)
        trap_col = "__leakage__total_touches_90d"
        original_trap = df[trap_col].copy()
        result = boost_leakage_trap(df, rng)
        pos_mask = df["converted"] == 1
        assert (result.loc[pos_mask, trap_col] >= original_trap[pos_mask]).all()

    def test_does_not_modify_input(self):
        df = _make_v5_df(n=500)
        original = df.copy()
        rng = np.random.RandomState(42)
        boost_leakage_trap(df, rng)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = boost_leakage_trap(df, np.random.RandomState(42))
        r2 = boost_leakage_trap(df, np.random.RandomState(42))
        pd.testing.assert_frame_equal(r1, r2)

    def test_boost_increases_mean_for_converted(self):
        """Mean trap value should be higher for converted leads after boost."""
        df = _make_v5_df(n=1000)
        rng = np.random.RandomState(42)
        trap_col = "__leakage__total_touches_90d"
        before_mean = df.loc[df["converted"] == 1, trap_col].mean()
        result = boost_leakage_trap(df, rng)
        after_mean = result.loc[result["converted"] == 1, trap_col].mean()
        assert after_mean > before_mean

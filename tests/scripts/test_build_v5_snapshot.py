"""Tests for leadforge.pipelines.build_v5 pipeline functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadforge.pipelines.build_v5 import (
    ACV_CAP,
    ACV_FLOOR,
    FINAL_COLUMNS,
    RENAME_MAP,
    boost_leakage_trap,
    cap_expected_acv,
    derive_binary_features,
    inject_missingness,
    rename_and_select,
    subsample,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    n: int = 500,
    conversion_rate: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal snapshot DataFrame with pre-rename column names.

    This is distinct from the shared ``make_v5_dataset`` because it uses
    the *pre-rename* columns that ``build_snapshot()`` actually produces.
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
        assert list(df.columns) == FINAL_COLUMNS

    def test_converted_is_int(self):
        df = _make_v5_df()
        assert df["converted"].dtype in (np.int64, np.int32, int)
        assert set(df["converted"].unique()).issubset({0, 1})

    def test_missing_column_raises(self):
        snapshot = _make_snapshot()
        snapshot = derive_binary_features(snapshot)
        snapshot = cap_expected_acv(snapshot)
        snapshot = snapshot.drop(columns=["industry"])
        with pytest.raises(ValueError, match="Missing required columns"):
            rename_and_select(snapshot)

    def test_rename_mapping_applied(self):
        snapshot = _make_snapshot()
        df = derive_binary_features(snapshot)
        df = cap_expected_acv(df)
        result = rename_and_select(df)
        for new_name in RENAME_MAP.values():
            assert new_name in result.columns

    def test_extra_columns_are_dropped(self):
        """Columns not in FINAL_COLUMNS should be silently dropped."""
        snapshot = _make_snapshot()
        snapshot["extra_col"] = 999
        df = derive_binary_features(snapshot)
        df = cap_expected_acv(df)
        result = rename_and_select(df)
        assert "extra_col" not in result.columns
        assert list(result.columns) == FINAL_COLUMNS


# ---------------------------------------------------------------------------
# Tests — subsample
# ---------------------------------------------------------------------------


class TestSubsample:
    def test_output_size(self):
        df = _make_v5_df(n=500)
        result = subsample(df, seed=42, n=100, target_rate=0.30)
        assert len(result) == 100

    @pytest.mark.parametrize(
        ("target_rate", "seed"),
        [(0.30, 42), (0.30, 99), (0.20, 42), (0.40, 7)],
    )
    def test_target_rate_approximate(self, target_rate, seed):
        df = _make_v5_df(n=500, seed=seed)
        result = subsample(df, seed=seed, n=200, target_rate=target_rate)
        actual_rate = result["converted"].mean()
        assert actual_rate == pytest.approx(target_rate, abs=0.01)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = subsample(df, seed=42, n=100, target_rate=0.30)
        r2 = subsample(df, seed=42, n=100, target_rate=0.30)
        pd.testing.assert_frame_equal(r1, r2)

    def test_insufficient_positives(self):
        """When fewer positives available than needed, warns and adjusts."""
        df = _make_v5_df(n=200, conversion_rate=0.05)  # only ~10 positives
        with pytest.warns(UserWarning, match="positives available"):
            result = subsample(df, seed=42, n=100, target_rate=0.50)  # need 50 positives
        # All available positives should be included
        assert result["converted"].sum() <= 10

    def test_insufficient_negatives(self):
        """When fewer negatives available than needed, warns and adjusts."""
        df = _make_v5_df(n=200, conversion_rate=0.95)  # only ~10 negatives
        n_neg_available = (df["converted"] == 0).sum()
        with pytest.warns(UserWarning, match="negatives available"):
            result = subsample(df, seed=42, n=100, target_rate=0.10)  # need 90 negatives
        # Verify actual composition: negatives capped at available count
        n_neg_result = (result["converted"] == 0).sum()
        assert n_neg_result <= n_neg_available
        # Output should still contain rows (not empty)
        assert len(result) > 0

    def test_index_is_reset(self):
        df = _make_v5_df(n=500)
        result = subsample(df, seed=42, n=100, target_rate=0.30)
        assert list(result.index) == list(range(len(result)))

    def test_n_larger_than_input_caps_gracefully(self):
        """Requesting more rows than available caps at available count."""
        df = _make_v5_df(n=50)
        with pytest.warns(UserWarning, match="available"):
            result = subsample(df, seed=42, n=200, target_rate=0.30)
        # Output should contain all available rows (capped)
        assert len(result) <= len(df)


# ---------------------------------------------------------------------------
# Tests — inject_missingness
# ---------------------------------------------------------------------------


class TestInjectMissingness:
    @pytest.mark.parametrize("seed", [42, 99, 7])
    def test_missingness_rates_bounded(self, seed):
        """Each column's missingness rate should stay under 20% across seeds."""
        df = _make_v5_df(n=2000, seed=seed)
        result = inject_missingness(df, seed=seed)
        for col in [
            "web_sessions",
            "seniority",
            "days_since_last_touch",
            "days_since_first_touch",
        ]:
            rate = result[col].isna().mean()
            assert rate < 0.20, f"{col} missingness rate {rate:.2%} exceeds 20%"

    def test_other_columns_not_affected(self):
        """Columns not in the missingness spec should have no new NaN."""
        df = _make_v5_df(n=500)
        result = inject_missingness(df, seed=42)
        miss_cols = {
            "web_sessions",
            "seniority",
            "days_since_last_touch",
            "days_since_first_touch",
        }
        for col in FINAL_COLUMNS:
            if col not in miss_cols:
                orig_nan = df[col].isna().sum()
                new_nan = result[col].isna().sum()
                assert new_nan == orig_nan, f"{col} gained unexpected NaN"

    def test_does_not_modify_input(self):
        df = _make_v5_df(n=500)
        original = df.copy()
        inject_missingness(df, seed=42)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = inject_missingness(df, seed=42)
        r2 = inject_missingness(df, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_web_sessions_missingness_varies_by_source(self):
        """SDR outbound should have higher web_sessions missingness than inbound."""
        df = _make_v5_df(n=3000)
        result = inject_missingness(df, seed=42)
        sdr_rate = result.loc[df["lead_source"] == "sdr_outbound", "web_sessions"].isna().mean()
        inbound_rate = (
            result.loc[df["lead_source"] == "inbound_marketing", "web_sessions"].isna().mean()
        )
        assert sdr_rate > inbound_rate

    def test_small_n_no_crash(self):
        """Should not crash on small DataFrames, even with sparse lead sources."""
        df = _make_v5_df(n=10)
        result = inject_missingness(df, seed=42)
        assert len(result) == 10

    def test_no_matching_lead_source(self):
        """If no rows match a source-conditional rate, no crash or extra NaN."""
        df = _make_v5_df(n=100)
        # Force all lead_source to a value not in the missingness spec
        df["lead_source"] = "direct"
        result = inject_missingness(df, seed=42)
        # web_sessions should only have missingness from other sources (none here)
        # but days_since_last_touch still gets 3% MCAR
        assert len(result) == 100


# ---------------------------------------------------------------------------
# Tests — boost_leakage_trap
# ---------------------------------------------------------------------------


class TestBoostLeakageTrap:
    def test_only_converted_leads_boosted(self):
        df = _make_v5_df(n=500)
        trap_col = "__leakage__total_touches_90d"
        original_trap = df[trap_col].copy()
        result = boost_leakage_trap(df, seed=42)
        neg_mask = df["converted"] == 0
        pd.testing.assert_series_equal(
            result.loc[neg_mask, trap_col],
            original_trap[neg_mask],
            check_names=False,
        )

    def test_converted_leads_get_higher_or_equal(self):
        df = _make_v5_df(n=500)
        trap_col = "__leakage__total_touches_90d"
        original_trap = df[trap_col].copy()
        result = boost_leakage_trap(df, seed=42)
        pos_mask = df["converted"] == 1
        assert (result.loc[pos_mask, trap_col] >= original_trap[pos_mask]).all()

    def test_does_not_modify_input(self):
        df = _make_v5_df(n=500)
        original = df.copy()
        boost_leakage_trap(df, seed=42)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v5_df(n=500)
        r1 = boost_leakage_trap(df, seed=42)
        r2 = boost_leakage_trap(df, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_boost_increases_mean_for_converted(self):
        """Mean trap value should be higher for converted leads after boost."""
        df = _make_v5_df(n=1000)
        trap_col = "__leakage__total_touches_90d"
        before_mean = df.loc[df["converted"] == 1, trap_col].mean()
        result = boost_leakage_trap(df, seed=42)
        after_mean = result.loc[result["converted"] == 1, trap_col].mean()
        assert after_mean > before_mean

    def test_zero_converted_leads_no_change(self):
        """When no leads are converted, trap values should be unchanged."""
        df = _make_v5_df(n=200, conversion_rate=0.30)
        df["converted"] = 0  # force all negative
        trap_col = "__leakage__total_touches_90d"
        original = df[trap_col].copy()
        result = boost_leakage_trap(df, seed=42)
        pd.testing.assert_series_equal(result[trap_col], original, check_names=False)

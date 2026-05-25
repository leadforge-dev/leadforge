"""Tests for leadforge.pipelines.build_v6 pipeline functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadforge.pipelines.build_v6 import (
    ACV_CAP,
    ACV_FLOOR,
    FINAL_COLUMNS_INSTRUCTOR,
    FINAL_COLUMNS_STUDENT,
    INSTRUCTOR_TRAP_COL,
    assign_acquisition_wave,
    boost_leakage_trap,
    derive_features,
    inject_missingness,
    rename_and_select,
    softcap_expected_acv,
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
    """Build a minimal snapshot DataFrame with pre-rename column names."""
    rng = np.random.RandomState(seed)
    n_pos = int(n * conversion_rate)
    n_neg = n - n_pos
    converted = np.array([1] * n_pos + [0] * n_neg)
    rng.shuffle(converted)

    return pd.DataFrame(
        {
            "lead_id": [f"lead_{i:06d}" for i in range(n)],
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
            "touches_days_0_7": rng.poisson(2, size=n),
            "touches_last_7_days": rng.poisson(2, size=n),
            "days_since_first_touch": rng.uniform(0, 14, size=n).round(1),
            "session_count": rng.poisson(4, size=n).astype(float),
            "activity_count": rng.poisson(3, size=n),
            "days_since_last_touch": rng.uniform(0, 14, size=n).round(1),
            "converted_within_90_days": converted,
        }
    )


def _make_v6_df(
    n: int = 500,
    conversion_rate: float = 0.30,
    seed: int = 42,
    instructor: bool = False,
) -> pd.DataFrame:
    """Build a DataFrame in v6 format (post-rename)."""
    rng = np.random.RandomState(seed)
    snapshot = _make_snapshot(n=n, conversion_rate=conversion_rate, seed=seed)
    df = derive_features(snapshot)
    df = softcap_expected_acv(df, seed=seed)
    df = assign_acquisition_wave(df, seed=seed)
    if instructor:
        df[INSTRUCTOR_TRAP_COL] = rng.poisson(10, size=n)
    return rename_and_select(df, instructor=instructor)


# ---------------------------------------------------------------------------
# Tests — derive_features
# ---------------------------------------------------------------------------


class TestDeriveFeatures:
    def test_opportunity_created_is_int(self):
        snapshot = _make_snapshot()
        result = derive_features(snapshot)
        assert result["opportunity_created"].dtype in (np.int64, np.int32, int)
        assert set(result["opportunity_created"].unique()).issubset({0, 1})

    def test_demo_completed_derived_from_page_views(self):
        snapshot = _make_snapshot()
        snapshot["demo_page_views"] = [0, 3, 0, 1, 0] * (len(snapshot) // 5)
        result = derive_features(snapshot)
        expected = (snapshot["demo_page_views"] > 0).astype(int)
        pd.testing.assert_series_equal(result["demo_completed"], expected, check_names=False)

    def test_does_not_modify_input(self):
        snapshot = _make_snapshot()
        original = snapshot.copy()
        derive_features(snapshot)
        pd.testing.assert_frame_equal(snapshot, original)


# ---------------------------------------------------------------------------
# Tests — softcap_expected_acv
# ---------------------------------------------------------------------------


class TestSoftcapExpectedACV:
    def test_floor_enforced(self):
        snapshot = _make_snapshot()
        snapshot["expected_acv"] = 1_000.0
        result = softcap_expected_acv(snapshot, seed=42)
        assert result["expected_acv"].min() >= ACV_FLOOR

    def test_cap_soft(self):
        """Values above cap should be pulled near the cap, not all clipped to it."""
        snapshot = _make_snapshot()
        snapshot["expected_acv"] = 200_000.0
        result = softcap_expected_acv(snapshot, seed=42)
        assert result["expected_acv"].max() <= ACV_CAP
        # Should NOT all be exactly the cap (soft winsorize adds noise)
        unique_vals = result["expected_acv"].nunique()
        assert unique_vals > 1

    def test_within_range_mostly_unchanged(self):
        snapshot = _make_snapshot()
        snapshot["expected_acv"] = 50_000.0
        result = softcap_expected_acv(snapshot, seed=42)
        assert (result["expected_acv"] == 50_000.0).all()

    def test_does_not_modify_input(self):
        snapshot = _make_snapshot()
        original = snapshot.copy()
        softcap_expected_acv(snapshot, seed=42)
        pd.testing.assert_frame_equal(snapshot, original)


# ---------------------------------------------------------------------------
# Tests — assign_acquisition_wave
# ---------------------------------------------------------------------------


class TestAssignAcquisitionWave:
    def test_three_waves(self):
        snapshot = _make_snapshot(n=300)
        result = assign_acquisition_wave(snapshot, seed=42)
        assert set(result["acquisition_wave"].unique()) == {"A", "B", "C"}

    def test_roughly_equal_distribution(self):
        snapshot = _make_snapshot(n=3000)
        result = assign_acquisition_wave(snapshot, seed=42)
        counts = result["acquisition_wave"].value_counts()
        for wave in ["A", "B", "C"]:
            assert 800 < counts[wave] < 1200  # ~1000 expected per wave


# ---------------------------------------------------------------------------
# Tests — rename_and_select
# ---------------------------------------------------------------------------


class TestRenameAndSelect:
    def test_student_columns(self):
        df = _make_v6_df()
        assert list(df.columns) == FINAL_COLUMNS_STUDENT

    def test_instructor_columns(self):
        df = _make_v6_df(instructor=True)
        assert list(df.columns) == FINAL_COLUMNS_INSTRUCTOR

    def test_converted_is_int(self):
        df = _make_v6_df()
        assert df["converted"].dtype in (np.int64, np.int32, int)

    def test_missing_column_raises(self):
        snapshot = _make_snapshot()
        snapshot = derive_features(snapshot)
        snapshot = softcap_expected_acv(snapshot, seed=42)
        snapshot = assign_acquisition_wave(snapshot, seed=42)
        snapshot = snapshot.drop(columns=["industry"])
        with pytest.raises(ValueError, match="Missing required columns"):
            rename_and_select(snapshot)


# ---------------------------------------------------------------------------
# Tests — subsample
# ---------------------------------------------------------------------------


class TestSubsample:
    def test_output_size(self):
        df = _make_v6_df(n=500)
        result = subsample(df, seed=42, n=100, target_rate=0.30)
        assert len(result) == 100

    @pytest.mark.parametrize("target_rate", [0.30, 0.20, 0.40])
    def test_target_rate_approximate(self, target_rate):
        df = _make_v6_df(n=500)
        result = subsample(df, seed=42, n=200, target_rate=target_rate)
        actual_rate = result["converted"].mean()
        assert actual_rate == pytest.approx(target_rate, abs=0.01)

    def test_deterministic_given_seed(self):
        df = _make_v6_df(n=500)
        r1 = subsample(df, seed=42, n=100, target_rate=0.30)
        r2 = subsample(df, seed=42, n=100, target_rate=0.30)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Tests — inject_missingness
# ---------------------------------------------------------------------------


class TestInjectMissingness:
    @pytest.mark.parametrize("seed", [42, 99, 7])
    def test_missingness_rates_bounded(self, seed):
        """Each column's missingness rate should stay under 10% across seeds."""
        df = _make_v6_df(n=2000, seed=seed)
        result = inject_missingness(df, seed=seed)
        for col in [
            "web_sessions",
            "seniority",
            "days_since_last_touch",
            "days_since_first_touch",
            "expected_acv",
        ]:
            rate = result[col].isna().mean()
            assert rate < 0.10, f"{col} missingness rate {rate:.2%} exceeds 10%"

    def test_expected_acv_gets_mcar(self):
        """expected_acv should have MCAR missingness."""
        df = _make_v6_df(n=5000)
        result = inject_missingness(df, seed=42)
        assert result["expected_acv"].isna().sum() > 0

    def test_other_columns_not_affected(self):
        """Columns not in the missingness spec should have no new NaN."""
        df = _make_v6_df(n=500)
        result = inject_missingness(df, seed=42)
        miss_cols = {
            "web_sessions",
            "seniority",
            "days_since_last_touch",
            "days_since_first_touch",
            "expected_acv",
        }
        for col in FINAL_COLUMNS_STUDENT:
            if col not in miss_cols:
                orig_nan = df[col].isna().sum()
                new_nan = result[col].isna().sum()
                assert new_nan == orig_nan, f"{col} gained unexpected NaN"

    def test_does_not_modify_input(self):
        df = _make_v6_df(n=500)
        original = df.copy()
        inject_missingness(df, seed=42)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v6_df(n=500)
        r1 = inject_missingness(df, seed=42)
        r2 = inject_missingness(df, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_web_sessions_missingness_varies_by_source(self):
        """SDR outbound should have higher web_sessions missingness than inbound."""
        df = _make_v6_df(n=3000)
        result = inject_missingness(df, seed=42)
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
        df = _make_v6_df(n=500, instructor=True)
        original_trap = df[INSTRUCTOR_TRAP_COL].copy()
        result = boost_leakage_trap(df, seed=42)
        neg_mask = df["converted"] == 0
        pd.testing.assert_series_equal(
            result.loc[neg_mask, INSTRUCTOR_TRAP_COL],
            original_trap[neg_mask],
            check_names=False,
        )

    def test_converted_leads_get_higher_or_equal(self):
        df = _make_v6_df(n=500, instructor=True)
        original_trap = df[INSTRUCTOR_TRAP_COL].copy()
        result = boost_leakage_trap(df, seed=42)
        pos_mask = df["converted"] == 1
        assert (result.loc[pos_mask, INSTRUCTOR_TRAP_COL] >= original_trap[pos_mask]).all()

    def test_does_not_modify_input(self):
        df = _make_v6_df(n=500, instructor=True)
        original = df.copy()
        boost_leakage_trap(df, seed=42)
        pd.testing.assert_frame_equal(df, original)

    def test_deterministic_given_seed(self):
        df = _make_v6_df(n=500, instructor=True)
        r1 = boost_leakage_trap(df, seed=42)
        r2 = boost_leakage_trap(df, seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_boost_increases_mean_for_converted(self):
        """Mean trap value should be higher for converted leads after boost."""
        df = _make_v6_df(n=1000, instructor=True)
        before_mean = df.loc[df["converted"] == 1, INSTRUCTOR_TRAP_COL].mean()
        result = boost_leakage_trap(df, seed=42)
        after_mean = result.loc[result["converted"] == 1, INSTRUCTOR_TRAP_COL].mean()
        assert after_mean > before_mean

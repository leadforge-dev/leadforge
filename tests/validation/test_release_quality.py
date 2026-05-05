"""Tests for :mod:`leadforge.validation.release_quality`.

The fast tests in this file exercise individual metric primitives and
the dataclass plumbing using hand-built train/test pairs that have
known answers.  The slow round-trip through ``Generator`` lives in
``tests/integration/test_release_quality_round_trip.py`` so the unit
suite stays under a second.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from leadforge.validation.release_quality import (
    _HEADLINE_FIELDS,
    CUMULATIVE_GAINS_PCTS,
    DEFAULT_MODEL_RANDOM_STATE,
    LABEL_COLUMN,
    PRECISION_KS,
    CalibrationBin,
    CohortShiftMetrics,
    CrossSeedTierMetrics,
    CrossTierOrdering,
    ReleaseQualityReport,
    TierMetrics,
    _aggregate_cross_seed,
    _calibration_bins,
    _cumulative_gains_curve,
    _expected_acv_capture,
    _lift_at_pct,
    _precision_at_k,
    _recall_at_k,
    _top_decile_rate,
    measure_cohort_shift_from_bundle,
    measure_release_quality,
    measure_tier_from_bundle,
    report_to_dict,
    report_to_json,
)

# ---------------------------------------------------------------------------
# Metric primitives — hand-built inputs with known answers.
# ---------------------------------------------------------------------------


class TestMetricPrimitives:
    def test_precision_at_k_perfect_ranker(self) -> None:
        # Top-K predictions should perfectly match the K positives.
        probs = np.array([0.9, 0.85, 0.8, 0.1, 0.05])
        y = np.array([1, 1, 1, 0, 0])
        assert _precision_at_k(probs, y, 3) == pytest.approx(1.0)
        assert _recall_at_k(probs, y, 3) == pytest.approx(1.0)

    def test_precision_at_k_random_score_returns_base_rate_in_expectation(self) -> None:
        # A constant score → ties break by stable order; precision@k
        # equals first-k label fraction, not necessarily base rate.
        # This is a structural sanity test — the function must not
        # raise and must return a value in [0, 1].
        probs = np.full(10, 0.5)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        p = _precision_at_k(probs, y, 5)
        assert 0.0 <= p <= 1.0

    def test_recall_at_k_no_positives_returns_nan(self) -> None:
        probs = np.array([0.9, 0.5, 0.1])
        y = np.array([0, 0, 0])
        assert math.isnan(_recall_at_k(probs, y, 2))

    def test_lift_at_pct_perfect_ranker_above_one(self) -> None:
        probs = np.array([0.9, 0.85, 0.8, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        # Top-30% (3 leads) all positive; base rate = 0.3 → lift = 1.0/0.3 ≈ 3.33.
        lift = _lift_at_pct(probs, y, 30.0)
        assert lift == pytest.approx(1.0 / 0.3, rel=1e-3)

    def test_lift_at_pct_zero_base_rate_returns_nan(self) -> None:
        probs = np.array([0.5, 0.5, 0.5])
        y = np.array([0, 0, 0])
        assert math.isnan(_lift_at_pct(probs, y, 50.0))

    def test_top_decile_rate_perfect_ranker(self) -> None:
        probs = np.linspace(1.0, 0.0, 10)  # descending
        y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Top-10% = top-1 = the single positive.
        assert _top_decile_rate(probs, y) == pytest.approx(1.0)

    def test_expected_acv_capture_perfect_ranker_captures_all(self) -> None:
        probs = np.array([0.9, 0.8, 0.5, 0.1, 0.05])
        y = np.array([1, 1, 0, 0, 0])
        acv = np.array([100, 200, 50, 50, 50], dtype=float)
        # Top-2 ranks both positives, capturing 100 + 200 of the
        # converted-ACV total of 300.
        assert _expected_acv_capture(probs, y, acv, 2) == pytest.approx(1.0)

    def test_expected_acv_capture_no_converted_returns_nan(self) -> None:
        probs = np.array([0.9, 0.5, 0.1])
        y = np.array([0, 0, 0])
        acv = np.array([100, 100, 100], dtype=float)
        assert math.isnan(_expected_acv_capture(probs, y, acv, 2))

    def test_calibration_bins_perfectly_calibrated_zero_error(self) -> None:
        # 1000 points where p == y_mean per bin (perfectly calibrated).
        rng = np.random.default_rng(42)
        probs = rng.uniform(0, 1, size=2000)
        y = (rng.uniform(0, 1, size=2000) < probs).astype(int)
        bins, max_err = _calibration_bins(probs, y, n_bins=10)
        assert len(bins) >= 5  # most bins populated
        # With 2000 samples the bin-mean-actual concentrates around the
        # bin midpoint; the max error should be well under 0.10.
        assert max_err < 0.10

    def test_calibration_bins_known_miscalibration(self) -> None:
        # Predict 0.9 always; truth is 0.1 base rate → max error 0.8.
        probs = np.full(500, 0.9)
        y = np.zeros(500, dtype=int)
        y[:50] = 1  # 10% positives
        bins, max_err = _calibration_bins(probs, y, n_bins=10)
        assert max_err == pytest.approx(0.8, abs=1e-6)
        # Only the bin containing 0.9 should be populated.
        populated = [b for b in bins if b.n > 0]
        assert len(populated) == 1

    def test_cumulative_gains_perfect_ranker_captures_at_top_k(self) -> None:
        # 100 leads, 30 positives at top → top-30% captures 100% of positives.
        probs = np.linspace(1.0, 0.0, 100)
        y = np.zeros(100, dtype=int)
        y[:30] = 1
        out = _cumulative_gains_curve(probs, y, (0.0, 10.0, 30.0, 50.0, 100.0))
        assert out["0"] == 0.0
        assert out["100"] == 1.0
        # Top-30% captures all 30 positives.
        assert out["30"] == pytest.approx(1.0)
        # Top-10% captures 10/30.
        assert out["10"] == pytest.approx(10 / 30, abs=1e-6)
        # Curve is monotonic non-decreasing.
        ordered = [out[f"{p:g}"] for p in (0.0, 10.0, 30.0, 50.0, 100.0)]
        assert ordered == sorted(ordered)

    def test_cumulative_gains_no_positives_returns_nan(self) -> None:
        out = _cumulative_gains_curve(
            np.array([0.9, 0.5, 0.1]), np.zeros(3, dtype=int), (0.0, 50.0, 100.0)
        )
        assert all(math.isnan(v) for v in out.values())


# ---------------------------------------------------------------------------
# Dataclass plumbing + JSON serialisation.
# ---------------------------------------------------------------------------


def _fixture_tier_metrics(tier: str, seed: int, **overrides: Any) -> TierMetrics:
    base: dict[str, Any] = {
        "tier": tier,
        "seed": seed,
        "n_train": 100,
        "n_test": 30,
        "base_rate": 0.3,
        "conversion_rate_train": 0.31,
        "conversion_rate_test": 0.30,
        "lr_auc": 0.85,
        "gbm_auc": 0.88,
        "gbm_minus_lr_auc": 0.03,
        "lr_average_precision": 0.62,
        "gbm_average_precision": 0.65,
        "precision_at_k": {"50": 0.66, "100": 0.55},
        "recall_at_k": {"50": 0.45, "100": 0.78},
        "lift_at_pct": {"1": 3.0, "5": 2.5, "10": 2.0},
        "top_decile_rate": 0.6,
        "cumulative_gains": {
            "0": 0.0,
            "10": 0.4,
            "20": 0.6,
            "50": 0.85,
            "100": 1.0,
        },
        "expected_acv_capture_at_k": {"50": 0.55, "100": 0.80},
        "brier_score": 0.12,
        "log_loss": 0.34,
        "calibration_max_bin_error": 0.18,
        "calibration_bins": [
            CalibrationBin(0.0, 0.1, 5, 0.05, 0.02),
            CalibrationBin(0.4, 0.5, 4, 0.45, 0.5),
        ],
        "baselines": {
            "source_only": 0.52,
            "engagement_only": 0.66,
            "post_snapshot_aggregates": 0.55,
            "id_only": 0.50,
        },
    }
    base.update(overrides)
    return TierMetrics(**base)


_DEFAULT_TIERS: tuple[str, ...] = ("intro", "intermediate", "advanced")


def _fixture_report(tiers: tuple[str, ...] = _DEFAULT_TIERS) -> ReleaseQualityReport:
    cross_seed: dict[str, CrossSeedTierMetrics] = {}
    cohort: dict[str, CohortShiftMetrics] = {}
    for tier in tiers:
        # Two seeds per tier — minimum for cross-seed spread.
        m1 = _fixture_tier_metrics(
            tier=tier,
            seed=42,
            lr_auc={"intro": 0.92, "intermediate": 0.88, "advanced": 0.85}.get(tier, 0.85),
            gbm_auc={"intro": 0.94, "intermediate": 0.91, "advanced": 0.88}.get(tier, 0.88),
            gbm_minus_lr_auc=0.03,
            lr_average_precision={"intro": 0.85, "intermediate": 0.65, "advanced": 0.40}.get(
                tier, 0.5
            ),
            conversion_rate_test={"intro": 0.42, "intermediate": 0.20, "advanced": 0.08}.get(
                tier, 0.2
            ),
        )
        m2 = _fixture_tier_metrics(tier=tier, seed=43)
        cross_seed[tier] = CrossSeedTierMetrics(
            tier=tier,
            seeds=[42, 43],
            per_seed=[m1, m2],
            medians={
                "lr_auc": (m1.lr_auc + m2.lr_auc) / 2,
                "gbm_auc": (m1.gbm_auc + m2.gbm_auc) / 2,
                "gbm_minus_lr_auc": 0.03,
                "lr_average_precision": m1.lr_average_precision,
                "gbm_average_precision": m1.gbm_average_precision,
                "brier_score": 0.12,
                "log_loss": 0.34,
                "calibration_max_bin_error": 0.18,
                "top_decile_rate": 0.6,
                "conversion_rate_test": m1.conversion_rate_test,
            },
            spreads=dict.fromkeys(
                ("lr_auc", "gbm_auc", "lr_average_precision", "brier_score"), 0.01
            ),
        )
        cohort[tier] = CohortShiftMetrics(
            tier=tier,
            seed=42,
            random_split_auc=0.85,
            cohort_split_auc=0.78,
            auc_degradation=0.07,
        )
    ordering = CrossTierOrdering(
        by_average_precision=["intro", "intermediate", "advanced"],
        by_precision_at_100=["intro", "intermediate", "advanced"],
        by_gbm_minus_lr=["intro", "intermediate", "advanced"],
        by_conversion_rate=["intro", "intermediate", "advanced"],
        average_precision_intro_gt_intermediate=True,
        average_precision_intermediate_gt_advanced=True,
        precision_at_100_intro_gt_intermediate=True,
        precision_at_100_intermediate_gt_advanced=True,
        conversion_rate_intro_gt_intermediate=True,
        conversion_rate_intermediate_gt_advanced=True,
        gbm_minus_lr_positive_in_every_tier=True,
    )
    return ReleaseQualityReport(
        release_id="leadforge-lead-scoring-v1",
        package_version="1.0.0",
        generation_timestamp="2026-05-06T12:00:00+00:00",
        seeds=[42, 43],
        tiers=cross_seed,
        cohort_shift=cohort,
        cross_tier_ordering=ordering,
    )


class TestJsonSerialisation:
    def test_report_to_json_round_trips(self) -> None:
        report = _fixture_report()
        s = report_to_json(report)
        d = json.loads(s)
        assert d["release_id"] == "leadforge-lead-scoring-v1"
        assert sorted(d["tiers"].keys()) == ["advanced", "intermediate", "intro"]
        assert d["tiers"]["intro"]["per_seed"][0]["lr_auc"] == pytest.approx(0.92)

    def test_report_to_dict_coerces_nan_to_none(self) -> None:
        # Build a TierMetrics with NaN in a nested dict and verify it
        # comes out as None — json.dumps would otherwise fail with
        # ``ValueError: Out of range float values are not JSON compliant``
        # under some encoder configurations.
        m = _fixture_tier_metrics(tier="intro", seed=42, calibration_max_bin_error=float("nan"))
        # Patch a NaN inside a nested float dict.
        m = TierMetrics(**{**m.__dict__, "lift_at_pct": {"1": float("nan"), "5": 2.0, "10": 1.5}})
        report = ReleaseQualityReport(
            release_id="x",
            package_version="1.0.0",
            generation_timestamp="2026-05-06T12:00:00+00:00",
            seeds=[42],
            tiers={
                "intro": CrossSeedTierMetrics(
                    tier="intro", seeds=[42], per_seed=[m], medians={}, spreads={}
                )
            },
            cohort_shift={},
            cross_tier_ordering=_fixture_report(("intro",)).cross_tier_ordering,
        )
        d = report_to_dict(report)
        # NaN survived as None.
        assert d["tiers"]["intro"]["per_seed"][0]["calibration_max_bin_error"] is None
        assert d["tiers"]["intro"]["per_seed"][0]["lift_at_pct"]["1"] is None
        # And the result is JSON-serialisable end-to-end.
        json.dumps(d)


# ---------------------------------------------------------------------------
# Cross-tier ordering computation
# ---------------------------------------------------------------------------


class TestCrossTierOrdering:
    def test_ordering_when_intro_is_easiest(self) -> None:
        report = _fixture_report()
        o = report.cross_tier_ordering
        assert o.by_average_precision[0] == "intro"
        assert o.by_conversion_rate[0] == "intro"
        assert o.average_precision_intro_gt_intermediate
        assert o.gbm_minus_lr_positive_in_every_tier

    def test_ordering_with_partial_release(self) -> None:
        # Only intro present — pairs that include a missing tier become
        # ``None``; the gbm-vs-lr "every tier" still resolves on the
        # finite intro median (positive in the fixture).  The previous
        # design defaulted these to ``True``, silently green-lighting
        # partial releases.
        from leadforge.validation.release_quality import _compute_cross_tier_ordering

        partial = _fixture_report(("intro",))
        o = _compute_cross_tier_ordering(partial.tiers)
        assert o.by_average_precision == ["intro"]
        assert o.average_precision_intro_gt_intermediate is None
        assert o.average_precision_intermediate_gt_advanced is None
        assert o.gbm_minus_lr_positive_in_every_tier is True

    def test_ordering_returns_none_on_empty_release(self) -> None:
        from leadforge.validation.release_quality import _compute_cross_tier_ordering

        o = _compute_cross_tier_ordering({})
        assert o.by_average_precision == []
        assert o.average_precision_intro_gt_intermediate is None
        assert o.gbm_minus_lr_positive_in_every_tier is None


class TestHeadlineFieldsRegistry:
    """Drift-guard for ``_HEADLINE_FIELDS``.

    Mirrors the meta-test pattern in ``test_leakage_probes.py`` —
    catches the failure mode where a new metric is added to
    :class:`TierMetrics` but the cross-seed aggregator forgets to
    include it.
    """

    def test_every_headline_field_is_a_scalar_float_on_tier_metrics(self) -> None:
        import typing

        # ``from __future__ import annotations`` stores annotations as
        # strings; ``get_type_hints`` resolves them back to real types.
        hints = typing.get_type_hints(TierMetrics)
        scalar_floats = {name for name, t in hints.items() if t is float}
        unknown = set(_HEADLINE_FIELDS) - scalar_floats
        assert not unknown, (
            f"_HEADLINE_FIELDS contains entries that are not scalar floats on "
            f"TierMetrics: {sorted(unknown)}"
        )

    def test_aggregator_emits_a_median_and_spread_per_field(self) -> None:
        per_seed = [_fixture_tier_metrics("intermediate", seed=42)]
        medians, spreads = _aggregate_cross_seed(per_seed)
        assert set(_HEADLINE_FIELDS) == set(medians.keys())
        assert set(_HEADLINE_FIELDS) == set(spreads.keys())


# ---------------------------------------------------------------------------
# Bundle-level measurement against a synthetic mini-bundle.
# ---------------------------------------------------------------------------


def _write_minimal_bundle(
    root: Path,
    *,
    n: int = 200,
    seed: int = 42,
    tier_signal: float = 1.0,
    include_cohort: bool = True,
) -> Path:
    """Hand-build a bundle directory with the minimal contract:
    manifest.json + tasks/<task>/{train,test,valid}.parquet.

    Tables/ is not required by ``measure_tier_from_bundle`` (only the
    task splits and manifest are read).
    """
    rng = np.random.default_rng(seed)
    n_train = int(n * 0.7)
    n_test = int(n * 0.15)
    n_valid = n - n_train - n_test

    def _make(n_rows: int, base_day: int) -> pd.DataFrame:
        # Generate a feature with explicit signal toward the label.
        latent = rng.normal(size=n_rows)
        engagement = latent * tier_signal + rng.normal(scale=0.5, size=n_rows)
        prob = 1.0 / (1.0 + np.exp(-engagement))
        y = pd.Series(rng.uniform(size=n_rows) < prob, dtype="boolean")
        if include_cohort:
            ts = pd.date_range("2026-01-01", periods=n_rows, freq="h") + pd.Timedelta(days=base_day)
        else:
            ts = pd.NaT
        df = pd.DataFrame(
            {
                "lead_id": [f"lead_{base_day:03d}_{i:05d}" for i in range(n_rows)],
                "account_id": [f"acct_{base_day:03d}_{i:05d}" for i in range(n_rows)],
                "contact_id": [f"cont_{base_day:03d}_{i:05d}" for i in range(n_rows)],
                "lead_created_at": pd.Series(ts).astype(str)
                if include_cohort
                else pd.Series([pd.NA] * n_rows, dtype="string"),
                "lead_source": rng.choice(
                    ["inbound_marketing", "sdr_outbound", "partner_referral"], size=n_rows
                ),
                "first_touch_channel": rng.choice(["seo", "ppc", "email"], size=n_rows),
                "industry": rng.choice(["fintech", "manufacturing", "healthcare"], size=n_rows),
                "region": rng.choice(["us", "uk", "eu"], size=n_rows),
                "employee_band": rng.choice(["50-200", "200-500"], size=n_rows),
                "estimated_revenue_band": rng.choice(["10m-50m", "50m-100m"], size=n_rows),
                "process_maturity_band": rng.choice(["low", "med", "high"], size=n_rows),
                "role_function": rng.choice(["finance", "ops"], size=n_rows),
                "seniority": rng.choice(["manager", "director"], size=n_rows),
                "buyer_role": rng.choice(["champion", "economic_buyer"], size=n_rows),
                "touch_count": pd.Series(
                    np.maximum(0, np.round(engagement * 3 + 5)).astype("int64"), dtype="Int64"
                ),
                "session_count": pd.Series(rng.integers(0, 10, size=n_rows), dtype="Int64"),
                "expected_acv": pd.Series(
                    rng.uniform(20_000, 100_000, size=n_rows), dtype="Float64"
                ),
                "total_touches_all": pd.Series(
                    np.maximum(
                        0, np.round(engagement * 3 + 5 + rng.normal(0, 1, size=n_rows))
                    ).astype("int64"),
                    dtype="Int64",
                ),
                LABEL_COLUMN: y,
            }
        )
        return df

    train = _make(n_train, base_day=0)
    valid = _make(n_valid, base_day=20)
    test = _make(n_test, base_day=30)

    task_dir = root / "tasks" / "converted_within_90_days"
    task_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(task_dir / "train.parquet", index=False)
    valid.to_parquet(task_dir / "valid.parquet", index=False)
    test.to_parquet(task_dir / "test.parquet", index=False)

    manifest = {
        "bundle_schema_version": "5",
        "package_version": "1.0.0",
        "recipe_id": "b2b_saas_procurement_v1",
        "seed": seed,
        "exposure_mode": "student_public",
        "difficulty": "intermediate",
        "n_accounts": n,
        "n_contacts": n * 3,
        "n_leads": n,
        "snapshot_day": 30,
        "primary_task": "converted_within_90_days",
        "label_window_days": 90,
        "tasks": {
            "converted_within_90_days": {
                "train_rows": n_train,
                "valid_rows": n_valid,
                "test_rows": n_test,
            }
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


class TestBundleMeasurement:
    def test_measure_tier_from_synthetic_bundle(self, tmp_path: Path) -> None:
        pytest.importorskip("sklearn")
        bundle = _write_minimal_bundle(tmp_path / "intermediate", n=400, seed=42)
        m = measure_tier_from_bundle(bundle, seed=42, tier_name="intermediate")
        assert m.tier == "intermediate"
        assert m.n_train > 0
        assert m.n_test > 0
        # Synthetic signal: LR AUC should clear chance comfortably.
        assert m.lr_auc > 0.6
        # Headline serialisation contract: every PRECISION_KS key is
        # present as a string-keyed entry.
        for k in PRECISION_KS:
            assert str(k) in m.precision_at_k
            assert str(k) in m.recall_at_k
        # Calibration bins integrate to the test size.
        assert sum(b.n for b in m.calibration_bins) == m.n_test
        # ID-only baseline ≈ chance — the synthetic IDs are
        # uncorrelated with the latent.  Allow generous slack: with
        # only 60 test rows (15% of 400) the AUC variance is wide.
        assert "id_only" in m.baselines
        assert m.baselines["id_only"] > 0.3
        assert m.baselines["id_only"] < 0.7
        # Cumulative gains carries one entry per CUMULATIVE_GAINS_PCTS.
        assert set(m.cumulative_gains.keys()) == {f"{p:g}" for p in CUMULATIVE_GAINS_PCTS}
        assert m.cumulative_gains["0"] == 0.0
        assert m.cumulative_gains["100"] == pytest.approx(1.0)
        # Curve is monotonic non-decreasing.
        sorted_pcts = sorted(float(k) for k in m.cumulative_gains)
        ys = [m.cumulative_gains[f"{p:g}"] for p in sorted_pcts]
        for prev, cur in zip(ys[:-1], ys[1:], strict=True):
            assert cur >= prev - 1e-9, f"cumulative gains decreased: {prev} -> {cur}"

    def test_measure_tier_raises_when_train_single_class(self, tmp_path: Path) -> None:
        pytest.importorskip("sklearn")
        bundle = tmp_path / "degenerate"
        # Build a bundle and overwrite the train split with all-zeros.
        _write_minimal_bundle(bundle, n=200, seed=42)
        train_path = bundle / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df[LABEL_COLUMN] = pd.Series([False] * len(df), dtype="boolean")
        df.to_parquet(train_path, index=False)
        with pytest.raises(ValueError, match="train split has fewer than two classes"):
            measure_tier_from_bundle(bundle, seed=42)

    def test_measure_cohort_shift_returns_random_auc_when_no_timestamp(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("sklearn")
        # Build a bundle without lead_created_at usable as datetime.
        bundle = _write_minimal_bundle(tmp_path / "no_cohort", n=200, seed=42)
        train_path = bundle / "tasks/converted_within_90_days/train.parquet"
        test_path = bundle / "tasks/converted_within_90_days/test.parquet"
        for p in (train_path, test_path):
            df = pd.read_parquet(p)
            df["lead_created_at"] = pd.Series(["not-a-date"] * len(df), dtype="string")
            df.to_parquet(p, index=False)
        cs = measure_cohort_shift_from_bundle(bundle, seed=42)
        assert not math.isnan(cs.random_split_auc)
        assert math.isnan(cs.cohort_split_auc)
        assert math.isnan(cs.auc_degradation)

    def test_model_random_state_decoupled_from_generation_seed(self, tmp_path: Path) -> None:
        """``model_random_state`` controls the sklearn ``random_state``;
        the ``seed`` argument is just the bundle's generation seed
        recorded for traceability.

        Two ``measure_tier_from_bundle`` calls on the *same* bundle with
        the same ``model_random_state`` but different ``seed`` arguments
        must produce identical AUCs (data is identical → model is
        identical → AUCs are identical).  Earlier versions of this
        function used ``seed`` for both, so the cross-seed sweep
        confounded data variance with model-RNG variance — that's the
        bug this test guards against.
        """
        pytest.importorskip("sklearn")
        bundle = _write_minimal_bundle(tmp_path / "fixed_data", n=400, seed=42)
        a = measure_tier_from_bundle(bundle, seed=1, model_random_state=DEFAULT_MODEL_RANDOM_STATE)
        b = measure_tier_from_bundle(bundle, seed=2, model_random_state=DEFAULT_MODEL_RANDOM_STATE)
        assert a.lr_auc == b.lr_auc
        assert a.gbm_auc == b.gbm_auc
        # The traceability seed differs.
        assert a.seed == 1
        assert b.seed == 2

    def test_cohort_shift_returns_well_formed_auc_pair(self, tmp_path: Path) -> None:
        """Cohort-shift evaluation returns finite AUCs in [0, 1] and a
        signed degradation when ``lead_created_at`` is parseable.

        We don't assert ``cohort_split_auc < random_split_auc`` on
        synthetic data — random vs chronological splits over a flat
        latent are both ~chance, so ordering is dominated by sample
        noise.  The behavioural ordering test lives in the round-trip
        integration suite where the engine produces a real time-shift.
        """
        pytest.importorskip("sklearn")
        bundle = _write_minimal_bundle(tmp_path / "cohort", n=500, seed=11)
        cs = measure_cohort_shift_from_bundle(bundle, seed=11)
        for auc in (cs.random_split_auc, cs.cohort_split_auc):
            assert 0.0 <= auc <= 1.0
            assert not math.isnan(auc)
        assert cs.auc_degradation == pytest.approx(
            cs.random_split_auc - cs.cohort_split_auc, abs=1e-9
        )


# ---------------------------------------------------------------------------
# Cross-seed orchestration on synthetic bundles.
# ---------------------------------------------------------------------------


class TestMeasureReleaseQuality:
    def test_orchestrator_aggregates_two_seeds(self, tmp_path: Path) -> None:
        pytest.importorskip("sklearn")
        b42 = _write_minimal_bundle(tmp_path / "intermediate__seed42", n=400, seed=42)
        b43 = _write_minimal_bundle(tmp_path / "intermediate__seed43", n=400, seed=43)
        report = measure_release_quality(
            {"intermediate": {42: b42, 43: b43}},
            generation_timestamp="2026-05-06T12:00:00+00:00",
        )
        assert sorted(report.tiers.keys()) == ["intermediate"]
        csm = report.tiers["intermediate"]
        assert csm.seeds == [42, 43]
        assert len(csm.per_seed) == 2
        assert "lr_auc" in csm.medians
        assert "lr_auc" in csm.spreads
        # Cohort shift was run for the canonical seed (smallest).
        assert "intermediate" in report.cohort_shift
        assert report.cohort_shift["intermediate"].seed == 42
        # JSON round trip works.
        d = report_to_dict(report)
        json.dumps(d)

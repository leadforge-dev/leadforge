"""Tests for the YAML-driven acceptance-band gate checker.

Covers the PR 3.3 extension to ``leadforge.validation.difficulty``:
:func:`load_bands`, :func:`check_release_bands`, :class:`GateFailure`,
and the parsing helpers.  The release-quality dataclasses are
constructed synthetically here; the round-trip integration test
covers the real measurement → band-check pipeline against a generated
bundle.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from leadforge.validation.difficulty import (
    AcceptanceBands,
    BandSpec,
    GateFailure,
    LeakageProbeBands,
    TierBands,
    _gate_id_for,
    _resolve_metric_value,
    check_release_bands,
    load_bands,
)
from leadforge.validation.leakage_probes import LeakageFinding, LeakageReport
from leadforge.validation.release_quality import (
    CalibrationBin,
    CohortShiftMetrics,
    CrossSeedTierMetrics,
    CrossTierOrdering,
    ReleaseQualityReport,
    TierMetrics,
)


def _make_tier_metrics(
    *,
    tier: str,
    seed: int,
    lr_auc: float = 0.85,
    gbm_auc: float = 0.88,
    lr_ap: float = 0.65,
    gbm_ap: float = 0.70,
    p_at_100: float = 0.75,
    brier: float = 0.18,
    cal_err: float = 0.04,
    rate: float = 0.20,
) -> TierMetrics:
    return TierMetrics(
        tier=tier,
        seed=seed,
        n_train=700,
        n_test=150,
        base_rate=rate,
        conversion_rate_train=rate,
        conversion_rate_test=rate,
        lr_auc=lr_auc,
        gbm_auc=gbm_auc,
        gbm_minus_lr_auc=gbm_auc - lr_auc,
        lr_average_precision=lr_ap,
        gbm_average_precision=gbm_ap,
        precision_at_k={"50": p_at_100, "100": p_at_100},
        recall_at_k={"50": 0.4, "100": 0.6},
        lift_at_pct={"1": 4.0, "5": 3.0, "10": 2.0},
        top_decile_rate=0.6,
        cumulative_gains={"0": 0.0, "10": 0.5, "100": 1.0},
        expected_acv_capture_at_k={"50": 0.4, "100": 0.6},
        brier_score=brier,
        log_loss=0.5,
        calibration_max_bin_error=cal_err,
        calibration_bins=[
            CalibrationBin(
                bin_lower=0.0, bin_upper=0.5, n=100, mean_predicted=0.2, mean_actual=0.18
            )
        ],
        baselines={"id_only": 0.5, "post_snapshot_aggregates": 0.7},
    )


def _make_cross_seed(tier: str, seeds: list[int], **kwargs: float) -> CrossSeedTierMetrics:
    per_seed = [_make_tier_metrics(tier=tier, seed=s, **kwargs) for s in seeds]
    # Trivial median + spread aggregator that mirrors the production one.
    medians = {
        "lr_auc": per_seed[0].lr_auc,
        "gbm_auc": per_seed[0].gbm_auc,
        "gbm_minus_lr_auc": per_seed[0].gbm_minus_lr_auc,
        "lr_average_precision": per_seed[0].lr_average_precision,
        "gbm_average_precision": per_seed[0].gbm_average_precision,
        "brier_score": per_seed[0].brier_score,
        "log_loss": per_seed[0].log_loss,
        "calibration_max_bin_error": per_seed[0].calibration_max_bin_error,
        "top_decile_rate": per_seed[0].top_decile_rate,
        "conversion_rate_test": per_seed[0].conversion_rate_test,
    }
    spreads = dict.fromkeys(medians, 0.0)
    return CrossSeedTierMetrics(
        tier=tier,
        seeds=seeds,
        per_seed=per_seed,
        medians=medians,
        spreads=spreads,
    )


def _make_report(
    *,
    intro: CrossSeedTierMetrics | None = None,
    intermediate: CrossSeedTierMetrics | None = None,
    advanced: CrossSeedTierMetrics | None = None,
    cohort_intro_deg: float = 0.05,
    cohort_inter_deg: float = 0.07,
    cohort_adv_deg: float = 0.09,
) -> ReleaseQualityReport:
    tiers: dict[str, CrossSeedTierMetrics] = {}
    if intro is not None:
        tiers["intro"] = intro
    if intermediate is not None:
        tiers["intermediate"] = intermediate
    if advanced is not None:
        tiers["advanced"] = advanced

    cohort: dict[str, CohortShiftMetrics] = {}
    for name, deg in (
        ("intro", cohort_intro_deg),
        ("intermediate", cohort_inter_deg),
        ("advanced", cohort_adv_deg),
    ):
        if name in tiers:
            cohort[name] = CohortShiftMetrics(
                tier=name,
                seed=42,
                random_split_auc=0.85,
                cohort_split_auc=0.85 - deg,
                auc_degradation=deg,
            )

    # Compute ordering booleans the way the production helper would, so
    # the test stays representative across changes.
    ap = {n: t.medians["lr_average_precision"] for n, t in tiers.items()}
    p100 = {
        n: float(t.per_seed[0].precision_at_k.get("100", float("nan"))) for n, t in tiers.items()
    }
    rate = {n: t.medians["conversion_rate_test"] for n, t in tiers.items()}

    def _gt(d: dict[str, float], a: str, b: str) -> bool | None:
        if a not in d or b not in d:
            return None
        if math.isnan(d[a]) or math.isnan(d[b]):
            return None
        return d[a] > d[b]

    finite_gbm_lr = [t.medians["gbm_minus_lr_auc"] for t in tiers.values()]
    gbm_lr_pos: bool | None = all(v > 0 for v in finite_gbm_lr) if finite_gbm_lr else None

    ordering = CrossTierOrdering(
        by_average_precision=sorted(tiers, key=lambda k: -ap[k]),
        by_precision_at_100=sorted(tiers, key=lambda k: -p100[k]),
        by_gbm_minus_lr=sorted(tiers, key=lambda k: -tiers[k].medians["gbm_minus_lr_auc"]),
        by_conversion_rate=sorted(tiers, key=lambda k: -rate[k]),
        average_precision_intro_gt_intermediate=_gt(ap, "intro", "intermediate"),
        average_precision_intermediate_gt_advanced=_gt(ap, "intermediate", "advanced"),
        precision_at_100_intro_gt_intermediate=_gt(p100, "intro", "intermediate"),
        precision_at_100_intermediate_gt_advanced=_gt(p100, "intermediate", "advanced"),
        conversion_rate_intro_gt_intermediate=_gt(rate, "intro", "intermediate"),
        conversion_rate_intermediate_gt_advanced=_gt(rate, "intermediate", "advanced"),
        gbm_minus_lr_positive_in_every_tier=gbm_lr_pos,
    )
    return ReleaseQualityReport(
        release_id="leadforge-lead-scoring-v1",
        package_version="1.0.0",
        generation_timestamp="2026-05-06T12:00:00+00:00",
        seeds=sorted({s for t in tiers.values() for s in t.seeds}),
        tiers=tiers,
        cohort_shift=cohort,
        cross_tier_ordering=ordering,
    )


_PASSING_BANDS_YAML = """
per_tier:
  intro:
    conversion_rate_test: {min: 0.30, max: 0.50}
    lr_auc: {min: 0.80, max: 0.97}
    gbm_minus_lr_auc: {min: 0.0}
    lr_average_precision: {min: 0.50, max: 0.97}
    precision_at_100: {min: 0.50, max: 1.0}
    brier_score: {max: 0.25}
    calibration_max_bin_error: {max: 0.30}
  intermediate:
    conversion_rate_test: {min: 0.13, max: 0.33}
    lr_auc: {min: 0.78, max: 0.97}
    gbm_minus_lr_auc: {min: -0.005}
    lr_average_precision: {min: 0.30, max: 0.85}
    precision_at_100: {min: 0.30, max: 0.95}
    brier_score: {max: 0.25}
    calibration_max_bin_error: {max: 0.30}
  advanced:
    conversion_rate_test: {min: 0.04, max: 0.20}
    lr_auc: {min: 0.70, max: 0.95}
    gbm_minus_lr_auc: {min: -0.02}
    lr_average_precision: {min: 0.10, max: 0.70}
    precision_at_100: {min: 0.10, max: 0.90}
    brier_score: {max: 0.25}
    calibration_max_bin_error: {max: 0.30}
cross_seed_spread:
  lr_auc: {max: 0.06}
  lr_average_precision: {max: 0.12}
cohort_shift:
  auc_degradation: {min: 0.0, max: 0.30}
cross_tier_required: [intro, intermediate, advanced]
leakage_probes:
  id_only_max_auc: 0.60
  label_drift_max: 0.10
  feature_subsets:
    post_snapshot_aggregates:
      max_auc: 0.95
      columns: [total_touches_all]
"""


@pytest.fixture
def passing_bands(tmp_path: Path) -> AcceptanceBands:
    p = tmp_path / "bands.yaml"
    p.write_text(_PASSING_BANDS_YAML)
    return load_bands(p)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestLoadBands:
    def test_round_trips_full_yaml(self, passing_bands: AcceptanceBands) -> None:
        assert set(passing_bands.per_tier) == {"intro", "intermediate", "advanced"}
        intro = passing_bands.per_tier["intro"]
        assert intro.bands["lr_auc"].min == pytest.approx(0.80)
        assert intro.bands["lr_auc"].max == pytest.approx(0.97)
        assert intro.bands["lr_auc"].gate == "G7.1.2"
        # Cross-seed spread is gate G8.1 by design.
        assert passing_bands.cross_seed_spread["lr_auc"].gate == "G8.1"
        # Cohort shift gate is G6.4.
        assert passing_bands.cohort_shift is not None
        assert passing_bands.cohort_shift.gate == "G6.4"
        # Required tiers preserved.
        assert passing_bands.cross_tier_required == ("intro", "intermediate", "advanced")
        # Leakage probe bands round-trip.
        lp = passing_bands.leakage_probes
        assert lp.id_only_max_auc == pytest.approx(0.60)
        assert lp.label_drift_max == pytest.approx(0.10)
        assert lp.feature_subsets["post_snapshot_aggregates"] == (
            pytest.approx(0.95),
            ("total_touches_all",),
        )

    def test_missing_optional_sections_default_to_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bands.yaml"
        p.write_text("per_tier:\n  intro:\n    lr_auc: {min: 0.8}\n")
        bands = load_bands(p)
        assert bands.cross_seed_spread == {}
        assert bands.cohort_shift is None
        assert bands.cross_tier_required == ()
        assert bands.leakage_probes.id_only_max_auc is None
        assert bands.leakage_probes.feature_subsets == {}

    def test_rejects_bare_scalar_band(self, tmp_path: Path) -> None:
        p = tmp_path / "bands.yaml"
        p.write_text("per_tier:\n  intro:\n    lr_auc: 0.8\n")
        with pytest.raises(ValueError, match="lr_auc"):
            load_bands(p)

    def test_rejects_missing_min_and_max(self, tmp_path: Path) -> None:
        p = tmp_path / "bands.yaml"
        p.write_text("per_tier:\n  intro:\n    lr_auc: {}\n")
        with pytest.raises(ValueError, match="min.*max"):
            load_bands(p)

    def test_rejects_bad_feature_subset_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "bands.yaml"
        p.write_text("leakage_probes:\n  feature_subsets:\n    bogus: {max_auc: 0.9}\n")
        with pytest.raises(ValueError, match="columns"):
            load_bands(p)


class TestGateIdResolution:
    @pytest.mark.parametrize(
        ("tier", "metric", "expected"),
        [
            ("intro", "lr_auc", "G7.1.2"),
            ("intermediate", "gbm_minus_lr_auc", "G7.2.4"),
            ("advanced", "calibration_max_bin_error", "G7.3.8"),
            ("intro", "precision_at_100", "G7.1.6"),
            ("intro", "conversion_rate_test", "G7.1.1"),
            ("unknown", "lr_auc", "G7.unknown.lr_auc"),
        ],
    )
    def test_resolves_gate_id(self, tier: str, metric: str, expected: str) -> None:
        assert _gate_id_for(tier, metric) == expected


class TestResolveMetricValue:
    def test_headline_metric_from_medians(self) -> None:
        csm = _make_cross_seed("intro", [42], lr_auc=0.91)
        assert _resolve_metric_value(csm, "lr_auc") == pytest.approx(0.91)

    def test_precision_at_k_from_per_seed(self) -> None:
        csm = _make_cross_seed("intro", [42, 43, 44], p_at_100=0.75)
        assert _resolve_metric_value(csm, "precision_at_100") == pytest.approx(0.75)

    def test_unknown_metric_returns_nan(self) -> None:
        csm = _make_cross_seed("intro", [42])
        assert math.isnan(_resolve_metric_value(csm, "nonexistent_metric"))


# ---------------------------------------------------------------------------
# Per-tier band check
# ---------------------------------------------------------------------------


class TestPerTierBands:
    def test_passing_report_yields_no_failures(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed(
                "intro",
                [42, 43, 44, 45, 46],
                lr_auc=0.92,
                gbm_auc=0.94,
                lr_ap=0.78,
                p_at_100=0.85,
                brier=0.15,
                cal_err=0.05,
                rate=0.42,
            ),
            intermediate=_make_cross_seed(
                "intermediate",
                [42, 43, 44, 45, 46],
                lr_auc=0.86,
                gbm_auc=0.88,
                lr_ap=0.55,
                p_at_100=0.65,
                brier=0.16,
                cal_err=0.05,
                rate=0.20,
            ),
            advanced=_make_cross_seed(
                "advanced",
                [42, 43, 44, 45, 46],
                lr_auc=0.78,
                gbm_auc=0.82,
                lr_ap=0.30,
                p_at_100=0.40,
                brier=0.10,
                cal_err=0.06,
                rate=0.08,
            ),
        )
        failures = check_release_bands(report, passing_bands)
        assert failures == [], failures

    def test_below_min_lr_auc_fails(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed(
                "intro", [42], lr_auc=0.50, lr_ap=0.78, p_at_100=0.85, rate=0.42
            ),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_auc=0.86, lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed(
                "advanced", [42], lr_auc=0.78, lr_ap=0.30, p_at_100=0.40, rate=0.08
            ),
        )
        failures = check_release_bands(report, passing_bands)
        gates = {f.gate for f in failures}
        assert "G7.1.2" in gates
        # No other per-tier failure when only intro lr_auc fails.
        intro_lr_failure = next(f for f in failures if f.gate == "G7.1.2" and f.tier == "intro")
        assert "lr_auc" in intro_lr_failure.message
        assert "0.5000" in intro_lr_failure.message

    def test_above_max_brier_fails(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed(
                "intro",
                [42],
                lr_auc=0.92,
                lr_ap=0.78,
                p_at_100=0.85,
                brier=0.40,
                rate=0.42,
            ),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_auc=0.86, lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed(
                "advanced", [42], lr_auc=0.78, lr_ap=0.30, p_at_100=0.40, rate=0.08
            ),
        )
        failures = check_release_bands(report, passing_bands)
        intro_brier = [f for f in failures if f.gate == "G7.1.7" and f.tier == "intro"]
        assert intro_brier, failures
        assert "above max" in intro_brier[0].message

    def test_missing_tier_in_report_fails(self, passing_bands: AcceptanceBands) -> None:
        # Report only carries `intermediate`; bands declare all three.
        report = _make_report(
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_auc=0.86, lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
        )
        failures = check_release_bands(report, passing_bands)
        gates = {(f.gate, f.tier) for f in failures}
        # Missing intro and advanced surface as their gate id with the
        # "absent from report" message.
        assert any(t == "intro" for _, t in gates)
        assert any(t == "advanced" for _, t in gates)
        # Regression guard: the missing-tier gate id must not double-prefix
        # ``G7.``.  Earlier code computed ``f"G7.{_GATE_PREFIX_BY_TIER.get(t)}"``
        # which yielded ``G7.G7.1`` because the prefix dict already carries
        # the leading ``G7.``.
        assert not any(g.startswith("G7.G7") for g, _ in gates)
        # The missing-tier gate id is exactly the tier's G7.* prefix.
        assert ("G7.1", "intro") in gates
        assert ("G7.3", "advanced") in gates


# ---------------------------------------------------------------------------
# Cross-seed spread
# ---------------------------------------------------------------------------


class TestCrossSeedSpread:
    def test_spread_within_tolerance_passes(self, passing_bands: AcceptanceBands) -> None:
        # All-zero spread (single seed) trivially passes.
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
        )
        failures = [f for f in check_release_bands(report, passing_bands) if f.gate == "G8.1"]
        assert failures == []

    def test_spread_exceeds_tolerance_fails(self, passing_bands: AcceptanceBands) -> None:
        csm_intro = _make_cross_seed(
            "intro", [42], lr_auc=0.92, lr_ap=0.78, p_at_100=0.85, rate=0.42
        )
        # Force a large spread on lr_auc; bands say max 0.06.
        bumped = CrossSeedTierMetrics(
            tier=csm_intro.tier,
            seeds=csm_intro.seeds,
            per_seed=csm_intro.per_seed,
            medians=csm_intro.medians,
            spreads={**csm_intro.spreads, "lr_auc": 0.20},
        )
        report = _make_report(
            intro=bumped,
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
        )
        failures = [f for f in check_release_bands(report, passing_bands) if f.gate == "G8.1"]
        assert any("cross-seed spread" in f.message for f in failures)


# ---------------------------------------------------------------------------
# Cohort shift
# ---------------------------------------------------------------------------


class TestCohortShift:
    def test_passing_degradation(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
            cohort_intro_deg=0.10,
            cohort_inter_deg=0.10,
            cohort_adv_deg=0.10,
        )
        failures = [f for f in check_release_bands(report, passing_bands) if f.gate == "G6.4"]
        assert failures == []

    def test_negative_degradation_fails(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
            cohort_intro_deg=-0.10,
            cohort_inter_deg=-0.10,
            cohort_adv_deg=-0.10,
        )
        failures = [f for f in check_release_bands(report, passing_bands) if f.gate == "G6.4"]
        assert len(failures) == 3
        assert all("below min" in f.message for f in failures)

    def test_nan_degradation_surfaces_explicit_failure(
        self, passing_bands: AcceptanceBands
    ) -> None:
        report = _make_report(
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
        )
        # Manually replace the cohort metric with a NaN one.
        intermediate_cohort = CohortShiftMetrics(
            tier="intermediate",
            seed=42,
            random_split_auc=0.85,
            cohort_split_auc=float("nan"),
            auc_degradation=float("nan"),
        )
        report = ReleaseQualityReport(
            release_id=report.release_id,
            package_version=report.package_version,
            generation_timestamp=report.generation_timestamp,
            seeds=report.seeds,
            tiers=report.tiers,
            cohort_shift={"intermediate": intermediate_cohort},
            cross_tier_ordering=report.cross_tier_ordering,
        )
        failures = [f for f in check_release_bands(report, passing_bands) if f.gate == "G6.4"]
        assert any("NaN" in f.message for f in failures)


# ---------------------------------------------------------------------------
# Cross-tier ordering
# ---------------------------------------------------------------------------


class TestCrossTierOrdering:
    def test_correct_ordering_passes(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
        )
        failures = [
            f for f in check_release_bands(report, passing_bands) if f.gate.startswith("G7.4")
        ]
        assert failures == []

    def test_inverted_ordering_fails(self, passing_bands: AcceptanceBands) -> None:
        # Advanced has higher AP than intro — the difficulty contract is broken.
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.20, p_at_100=0.40, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.80, p_at_100=0.85, rate=0.08),
        )
        failures = [
            f for f in check_release_bands(report, passing_bands) if f.gate.startswith("G7.4")
        ]
        gates = {f.gate for f in failures}
        assert "G7.4.1" in gates  # AP ordering broken.
        assert "G7.4.2" in gates  # P@100 ordering broken.

    def test_partial_release_with_required_tiers_fails(
        self, passing_bands: AcceptanceBands
    ) -> None:
        # cross_tier_required = [intro, intermediate, advanced] but only
        # `intermediate` is present.  None ordering bools become failures.
        report = _make_report(
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
        )
        ordering_failures = [
            f for f in check_release_bands(report, passing_bands) if f.gate.startswith("G7.4")
        ]
        # The intro/intermediate and intermediate/advanced pairs both
        # surface as required-but-undefined.
        assert any("ordering" in f.message and "undefined" in f.message for f in ordering_failures)

    def test_partial_release_without_required_tiers_skips(self) -> None:
        # cross_tier_required is empty — None ordering bools are silently
        # skipped (not failures).
        bands = AcceptanceBands(
            per_tier={
                "intermediate": TierBands(
                    tier="intermediate",
                    bands={
                        "lr_auc": BandSpec(metric="lr_auc", gate="G7.2.2", min=0.7, max=1.0),
                    },
                )
            },
            cross_seed_spread={},
            cohort_shift=None,
            cross_tier_required=(),
            leakage_probes=LeakageProbeBands(
                id_only_max_auc=None, label_drift_max=None, feature_subsets={}
            ),
        )
        report = _make_report(
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_auc=0.86, lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
        )
        failures = [f for f in check_release_bands(report, bands) if f.gate.startswith("G7.4")]
        assert failures == []


# ---------------------------------------------------------------------------
# Leakage findings → gate failures
# ---------------------------------------------------------------------------


class TestLeakageReports:
    def test_findings_become_gate_failures(self, passing_bands: AcceptanceBands) -> None:
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
        )
        leak = {
            "intermediate": LeakageReport(
                findings=(
                    LeakageFinding(
                        channel="id_only_baseline",
                        detail="cols=lead_id",
                        message="AUC 0.85 > max 0.60",
                    ),
                )
            )
        }
        failures = check_release_bands(report, passing_bands, leakage_reports=leak)
        leakage_failures = [f for f in failures if f.gate == "G5.3"]
        assert len(leakage_failures) == 1
        assert leakage_failures[0].tier == "intermediate"
        assert "id_only_baseline" in leakage_failures[0].message

    def test_split_label_drift_does_not_collide_with_g6_4(
        self, passing_bands: AcceptanceBands
    ) -> None:
        """``split_label_drift`` findings must NOT be mapped to G6.4.

        G6.4 is the cohort/time-shift AUC degradation gate.  Earlier
        code mapped split-label-drift findings to G6.4 too, which would
        group unrelated failures under one gate id and confuse the CLI
        output.  The mapping was removed; the channel now falls through
        to ``leakage:split_label_drift``.
        """
        report = _make_report(
            intro=_make_cross_seed("intro", [42], lr_ap=0.78, p_at_100=0.85, rate=0.42),
            intermediate=_make_cross_seed(
                "intermediate", [42], lr_ap=0.55, p_at_100=0.65, rate=0.20
            ),
            advanced=_make_cross_seed("advanced", [42], lr_ap=0.30, p_at_100=0.40, rate=0.08),
        )
        leak = {
            "intermediate": LeakageReport(
                findings=(
                    LeakageFinding(
                        channel="split_label_drift",
                        detail="train↔test",
                        message="drift 0.15",
                    ),
                )
            )
        }
        failures = check_release_bands(report, passing_bands, leakage_reports=leak)
        gates = {f.gate for f in failures}
        assert "G6.4" not in gates  # Reserved for cohort-shift gate.
        assert "leakage:split_label_drift" in gates


# ---------------------------------------------------------------------------
# GateFailure formatting smoke test (the dataclass is dirt-simple but the
# CLI's format_failures consumes it; this test pins the field shape).
# ---------------------------------------------------------------------------


def test_gate_failure_is_immutable() -> None:
    f = GateFailure(gate="G7.1.2", tier="intro", message="oops")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.message = "bypassed"  # type: ignore[misc]
    assert f.gate == "G7.1.2"

"""Tests for :mod:`leadforge.validation.reporting`.

The renderer is matplotlib-Agg-only and deterministic.  We don't visual-
diff the figures (out of scope per the PR plan); we just assert each
contract file is created with non-empty bytes and that the markdown
report cites every JSON path it surfaces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from leadforge.validation.release_quality import (
    CalibrationBin,
    CohortShiftMetrics,
    CrossSeedTierMetrics,
    CrossTierOrdering,
    ReleaseQualityReport,
    TierMetrics,
)
from leadforge.validation.reporting import (
    CALIBRATION_FIGURE,
    COHORT_SHIFT_FIGURE,
    FIGURES_DIRNAME,
    LEAKAGE_DELTA_FIGURE,
    LIFT_CURVE_FIGURE_TEMPLATE,
    REPORT_JSON,
    REPORT_MD,
    VALUE_CAPTURE_FIGURE,
    render_report,
)


def _tier_metrics(tier: str, seed: int, **overrides: object) -> TierMetrics:
    base: dict[str, object] = {
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
            CalibrationBin(0.4, 0.5, 4, 0.45, 0.50),
            CalibrationBin(0.8, 0.9, 6, 0.85, 0.83),
        ],
        "baselines": {
            "source_only": 0.52,
            "engagement_only": 0.66,
            "post_snapshot_aggregates": 0.55,
            "id_only": 0.50,
        },
    }
    base.update(overrides)
    return TierMetrics(**base)  # type: ignore[arg-type]


_DEFAULT_TIERS: tuple[str, ...] = ("intro", "intermediate", "advanced")


def _build_report(tiers: tuple[str, ...] = _DEFAULT_TIERS) -> ReleaseQualityReport:
    cs_tiers: dict[str, CrossSeedTierMetrics] = {}
    cohort: dict[str, CohortShiftMetrics] = {}
    for tier in tiers:
        m1 = _tier_metrics(tier=tier, seed=42)
        m2 = _tier_metrics(tier=tier, seed=43, lr_auc=0.86, gbm_auc=0.89)
        cs_tiers[tier] = CrossSeedTierMetrics(
            tier=tier,
            seeds=[42, 43],
            per_seed=[m1, m2],
            medians={
                "lr_auc": 0.855,
                "gbm_auc": 0.885,
                "gbm_minus_lr_auc": 0.03,
                "lr_average_precision": 0.62,
                "gbm_average_precision": 0.65,
                "brier_score": 0.12,
                "log_loss": 0.34,
                "calibration_max_bin_error": 0.18,
                "top_decile_rate": 0.6,
                "conversion_rate_test": 0.30,
            },
            spreads={
                "lr_auc": 0.01,
                "gbm_auc": 0.01,
                "gbm_minus_lr_auc": 0.0,
                "lr_average_precision": 0.0,
                "brier_score": 0.0,
                "calibration_max_bin_error": 0.0,
                "top_decile_rate": 0.0,
                "conversion_rate_test": 0.0,
            },
        )
        cohort[tier] = CohortShiftMetrics(
            tier=tier, seed=42, random_split_auc=0.85, cohort_split_auc=0.78, auc_degradation=0.07
        )
    tier_list = list(tiers)
    ordering = CrossTierOrdering(
        by_average_precision=tier_list,
        by_precision_at_100=tier_list,
        by_gbm_minus_lr=tier_list,
        by_conversion_rate=tier_list,
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
        tiers=cs_tiers,
        cohort_shift=cohort,
        cross_tier_ordering=ordering,
    )


class TestRenderReport:
    def test_writes_every_contract_file(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "release/validation"
        written = render_report(report, out)
        assert (out / REPORT_JSON).exists()
        assert (out / REPORT_MD).exists()
        for tier in ("intro", "intermediate", "advanced"):
            f = out / FIGURES_DIRNAME / LIFT_CURVE_FIGURE_TEMPLATE.format(tier=tier)
            assert f.exists()
            assert f.stat().st_size > 0
        for fig in (
            CALIBRATION_FIGURE,
            LEAKAGE_DELTA_FIGURE,
            COHORT_SHIFT_FIGURE,
            VALUE_CAPTURE_FIGURE,
        ):
            f = out / FIGURES_DIRNAME / fig
            assert f.exists()
            assert f.stat().st_size > 0
        # Returned mapping covers every artefact name we just inspected.
        assert set(written) >= {
            "json",
            "md",
            "lift_curve_intro",
            "lift_curve_intermediate",
            "lift_curve_advanced",
            "calibration",
            "leakage_delta",
            "cohort_shift",
            "value_capture",
        }

    def test_json_is_well_formed(self, tmp_path: Path) -> None:
        report = _build_report()
        out = tmp_path / "v"
        render_report(report, out)
        d = json.loads((out / REPORT_JSON).read_text())
        assert d["release_id"] == "leadforge-lead-scoring-v1"
        assert "tiers" in d
        assert "cross_tier_ordering" in d
        # Ordering booleans round-trip as JSON true/false (not Python str).
        assert d["cross_tier_ordering"]["gbm_minus_lr_positive_in_every_tier"] is True

    def test_markdown_cites_json_paths_for_every_metric_cell(self, tmp_path: Path) -> None:
        """G10.6 — every claim has a backing JSON reference."""
        report = _build_report()
        out = tmp_path / "v"
        render_report(report, out)
        md = (out / REPORT_MD).read_text()
        # Every numeric cell should be followed by ``(`$.<path>`)``.
        # Find table rows under the per-tier headline section.
        # We assert at least one citation exists per tier.
        for tier in ("intro", "intermediate", "advanced"):
            assert re.search(
                rf"\| {tier} \|.*\$\.tiers\.{tier}\.medians\.lr_auc",
                md,
            ), f"missing JSON citation for {tier}.medians.lr_auc"
        # Cohort shift section cites every degradation value.
        for tier in ("intro", "intermediate", "advanced"):
            assert f"$.cohort_shift.{tier}.auc_degradation" in md
        # Cross-tier ordering booleans cite their JSON keys.
        assert "$.cross_tier_ordering.gbm_minus_lr_positive_in_every_tier" in md

    def test_partial_release_renders_partial_outputs(self, tmp_path: Path) -> None:
        """One-tier reports skip lift curves for the missing tiers and
        also skip calibration (which is intermediate-only)."""
        report = _build_report(("intro",))
        out = tmp_path / "v"
        render_report(report, out)
        assert (out / FIGURES_DIRNAME / LIFT_CURVE_FIGURE_TEMPLATE.format(tier="intro")).exists()
        assert not (
            out / FIGURES_DIRNAME / LIFT_CURVE_FIGURE_TEMPLATE.format(tier="intermediate")
        ).exists()
        # Calibration figure only renders for the intermediate tier per
        # the design contract; intro-only reports skip it.
        assert not (out / FIGURES_DIRNAME / CALIBRATION_FIGURE).exists()

    def test_render_is_deterministic_given_same_input(self, tmp_path: Path) -> None:
        """Two consecutive renders of the same report produce
        byte-identical JSON.  Markdown and figures are also stable but
        figures depend on the matplotlib version's font cache, so we
        only assert byte-equality on the text artefacts."""
        report = _build_report()
        a = tmp_path / "a"
        b = tmp_path / "b"
        render_report(report, a)
        render_report(report, b)
        assert (a / REPORT_JSON).read_bytes() == (b / REPORT_JSON).read_bytes()
        assert (a / REPORT_MD).read_bytes() == (b / REPORT_MD).read_bytes()


class TestNanRenderingIsClean:
    def test_nan_metrics_render_as_n_a_in_markdown(self, tmp_path: Path) -> None:
        report = _build_report()
        # Patch one cohort entry to NaN.
        report.cohort_shift["intro"] = CohortShiftMetrics(
            tier="intro",
            seed=42,
            random_split_auc=0.85,
            cohort_split_auc=float("nan"),
            auc_degradation=float("nan"),
        )
        out = tmp_path / "v"
        render_report(report, out)
        md = (out / REPORT_MD).read_text()
        # The intro cohort row should carry an n/a marker rather than
        # the literal string ``nan``.
        intro_row = next(
            line
            for line in md.splitlines()
            if line.startswith("| intro ") and "$.cohort_shift" in line
        )
        assert "_n/a_" in intro_row
        assert "nan" not in intro_row.lower().replace("$.cohort_shift.intro.", "")
        # JSON converted NaN to null.
        d = json.loads((out / REPORT_JSON).read_text())
        assert d["cohort_shift"]["intro"]["cohort_split_auc"] is None

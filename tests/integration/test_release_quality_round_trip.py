"""End-to-end: ``Generator`` → ``release_quality.measure`` → ``reporting.render``.

Slow integration test (one full simulation, twice — N=2 seeds) that
verifies the PR 3.2 modules compose against the real bundle writer.

The unit suites in ``tests/validation/test_release_quality.py`` and
``tests/validation/test_reporting.py`` cover individual primitives; this
test asserts they actually plug together.

Tiny population sizes per the PR plan (the per-tier sweep with full
counts is PR 3.3's driver).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.release_quality import (
    TierBuildSpec,
    measure_release_quality,
    regenerate_tier_for_seeds,
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

_SMALL = {"n_leads": 50, "n_accounts": 25, "n_contacts": 75}


@pytest.fixture(scope="module")
def small_intermediate_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full Generator run at the smallest size that still produces
    a non-degenerate label distribution.  Reused across the round-trip
    tests to amortise the simulation cost."""
    pytest.importorskip("sklearn")
    out = tmp_path_factory.mktemp("rq_intermediate") / "bundle"
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=42,
        exposure_mode="student_public",
        difficulty="intermediate",
    )
    gen.generate(**_SMALL).save(str(out))
    return out


def test_round_trip_against_real_bundle(small_intermediate_bundle: Path, tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    report = measure_release_quality(
        {"intermediate": {42: small_intermediate_bundle}},
        generation_timestamp="2026-05-06T12:00:00+00:00",
    )
    out = tmp_path / "release/validation"
    written = render_report(report, out)

    # Every promised file is produced and non-empty for a single-tier
    # report.  Lift curve renders only for ``intermediate`` here (the
    # other two tier names aren't in the report).
    assert (out / REPORT_JSON).exists()
    assert (out / REPORT_JSON).stat().st_size > 0
    assert (out / REPORT_MD).exists()
    assert (out / REPORT_MD).stat().st_size > 0
    for fig in (
        LIFT_CURVE_FIGURE_TEMPLATE.format(tier="intermediate"),
        CALIBRATION_FIGURE,
        LEAKAGE_DELTA_FIGURE,
        COHORT_SHIFT_FIGURE,
        VALUE_CAPTURE_FIGURE,
    ):
        f = out / FIGURES_DIRNAME / fig
        assert f.exists(), f"missing figure: {fig}"
        assert f.stat().st_size > 0, f"empty figure: {fig}"
    # JSON shape is the agreed contract.
    d = json.loads((out / REPORT_JSON).read_text())
    assert d["release_id"] == "leadforge-lead-scoring-v1"
    tier = d["tiers"]["intermediate"]
    assert tier["seeds"] == [42]
    per_seed = tier["per_seed"][0]
    for fld in (
        "lr_auc",
        "gbm_auc",
        "gbm_minus_lr_auc",
        "lr_average_precision",
        "brier_score",
        "log_loss",
        "calibration_max_bin_error",
        "calibration_bins",
        "baselines",
        "precision_at_k",
        "recall_at_k",
        "lift_at_pct",
        "expected_acv_capture_at_k",
    ):
        assert fld in per_seed, f"missing field in per_seed payload: {fld}"
    # Cohort shift entry carries the canonical seed.
    assert d["cohort_shift"]["intermediate"]["seed"] == 42
    # Markdown exists with at least the GBM-vs-LR ordering bool citation.
    md = (out / REPORT_MD).read_text()
    assert "$.cross_tier_ordering.gbm_minus_lr_positive_in_every_tier" in md
    assert "intermediate" in md
    assert sorted(written.keys()) == sorted(
        {
            "json",
            "md",
            "lift_curve_intermediate",
            "calibration",
            "leakage_delta",
            "cohort_shift",
            "value_capture",
        }
    )


def test_regenerate_tier_for_seeds_n2(small_intermediate_bundle: Path, tmp_path: Path) -> None:
    """``regenerate_tier_for_seeds`` produces one bundle dir per seed
    and is idempotent — a re-run reuses the existing manifest.

    N=2 here per the PR plan; the full N=5 release-time sweep is PR
    3.3's driver.
    """
    pytest.importorskip("sklearn")
    spec = TierBuildSpec.from_bundle(small_intermediate_bundle, name="intermediate_small")
    # Override population sizes so the rebuild stays fast; tier from the
    # canonical bundle has the same difficulty/recipe/exposure_mode.
    spec = TierBuildSpec(
        name=spec.name,
        recipe_id=spec.recipe_id,
        difficulty=spec.difficulty,
        n_leads=_SMALL["n_leads"],
        n_accounts=_SMALL["n_accounts"],
        n_contacts=_SMALL["n_contacts"],
        snapshot_day=spec.snapshot_day,
        primary_task=spec.primary_task,
        label_window_days=spec.label_window_days,
        exposure_mode=spec.exposure_mode,
    )
    workdir = tmp_path / "regen"
    out = regenerate_tier_for_seeds(spec, [42, 43], workdir)
    assert sorted(out.keys()) == [42, 43]
    for seed, p in out.items():
        assert (p / "manifest.json").exists()
        manifest = json.loads((p / "manifest.json").read_text())
        assert manifest["seed"] == seed
    # Idempotent re-run returns the same paths.
    out2 = regenerate_tier_for_seeds(spec, [42, 43], workdir)
    assert out == out2


def test_full_release_quality_n2(small_intermediate_bundle: Path, tmp_path: Path) -> None:
    """N=2 cross-seed sweep through the full orchestrator + renderer.

    This exercises everything PR 3.3's driver will call: regenerate →
    measure_release_quality → render_report.  At N=2 we only assert
    structural shape; PR 3.3 calibrates the bands themselves.
    """
    pytest.importorskip("sklearn")
    spec = TierBuildSpec(
        name="intermediate",
        recipe_id="b2b_saas_procurement_v1",
        difficulty="intermediate",
        n_leads=_SMALL["n_leads"],
        n_accounts=_SMALL["n_accounts"],
        n_contacts=_SMALL["n_contacts"],
        snapshot_day=30,
        primary_task="converted_within_90_days",
        label_window_days=90,
        exposure_mode="student_public",
    )
    workdir = tmp_path / "regen"
    bundles = regenerate_tier_for_seeds(spec, [42, 43], workdir)
    report = measure_release_quality(
        {"intermediate": bundles}, generation_timestamp="2026-05-06T12:00:00+00:00"
    )
    csm = report.tiers["intermediate"]
    assert csm.seeds == [42, 43]
    assert len(csm.per_seed) == 2
    assert "lr_auc" in csm.medians
    assert "lr_auc" in csm.spreads
    # Each seed's TierMetrics is a separate measurement, so the spread
    # is non-negative and finite.
    assert csm.spreads["lr_auc"] >= 0.0
    out = tmp_path / "release/validation"
    written = render_report(report, out)
    assert (out / REPORT_JSON).exists()
    assert (out / FIGURES_DIRNAME / LIFT_CURVE_FIGURE_TEMPLATE.format(tier="intermediate")).exists()
    assert "lift_curve_intermediate" in written

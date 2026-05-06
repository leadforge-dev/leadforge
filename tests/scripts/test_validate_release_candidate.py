"""Tests for ``scripts/validate_release_candidate.py``.

Two layers:

* Unit tests against the driver helpers (``parse_args``,
  ``build_tier_spec``, ``regenerate_or_load``, ``run_tier_leakage_probes``,
  ``format_failures``, ``format_summary``) — fast, mocked at the
  ``measure_release_quality`` / ``regenerate_tier_for_seeds`` boundary.
* One integration test that runs the full ``run_validation`` pipeline
  end-to-end at ``--quick`` size against a real Generator run; gated on
  sklearn availability.

Pattern follows ``tests/scripts/test_probe_relational_leakage.py`` —
loads the script as a module via ``importlib`` so the helpers can be
unit-tested directly.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_release_candidate.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("validate_release_candidate", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
driver = importlib.util.module_from_spec(_spec)
sys.modules["validate_release_candidate"] = driver
_spec.loader.exec_module(driver)


# ---------------------------------------------------------------------------
# Mock fixtures and helpers
# ---------------------------------------------------------------------------


_BANDS_YAML = """
per_tier:
  intro:
    lr_auc: {min: 0.70, max: 0.99}
    conversion_rate_test: {min: 0.20, max: 0.60}
  intermediate:
    lr_auc: {min: 0.70, max: 0.99}
    conversion_rate_test: {min: 0.05, max: 0.50}
  advanced:
    lr_auc: {min: 0.60, max: 0.99}
    conversion_rate_test: {min: 0.0, max: 0.30}
cross_seed_spread:
  lr_auc: {max: 0.30}
cohort_shift:
  auc_degradation: {min: -0.30, max: 0.50}
cross_tier_required: [intro, intermediate, advanced]
leakage_probes:
  id_only_max_auc: 0.99
  feature_subsets: {}
"""


@pytest.fixture
def bands_path(tmp_path: Path) -> Path:
    p = tmp_path / "bands.yaml"
    p.write_text(_BANDS_YAML)
    return p


def _write_minimal_bundle(target: Path, *, seed: int, difficulty: str) -> None:
    """Write the smallest manifest+task layout the driver reads."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": "5",
                "package_version": "1.0.0",
                "recipe_id": "b2b_saas_procurement_v1",
                "seed": seed,
                "exposure_mode": "student_public",
                "difficulty": difficulty,
                "n_accounts": 25,
                "n_contacts": 75,
                "n_leads": 50,
                "horizon_days": 90,
                "primary_task": "converted_within_90_days",
                "label_window_days": 90,
                "snapshot_day": 30,
            }
        )
    )
    task_dir = target / "tasks" / "converted_within_90_days"
    task_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "lead_id": [f"lead_{i:04d}" for i in range(20)],
            "industry": ["saas", "fintech"] * 10,
            "expected_acv": [50_000.0] * 20,
            "converted_within_90_days": [True, False] * 10,
        }
    )
    for split in ("train", "valid", "test"):
        df.to_parquet(task_dir / f"{split}.parquet", index=False)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_default_seeds_and_paths(self) -> None:
        args = driver.parse_args([])
        assert args.seeds == list(driver.DEFAULT_SEEDS)
        assert args.cohort_canonical_seed == driver.DEFAULT_COHORT_CANONICAL_SEED
        assert args.release_dir == driver.DEFAULT_RELEASE_DIR
        assert args.workdir == driver.DEFAULT_WORKDIR
        assert args.out_dir == driver.DEFAULT_OUT_DIR
        assert args.bands == driver.DEFAULT_BANDS
        assert args.quick is False
        assert args.no_rebuild is False
        assert args.tiers == list(driver.TIERS)

    def test_quick_overrides_seed_list(self) -> None:
        args = driver.parse_args(["--quick", "--seeds", "100", "200", "300"])
        config = driver._config_from_args(args)
        assert config.quick is True
        # --quick replaces user-provided seeds with QUICK_SEEDS.
        assert config.seeds == driver.QUICK_SEEDS

    def test_canonical_seed_outside_sweep_falls_back(self) -> None:
        args = driver.parse_args(["--seeds", "10", "11", "--cohort-canonical-seed", "99"])
        config = driver._config_from_args(args)
        assert config.cohort_canonical_seed == 10  # smallest seed in sweep.

    def test_tiers_subset(self) -> None:
        args = driver.parse_args(["--tiers", "intermediate"])
        assert args.tiers == ["intermediate"]


# ---------------------------------------------------------------------------
# build_tier_spec
# ---------------------------------------------------------------------------


class TestBuildTierSpec:
    def test_full_size_reads_manifest(self, tmp_path: Path) -> None:
        release = tmp_path / "release"
        intro = release / "intro"
        _write_minimal_bundle(intro, seed=42, difficulty="intro")
        spec = driver.build_tier_spec(release, "intro", quick=False)
        assert spec.name == "intro"
        assert spec.recipe_id == "b2b_saas_procurement_v1"
        assert spec.n_leads == 50
        assert spec.snapshot_day == 30

    def test_quick_overrides_population(self, tmp_path: Path) -> None:
        release = tmp_path / "release"
        intro = release / "intro"
        _write_minimal_bundle(intro, seed=42, difficulty="intro")
        # Manifest declares n_leads=50; --quick swaps in QUICK_POPULATION.
        spec = driver.build_tier_spec(release, "intro", quick=True)
        assert spec.n_leads == driver.QUICK_POPULATION["n_leads"]
        assert spec.n_accounts == driver.QUICK_POPULATION["n_accounts"]
        assert spec.n_contacts == driver.QUICK_POPULATION["n_contacts"]

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest"):
            driver.build_tier_spec(tmp_path / "release", "intro", quick=False)


# ---------------------------------------------------------------------------
# regenerate_or_load
# ---------------------------------------------------------------------------


class TestRegenerateOrLoad:
    def test_no_rebuild_with_existing_bundles(self, tmp_path: Path) -> None:
        workdir = tmp_path / "workdir"
        bundle = workdir / "intro__seed42"
        _write_minimal_bundle(bundle, seed=42, difficulty="intro")
        spec = driver.TierBuildSpec(
            name="intro",
            recipe_id="b2b_saas_procurement_v1",
            difficulty="intro",
            n_leads=50,
            n_accounts=25,
            n_contacts=75,
            snapshot_day=30,
        )
        out = driver.regenerate_or_load(spec, [42], workdir, no_rebuild=True)
        assert out == {42: bundle}

    def test_no_rebuild_with_missing_bundles_raises(self, tmp_path: Path) -> None:
        workdir = tmp_path / "workdir"
        spec = driver.TierBuildSpec(
            name="intro",
            recipe_id="b2b_saas_procurement_v1",
            difficulty="intro",
            n_leads=50,
            n_accounts=25,
            n_contacts=75,
            snapshot_day=30,
        )
        with pytest.raises(FileNotFoundError, match="missing"):
            driver.regenerate_or_load(spec, [42, 43], workdir, no_rebuild=True)

    def test_with_rebuild_calls_generator(self, tmp_path: Path) -> None:
        workdir = tmp_path / "workdir"
        spec = driver.TierBuildSpec(
            name="intro",
            recipe_id="b2b_saas_procurement_v1",
            difficulty="intro",
            n_leads=50,
            n_accounts=25,
            n_contacts=75,
            snapshot_day=30,
        )
        with mock.patch.object(
            driver,
            "regenerate_tier_for_seeds",
            return_value={42: workdir / "intro__seed42", 43: workdir / "intro__seed43"},
        ) as fake:
            out = driver.regenerate_or_load(spec, [42, 43], workdir, no_rebuild=False)
        fake.assert_called_once()
        assert sorted(out.keys()) == [42, 43]


# ---------------------------------------------------------------------------
# run_tier_leakage_probes
# ---------------------------------------------------------------------------


class TestRunTierLeakageProbes:
    def test_skips_when_no_splits(self, tmp_path: Path, bands_path: Path) -> None:
        bundle = tmp_path / "empty"
        bundle.mkdir()
        bands = driver.load_bands(bands_path)
        report = driver.run_tier_leakage_probes(bundle, bands=bands)
        # No manifest at all: skips silently.
        assert report.findings == ()

    def test_runs_against_real_splits(self, tmp_path: Path, bands_path: Path) -> None:
        pytest.importorskip("sklearn")
        bundle = tmp_path / "bundle"
        _write_minimal_bundle(bundle, seed=42, difficulty="intro")
        bands = driver.load_bands(bands_path)
        report = driver.run_tier_leakage_probes(bundle, bands=bands)
        # The mocked bundle has lead_ids that don't repeat across splits
        # (we wrote the same df for every split, so every lead_id IS in
        # train+valid+test) — id_only baseline runs with max_auc=0.99
        # which is permissive, so no findings expected at this scale.
        assert isinstance(report.findings, tuple)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_failures_groups_by_gate(self) -> None:
        from leadforge.validation.difficulty import GateFailure

        text = driver.format_failures(
            [
                GateFailure(gate="G7.1.2", tier="intro", message="lr_auc below"),
                GateFailure(gate="G7.1.2", tier="intermediate", message="lr_auc below"),
                GateFailure(gate="G6.4", tier="intro", message="cohort skew"),
            ]
        )
        # Gates are alphabetically sorted; G6.4 before G7.1.2.
        assert text.index("[G6.4]") < text.index("[G7.1.2]")
        assert text.count("[G7.1.2]") == 1
        assert "intro" in text
        assert "intermediate" in text

    def test_format_failures_empty(self) -> None:
        assert driver.format_failures([]) == ""

    def test_format_summary_contains_pass_or_fail_marker(self) -> None:
        from leadforge.validation.difficulty import GateFailure
        from leadforge.validation.leakage_probes import LeakageReport
        from leadforge.validation.release_quality import (
            CrossTierOrdering,
            ReleaseQualityReport,
        )

        report = ReleaseQualityReport(
            release_id="x",
            package_version="0.0",
            generation_timestamp="2026-01-01T00:00:00+00:00",
            seeds=[42],
            tiers={},
            cohort_shift={},
            cross_tier_ordering=CrossTierOrdering(
                by_average_precision=[],
                by_precision_at_100=[],
                by_gbm_minus_lr=[],
                by_conversion_rate=[],
                average_precision_intro_gt_intermediate=None,
                average_precision_intermediate_gt_advanced=None,
                precision_at_100_intro_gt_intermediate=None,
                precision_at_100_intermediate_gt_advanced=None,
                conversion_rate_intro_gt_intermediate=None,
                conversion_rate_intermediate_gt_advanced=None,
                gbm_minus_lr_positive_in_every_tier=None,
            ),
        )
        passing = driver.DriverResult(
            report=report, leakage_reports={"intro": LeakageReport(())}, failures=[]
        )
        assert "PASS" in driver.format_summary(passing)
        failing = driver.DriverResult(
            report=report,
            leakage_reports={"intro": LeakageReport(())},
            failures=[GateFailure(gate="G7.1.2", tier="intro", message="x")],
        )
        assert "FAIL" in driver.format_summary(failing)


# ---------------------------------------------------------------------------
# run_validation — pipeline shape (mocked)
# ---------------------------------------------------------------------------


class TestRunValidationMocked:
    def test_pipeline_writes_outputs_and_runs_probes(
        self, tmp_path: Path, bands_path: Path
    ) -> None:
        """Mocks measure_release_quality + regenerate; checks that
        render_report is invoked and the gate-checker output is plumbed
        into the DriverResult."""
        from leadforge.validation.leakage_probes import LeakageReport
        from leadforge.validation.release_quality import (
            CalibrationBin,
            CohortShiftMetrics,
            CrossSeedTierMetrics,
            CrossTierOrdering,
            ReleaseQualityReport,
            TierMetrics,
        )

        release = tmp_path / "release"
        for tier in driver.TIERS:
            _write_minimal_bundle(release / tier, seed=42, difficulty=tier)
        workdir = tmp_path / "workdir"
        for tier in driver.TIERS:
            for seed in (42, 43):
                _write_minimal_bundle(workdir / f"{tier}__seed{seed}", seed=seed, difficulty=tier)

        # Build a synthetic ReleaseQualityReport.  Each tier just gets one
        # seed of trivial metrics; the band check should pass against
        # _BANDS_YAML.
        def _per_seed(tier: str, seed: int, *, lr_auc: float, rate: float) -> TierMetrics:
            return TierMetrics(
                tier=tier,
                seed=seed,
                n_train=20,
                n_test=20,
                base_rate=rate,
                conversion_rate_train=rate,
                conversion_rate_test=rate,
                lr_auc=lr_auc,
                gbm_auc=lr_auc + 0.01,
                gbm_minus_lr_auc=0.01,
                lr_average_precision=0.5,
                gbm_average_precision=0.55,
                precision_at_k={"50": 0.5, "100": 0.5},
                recall_at_k={"50": 0.5, "100": 0.5},
                lift_at_pct={"1": 2.0, "5": 1.5, "10": 1.2},
                top_decile_rate=0.5,
                cumulative_gains={"0": 0.0, "10": 0.4, "100": 1.0},
                expected_acv_capture_at_k={"50": 0.4, "100": 0.6},
                brier_score=0.18,
                log_loss=0.5,
                calibration_max_bin_error=0.1,
                calibration_bins=[
                    CalibrationBin(
                        bin_lower=0.0, bin_upper=0.5, n=10, mean_predicted=0.2, mean_actual=0.2
                    )
                ],
                baselines={"id_only": 0.5},
            )

        tier_data = {
            "intro": (0.85, 0.42),
            "intermediate": (0.85, 0.20),
            "advanced": (0.80, 0.08),
        }
        tiers: dict[str, CrossSeedTierMetrics] = {}
        cohort: dict[str, CohortShiftMetrics] = {}
        for name, (lr_auc, rate) in tier_data.items():
            per_seed = [_per_seed(name, s, lr_auc=lr_auc, rate=rate) for s in (42, 43)]
            tiers[name] = CrossSeedTierMetrics(
                tier=name,
                seeds=[42, 43],
                per_seed=per_seed,
                medians={
                    "lr_auc": lr_auc,
                    "gbm_auc": lr_auc + 0.01,
                    "gbm_minus_lr_auc": 0.01,
                    "lr_average_precision": 0.5,
                    "gbm_average_precision": 0.55,
                    "brier_score": 0.18,
                    "log_loss": 0.5,
                    "calibration_max_bin_error": 0.1,
                    "top_decile_rate": 0.5,
                    "conversion_rate_test": rate,
                },
                spreads={
                    "lr_auc": 0.0,
                    "gbm_auc": 0.0,
                    "gbm_minus_lr_auc": 0.0,
                    "lr_average_precision": 0.0,
                    "gbm_average_precision": 0.0,
                    "brier_score": 0.0,
                    "log_loss": 0.0,
                    "calibration_max_bin_error": 0.0,
                    "top_decile_rate": 0.0,
                    "conversion_rate_test": 0.0,
                },
            )
            cohort[name] = CohortShiftMetrics(
                tier=name,
                seed=42,
                random_split_auc=lr_auc,
                cohort_split_auc=lr_auc - 0.05,
                auc_degradation=0.05,
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
        synthetic_report = ReleaseQualityReport(
            release_id="leadforge-lead-scoring-v1",
            package_version="1.0.0",
            generation_timestamp="2026-05-06T12:00:00+00:00",
            seeds=[42, 43],
            tiers=tiers,
            cohort_shift=cohort,
            cross_tier_ordering=ordering,
        )

        config = driver.DriverConfig(
            release_dir=release,
            workdir=workdir,
            out_dir=tmp_path / "out",
            bands_path=bands_path,
            seeds=(42, 43),
            cohort_canonical_seed=42,
            tiers=driver.TIERS,
            quick=False,
            no_rebuild=True,
        )

        with (
            mock.patch.object(driver, "measure_release_quality", return_value=synthetic_report),
            mock.patch.object(driver, "run_tier_leakage_probes", return_value=LeakageReport(())),
        ):
            result = driver.run_validation(config)

        assert isinstance(result, driver.DriverResult)
        assert result.failures == []
        # render_report wrote the artefacts.
        out = tmp_path / "out"
        assert (out / "validation_report.json").exists()
        assert (out / "validation_report.md").exists()
        assert (out / "figures").is_dir()


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------


class TestMain:
    def test_pre_flight_missing_release_dir_returns_2(
        self, tmp_path: Path, bands_path: Path
    ) -> None:
        rc = driver.main(
            [
                "--release-dir",
                str(tmp_path / "nonexistent"),
                "--workdir",
                str(tmp_path / "workdir"),
                "--out-dir",
                str(tmp_path / "out"),
                "--bands",
                str(bands_path),
                "--no-rebuild",
            ]
        )
        assert rc == 2

    def test_invocation_with_dash_h(self) -> None:
        # Smoke-check the help screen renders without crashing.
        rc = subprocess.run(  # noqa: S603 — args are repo-internal constants
            [sys.executable, str(_SCRIPT_PATH), "--help"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert rc.returncode == 0
        assert "validate_release_candidate" in rc.stdout
        assert "--quick" in rc.stdout


# ---------------------------------------------------------------------------
# End-to-end --quick run against a real Generator
# ---------------------------------------------------------------------------


def test_quick_end_to_end(tmp_path: Path, bands_path: Path) -> None:
    """Real Generator run at QUICK size.  Slow (~30s) but covers the
    full pipeline once.  Skips when sklearn is not installed; the band
    YAML is permissive enough that tiny bundles still pass."""
    pytest.importorskip("sklearn")
    from leadforge.api.generator import Generator

    release = tmp_path / "release"
    for tier in driver.TIERS:
        out = release / tier
        Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            exposure_mode="student_public",
            difficulty=tier,
        ).generate(**driver.QUICK_POPULATION).save(str(out))

    config = driver.DriverConfig(
        release_dir=release,
        workdir=tmp_path / "workdir",
        out_dir=tmp_path / "out",
        bands_path=bands_path,
        seeds=driver.QUICK_SEEDS,
        cohort_canonical_seed=42,
        tiers=driver.TIERS,
        quick=True,
        no_rebuild=False,
    )
    result = driver.run_validation(config)
    # Don't assert pass / fail at QUICK size — the bands here are
    # designed for the full release. Just assert the pipeline produced a
    # report and figures.
    assert result.report.tiers
    assert (tmp_path / "out" / "validation_report.json").exists()
    assert (tmp_path / "out" / "figures").is_dir()

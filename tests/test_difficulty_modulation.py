"""Tests for difficulty profile modulation in the simulation engine."""

from __future__ import annotations

import pytest

from leadforge.api.generator import Generator
from leadforge.core.models import DifficultyParams, GenerationConfig
from leadforge.mechanisms.policies import assign_mechanisms

_MEDIUM = {"n_leads": 500, "n_accounts": 200, "n_contacts": 600}


class TestDifficultyParams:
    """Unit tests for DifficultyParams dataclass."""

    def test_construction(self) -> None:
        dp = DifficultyParams(
            signal_strength=0.90,
            noise_scale=0.10,
            missing_rate=0.02,
            outlier_rate=0.01,
            conversion_rate_lo=0.30,
            conversion_rate_hi=0.45,
            committee_friction=0.10,
        )
        assert dp.signal_strength == 0.90
        assert dp.conversion_rate_lo == 0.30

    def test_on_generation_config(self) -> None:
        dp = DifficultyParams(
            signal_strength=0.70,
            noise_scale=0.30,
            missing_rate=0.08,
            outlier_rate=0.04,
            conversion_rate_lo=0.18,
            conversion_rate_hi=0.28,
            committee_friction=0.30,
        )
        config = GenerationConfig(difficulty_params=dp)
        assert config.difficulty_params is dp

    def test_defaults_to_none(self) -> None:
        config = GenerationConfig()
        assert config.difficulty_params is None


class TestAssignMechanismsWithDifficulty:
    """Unit tests for difficulty-aware mechanism assignment."""

    def test_without_difficulty_unchanged(self) -> None:
        """Without difficulty_params, behavior matches original."""
        import random

        m = assign_mechanisms("fit_dominant", random.Random(42))  # noqa: S311
        # Original base_rate for fit_dominant is 0.008.
        assert m.conversion_hazard._base_rate == pytest.approx(0.008)

    def test_with_difficulty_params_changes_hazard(self) -> None:
        """With difficulty_params, hazard rates are modulated."""
        import random

        dp = DifficultyParams(
            signal_strength=0.70,
            noise_scale=0.30,
            missing_rate=0.08,
            outlier_rate=0.04,
            conversion_rate_lo=0.18,
            conversion_rate_hi=0.28,
            committee_friction=0.30,
        )
        m = assign_mechanisms("fit_dominant", random.Random(42), difficulty_params=dp)  # noqa: S311
        # Should be different from the default 0.008.
        assert m.conversion_hazard._base_rate != pytest.approx(0.008)
        # Should be lower (targeting ~23% vs baseline ~70%).
        assert m.conversion_hazard._base_rate < 0.008

    def test_intro_higher_than_advanced(self) -> None:
        """Intro difficulty produces higher hazard rates than advanced."""
        import random

        intro_dp = DifficultyParams(
            signal_strength=0.90,
            noise_scale=0.10,
            missing_rate=0.02,
            outlier_rate=0.01,
            conversion_rate_lo=0.30,
            conversion_rate_hi=0.45,
            committee_friction=0.10,
        )
        advanced_dp = DifficultyParams(
            signal_strength=0.50,
            noise_scale=0.55,
            missing_rate=0.18,
            outlier_rate=0.08,
            conversion_rate_lo=0.08,
            conversion_rate_hi=0.15,
            committee_friction=0.55,
        )
        m_intro = assign_mechanisms("fit_dominant", random.Random(42), difficulty_params=intro_dp)  # noqa: S311
        m_adv = assign_mechanisms("fit_dominant", random.Random(42), difficulty_params=advanced_dp)  # noqa: S311
        assert m_intro.conversion_hazard._base_rate > m_adv.conversion_hazard._base_rate


class TestConversionRateModulation:
    """Integration tests verifying conversion rates fall within declared ranges."""

    @pytest.mark.parametrize(
        ("difficulty", "lo", "hi"),
        [
            ("intro", 0.30, 0.45),
            ("intermediate", 0.18, 0.28),
            ("advanced", 0.08, 0.15),
        ],
    )
    def test_rate_within_range(self, difficulty: str, lo: float, hi: float) -> None:
        """Conversion rate falls within target range (±tolerance)."""
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            difficulty=difficulty,
        )
        bundle = gen.generate(**_MEDIUM)
        leads = bundle.simulation_result.leads
        rate = sum(1 for lead in leads if lead.current_stage == "closed_won") / len(leads)
        # Allow 8% tolerance for small-sample variance.
        tolerance = 0.08
        assert rate >= lo - tolerance, f"{difficulty} rate {rate:.2%} below {lo - tolerance:.2%}"
        assert rate <= hi + tolerance, f"{difficulty} rate {rate:.2%} above {hi + tolerance:.2%}"

    def test_ordering(self) -> None:
        """Intro > intermediate > advanced in conversion rate."""
        rates = {}
        for difficulty in ("intro", "intermediate", "advanced"):
            gen = Generator.from_recipe(
                "b2b_saas_procurement_v1",
                seed=42,
                difficulty=difficulty,
            )
            bundle = gen.generate(**_MEDIUM)
            leads = bundle.simulation_result.leads
            rates[difficulty] = sum(
                1 for lead in leads if lead.current_stage == "closed_won"
            ) / len(leads)
        assert rates["intro"] > rates["intermediate"] > rates["advanced"]


class TestDeterminism:
    """Determinism tests for difficulty modulation."""

    def test_same_seed_same_difficulty_identical(self) -> None:
        """Same seed + difficulty produces identical results."""
        results = []
        for _ in range(2):
            gen = Generator.from_recipe(
                "b2b_saas_procurement_v1",
                seed=42,
                difficulty="intermediate",
            )
            bundle = gen.generate(n_leads=100, n_accounts=50, n_contacts=150)
            leads = bundle.simulation_result.leads
            stages = [lead.current_stage for lead in leads]
            results.append(stages)
        assert results[0] == results[1]


class TestSnapshotDistortions:
    """Tests for noise and missingness injection in snapshot."""

    def test_intro_has_minimal_noise(self) -> None:
        """Intro tier has low noise and minimal missingness."""
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            difficulty="intro",
        )
        bundle = gen.generate(n_leads=200, n_accounts=80, n_contacts=240)
        bundle.save("/tmp/test_intro_distortion")

        import pandas as pd

        df = pd.read_parquet(
            "/tmp/test_intro_distortion/tasks/converted_within_90_days/train.parquet"
        )
        # Intro has 2% missing rate, so very few NaN values expected.
        total_cells = df.select_dtypes(include="number").size
        missing_frac = df.select_dtypes(include="number").isna().sum().sum() / total_cells
        assert missing_frac < 0.10  # well below 10%

    def test_advanced_has_more_missingness(self) -> None:
        """Advanced tier has substantially more missing values than intro."""
        import pandas as pd

        for diff, out_path in [
            ("intro", "/tmp/test_miss_intro"),
            ("advanced", "/tmp/test_miss_adv"),
        ]:
            gen = Generator.from_recipe(
                "b2b_saas_procurement_v1",
                seed=42,
                difficulty=diff,
            )
            bundle = gen.generate(n_leads=200, n_accounts=80, n_contacts=240)
            bundle.save(out_path)

        df_intro = pd.read_parquet(
            "/tmp/test_miss_intro/tasks/converted_within_90_days/train.parquet"
        )
        df_adv = pd.read_parquet("/tmp/test_miss_adv/tasks/converted_within_90_days/train.parquet")
        miss_intro = df_intro.select_dtypes(include="number").isna().sum().sum()
        miss_adv = df_adv.select_dtypes(include="number").isna().sum().sum()
        assert miss_adv > miss_intro

"""Tests for leadforge.validation.drift."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from leadforge.api.generator import Generator
from leadforge.validation.drift import check_cross_seed_stability

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


@pytest.fixture(scope="module")
def multi_seed_bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    bundles: dict[int, Path] = {}
    for seed in (1, 2, 3):
        out = tmp_path_factory.mktemp(f"seed_{seed}")
        Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=seed, exposure_mode="student_public"
        ).generate(**_SMALL).save(str(out))
        bundles[seed] = out
    return bundles


class TestCrossSeedStability:
    def test_similar_seeds_pass(self, multi_seed_bundles: dict[int, Path]) -> None:
        errors = check_cross_seed_stability(multi_seed_bundles)
        assert errors == [], f"Unexpected drift errors: {errors}"

    def test_single_seed_skips(self, multi_seed_bundles: dict[int, Path]) -> None:
        first_seed = next(iter(multi_seed_bundles))
        errors = check_cross_seed_stability({first_seed: multi_seed_bundles[first_seed]})
        assert errors == []

    def test_detects_zero_conversion_seed(
        self, tmp_path: Path, multi_seed_bundles: dict[int, Path]
    ) -> None:
        """A seed with 0% conversion should be flagged as degenerate."""
        # Copy one real bundle, then corrupt its train split to all-False.
        first_seed = next(iter(multi_seed_bundles))
        real = multi_seed_bundles[first_seed]
        fake = tmp_path / "zero_conv"
        shutil.copytree(real, fake)
        train_path = fake / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["converted_within_90_days"] = False
        df.to_parquet(train_path)

        bundles = {first_seed: real, 999: fake}
        errors = check_cross_seed_stability(bundles)
        assert any("0% conversion" in e and "999" in e for e in errors)

    def test_detects_full_conversion_seed(
        self, tmp_path: Path, multi_seed_bundles: dict[int, Path]
    ) -> None:
        """A seed with 100% conversion should be flagged as degenerate."""
        first_seed = next(iter(multi_seed_bundles))
        real = multi_seed_bundles[first_seed]
        fake = tmp_path / "full_conv"
        shutil.copytree(real, fake)
        train_path = fake / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        df["converted_within_90_days"] = True
        df.to_parquet(train_path)

        bundles = {first_seed: real, 998: fake}
        errors = check_cross_seed_stability(bundles)
        assert any("100% conversion" in e and "998" in e for e in errors)

    def test_detects_wide_rate_spread(
        self, tmp_path: Path, multi_seed_bundles: dict[int, Path]
    ) -> None:
        """A >5x spread in conversion rates should be flagged."""
        first_seed = next(iter(multi_seed_bundles))
        real = multi_seed_bundles[first_seed]
        fake = tmp_path / "high_rate"
        shutil.copytree(real, fake)
        train_path = fake / "tasks/converted_within_90_days/train.parquet"
        df = pd.read_parquet(train_path)
        # Set all rows to True → 100% rate vs real's ~10-30%.
        df["converted_within_90_days"] = True
        df.to_parquet(train_path)

        bundles = {first_seed: real, 997: fake}
        errors = check_cross_seed_stability(bundles)
        assert any("spread too wide" in e for e in errors)

    def test_detects_single_stage_seed(
        self, tmp_path: Path, multi_seed_bundles: dict[int, Path]
    ) -> None:
        """A seed where all leads are in one stage should be flagged."""
        first_seed = next(iter(multi_seed_bundles))
        real = multi_seed_bundles[first_seed]
        fake = tmp_path / "one_stage"
        shutil.copytree(real, fake)
        leads_path = fake / "tables/leads.parquet"
        df = pd.read_parquet(leads_path)
        df["current_stage"] = "mql"
        df.to_parquet(leads_path)

        bundles = {first_seed: real, 996: fake}
        errors = check_cross_seed_stability(bundles)
        assert any("only 1 funnel stage" in e and "996" in e for e in errors)

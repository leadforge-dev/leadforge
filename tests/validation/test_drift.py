"""Tests for leadforge.validation.drift."""

from __future__ import annotations

from pathlib import Path

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

"""Tests for leadforge.api.generator — Generator.from_recipe."""

import pytest

from leadforge.api.generator import Generator
from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidRecipeError


def test_from_recipe_returns_generator() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert isinstance(gen, Generator)


def test_from_recipe_config_recipe_id() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=1)
    assert gen.config.recipe_id == "b2b_saas_procurement_v1"


def test_from_recipe_seed_propagates() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=777)
    assert gen.config.seed == 777


def test_from_recipe_exposure_mode_propagates() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", exposure_mode="research_instructor")
    assert gen.config.exposure_mode == ExposureMode.research_instructor


def test_from_recipe_difficulty_propagates() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", difficulty="advanced")
    assert gen.config.difficulty == DifficultyProfile.advanced


def test_from_recipe_population_override() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", n_leads=123, n_accounts=50)
    assert gen.config.n_leads == 123
    assert gen.config.n_accounts == 50


def test_from_recipe_deterministic_config() -> None:
    """Same args must produce identical configs."""
    from leadforge.core.hashing import hash_config

    gen1 = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    gen2 = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert hash_config(gen1.config) == hash_config(gen2.config)


def test_from_recipe_different_seeds_different_configs() -> None:
    from leadforge.core.hashing import hash_config

    gen1 = Generator.from_recipe("b2b_saas_procurement_v1", seed=1)
    gen2 = Generator.from_recipe("b2b_saas_procurement_v1", seed=2)
    assert hash_config(gen1.config) != hash_config(gen2.config)


def test_from_recipe_invalid_id_raises() -> None:
    with pytest.raises(InvalidRecipeError):
        Generator.from_recipe("does_not_exist")


def test_generate_not_implemented() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    with pytest.raises(NotImplementedError):
        gen.generate()


def test_from_recipe_config_has_package_version() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    assert gen.config.package_version  # non-empty string


def test_from_recipe_override_dict() -> None:
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        override={"n_leads": 4242},
        # explicit kwarg wins
        n_leads=9999,
    )
    assert gen.config.n_leads == 9999


def test_from_recipe_override_dict_applies_seed_and_output_path() -> None:
    """Layer 2: override dict should set seed / output_path when not explicitly passed."""
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        override={"seed": 5678, "output_path": "/tmp/override"},
    )
    assert gen.config.seed == 5678
    assert gen.config.output_path == "/tmp/override"


def test_from_recipe_explicit_seed_beats_override_dict() -> None:
    """Layer 1: explicit seed / output_path kwargs beat override dict."""
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        override={"seed": 5678, "output_path": "/tmp/override"},
        seed=42,
        output_path="/tmp/explicit",
    )
    assert gen.config.seed == 42
    assert gen.config.output_path == "/tmp/explicit"

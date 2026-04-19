"""Tests for leadforge.api.recipes — Recipe model and config resolution."""

import pytest

from leadforge.api.recipes import Recipe
from leadforge.core.enums import DifficultyProfile, ExposureMode
from leadforge.core.exceptions import InvalidRecipeError
from leadforge.core.models import GenerationConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_DICT = {
    "id": "test_recipe_v1",
    "title": "Test Recipe",
    "vertical": "test_vertical",
    "description": "A recipe for testing.",
    "primary_task": "converted_within_90_days",
    "supported_modes": ["student_public", "research_instructor"],
    "supported_difficulty": ["intro", "intermediate", "advanced"],
    "default_population": {"n_accounts": 100, "n_contacts": 300, "n_leads": 500},
    "horizon_days": 90,
}


# ---------------------------------------------------------------------------
# Recipe.from_dict
# ---------------------------------------------------------------------------


def test_from_dict_roundtrip() -> None:
    recipe = Recipe.from_dict(VALID_DICT)
    assert recipe.id == "test_recipe_v1"
    assert recipe.primary_task == "converted_within_90_days"
    assert ExposureMode.student_public in recipe.supported_modes
    assert DifficultyProfile.intermediate in recipe.supported_difficulty
    assert recipe.default_population["n_leads"] == 500
    assert recipe.horizon_days == 90


def test_from_dict_missing_key_raises() -> None:
    bad = {k: v for k, v in VALID_DICT.items() if k != "primary_task"}
    with pytest.raises(InvalidRecipeError, match="missing required keys"):
        Recipe.from_dict(bad)


def test_from_dict_invalid_mode_raises() -> None:
    bad = {**VALID_DICT, "supported_modes": ["student_public", "not_a_mode"]}
    with pytest.raises(InvalidRecipeError, match="Invalid exposure mode"):
        Recipe.from_dict(bad)


def test_from_dict_invalid_difficulty_raises() -> None:
    bad = {**VALID_DICT, "supported_difficulty": ["easy"]}
    with pytest.raises(InvalidRecipeError, match="Invalid difficulty profile"):
        Recipe.from_dict(bad)


def test_from_dict_invalid_population_raises() -> None:
    bad = {**VALID_DICT, "default_population": {"n_leads": "five_thousand"}}
    with pytest.raises(InvalidRecipeError, match="default_population"):
        Recipe.from_dict(bad)


# ---------------------------------------------------------------------------
# Config resolution / precedence
# ---------------------------------------------------------------------------


def test_resolve_config_returns_generation_config() -> None:
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config()
    assert isinstance(config, GenerationConfig)


def test_resolve_config_recipe_defaults_used() -> None:
    """Layer 3: recipe default_population should flow into config."""
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config()
    assert config.n_accounts == 100
    assert config.n_contacts == 300
    assert config.n_leads == 500
    assert config.horizon_days == 90


def test_resolve_config_explicit_kwargs_override_recipe() -> None:
    """Layer 1: explicit kwargs win over recipe defaults."""
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config(n_leads=9999, horizon_days=30)
    assert config.n_leads == 9999
    assert config.horizon_days == 30
    # Non-overridden values still come from recipe
    assert config.n_accounts == 100


def test_resolve_config_override_dict_beats_recipe() -> None:
    """Layer 2: override dict beats recipe defaults but loses to explicit kwargs."""
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config(
        override={"n_leads": 7777, "n_accounts": 50},
        n_leads=8888,  # explicit kwargs win
    )
    assert config.n_leads == 8888
    assert config.n_accounts == 50  # from override dict


def test_resolve_config_seed_propagates() -> None:
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config(seed=999)
    assert config.seed == 999


def test_resolve_config_override_dict_applies_seed_and_output_path() -> None:
    """Layer 2: override dict should set seed / output_path when not explicitly passed."""
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config(override={"seed": 1234, "output_path": "/tmp/override"})
    assert config.seed == 1234
    assert config.output_path == "/tmp/override"


def test_resolve_config_explicit_seed_beats_override_dict() -> None:
    """Layer 1: explicit seed / output_path kwargs beat override dict."""
    recipe = Recipe.from_dict(VALID_DICT)
    config = recipe.resolve_config(
        override={"seed": 1234, "output_path": "/tmp/override"},
        seed=999,
        output_path="/tmp/explicit",
    )
    assert config.seed == 999
    assert config.output_path == "/tmp/explicit"


def test_resolve_config_unsupported_mode_raises() -> None:
    limited = {**VALID_DICT, "supported_modes": ["student_public"]}
    recipe = Recipe.from_dict(limited)
    with pytest.raises(InvalidRecipeError, match="not supported"):
        recipe.resolve_config(exposure_mode="research_instructor")


def test_resolve_config_unsupported_difficulty_raises() -> None:
    limited = {**VALID_DICT, "supported_difficulty": ["intermediate"]}
    recipe = Recipe.from_dict(limited)
    with pytest.raises(InvalidRecipeError, match="not supported"):
        recipe.resolve_config(difficulty="advanced")


# ---------------------------------------------------------------------------
# Real recipe loading via registry
# ---------------------------------------------------------------------------


def test_real_recipe_loads_and_parses() -> None:
    from leadforge.recipes.registry import load_recipe

    raw = load_recipe("b2b_saas_procurement_v1")
    recipe = Recipe.from_dict(raw)
    assert recipe.id == "b2b_saas_procurement_v1"
    assert DifficultyProfile.intermediate in recipe.supported_difficulty


def test_real_recipe_narrative_loads() -> None:
    from leadforge.recipes.registry import load_recipe

    recipe = Recipe.from_dict(load_recipe("b2b_saas_procurement_v1"))
    narrative = recipe.load_narrative()
    assert "company" in narrative
    assert "personas" in narrative


def test_real_recipe_difficulty_profiles_load() -> None:
    from leadforge.recipes.registry import load_recipe

    recipe = Recipe.from_dict(load_recipe("b2b_saas_procurement_v1"))
    profiles = recipe.load_difficulty_profiles()
    assert "intro" in profiles
    assert "intermediate" in profiles
    assert "advanced" in profiles

"""Tests for the recipe registry."""

import pytest

from leadforge.core.exceptions import InvalidRecipeError
from leadforge.recipes.registry import list_recipes, load_recipe


def test_list_recipes_returns_list() -> None:
    recipes = list_recipes()
    assert isinstance(recipes, list)
    assert len(recipes) >= 1


def test_list_recipes_contains_v1() -> None:
    ids = [r["id"] for r in list_recipes()]
    assert "b2b_saas_procurement_v1" in ids


def test_v1_recipe_has_required_fields() -> None:
    recipe = load_recipe("b2b_saas_procurement_v1")
    for field in ("id", "title", "primary_task", "supported_modes", "supported_difficulty"):
        assert field in recipe, f"Missing field: {field}"


def test_v1_recipe_primary_task() -> None:
    recipe = load_recipe("b2b_saas_procurement_v1")
    assert recipe["primary_task"] == "converted_within_90_days"


def test_v1_recipe_supported_modes() -> None:
    recipe = load_recipe("b2b_saas_procurement_v1")
    assert "student_public" in recipe["supported_modes"]
    assert "research_instructor" in recipe["supported_modes"]


def test_load_unknown_recipe_raises() -> None:
    with pytest.raises(InvalidRecipeError, match="not found"):
        load_recipe("nonexistent_recipe_xyz")

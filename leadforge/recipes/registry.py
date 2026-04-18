"""Recipe registry: discovery and loading of generation recipes."""

from pathlib import Path
from typing import Any

import yaml

from leadforge.core.exceptions import InvalidRecipeError

_RECIPES_DIR = Path(__file__).parent


def list_recipes() -> list[dict[str, Any]]:
    """Return metadata for all available recipes, sorted by ID."""
    recipes = []
    for entry in sorted(_RECIPES_DIR.iterdir()):
        recipe_file = entry / "recipe.yaml"
        if entry.is_dir() and recipe_file.exists():
            with recipe_file.open() as fh:
                recipes.append(yaml.safe_load(fh))
    return recipes


def load_recipe(recipe_id: str) -> dict[str, Any]:
    """Load and return a recipe by ID.

    Raises:
        InvalidRecipeError: if the recipe does not exist.
    """
    recipe_file = _RECIPES_DIR / recipe_id / "recipe.yaml"
    if not recipe_file.exists():
        raise InvalidRecipeError(
            f"Recipe '{recipe_id}' not found. "
            f"Run 'leadforge list-recipes' to see available recipes."
        )
    with recipe_file.open() as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]

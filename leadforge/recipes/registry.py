"""Recipe registry: discovery and loading of generation recipes."""

from pathlib import Path
from typing import Any

import yaml

from leadforge.core.exceptions import InvalidRecipeError

_RECIPES_DIR = Path(__file__).parent


def _parse_and_validate(path: Path) -> dict[str, Any]:
    """Parse a recipe YAML file and validate it is a well-formed dict."""
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "id" not in data:
        raise InvalidRecipeError(
            f"Recipe file '{path}' is malformed: expected a YAML mapping with an 'id' key."
        )
    return data  # type: ignore[return-value]


def list_recipes() -> list[dict[str, Any]]:
    """Return metadata for all available recipes, sorted by ID."""
    recipes = []
    for entry in _RECIPES_DIR.iterdir():
        recipe_file = entry / "recipe.yaml"
        if entry.is_dir() and recipe_file.exists():
            recipes.append(_parse_and_validate(recipe_file))
    return sorted(recipes, key=lambda recipe: recipe["id"])


def load_recipe(recipe_id: str) -> dict[str, Any]:
    """Load and return a recipe by ID.

    Raises:
        InvalidRecipeError: if the recipe does not exist, the ID is invalid,
            or the recipe file is malformed.
    """
    # Guard against path traversal (e.g. recipe_id = "../secret")
    base_dir = _RECIPES_DIR.resolve()
    recipe_dir = (base_dir / recipe_id).resolve()
    if not recipe_dir.is_relative_to(base_dir):
        raise InvalidRecipeError(f"Recipe ID '{recipe_id}' is invalid.")

    recipe_file = recipe_dir / "recipe.yaml"
    if not recipe_file.exists():
        raise InvalidRecipeError(
            f"Recipe '{recipe_id}' not found. "
            f"Run 'leadforge list-recipes' to see available recipes."
        )
    return _parse_and_validate(recipe_file)

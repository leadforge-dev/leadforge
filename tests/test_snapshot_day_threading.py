"""Tests for ``snapshot_day`` threading through the generation pipeline.

Verifies all four precedence layers (package default → recipe → override
dict → explicit kwarg) and the validation paths in
:meth:`GenerationConfig.__post_init__`.

Companion to :mod:`tests.test_primary_task_threading`, which covers the
sibling fields ``primary_task`` and ``label_window_days`` introduced in
PR #36 / PR #43.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.api.recipes import Recipe
from leadforge.core.exceptions import InvalidConfigError, InvalidRecipeError
from leadforge.core.models import GenerationConfig
from leadforge.core.serialization import load_json

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


# ---------------------------------------------------------------------------
# Recipe.from_dict parsing
# ---------------------------------------------------------------------------


def _base_recipe_dict(**extra) -> dict:
    """Minimal valid recipe payload for from_dict tests."""
    payload = {
        "id": "test_recipe",
        "title": "Test recipe",
        "vertical": "test_vertical",
        "description": "test",
        "primary_task": "converted_within_90_days",
        "supported_modes": ["student_public", "research_instructor"],
        "supported_difficulty": ["intro", "intermediate", "advanced"],
        "default_population": {"n_accounts": 1, "n_contacts": 1, "n_leads": 1},
        "horizon_days": 90,
    }
    payload.update(extra)
    return payload


class TestRecipeFromDictSnapshotDay:
    def test_absent_key_yields_none(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict())
        assert recipe.snapshot_day is None

    def test_positive_int_round_trips(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict(snapshot_day=30))
        assert recipe.snapshot_day == 30

    def test_zero_rejected(self) -> None:
        with pytest.raises(InvalidRecipeError, match="snapshot_day"):
            Recipe.from_dict(_base_recipe_dict(snapshot_day=0))

    def test_negative_rejected(self) -> None:
        with pytest.raises(InvalidRecipeError, match="snapshot_day"):
            Recipe.from_dict(_base_recipe_dict(snapshot_day=-5))

    def test_string_rejected(self) -> None:
        with pytest.raises(InvalidRecipeError, match="snapshot_day"):
            Recipe.from_dict(_base_recipe_dict(snapshot_day="30"))

    def test_bool_rejected(self) -> None:
        # bool is an int subclass; the validator must reject it explicitly.
        with pytest.raises(InvalidRecipeError, match="snapshot_day"):
            Recipe.from_dict(_base_recipe_dict(snapshot_day=True))

    def test_explicit_null_treated_as_none(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict(snapshot_day=None))
        assert recipe.snapshot_day is None


# ---------------------------------------------------------------------------
# resolve_config precedence
# ---------------------------------------------------------------------------


class TestResolveConfigSnapshotDayPrecedence:
    def test_package_default_when_recipe_silent(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict())  # no snapshot_day
        cfg = recipe.resolve_config()
        assert cfg.snapshot_day is None  # package default

    def test_recipe_default_overrides_package(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict(snapshot_day=20))
        cfg = recipe.resolve_config()
        assert cfg.snapshot_day == 20

    def test_override_dict_beats_recipe(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict(snapshot_day=20))
        cfg = recipe.resolve_config(override={"snapshot_day": 45})
        assert cfg.snapshot_day == 45

    def test_kwarg_beats_override_and_recipe(self) -> None:
        recipe = Recipe.from_dict(_base_recipe_dict(snapshot_day=20))
        cfg = recipe.resolve_config(override={"snapshot_day": 45}, snapshot_day=60)
        assert cfg.snapshot_day == 60

    def test_b2b_recipe_pins_thirty(self) -> None:
        """Sanity check that the shipped recipe applies snapshot_day=30."""
        from leadforge.recipes.registry import load_recipe

        recipe = Recipe.from_dict(load_recipe("b2b_saas_procurement_v1"))
        cfg = recipe.resolve_config()
        assert cfg.snapshot_day == 30


# ---------------------------------------------------------------------------
# GenerationConfig validation
# ---------------------------------------------------------------------------


class TestGenerationConfigValidation:
    def test_none_is_valid(self) -> None:
        cfg = GenerationConfig(snapshot_day=None)
        assert cfg.snapshot_day is None

    def test_positive_int_is_valid(self) -> None:
        cfg = GenerationConfig(snapshot_day=30)
        assert cfg.snapshot_day == 30

    def test_zero_rejected(self) -> None:
        with pytest.raises(InvalidConfigError, match="snapshot_day"):
            GenerationConfig(snapshot_day=0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(InvalidConfigError, match="snapshot_day"):
            GenerationConfig(snapshot_day=-1)

    def test_string_rejected(self) -> None:
        with pytest.raises(InvalidConfigError, match="snapshot_day"):
            GenerationConfig(snapshot_day="30")  # type: ignore[arg-type]

    def test_bool_rejected(self) -> None:
        with pytest.raises(InvalidConfigError, match="snapshot_day"):
            GenerationConfig(snapshot_day=True)  # type: ignore[arg-type]

    def test_exceeds_horizon_rejected(self) -> None:
        with pytest.raises(InvalidConfigError, match="horizon_days"):
            GenerationConfig(horizon_days=20, label_window_days=20, snapshot_day=21)

    def test_exceeds_label_window_rejected(self) -> None:
        """A snapshot anchored after the label closes is nonsensical: the
        feature window would extend past events the label was scored on.
        Catch this at config time, not at the consumer end."""
        with pytest.raises(InvalidConfigError, match="label_window_days"):
            GenerationConfig(horizon_days=120, label_window_days=90, snapshot_day=100)

    def test_equal_to_horizon_is_valid(self) -> None:
        # Equal is the boundary case — this is "full-horizon" expressed
        # explicitly rather than via None.  Allowed.
        cfg = GenerationConfig(horizon_days=90, label_window_days=90, snapshot_day=90)
        assert cfg.snapshot_day == 90


# ---------------------------------------------------------------------------
# End-to-end: kwarg → manifest field
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def custom_snapshot_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bundle generated with snapshot_day overridden via the explicit kwarg."""
    out = tmp_path_factory.mktemp("custom_snapshot")
    Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=42,
        exposure_mode="student_public",
        snapshot_day=20,
    ).generate(**_SMALL).save(str(out))
    return out


class TestEndToEnd:
    def test_kwarg_overrides_recipe_in_manifest(self, custom_snapshot_bundle: Path) -> None:
        manifest = load_json(custom_snapshot_bundle / "manifest.json")
        assert manifest["snapshot_day"] == 20  # not 30 (the recipe default)

"""Tests for leadforge.narrative.dataset_card."""

from leadforge.api.generator import Generator
from leadforge.core.models import GenerationConfig, WorldSpec
from leadforge.narrative.dataset_card import render_dataset_card


def _make_world_spec(**kwargs: object) -> WorldSpec:
    return WorldSpec(config=GenerationConfig(**kwargs))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Rendering without narrative (stub mode)
# ---------------------------------------------------------------------------


def test_card_returns_string() -> None:
    spec = _make_world_spec()
    card = render_dataset_card(spec)
    assert isinstance(card, str)
    assert len(card) > 0


def test_card_contains_recipe_id() -> None:
    spec = _make_world_spec(recipe_id="b2b_saas_procurement_v1")
    assert "b2b_saas_procurement_v1" in render_dataset_card(spec)


def test_card_contains_seed() -> None:
    spec = _make_world_spec(seed=99)
    assert "99" in render_dataset_card(spec)


def test_card_contains_exposure_mode() -> None:
    spec = _make_world_spec()
    assert "student_public" in render_dataset_card(spec)


def test_card_contains_primary_task() -> None:
    assert "converted_within_90_days" in render_dataset_card(_make_world_spec())


def test_card_contains_label_definition() -> None:
    card = render_dataset_card(_make_world_spec())
    assert "closed_won" in card
    assert "90 days" in card


def test_card_contains_use_cases() -> None:
    card = render_dataset_card(_make_world_spec())
    assert "use cases" in card.lower()


def test_card_contains_caveats() -> None:
    card = render_dataset_card(_make_world_spec())
    assert "synthetic" in card.lower()


def test_card_no_narrative_shows_stub() -> None:
    spec = WorldSpec(config=GenerationConfig(), narrative=None)
    assert "not available" in render_dataset_card(spec).lower()


# ---------------------------------------------------------------------------
# Rendering with narrative (full mode)
# ---------------------------------------------------------------------------


def test_card_with_narrative_contains_company_name() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    card = render_dataset_card(gen.world_spec)
    assert "Veridian Technologies" in card


def test_card_with_narrative_contains_product_name() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    card = render_dataset_card(gen.world_spec)
    assert "Veridian Procure" in card


def test_card_with_narrative_contains_geographies() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    card = render_dataset_card(gen.world_spec)
    assert "US" in card


def test_card_with_narrative_contains_personas() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    card = render_dataset_card(gen.world_spec)
    assert "vp_finance" in card


# ---------------------------------------------------------------------------
# Generator integration
# ---------------------------------------------------------------------------


def test_generator_world_spec_has_narrative() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert gen.world_spec.narrative is not None


def test_generator_world_spec_config_matches() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=7)
    assert gen.world_spec.config is gen.config


def test_generator_world_spec_is_world_spec() -> None:
    from leadforge.core.models import WorldSpec

    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    assert isinstance(gen.world_spec, WorldSpec)

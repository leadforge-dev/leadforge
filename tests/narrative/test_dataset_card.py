"""Tests for leadforge.narrative.dataset_card."""

from leadforge.api.generator import Generator
from leadforge.core.models import GenerationConfig, WorldSpec
from leadforge.narrative.dataset_card import render_dataset_card
from leadforge.schema.tasks import SplitSpec, TaskManifest, task_manifest_for_config


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
    task = task_manifest_for_config()
    card = render_dataset_card(_make_world_spec(), task_manifest=task)
    assert "closed_won" in card
    assert "90 days" in card


def test_card_renders_custom_primary_task() -> None:
    spec = _make_world_spec(primary_task="churned_within_60_days")
    task = task_manifest_for_config("churned_within_60_days", 60)
    card = render_dataset_card(spec, task_manifest=task)
    assert "`churned_within_60_days`" in card
    assert "converted_within_90_days" not in card


def test_card_renders_custom_label_window_days() -> None:
    spec = _make_world_spec(label_window_days=60)
    task = task_manifest_for_config(label_window_days=60)
    card = render_dataset_card(spec, task_manifest=task)
    assert "60" in card


def test_card_renders_custom_task_and_window() -> None:
    spec = _make_world_spec(primary_task="upgraded_within_30_days", label_window_days=30)
    task = task_manifest_for_config("upgraded_within_30_days", 30)
    card = render_dataset_card(spec, task_manifest=task)
    assert "`upgraded_within_30_days`" in card
    assert "30" in card


def test_card_contains_use_cases() -> None:
    # Card must explain what the dataset is intended for (section heading may vary).
    card = render_dataset_card(_make_world_spec())
    card_lower = card.lower()
    has_use_cases = "use cases" in card_lower
    has_intended = "intended" in card_lower
    assert has_use_cases or has_intended


def test_card_contains_caveats() -> None:
    card = render_dataset_card(_make_world_spec())
    assert "synthetic" in card.lower()


def test_card_no_narrative_shows_stub() -> None:
    spec = WorldSpec(config=GenerationConfig(), narrative=None)
    assert "unavailable" in render_dataset_card(spec).lower()


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
    # Persona information should appear; new format uses human titles alongside role keys.
    gen = Generator.from_recipe("b2b_saas_procurement_v1")
    card = render_dataset_card(gen.world_spec)
    assert "vp_finance" in card  # role key included as machine-readable anchor


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


# ---------------------------------------------------------------------------
# Task manifest threading (issue #38)
# ---------------------------------------------------------------------------


def test_card_uses_task_manifest_description() -> None:
    """When a TaskManifest is provided, its description replaces default prose."""
    spec = _make_world_spec(primary_task="churned_within_60_days", label_window_days=60)
    task = TaskManifest(
        task_id="churned_within_60_days",
        label_column="churned_within_60_days",
        label_window_days=60,
        primary_table="leads",
        split=SplitSpec(train=0.7, valid=0.15, test=0.15),
        description=(
            "A lead is considered churned if a `churn` event is recorded "
            "within 60 days of the snapshot anchor date."
        ),
    )
    card = render_dataset_card(spec, task_manifest=task)
    assert "churned" in card
    assert "churn" in card
    assert "closed_won" not in card


def test_card_default_task_manifest_has_conversion_prose() -> None:
    """Default task manifest produces conversion-specific prose."""
    spec = _make_world_spec()
    task = task_manifest_for_config()
    card = render_dataset_card(spec, task_manifest=task)
    assert "closed_won" in card
    assert "90 days" in card


def test_card_without_task_manifest_uses_generic_fallback() -> None:
    """Without a TaskManifest, the card uses a task-agnostic fallback."""
    spec = _make_world_spec()
    card = render_dataset_card(spec)
    assert "event-derived" in card
    assert "closed_won" not in card


def test_card_task_manifest_empty_description_uses_generic_fallback() -> None:
    """A TaskManifest with empty description falls back to generic prose."""
    spec = _make_world_spec()
    task = TaskManifest(
        task_id="converted_within_90_days",
        label_column="converted_within_90_days",
        label_window_days=90,
        primary_table="leads",
        split=SplitSpec(train=0.7, valid=0.15, test=0.15),
        description="",
    )
    card = render_dataset_card(spec, task_manifest=task)
    assert "event-derived" in card
    assert "closed_won" not in card


def test_card_non_default_task_via_factory_has_generic_prose() -> None:
    """task_manifest_for_config with non-default task produces generic description."""
    spec = _make_world_spec(primary_task="churned_within_60_days", label_window_days=60)
    task = task_manifest_for_config("churned_within_60_days", 60)
    card = render_dataset_card(spec, task_manifest=task)
    assert "`churned_within_60_days`" in card
    assert "60-day" in card
    assert "closed_won" not in card


# ---------------------------------------------------------------------------
# Table inventory
# ---------------------------------------------------------------------------


def test_card_table_inventory_with_counts() -> None:
    """When table_counts is provided, the card renders a row-count table."""
    counts = {"accounts": 1500, "contacts": 4200, "leads": 5000}
    card = render_dataset_card(_make_world_spec(), table_counts=counts)
    assert "| accounts | 1,500 |" in card
    assert "| leads | 5,000 |" in card


def test_card_table_inventory_without_counts() -> None:
    """Without table_counts, the card shows a placeholder."""
    card = render_dataset_card(_make_world_spec())
    assert "not available" in card.lower()


def test_card_table_inventory_empty_dict_renders_empty_table() -> None:
    """An empty dict should render the table header with no rows, not the placeholder."""
    card = render_dataset_card(_make_world_spec(), table_counts={})
    assert "| Table | Rows |" in card
    assert "not available" not in card.lower()


# ---------------------------------------------------------------------------
# Feature categories
# ---------------------------------------------------------------------------


def test_card_feature_categories_rendered() -> None:
    """Feature categories are always rendered from LEAD_SNAPSHOT_FEATURES."""
    card = render_dataset_card(_make_world_spec())
    assert "| Category | Count | Examples |" in card
    card_lower = card.lower()
    assert "account" in card_lower
    assert "engagement" in card_lower
    assert "sales" in card_lower
    assert "target" in card_lower


def test_card_leakage_flagged_columns() -> None:
    """Leakage-flagged columns are listed in the feature categories section."""
    card = render_dataset_card(_make_world_spec())
    assert "`total_touches_all`" in card
    assert "`current_stage`" in card
    # Phrasing may vary; key invariant is that leakage is mentioned with the column names.
    assert "leakage" in card.lower() or "Leakage-flagged" in card

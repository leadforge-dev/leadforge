"""resolve_config carries the lifecycle config fields (LTV-Po.2a)."""

from __future__ import annotations

from leadforge.api.recipes import Recipe
from leadforge.core.enums import DifficultyProfile, ExposureMode


def _recipe(**pop) -> Recipe:
    return Recipe(
        id="tmp_lifecycle",
        title="t",
        vertical="v",
        scheme="lifecycle",
        description="d",
        primary_task="ltv_revenue_365d",
        supported_modes=(ExposureMode.student_public, ExposureMode.research_instructor),
        supported_difficulty=(DifficultyProfile.intro,),
        default_population=dict(pop),
        horizon_days=90,
        label_window_days=None,
        snapshot_day=None,
    )


def test_n_customers_flows_from_default_population() -> None:
    cfg = _recipe(n_customers=2500).resolve_config(seed=1, difficulty="intro")
    assert cfg.n_customers == 2500


def test_n_customers_kwarg_overrides_recipe() -> None:
    cfg = _recipe(n_customers=2500).resolve_config(seed=1, difficulty="intro", n_customers=300)
    assert cfg.n_customers == 300


def test_lifecycle_fields_via_override() -> None:
    cfg = _recipe(n_customers=100).resolve_config(
        seed=1,
        difficulty="intro",
        override={"early_tenure_weeks": 8, "observation_date": "2026-06-01"},
    )
    assert cfg.early_tenure_weeks == 8
    assert cfg.observation_date == "2026-06-01"


def test_defaults_when_recipe_omits_lifecycle_fields() -> None:
    # A lead-scoring-style recipe (no n_customers) → lifecycle fields keep their
    # GenerationConfig defaults; this is why lead-scoring resolution is unchanged.
    cfg = _recipe(n_accounts=10, n_contacts=30, n_leads=30).resolve_config(
        seed=1, difficulty="intro"
    )
    assert cfg.n_customers == 1500  # GenerationConfig default
    assert cfg.forward_windows_days == (90, 365, 730)
    assert cfg.early_tenure_weeks == 4
    assert cfg.observation_date is None

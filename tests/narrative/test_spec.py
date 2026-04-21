"""Tests for leadforge.narrative.spec — NarrativeSpec and sub-models."""

import dataclasses

import pytest

from leadforge.core.exceptions import InvalidRecipeError
from leadforge.narrative.spec import (
    CompanySpec,
    MarketSpec,
    NarrativeSpec,
    ProductSpec,
)

# ---------------------------------------------------------------------------
# Minimal valid payloads
# ---------------------------------------------------------------------------

COMPANY = {
    "name": "Acme Corp",
    "founded_year": 2015,
    "hq_city": "Boston",
    "hq_country": "US",
    "stage": "Series A",
    "employee_range": [50, 120],
}

PRODUCT = {
    "name": "Acme Product",
    "category": "AP Automation",
    "deployment": "cloud_saas",
    "pricing_model": "per_seat_annual",
    "acv_range_usd": [10000, 80000],
    "contract_terms_months": [12, 24],
    "free_trial_available": True,
    "demo_available": True,
}

MARKET = {
    "icp_employee_range": [100, 1000],
    "icp_industries": ["manufacturing", "logistics"],
    "geographies": ["US"],
    "avg_deal_size_usd": 30000,
    "avg_sales_cycle_days": 40,
}

GTM = {
    "channels": ["inbound_marketing", "sdr_outbound"],
    "inbound_share": 0.6,
    "outbound_share": 0.3,
    "partner_share": 0.1,
}

PERSONA = {
    "role": "vp_finance",
    "title_variants": ["VP Finance", "CFO"],
    "decision_authority": "economic_buyer",
    "typical_involvement": "late_stage",
}

FUNNEL_STAGE = {"name": "mql", "label": "Marketing Qualified Lead"}

VALID_NARRATIVE = {
    "company": COMPANY,
    "product": PRODUCT,
    "market": MARKET,
    "gtm_motion": GTM,
    "personas": [PERSONA],
    "funnel_stages": [FUNNEL_STAGE],
}


# ---------------------------------------------------------------------------
# NarrativeSpec.from_dict — happy path
# ---------------------------------------------------------------------------


def test_narrative_spec_roundtrip() -> None:
    spec = NarrativeSpec.from_dict(VALID_NARRATIVE)
    assert spec.company.name == "Acme Corp"
    assert spec.product.name == "Acme Product"
    assert spec.market.avg_deal_size_usd == 30000
    assert spec.gtm_motion.inbound_share == pytest.approx(0.6)
    assert len(spec.personas) == 1
    assert spec.personas[0].role == "vp_finance"
    assert len(spec.funnel_stages) == 1
    assert spec.funnel_stages[0].name == "mql"


def test_narrative_spec_frozen() -> None:
    spec = NarrativeSpec.from_dict(VALID_NARRATIVE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.company = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NarrativeSpec.from_dict — validation errors
# ---------------------------------------------------------------------------


def test_narrative_missing_key_raises() -> None:
    bad = {k: v for k, v in VALID_NARRATIVE.items() if k != "company"}
    with pytest.raises(InvalidRecipeError, match="missing required keys"):
        NarrativeSpec.from_dict(bad)


def test_narrative_personas_not_list_raises() -> None:
    bad = {**VALID_NARRATIVE, "personas": "not_a_list"}
    with pytest.raises(InvalidRecipeError, match="personas"):
        NarrativeSpec.from_dict(bad)


def test_narrative_funnel_not_list_raises() -> None:
    bad = {**VALID_NARRATIVE, "funnel_stages": {"name": "mql"}}
    with pytest.raises(InvalidRecipeError, match="funnel_stages"):
        NarrativeSpec.from_dict(bad)


# ---------------------------------------------------------------------------
# CompanySpec
# ---------------------------------------------------------------------------


def test_company_bool_founded_year_raises() -> None:
    bad = {**COMPANY, "founded_year": True}
    with pytest.raises(InvalidRecipeError, match="founded_year"):
        CompanySpec.from_dict(bad)


def test_company_bad_employee_range_raises() -> None:
    bad = {**COMPANY, "employee_range": [50]}  # wrong length
    with pytest.raises(InvalidRecipeError, match="employee_range"):
        CompanySpec.from_dict(bad)


# ---------------------------------------------------------------------------
# ProductSpec
# ---------------------------------------------------------------------------


def test_product_bad_acv_range_raises() -> None:
    bad = {**PRODUCT, "acv_range_usd": "10000-80000"}
    with pytest.raises(InvalidRecipeError, match="acv_range_usd"):
        ProductSpec.from_dict(bad)


def test_product_bad_contract_terms_raises() -> None:
    bad = {**PRODUCT, "contract_terms_months": [12, "twenty-four"]}
    with pytest.raises(InvalidRecipeError, match="contract_terms_months"):
        ProductSpec.from_dict(bad)


# ---------------------------------------------------------------------------
# MarketSpec
# ---------------------------------------------------------------------------


def test_market_bool_avg_deal_size_raises() -> None:
    bad = {**MARKET, "avg_deal_size_usd": True}
    with pytest.raises(InvalidRecipeError, match="avg_deal_size_usd"):
        MarketSpec.from_dict(bad)


def test_market_zero_sales_cycle_raises() -> None:
    bad = {**MARKET, "avg_sales_cycle_days": 0}
    with pytest.raises(InvalidRecipeError, match="avg_sales_cycle_days"):
        MarketSpec.from_dict(bad)


# ---------------------------------------------------------------------------
# Real recipe round-trip
# ---------------------------------------------------------------------------


def test_real_narrative_yaml_parses() -> None:
    """The shipped narrative.yaml must parse without errors."""
    from leadforge.api.recipes import Recipe
    from leadforge.recipes.registry import load_recipe

    recipe = Recipe.from_dict(load_recipe("b2b_saas_procurement_v1"))
    data = recipe.load_narrative()
    spec = NarrativeSpec.from_dict(data)
    assert spec.company.name == "Veridian Technologies"
    assert len(spec.personas) >= 1
    assert len(spec.funnel_stages) >= 1


def test_real_narrative_tuples_are_immutable() -> None:
    from leadforge.api.recipes import Recipe
    from leadforge.recipes.registry import load_recipe

    recipe = Recipe.from_dict(load_recipe("b2b_saas_procurement_v1"))
    spec = NarrativeSpec.from_dict(recipe.load_narrative())
    assert isinstance(spec.personas, tuple)
    assert isinstance(spec.funnel_stages, tuple)
    assert isinstance(spec.market.icp_industries, tuple)

"""Narrative-driven firmographics for the lifecycle population (LTV-Po.1)."""

from __future__ import annotations

import pytest

from leadforge.narrative.spec import MarketSpec, NarrativeSpec
from leadforge.schemes.lifecycle.population import (
    _GEOGRAPHIES,
    _ICP_INDUSTRIES,
    build_customer_population,
)


def _narrative(industries: tuple[str, ...], geographies: tuple[str, ...]) -> NarrativeSpec:
    # The population builder reads only ``narrative.market``; the other sub-specs
    # are irrelevant here, so they are left as None/empty (never dereferenced).
    market = MarketSpec(
        icp_employee_range=(200, 2000),
        icp_industries=industries,
        geographies=geographies,
        avg_deal_size_usd=25000,
        avg_sales_cycle_days=60,
    )
    return NarrativeSpec(
        company=None,  # type: ignore[arg-type]
        product=None,  # type: ignore[arg-type]
        market=market,
        gtm_motion=None,  # type: ignore[arg-type]
        personas=(),
        funnel_stages=(),
    )


def test_narrative_drives_industries_and_regions() -> None:
    industries = ("Aerospace", "Maritime Logistics")
    geographies = ("Antarctica",)
    pop = build_customer_population(
        80, 7, motif_family="product_led_retention", narrative=_narrative(industries, geographies)
    )
    seen_ind = {a.industry for a in pop.accounts}
    seen_geo = {a.region for a in pop.accounts}
    assert seen_ind <= set(industries)
    assert seen_geo == set(geographies)
    # And they are NOT the built-in defaults.
    assert seen_ind.isdisjoint(set(_ICP_INDUSTRIES))


def test_no_narrative_uses_builtin_defaults() -> None:
    pop = build_customer_population(80, 7, motif_family="product_led_retention", narrative=None)
    assert {a.industry for a in pop.accounts} <= set(_ICP_INDUSTRIES)
    assert {a.region for a in pop.accounts} <= set(_GEOGRAPHIES)


def test_empty_narrative_vocab_rejected() -> None:
    with pytest.raises(ValueError, match="icp_industries and geographies"):
        build_customer_population(10, 1, narrative=_narrative((), ("US",)))


def test_narrative_population_deterministic() -> None:
    nar = _narrative(("A", "B", "C"), ("X", "Y"))
    a = build_customer_population(60, 3, narrative=nar)
    b = build_customer_population(60, 3, narrative=nar)
    assert [r.to_dict() for r in a.accounts] == [r.to_dict() for r in b.accounts]

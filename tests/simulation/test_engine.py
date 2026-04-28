"""Tests for simulation/engine.py and simulation/state.py."""

from __future__ import annotations

import pytest

from leadforge.core.models import GenerationConfig
from leadforge.schema.entities import (
    CustomerRow,
    LeadRow,
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    SubscriptionRow,
    TouchRow,
)
from leadforge.simulation.engine import SimulationResult, _plan_from_acv, simulate_world
from leadforge.simulation.population import build_population
from leadforge.simulation.state import LeadSimState
from leadforge.structure.sampler import sample_hidden_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(seed: int = 42, n_leads: int = 50) -> GenerationConfig:
    """Return a small GenerationConfig suitable for unit tests."""
    return GenerationConfig(seed=seed, n_accounts=20, n_contacts=60, n_leads=n_leads)


def _make_narrative():
    """Return the default recipe narrative via Generator."""
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert gen.world_spec.narrative is not None
    return gen.world_spec.narrative


def _run_sim(seed: int = 42, n_leads: int = 50, motif: str | None = None) -> SimulationResult:
    config = _make_config(seed=seed, n_leads=n_leads)
    narrative = _make_narrative()
    graph = sample_hidden_graph(seed, motif_family_name=motif)
    pop = build_population(config, narrative, graph)
    return simulate_world(config, pop, graph)


# ---------------------------------------------------------------------------
# LeadSimState unit tests
# ---------------------------------------------------------------------------


class TestLeadSimState:
    def test_initial_not_terminal(self) -> None:
        s = LeadSimState("lead_000001", "mql")
        assert not s.is_terminal

    def test_mark_converted(self) -> None:
        s = LeadSimState("lead_000001", "mql")
        s.mark_converted(10)
        assert s.converted
        assert s.conversion_day == 10
        assert s.current_stage == "closed_won"
        assert s.is_terminal

    def test_mark_churned(self) -> None:
        s = LeadSimState("lead_000001", "mql")
        s.mark_churned(5)
        assert s.churned
        assert s.churn_day == 5
        assert s.current_stage == "closed_lost"
        assert s.is_terminal

    def test_advance_stage_resets_dwell(self) -> None:
        s = LeadSimState("lead_000001", "mql")
        s.dwell_days = 7
        s.advance_stage("sal", 7)
        assert s.current_stage == "sal"
        assert s.dwell_days == 0

    def test_advance_stage_records_sql_day(self) -> None:
        s = LeadSimState("lead_000001", "sal")
        s.advance_stage("sql", 12)
        assert s.sql_day == 12

    def test_advance_stage_sql_day_not_overwritten(self) -> None:
        s = LeadSimState("lead_000001", "sql")
        s.sql_day = 5
        # Advancing to a deeper stage should not overwrite sql_day.
        s.advance_stage("demo_scheduled", 8)
        assert s.sql_day == 5


# ---------------------------------------------------------------------------
# SimulationResult structure
# ---------------------------------------------------------------------------


class TestSimulationResultTypes:
    def test_result_contains_correct_types(self) -> None:
        result = _run_sim()
        assert all(isinstance(r, LeadRow) for r in result.leads)
        assert all(isinstance(r, TouchRow) for r in result.touches)
        assert all(isinstance(r, SessionRow) for r in result.sessions)
        assert all(isinstance(r, SalesActivityRow) for r in result.sales_activities)
        assert all(isinstance(r, OpportunityRow) for r in result.opportunities)
        assert all(isinstance(r, CustomerRow) for r in result.customers)
        assert all(isinstance(r, SubscriptionRow) for r in result.subscriptions)

    def test_lead_count_preserved(self) -> None:
        result = _run_sim(n_leads=50)
        assert len(result.leads) == 50

    def test_touches_non_empty(self) -> None:
        # With 50 leads over 90 days, some touches must be emitted.
        result = _run_sim(n_leads=50)
        assert len(result.touches) > 0

    def test_sessions_non_empty(self) -> None:
        result = _run_sim(n_leads=50)
        assert len(result.sessions) > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        r1 = _run_sim(seed=7)
        r2 = _run_sim(seed=7)
        assert len(r1.leads) == len(r2.leads)
        assert len(r1.touches) == len(r2.touches)
        assert len(r1.sessions) == len(r2.sessions)
        assert len(r1.opportunities) == len(r2.opportunities)
        assert [row.converted_within_90_days for row in r1.leads] == [
            row.converted_within_90_days for row in r2.leads
        ]

    def test_different_seeds_differ(self) -> None:
        r1 = _run_sim(seed=1)
        r2 = _run_sim(seed=2)
        # With large enough populations the touch counts should differ.
        assert len(r1.touches) != len(r2.touches) or len(r1.sessions) != len(r2.sessions)


# ---------------------------------------------------------------------------
# Lead outcomes
# ---------------------------------------------------------------------------


class TestLeadOutcomes:
    def test_converted_within_90_days_is_bool(self) -> None:
        result = _run_sim()
        assert all(isinstance(row.converted_within_90_days, bool) for row in result.leads)

    def test_all_leads_are_mql(self) -> None:
        result = _run_sim()
        assert all(row.is_mql for row in result.leads)

    def test_converted_leads_have_timestamp(self) -> None:
        result = _run_sim(n_leads=100)
        for lead in result.leads:
            if lead.converted_within_90_days:
                assert lead.conversion_timestamp is not None
            else:
                assert lead.conversion_timestamp is None

    def test_converted_leads_at_closed_won(self) -> None:
        result = _run_sim(n_leads=100)
        for lead in result.leads:
            if lead.converted_within_90_days:
                assert lead.current_stage == "closed_won"

    def test_some_leads_convert(self) -> None:
        result = _run_sim(n_leads=200)
        n_conv = sum(row.converted_within_90_days for row in result.leads)
        assert n_conv > 0, "Expected at least one conversion in 200-lead sim"

    def test_some_leads_do_not_convert(self) -> None:
        result = _run_sim(n_leads=200)
        n_not_conv = sum(not row.converted_within_90_days for row in result.leads)
        assert n_not_conv > 0, "Expected at least one non-conversion in 200-lead sim"

    def test_sql_leads_are_flagged(self) -> None:
        result = _run_sim(n_leads=100)
        for lead in result.leads:
            if lead.current_stage in {
                "sql",
                "demo_scheduled",
                "demo_completed",
                "proposal_sent",
                "negotiation",
                "closed_won",
            }:
                assert lead.is_sql

    def test_converted_leads_also_sql(self) -> None:
        result = _run_sim(n_leads=100)
        for lead in result.leads:
            if lead.converted_within_90_days:
                assert lead.is_sql


# ---------------------------------------------------------------------------
# Opportunities
# ---------------------------------------------------------------------------


class TestOpportunities:
    def test_sql_leads_have_opportunity(self) -> None:
        result = _run_sim(n_leads=100)
        opp_lead_ids = {o.lead_id for o in result.opportunities}
        for lead in result.leads:
            if lead.is_sql:
                assert lead.lead_id in opp_lead_ids

    def test_non_sql_leads_no_opportunity(self) -> None:
        result = _run_sim(n_leads=100)
        opp_lead_ids = {o.lead_id for o in result.opportunities}
        for lead in result.leads:
            if not lead.is_sql:
                assert lead.lead_id not in opp_lead_ids

    def test_opportunity_acv_positive(self) -> None:
        result = _run_sim(n_leads=100)
        assert all(o.estimated_acv > 0 for o in result.opportunities)

    def test_converted_opportunity_has_close_outcome(self) -> None:
        result = _run_sim(n_leads=100)
        converted_ids = {row.lead_id for row in result.leads if row.converted_within_90_days}
        for opp in result.opportunities:
            if opp.lead_id in converted_ids:
                assert opp.close_outcome == "closed_won"
                assert opp.closed_at is not None

    def test_opportunity_ids_unique(self) -> None:
        result = _run_sim(n_leads=100)
        ids = [o.opportunity_id for o in result.opportunities]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Customers and subscriptions
# ---------------------------------------------------------------------------


class TestCustomersAndSubscriptions:
    def test_customer_per_conversion(self) -> None:
        result = _run_sim(n_leads=100)
        n_conv = sum(row.converted_within_90_days for row in result.leads)
        assert len(result.customers) == n_conv

    def test_subscription_per_customer(self) -> None:
        result = _run_sim(n_leads=100)
        assert len(result.subscriptions) == len(result.customers)

    def test_customer_account_fk(self) -> None:
        config = _make_config(n_leads=50)
        narrative = _make_narrative()
        graph = sample_hidden_graph(42)
        pop = build_population(config, narrative, graph)
        result = simulate_world(config, pop, graph)
        acct_ids = {a.account_id for a in pop.accounts}
        for cust in result.customers:
            assert cust.account_id in acct_ids

    def test_subscription_status_active(self) -> None:
        result = _run_sim(n_leads=100)
        assert all(s.subscription_status == "active" for s in result.subscriptions)

    def test_subscription_plan_valid(self) -> None:
        result = _run_sim(n_leads=100)
        valid_plans = {"starter", "growth", "enterprise"}
        assert all(s.plan_name in valid_plans for s in result.subscriptions)


# ---------------------------------------------------------------------------
# Touch / session / activity integrity
# ---------------------------------------------------------------------------


class TestEventIntegrity:
    def test_touch_lead_fk(self) -> None:
        config = _make_config(n_leads=50)
        narrative = _make_narrative()
        graph = sample_hidden_graph(42)
        pop = build_population(config, narrative, graph)
        result = simulate_world(config, pop, graph)
        lead_ids = {row.lead_id for row in result.leads}
        for touch in result.touches:
            assert touch.lead_id in lead_ids

    def test_session_lead_fk(self) -> None:
        config = _make_config(n_leads=50)
        narrative = _make_narrative()
        graph = sample_hidden_graph(42)
        pop = build_population(config, narrative, graph)
        result = simulate_world(config, pop, graph)
        lead_ids = {row.lead_id for row in result.leads}
        for sess in result.sessions:
            assert sess.lead_id in lead_ids

    def test_activity_rep_id_non_empty(self) -> None:
        result = _run_sim(n_leads=100)
        for act in result.sales_activities:
            assert act.rep_id

    def test_touch_ids_unique(self) -> None:
        result = _run_sim(n_leads=50)
        ids = [t.touch_id for t in result.touches]
        assert len(ids) == len(set(ids))

    def test_session_duration_positive(self) -> None:
        result = _run_sim(n_leads=50)
        assert all(s.session_duration_seconds > 0 for s in result.sessions)

    def test_session_page_views_positive(self) -> None:
        result = _run_sim(n_leads=50)
        assert all(s.page_views > 0 for s in result.sessions)


# ---------------------------------------------------------------------------
# Motif family variation
# ---------------------------------------------------------------------------


class TestMotifVariation:
    @pytest.mark.parametrize(
        "motif",
        [
            "fit_dominant",
            "intent_dominant",
            "sales_execution_sensitive",
            "demo_trial_mediated",
            "buying_committee_friction",
        ],
    )
    def test_all_motifs_complete_without_error(self, motif: str) -> None:
        result = _run_sim(n_leads=30, motif=motif)
        assert len(result.leads) == 30
        # fit_dominant should have higher conversion than buying_committee_friction
        # — don't assert exact ordering since it's stochastic at n=30.

    def test_fit_dominant_higher_conversion_than_friction(self) -> None:
        # Law of large numbers: fit_dominant should convert more than friction.
        fit = _run_sim(seed=99, n_leads=300, motif="fit_dominant")
        friction = _run_sim(seed=99, n_leads=300, motif="buying_committee_friction")
        fit_rate = sum(row.converted_within_90_days for row in fit.leads) / 300
        fric_rate = sum(row.converted_within_90_days for row in friction.leads) / 300
        assert fit_rate > fric_rate


# ---------------------------------------------------------------------------
# _plan_from_acv helper
# ---------------------------------------------------------------------------


class TestPlanFromAcv:
    def test_starter(self) -> None:
        assert _plan_from_acv(10_000) == "starter"
        assert _plan_from_acv(29_999) == "starter"

    def test_growth(self) -> None:
        assert _plan_from_acv(30_000) == "growth"
        assert _plan_from_acv(79_999) == "growth"

    def test_enterprise(self) -> None:
        assert _plan_from_acv(80_000) == "enterprise"
        assert _plan_from_acv(200_000) == "enterprise"

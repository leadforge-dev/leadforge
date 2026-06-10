"""Tests for simulation/engine.py and simulation/state.py."""

from __future__ import annotations

import pytest

from leadforge.core.models import GenerationConfig
from leadforge.core.rng import RNGRoot
from leadforge.schema.entities import (
    CustomerRow,
    LeadRow,
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    SubscriptionRow,
    TouchRow,
)
from leadforge.schemes.lead_scoring.simulation.engine import (
    SimulationResult,
    _plan_from_acv,
    simulate_world,
)
from leadforge.schemes.lead_scoring.simulation.population import build_population
from leadforge.schemes.lead_scoring.simulation.state import LeadSimState
from leadforge.schemes.lead_scoring.structure.sampler import sample_hidden_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    seed: int = 42, n_leads: int = 50, label_window_days: int = 90
) -> GenerationConfig:
    """Return a small GenerationConfig suitable for unit tests."""
    return GenerationConfig(
        seed=seed,
        n_accounts=20,
        n_contacts=60,
        n_leads=n_leads,
        label_window_days=label_window_days,
    )


def _make_narrative():
    """Return the default recipe narrative via Generator."""
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert gen.world_spec.narrative is not None
    return gen.world_spec.narrative


def _run_sim(seed: int = 42, n_leads: int = 50, motif: str | None = None) -> SimulationResult:
    config = _make_config(seed=seed, n_leads=n_leads)
    narrative = _make_narrative()
    graph = sample_hidden_graph(RNGRoot(seed), motif_family_name=motif)
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
        r1 = _run_sim(seed=1, n_leads=200)
        r2 = _run_sim(seed=2, n_leads=200)
        sig1 = [
            (row.lead_id, row.converted_within_90_days, row.current_stage, row.conversion_timestamp)
            for row in r1.leads
        ]
        sig2 = [
            (row.lead_id, row.converted_within_90_days, row.current_stage, row.conversion_timestamp)
            for row in r2.leads
        ]
        assert (
            sig1 != sig2
            or len(r1.touches) != len(r2.touches)
            or len(r1.sessions) != len(r2.sessions)
            or len(r1.opportunities) != len(r2.opportunities)
        )


# ---------------------------------------------------------------------------
# Lead outcomes
# ---------------------------------------------------------------------------


class TestLeadOutcomes:
    def test_converted_within_90_days_is_bool(self) -> None:
        result = _run_sim()
        assert all(isinstance(row.converted_within_90_days, bool) for row in result.leads)

    def test_no_lead_is_initialised_pre_mql(self) -> None:
        """All leads enter the simulation at MQL stage; pre-MQL stages don't
        appear.  ``is_mql`` was removed from the schema in v3 because of
        this invariant — but the invariant itself still holds."""
        result = _run_sim()
        # Stages that would mean "not yet MQL" — none should be present in
        # initial population, nor in final state because the funnel only
        # advances forward.
        pre_mql_stages = {"awareness", "interest"}
        assert not any(row.current_stage in pre_mql_stages for row in result.leads)

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

    def test_most_converted_leads_are_sql(self) -> None:
        """Most converted leads should have reached SQL, but direct conversion
        allows some non-SQL leads to convert too."""
        result = _run_sim(n_leads=500, seed=42)
        converted = [lead for lead in result.leads if lead.converted_within_90_days]
        sql_converted = [lead for lead in converted if lead.is_sql]
        # The vast majority of conversions should still go through SQL.
        assert len(sql_converted) / len(converted) > 0.8


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

    def test_non_sql_non_converted_leads_no_opportunity(self) -> None:
        """Non-SQL leads that did NOT convert should have no opportunity.
        Direct-converted non-SQL leads get an opportunity at conversion time."""
        result = _run_sim(n_leads=100)
        opp_lead_ids = {o.lead_id for o in result.opportunities}
        for lead in result.leads:
            if not lead.is_sql and not lead.converted_within_90_days:
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
        graph = sample_hidden_graph(RNGRoot(42))
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
        graph = sample_hidden_graph(RNGRoot(42))
        pop = build_population(config, narrative, graph)
        result = simulate_world(config, pop, graph)
        lead_ids = {row.lead_id for row in result.leads}
        for touch in result.touches:
            assert touch.lead_id in lead_ids

    def test_session_lead_fk(self) -> None:
        config = _make_config(n_leads=50)
        narrative = _make_narrative()
        graph = sample_hidden_graph(RNGRoot(42))
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


# ---------------------------------------------------------------------------
# Direct conversion (pre-SQL bypass)
# ---------------------------------------------------------------------------


class TestDirectConversion:
    """Verify the rare direct-conversion path for pre-SQL leads."""

    def test_some_non_sql_leads_convert(self) -> None:
        """With enough leads, at least one non-SQL lead should convert."""
        result = _run_sim(seed=42, n_leads=2000)
        non_sql_converted = [
            lead for lead in result.leads if lead.converted_within_90_days and not lead.is_sql
        ]
        assert len(non_sql_converted) > 0, (
            "Expected at least one non-SQL conversion in 2000-lead sim"
        )

    def test_non_sql_conversion_rate_much_lower(self) -> None:
        """Non-SQL conversion rate should be significantly lower than SQL."""
        result = _run_sim(seed=42, n_leads=2000)
        sql_leads = [lead for lead in result.leads if lead.is_sql]
        non_sql_leads = [lead for lead in result.leads if not lead.is_sql]
        assert len(sql_leads) > 0
        assert len(non_sql_leads) > 0

        sql_rate = sum(lead.converted_within_90_days for lead in sql_leads) / len(sql_leads)
        non_sql_rate = sum(lead.converted_within_90_days for lead in non_sql_leads) / len(
            non_sql_leads
        )
        # Non-SQL rate should be at least 5x lower than SQL rate.
        assert non_sql_rate < sql_rate / 5, (
            f"Non-SQL rate {non_sql_rate:.4f} not much lower than SQL rate {sql_rate:.4f}"
        )

    def test_direct_conversion_deterministic(self) -> None:
        """Direct conversion path preserves full determinism."""
        r1 = _run_sim(seed=77, n_leads=500)
        r2 = _run_sim(seed=77, n_leads=500)
        labels1 = [row.converted_within_90_days for row in r1.leads]
        labels2 = [row.converted_within_90_days for row in r2.leads]
        assert labels1 == labels2

    def test_direct_converted_lead_has_opportunity(self) -> None:
        """A direct-converted non-SQL lead should still get an opportunity row."""
        result = _run_sim(seed=42, n_leads=2000)
        opp_lead_ids = {o.lead_id for o in result.opportunities}
        non_sql_converted = [
            lead for lead in result.leads if lead.converted_within_90_days and not lead.is_sql
        ]
        for lead in non_sql_converted:
            assert lead.lead_id in opp_lead_ids

    def test_direct_converted_lead_has_customer_and_subscription(self) -> None:
        """A direct-converted non-SQL lead should get customer + subscription rows."""
        result = _run_sim(seed=42, n_leads=2000)
        cust_opp_ids = {c.opportunity_id for c in result.customers}
        sub_cust_ids = {s.customer_id for s in result.subscriptions}
        opp_by_lead = {o.lead_id: o for o in result.opportunities}
        cust_by_opp = {c.opportunity_id: c for c in result.customers}
        non_sql_converted = [
            lead for lead in result.leads if lead.converted_within_90_days and not lead.is_sql
        ]
        for lead in non_sql_converted:
            opp = opp_by_lead.get(lead.lead_id)
            assert opp is not None, f"No opportunity for direct-converted lead {lead.lead_id}"
            assert opp.opportunity_id in cust_opp_ids
            cust = cust_by_opp[opp.opportunity_id]
            assert cust.customer_id in sub_cust_ids


# ---------------------------------------------------------------------------
# label_window_days affects label derivation
# ---------------------------------------------------------------------------


class TestLabelWindowDays:
    """Verify that label_window_days gates converted_within_90_days."""

    def _run_with_window(
        self, label_window_days: int, seed: int = 42, n_leads: int = 200
    ) -> SimulationResult:
        config = _make_config(seed=seed, n_leads=n_leads, label_window_days=label_window_days)
        narrative = _make_narrative()
        graph = sample_hidden_graph(RNGRoot(seed))
        pop = build_population(config, narrative, graph)
        return simulate_world(config, pop, graph)

    def test_default_90_unchanged(self) -> None:
        """label_window_days=90 (default) matches the old behavior exactly."""
        r_default = _run_sim(seed=42, n_leads=200)
        r_explicit = self._run_with_window(label_window_days=90, seed=42, n_leads=200)
        labels_default = [row.converted_within_90_days for row in r_default.leads]
        labels_explicit = [row.converted_within_90_days for row in r_explicit.leads]
        assert labels_default == labels_explicit

    def test_shorter_window_fewer_conversions(self) -> None:
        """A 30-day window should produce fewer (or equal) conversions than 90."""
        r90 = self._run_with_window(label_window_days=90, seed=42, n_leads=300)
        r30 = self._run_with_window(label_window_days=30, seed=42, n_leads=300)
        conv90 = sum(row.converted_within_90_days for row in r90.leads)
        conv30 = sum(row.converted_within_90_days for row in r30.leads)
        assert conv30 <= conv90
        # With 300 leads over 90 days, there should be *strictly* fewer at 30.
        assert conv30 < conv90

    def test_window_1_almost_no_conversions(self) -> None:
        """With a 1-day window, nearly no leads can convert (need negotiation first)."""
        r = self._run_with_window(label_window_days=1, seed=42, n_leads=200)
        conv = sum(row.converted_within_90_days for row in r.leads)
        # All leads start at mql; reaching negotiation + closed_won in <1 day
        # is essentially impossible.
        assert conv == 0

    def test_late_conversions_excluded(self) -> None:
        """Leads that convert after the label window should not be labeled positive.

        With the inclusive boundary fix (<= label_window_days), a conversion on
        day label_window_days is now correctly labeled positive.
        """
        from datetime import date

        r30 = self._run_with_window(label_window_days=30, seed=42, n_leads=300)
        for lead in r30.leads:
            if lead.converted_within_90_days:
                # If labeled positive, conversion day must be <= label_window_days.
                assert lead.conversion_timestamp is not None
                created = date.fromisoformat(lead.lead_created_at)
                converted = date.fromisoformat(lead.conversion_timestamp)
                day_offset = (converted - created).days
                assert day_offset <= 30, (
                    f"Lead {lead.lead_id} labeled positive but converted on day {day_offset}"
                )

    def test_conversion_timestamp_still_set_outside_window(self) -> None:
        """conversion_timestamp is set for all actual conversions, even outside window."""
        r30 = self._run_with_window(label_window_days=30, seed=42, n_leads=300)
        # Some leads should have conversion_timestamp but label=False
        # (they converted after day 30 but before day 90).
        late_conversions = [
            lead
            for lead in r30.leads
            if lead.conversion_timestamp is not None and not lead.converted_within_90_days
        ]
        # With 300 leads, there should be at least some late conversions.
        assert len(late_conversions) > 0

    def test_event_count_unchanged_by_window(self) -> None:
        """label_window_days does not affect event generation (only the label)."""
        r90 = self._run_with_window(label_window_days=90, seed=42, n_leads=100)
        r30 = self._run_with_window(label_window_days=30, seed=42, n_leads=100)
        # Simulation runs identically; only label derivation differs.
        assert len(r90.touches) == len(r30.touches)
        assert len(r90.sessions) == len(r30.sessions)
        assert len(r90.sales_activities) == len(r30.sales_activities)
        assert len(r90.opportunities) == len(r30.opportunities)

    def test_bundle_round_trip_respects_window(self, tmp_path) -> None:
        """Full pipeline: generate → save → read task split reflects label_window_days."""
        import pandas as pd

        from leadforge.api.generator import Generator

        # primary_task name stays the default; only label_window_days changes.
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            label_window_days=30,
        )
        bundle = gen.generate(n_accounts=20, n_contacts=60, n_leads=200)
        out = str(tmp_path / "bundle")
        bundle.save(out, generation_timestamp="2025-01-01T00:00:00Z")

        task_dir = f"{out}/tasks/converted_within_90_days"
        train = pd.read_parquet(f"{task_dir}/train.parquet")
        conv_30 = int(train["converted_within_90_days"].sum())

        # Compare with default 90-day window.
        gen90 = Generator.from_recipe(
            "b2b_saas_procurement_v1",
            seed=42,
            label_window_days=90,
        )
        bundle90 = gen90.generate(n_accounts=20, n_contacts=60, n_leads=200)
        out90 = str(tmp_path / "bundle90")
        bundle90.save(out90, generation_timestamp="2025-01-01T00:00:00Z")

        task_dir90 = f"{out90}/tasks/converted_within_90_days"
        train90 = pd.read_parquet(f"{task_dir90}/train.parquet")
        conv_90 = int(train90["converted_within_90_days"].sum())

        assert conv_30 <= conv_90

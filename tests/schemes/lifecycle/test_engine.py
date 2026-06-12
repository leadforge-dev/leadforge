"""Tests for the weekly lifecycle simulation engine (LTV-Pk)."""

from datetime import date, timedelta

import pytest

from leadforge.schemes.lifecycle.engine import (
    LifecycleSimulationResult,
    simulate_lifecycle,
)
from leadforge.schemes.lifecycle.hazards import is_renewal_week
from leadforge.schemes.lifecycle.population import (
    LIFECYCLE_MOTIF_FAMILIES,
    build_customer_population,
)

_POP_SEED = 11
_SIM_SEED = 99
_N = 150


@pytest.fixture(scope="module")
def population():
    return build_customer_population(_N, _POP_SEED, motif_family="product_led_retention")


@pytest.fixture(scope="module")
def sim(population):
    return simulate_lifecycle(population, _SIM_SEED)


# ---------------------------------------------------------------------------
# Shape + validation
# ---------------------------------------------------------------------------


def test_returns_result_type(sim) -> None:
    assert isinstance(sim, LifecycleSimulationResult)


def test_one_subscription_per_customer(population, sim) -> None:
    assert len(sim.subscriptions) == len(population.customers)
    assert {s.customer_id for s in sim.subscriptions} == {
        c.customer_id for c in population.customers
    }


def test_rejects_population_without_motif(population) -> None:
    import dataclasses

    broken = dataclasses.replace(population, motif_family="")
    with pytest.raises(ValueError, match="motif_family"):
        simulate_lifecycle(broken, _SIM_SEED)


def test_result_records_simulation_horizon(population, sim) -> None:
    # Downstream builders verify target-window coverage off these fields.
    assert sim.forward_window_days == 730
    assert sim.early_tenure_weeks == 4
    custom = simulate_lifecycle(population, _SIM_SEED, forward_window_days=90)
    assert custom.forward_window_days == 90


def test_rejects_bad_windows(population) -> None:
    with pytest.raises(ValueError, match="forward_window_days"):
        simulate_lifecycle(population, _SIM_SEED, forward_window_days=0)
    with pytest.raises(ValueError, match="early_tenure_weeks"):
        simulate_lifecycle(population, _SIM_SEED, early_tenure_weeks=-1)


# ---------------------------------------------------------------------------
# Determinism + per-customer independence
# ---------------------------------------------------------------------------


def test_deterministic_under_same_seeds(population) -> None:
    a = simulate_lifecycle(population, _SIM_SEED)
    b = simulate_lifecycle(population, _SIM_SEED)
    assert [s.to_dict() for s in a.subscriptions] == [s.to_dict() for s in b.subscriptions]
    assert [e.to_dict() for e in a.subscription_events] == [
        e.to_dict() for e in b.subscription_events
    ]
    assert [i.to_dict() for i in a.invoices] == [i.to_dict() for i in b.invoices]
    assert [h.to_dict() for h in a.health_signals] == [h.to_dict() for h in b.health_signals]


def test_different_sim_seed_changes_outcomes(population) -> None:
    a = simulate_lifecycle(population, 1)
    b = simulate_lifecycle(population, 2)
    assert [s.subscription_status for s in a.subscriptions] != [
        s.subscription_status for s in b.subscriptions
    ]


def test_per_customer_trajectories_independent_of_other_customers(population) -> None:
    """Per-customer RNG substreams: a customer's trajectory is invariant to the
    rest of the population (same customer_id + latents + seed → same draws)."""
    import dataclasses

    full = simulate_lifecycle(population, _SIM_SEED)
    target = population.customers[7]
    # Re-simulate with ONLY this customer present.
    solo_pop = dataclasses.replace(population, customers=[target])
    solo = simulate_lifecycle(solo_pop, _SIM_SEED)

    full_events = [
        (e.event_type, e.event_timestamp, e.mrr_before, e.mrr_after)
        for e in full.subscription_events
        if e.customer_id == target.customer_id
    ]
    solo_events = [
        (e.event_type, e.event_timestamp, e.mrr_before, e.mrr_after)
        for e in solo.subscription_events
    ]
    assert full_events == solo_events


# ---------------------------------------------------------------------------
# Cadences + full-window coverage
# ---------------------------------------------------------------------------


def test_weekly_health_cadence(population, sim) -> None:
    # Every active week emits exactly one health signal: consecutive weekly dates.
    by_cust: dict[str, list[str]] = {}
    for h in sim.health_signals:
        by_cust.setdefault(h.customer_id, []).append(h.period_start)
    for cust_id, dates in by_cust.items():
        parsed = [date.fromisoformat(d) for d in dates]
        for prev, nxt in zip(parsed, parsed[1:], strict=False):
            assert (nxt - prev).days == 7, f"{cust_id}: gap {prev} → {nxt}"


def test_monthly_invoice_cadence(population, sim) -> None:
    # ~12 invoices per 52 active weeks (one per month boundary).
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    subs = {s.customer_id: s for s in sim.subscriptions}
    inv_by_cust: dict[str, int] = {}
    for inv in sim.invoices:
        inv_by_cust[inv.customer_id] = inv_by_cust.get(inv.customer_id, 0) + 1
    for cust_id, n_inv in inv_by_cust.items():
        sub = subs[cust_id]
        end = (
            date.fromisoformat(sub.churn_at)
            if sub.churn_at
            else max(
                date.fromisoformat(h.period_start)
                for h in sim.health_signals
                if h.customer_id == cust_id
            )
        )
        active_weeks = (end - starts[cust_id]).days // 7 + 1
        expected_months = int(active_weeks / (52 / 12)) + 1
        assert abs(n_inv - expected_months) <= 1, (
            f"{cust_id}: {n_inv} invoices over {active_weeks} active weeks"
        )


def test_nps_quarterly_only(sim) -> None:
    starts: dict[str, date] = {}
    for h in sim.health_signals:
        starts.setdefault(h.customer_id, date.fromisoformat(h.period_start))
    for h in sim.health_signals:
        week = (date.fromisoformat(h.period_start) - starts[h.customer_id]).days // 7
        if h.nps_score is not None:
            assert week > 0, f"nps at week {week}"
            assert week % 13 == 0, f"nps at non-quarterly week {week}"
            assert 0 <= h.nps_score <= 10
        else:
            assert week == 0 or week % 13 != 0


def test_active_customers_simulated_through_full_window(population, sim) -> None:
    # D6: every still-active customer has health coverage through obs + 730d.
    obs = date.fromisoformat(population.observation_date)
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    min_end = {
        cid: max(obs, start + timedelta(weeks=4)) + timedelta(days=730)
        for cid, start in starts.items()
    }
    last_signal: dict[str, date] = {}
    for h in sim.health_signals:
        d = date.fromisoformat(h.period_start)
        if h.customer_id not in last_signal or d > last_signal[h.customer_id]:
            last_signal[h.customer_id] = d
    for sub in sim.subscriptions:
        if sub.subscription_status == "active":
            assert last_signal[sub.customer_id] >= min_end[sub.customer_id] - timedelta(days=7), (
                f"{sub.customer_id} active but coverage ends {last_signal[sub.customer_id]}"
            )


# ---------------------------------------------------------------------------
# Event + terminal-state consistency
# ---------------------------------------------------------------------------


def test_event_fk_integrity(population, sim) -> None:
    cust_ids = {c.customer_id for c in population.customers}
    sub_ids = {s.subscription_id for s in sim.subscriptions}
    for e in sim.subscription_events:
        assert e.customer_id in cust_ids
        assert e.subscription_id in sub_ids


def test_churned_subscriptions_consistent(sim) -> None:
    churn_events = {e.customer_id for e in sim.subscription_events if e.event_type == "churn"}
    for sub in sim.subscriptions:
        if sub.subscription_status == "churned":
            assert sub.churn_at is not None
            assert sub.subscription_end_at == sub.churn_at
            assert sub.churn_reason in ("voluntary", "non_renewal", "payment_failure")
            assert sub.customer_id in churn_events
        else:
            assert sub.churn_at is None
            assert sub.churn_reason is None
            assert sub.customer_id not in churn_events


def test_no_events_after_churn(sim) -> None:
    churn_date: dict[str, str] = {}
    for e in sim.subscription_events:
        if e.event_type == "churn":
            churn_date[e.customer_id] = e.event_timestamp
    for e in sim.subscription_events:
        if e.customer_id in churn_date:
            assert e.event_timestamp <= churn_date[e.customer_id]
    for h in sim.health_signals:
        if h.customer_id in churn_date:
            assert h.period_start <= churn_date[h.customer_id]


def test_mrr_chain_consistency(population, sim) -> None:
    initial = {c.customer_id: c.initial_mrr for c in population.customers}
    expansions: dict[str, list] = {}
    for e in sim.subscription_events:
        if e.event_type == "expansion":
            expansions.setdefault(e.customer_id, []).append(e)
            assert e.mrr_after > e.mrr_before
    for sub in sim.subscriptions:
        chain = expansions.get(sub.customer_id, [])
        mrr = initial[sub.customer_id]
        for e in chain:
            assert e.mrr_before == mrr
            mrr = e.mrr_after
        assert sub.current_mrr == mrr
        assert sub.expansion_count == len(chain)


def test_renewal_events_only_on_anniversaries(population, sim) -> None:
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    terms = {c.customer_id: c.contract_term_months for c in population.customers}
    renewal_counts: dict[str, int] = {}
    for e in sim.subscription_events:
        if e.event_type == "renewal":
            week = (date.fromisoformat(e.event_timestamp) - starts[e.customer_id]).days // 7
            assert is_renewal_week(week, terms[e.customer_id]), (
                f"renewal at non-anniversary week {week} (term {terms[e.customer_id]}mo)"
            )
            renewal_counts[e.customer_id] = renewal_counts.get(e.customer_id, 0) + 1
    for sub in sim.subscriptions:
        assert sub.renewal_count == renewal_counts.get(sub.customer_id, 0)


def test_payment_failures_resolve_or_censor(population, sim) -> None:
    # Every failed invoice ends as recovered / written_off, unless the customer
    # churned (or the window ended) before the dunning period elapsed.
    churned = {s.customer_id: s for s in sim.subscriptions if s.churn_at}
    for inv in sim.invoices:
        assert inv.payment_status in ("paid", "failed", "recovered", "written_off")
        if inv.payment_status == "written_off":
            sub = churned.get(inv.customer_id)
            assert sub is not None
            assert sub.churn_reason == "payment_failure"


def test_written_off_customers_churn_with_payment_reason(sim) -> None:
    wo_customers = {i.customer_id for i in sim.invoices if i.payment_status == "written_off"}
    reasons = {s.customer_id: s.churn_reason for s in sim.subscriptions}
    for cid in wo_customers:
        assert reasons[cid] == "payment_failure"


def test_account_latents_influence_outcomes(population) -> None:
    """Regression: account latents must NOT be shadowed by customer latents.

    The merge blends the shared traits 50/50 (account-level random effect), so
    swinging an account's latent_budget_stability between the extremes must
    change its customers' simulated trajectories.
    """
    import copy

    lo_pop = copy.deepcopy(population)
    hi_pop = copy.deepcopy(population)
    for traits in lo_pop.latent_state.account_latents.values():
        traits["latent_budget_stability"] = 0.0
    for traits in hi_pop.latent_state.account_latents.values():
        traits["latent_budget_stability"] = 1.0

    lo_sim = simulate_lifecycle(lo_pop, _SIM_SEED)
    hi_sim = simulate_lifecycle(hi_pop, _SIM_SEED)
    lo_failed = sum(1 for i in lo_sim.invoices if i.payment_status != "paid")
    hi_failed = sum(1 for i in hi_sim.invoices if i.payment_status != "paid")
    assert lo_failed > hi_failed, (
        f"account-level budget stability had no effect: lo={lo_failed}, hi={hi_failed}"
    )


def test_dangling_failed_invoices_are_censoring_only(population, sim) -> None:
    """An invoice may end at status 'failed' only when the customer churned for
    another reason mid-dunning or the simulation window ended first — never
    because a second failure was silently dropped while one was pending."""
    from leadforge.schemes.lifecycle.mechanisms import assign_lifecycle_mechanisms

    dunning_weeks = assign_lifecycle_mechanisms(
        population.motif_family
    ).payment_failure.dunning_weeks
    obs = date.fromisoformat(population.observation_date)
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    subs = {s.customer_id: s for s in sim.subscriptions}
    for inv in sim.invoices:
        if inv.payment_status != "failed":
            continue
        inv_date = date.fromisoformat(inv.invoice_date)
        resolution_due = inv_date + timedelta(weeks=dunning_weeks)
        sub = subs[inv.customer_id]
        window_end = max(obs, starts[inv.customer_id] + timedelta(weeks=4)) + timedelta(days=730)
        churned_first = (
            sub.churn_at is not None and date.fromisoformat(sub.churn_at) <= resolution_due
        )
        censored = resolution_due > window_end
        assert churned_first or censored, (
            f"invoice {inv.invoice_id} dangling 'failed' without churn/censoring"
        )


def test_recorded_depth_is_round_tripped(sim) -> None:
    # The stored observable is the exact value the expansion hazard consumed.
    for h in sim.health_signals:
        assert h.feature_depth_score == round(h.feature_depth_score, 4)


# ---------------------------------------------------------------------------
# Calibration: simulated first-year churn per motif (engine-calibrated bands)
# ---------------------------------------------------------------------------

_CHURN_BANDS = {
    "product_led_retention": (0.13, 0.27),
    "relationship_led_retention": (0.18, 0.33),
    "expansion_led_growth": (0.09, 0.23),
    "payment_fragile": (0.29, 0.45),
    "churner_dominated": (0.34, 0.50),
}


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_first_year_churn_within_calibrated_band(motif: str) -> None:
    pop = build_customer_population(600, _POP_SEED, motif_family=motif)
    result = simulate_lifecycle(pop, _SIM_SEED)
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in pop.customers}
    yr1 = sum(
        1
        for s in result.subscriptions
        if s.churn_at and (date.fromisoformat(s.churn_at) - starts[s.customer_id]).days // 7 <= 52
    )
    rate = yr1 / len(pop.customers)
    lo, hi = _CHURN_BANDS[motif]
    assert lo <= rate <= hi, f"{motif}: year-1 churn {rate:.1%} outside [{lo:.0%}, {hi:.0%}]"


def test_expansion_world_expands_most() -> None:
    counts = {}
    for motif in ("expansion_led_growth", "churner_dominated"):
        pop = build_customer_population(400, _POP_SEED, motif_family=motif)
        result = simulate_lifecycle(pop, _SIM_SEED)
        counts[motif] = sum(1 for e in result.subscription_events if e.event_type == "expansion")
    assert counts["expansion_led_growth"] > 2 * counts["churner_dominated"]


def test_most_customers_active_at_observation_date(population, sim) -> None:
    # Starts are within ~56 weeks of obs and annual churn ~20%, so a clear
    # majority must still be active at the observation date.
    obs = date.fromisoformat(population.observation_date)
    active_at_obs = sum(
        1 for s in sim.subscriptions if s.churn_at is None or date.fromisoformat(s.churn_at) > obs
    )
    assert active_at_obs / len(sim.subscriptions) > 0.6

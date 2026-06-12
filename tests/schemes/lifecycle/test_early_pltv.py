"""Tests for the tenure-anchored early-pLTV snapshot builder (LTV-Pm)."""

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from leadforge.core.models import DifficultyParams
from leadforge.schemes.lifecycle.engine import (
    LifecycleSimulationResult,
    simulate_lifecycle,
)
from leadforge.schemes.lifecycle.features import CUSTOMER_SNAPSHOT_FEATURES
from leadforge.schemes.lifecycle.population import build_customer_population
from leadforge.schemes.lifecycle.snapshots import (
    DEFAULT_EARLY_TENURE_WEEKS,
    FORWARD_WINDOWS_DAYS,
    build_customer_snapshot,
    build_early_pltv_snapshot,
)

_POP_SEED = 11
_SIM_SEED = 99
_N = 200
_ET = DEFAULT_EARLY_TENURE_WEEKS  # 4

_FEATURE_COLS = [
    f.name for f in CUSTOMER_SNAPSHOT_FEATURES if not f.is_target and not f.leakage_risk
]
_TARGET_COLS = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES if f.is_target]


@pytest.fixture(scope="module")
def population():
    return build_customer_population(_N, _POP_SEED, motif_family="expansion_led_growth")


@pytest.fixture(scope="module")
def sim(population):
    return simulate_lifecycle(population, _SIM_SEED)


@pytest.fixture(scope="module")
def early(population, sim):
    return build_early_pltv_snapshot(population, sim, early_tenure_weeks=_ET)


def _cutoff_for(customer) -> date:
    return date.fromisoformat(customer.customer_start_at) + timedelta(weeks=_ET)


# ---------------------------------------------------------------------------
# Shape + the defining tenure-anchored property
# ---------------------------------------------------------------------------


def test_columns_and_dtypes_match_catalog(early) -> None:
    assert list(early.columns) == [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        assert str(early[f.name].dtype) == f.dtype, f.name


def test_tenure_is_constant_at_anchor(early) -> None:
    # The defining property of the regime: every row observed at the same tenure.
    assert set(early["tenure_weeks"].unique()) == {_ET}


@pytest.mark.parametrize("pop_seed", [1, 7, 42])
def test_structurally_degenerate_columns_at_short_anchor(pop_seed: int) -> None:
    """Pin the columns that are dead by *construction* at a sub-13-week anchor,
    so the LTV-Pp validation harness has a tracked exemption list and a future
    change that revives one of them forces a conscious update here.

    These are cadence consequences, not seed accidents: first renewal at week
    52, first NPS at week 13, tenure fixed at the anchor.
    """
    pop = build_customer_population(250, pop_seed, motif_family="payment_fragile")
    sim = simulate_lifecycle(pop, pop_seed * 2 + 1)
    snap = build_early_pltv_snapshot(pop, sim, early_tenure_weeks=_ET)

    assert snap["tenure_weeks"].nunique(dropna=True) == 1  # constant = anchor
    assert set(snap["renewal_count"].unique()) == {0}  # first renewal at week 52
    assert snap["last_nps_score"].isna().all()  # first NPS at week 13


def test_deterministic(population, sim, early) -> None:
    again = build_early_pltv_snapshot(population, sim, early_tenure_weeks=_ET)
    pd.testing.assert_frame_equal(early, again)


# ---------------------------------------------------------------------------
# Per-customer cutoff correctness + eligibility cohort
# ---------------------------------------------------------------------------


def test_eligibility_is_survival_to_anchor(population, sim, early) -> None:
    """Included iff the customer did not churn at or before start + anchor."""
    churn = {s.customer_id: s.churn_at for s in sim.subscriptions}
    expected = {
        c.customer_id
        for c in population.customers
        if churn[c.customer_id] is None or date.fromisoformat(churn[c.customer_id]) > _cutoff_for(c)
    }
    assert set(early["customer_id"]) == expected


def test_onboarding_churners_excluded(population, sim, early) -> None:
    churn = {s.customer_id: s.churn_at for s in sim.subscriptions}
    onboarding_churners = {
        c.customer_id
        for c in population.customers
        if churn[c.customer_id] is not None
        and date.fromisoformat(churn[c.customer_id]) <= _cutoff_for(c)
    }
    assert onboarding_churners, "fixture should have some onboarding churn"
    assert onboarding_churners.isdisjoint(set(early["customer_id"]))


def test_cohort_differs_from_calendar_regime(population, sim, early) -> None:
    """The early cohort keeps customers who churned *after* their tenure anchor
    but before the calendar observation_date — they are cold-start customers
    with a real (often low) forward value, dropped by the calendar regime."""
    cal = build_customer_snapshot(population, sim)
    obs = date.fromisoformat(population.observation_date)
    churn = {s.customer_id: s.churn_at for s in sim.subscriptions}
    early_only_expected = {
        c.customer_id
        for c in population.customers
        if churn[c.customer_id] is not None
        and _cutoff_for(c) < date.fromisoformat(churn[c.customer_id]) <= obs
    }
    early_ids, cal_ids = set(early["customer_id"]), set(cal["customer_id"])
    assert early_only_expected, "fixture should have post-anchor pre-obs churn"
    assert early_only_expected == early_ids - cal_ids


def test_late_starter_cutoff_may_exceed_observation_date(population, sim, early) -> None:
    obs = date.fromisoformat(population.observation_date)
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    anchored_after_obs = [
        cid for cid in early["customer_id"] if starts[cid] + timedelta(weeks=_ET) > obs
    ]
    # Valid because each customer's forward windows are fully simulated relative
    # to its own start (engine D6), not the calendar anchor — so the builder
    # does not require the tenure cutoff to fall on or before observation_date.
    assert anchored_after_obs


# ---------------------------------------------------------------------------
# Short-tenure sparsity (the cold-start signal)
# ---------------------------------------------------------------------------


def test_nps_entirely_null_before_first_survey(early) -> None:
    # First quarterly NPS lands at week 13; at a 4-week anchor nobody has one.
    assert early["last_nps_score"].isna().all()


def test_health_aggregates_use_only_pre_anchor_signals(population, sim, early) -> None:
    for _, row in early.head(15).iterrows():
        customer = next(c for c in population.customers if c.customer_id == row["customer_id"])
        cutoff = _cutoff_for(customer)
        signals = [
            h
            for h in sim.health_signals
            if h.customer_id == row["customer_id"] and date.fromisoformat(h.period_start) <= cutoff
        ]
        assert signals
        assert row["avg_active_users_l12w"] == pytest.approx(
            sum(h.active_users for h in signals) / len(signals)
        )


# ---------------------------------------------------------------------------
# Snapshot safety: features see nothing after each customer's own cutoff
# ---------------------------------------------------------------------------


def test_features_identical_under_per_customer_censoring(population, sim, early) -> None:
    """Delete every event after each customer's own tenure cutoff and rebuild;
    non-target, non-trap features must be unchanged. Any feature that moves
    leaks across the (per-customer) anchor."""
    cutoff_iso = {c.customer_id: _cutoff_for(c).isoformat() for c in population.customers}
    censored = LifecycleSimulationResult(
        subscriptions=sim.subscriptions,
        subscription_events=[
            e for e in sim.subscription_events if e.event_timestamp <= cutoff_iso[e.customer_id]
        ],
        health_signals=[
            h for h in sim.health_signals if h.period_start <= cutoff_iso[h.customer_id]
        ],
        invoices=[i for i in sim.invoices if i.invoice_date <= cutoff_iso[i.customer_id]],
        forward_window_days=sim.forward_window_days,
        early_tenure_weeks=sim.early_tenure_weeks,
    )
    rebuilt = build_early_pltv_snapshot(population, censored, early_tenure_weeks=_ET)
    pd.testing.assert_frame_equal(early[_FEATURE_COLS], rebuilt[_FEATURE_COLS])


# ---------------------------------------------------------------------------
# Targets recomputed off the tenure anchor
# ---------------------------------------------------------------------------


def test_ltv_targets_match_invoice_table_per_customer_cutoff(population, sim, early) -> None:
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    for _, row in early.iterrows():
        cutoff = starts[row["customer_id"]] + timedelta(weeks=_ET)
        for window in FORWARD_WINDOWS_DAYS:
            bound = cutoff + timedelta(days=window)
            expected = sum(
                i.amount_usd
                for i in sim.invoices
                if i.customer_id == row["customer_id"]
                and i.payment_status in ("paid", "recovered")
                and cutoff < date.fromisoformat(i.invoice_date) <= bound
            )
            assert row[f"ltv_revenue_{window}d"] == float(expected)


def test_ltv_windows_monotone(early) -> None:
    assert (early["ltv_revenue_90d"] <= early["ltv_revenue_365d"]).all()
    assert (early["ltv_revenue_365d"] <= early["ltv_revenue_730d"]).all()


def test_targets_are_right_skewed(early) -> None:
    for window in FORWARD_WINDOWS_DAYS:
        col = early[f"ltv_revenue_{window}d"]
        assert (col >= 0).all()
        assert col.mean() > col.median()


def test_trap_diverges_strongly_in_early_regime(early) -> None:
    """The mrr_change_full_period trap is *more* leaky here than in the calendar
    regime: at a 4-week anchor almost no expansion has happened, so the valid
    mrr_change_at_snapshot is ~0 while the trap captures the whole future
    expansion path that drives the targets."""
    valid_zero = (early["mrr_change_at_snapshot"] == 0).mean()
    diverges = (early["mrr_change_full_period"] != early["mrr_change_at_snapshot"]).mean()
    assert valid_zero > 0.8  # cold start: little expansion yet
    assert diverges > 0.10


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_nonpositive_anchor(population, sim) -> None:
    with pytest.raises(ValueError, match="early_tenure_weeks must be >= 1"):
        build_early_pltv_snapshot(population, sim, early_tenure_weeks=0)


def test_rejects_anchor_beyond_simulated_tenure(population, sim) -> None:
    with pytest.raises(ValueError, match="exceeds the sim's recorded"):
        build_early_pltv_snapshot(population, sim, early_tenure_weeks=sim.early_tenure_weeks + 1)


def test_rejects_short_forward_window(population, sim) -> None:
    short = replace(sim, forward_window_days=365)
    with pytest.raises(ValueError, match="forward_window_days"):
        build_early_pltv_snapshot(population, short)


def test_rejects_population_sim_mismatch(population, sim) -> None:
    mismatched = replace(sim, subscriptions=sim.subscriptions[1:])
    with pytest.raises(ValueError, match="population/sim mismatch"):
        build_early_pltv_snapshot(population, mismatched)


def test_rejects_missing_observation_date(population, sim) -> None:
    broken = replace(population, observation_date="")
    with pytest.raises(ValueError, match="observation_date"):
        build_early_pltv_snapshot(broken, sim)


# ---------------------------------------------------------------------------
# Distortions reuse the shared machinery (targets/trap stay intact)
# ---------------------------------------------------------------------------


def test_distortions_leave_targets_and_trap_intact(population, sim, early) -> None:
    params = DifficultyParams(
        signal_strength=1.0,
        noise_scale=0.5,
        missing_rate=0.3,
        outlier_rate=0.02,
        conversion_rate_lo=0.02,
        conversion_rate_hi=0.4,
        committee_friction=0.5,
    )
    distorted = build_early_pltv_snapshot(
        population, sim, early_tenure_weeks=_ET, difficulty_params=params, seed=7
    )
    pd.testing.assert_frame_equal(distorted[_TARGET_COLS], early[_TARGET_COLS])
    pd.testing.assert_series_equal(
        distorted["mrr_change_full_period"], early["mrr_change_full_period"]
    )

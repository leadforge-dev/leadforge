"""Tests for the calendar-anchored customer snapshot builder (LTV-Pl)."""

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
from leadforge.schemes.lifecycle.hazards import is_renewal_week
from leadforge.schemes.lifecycle.population import build_customer_population
from leadforge.schemes.lifecycle.snapshots import (
    CHURN_WINDOW_DAYS,
    FORWARD_WINDOWS_DAYS,
    build_customer_snapshot,
)

_POP_SEED = 11
_SIM_SEED = 99
_N = 150

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
def snapshot(population, sim):
    return build_customer_snapshot(population, sim)


def _difficulty_params(**overrides) -> DifficultyParams:
    defaults = {
        "signal_strength": 1.0,
        "noise_scale": 0.3,
        "missing_rate": 0.10,
        "outlier_rate": 0.02,
        "conversion_rate_lo": 0.02,
        "conversion_rate_hi": 0.4,
        "committee_friction": 0.5,
    }
    defaults.update(overrides)
    return DifficultyParams(**defaults)


# ---------------------------------------------------------------------------
# Shape + schema
# ---------------------------------------------------------------------------


def test_one_row_per_active_at_cutoff_customer(population, sim, snapshot) -> None:
    cutoff = population.observation_date
    active = {s.customer_id for s in sim.subscriptions if s.churn_at is None or s.churn_at > cutoff}
    assert set(snapshot["customer_id"]) == active
    assert len(snapshot) == len(active)


def test_churned_before_cutoff_excluded(population, sim, snapshot) -> None:
    cutoff = population.observation_date
    churned_early = {
        s.customer_id for s in sim.subscriptions if s.churn_at is not None and s.churn_at <= cutoff
    }
    assert churned_early, "fixture world should have some pre-cutoff churn"
    assert churned_early.isdisjoint(set(snapshot["customer_id"]))


def test_columns_match_catalog_order(snapshot) -> None:
    assert list(snapshot.columns) == [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]


def test_dtypes_match_catalog(snapshot) -> None:
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        assert str(snapshot[f.name].dtype) == f.dtype, f.name


def test_deterministic(population, sim, snapshot) -> None:
    again = build_customer_snapshot(population, sim)
    pd.testing.assert_frame_equal(snapshot, again)


def test_empty_when_cutoff_precedes_all_starts(population, sim) -> None:
    earliest = min(date.fromisoformat(c.customer_start_at) for c in population.customers)
    snap = build_customer_snapshot(population, sim, cutoff=earliest - timedelta(days=1))
    assert len(snap) == 0
    assert list(snap.columns) == [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_cutoff_after_observation_date(population, sim) -> None:
    late = date.fromisoformat(population.observation_date) + timedelta(days=1)
    with pytest.raises(ValueError, match="censored"):
        build_customer_snapshot(population, sim, cutoff=late)


def test_rejects_population_without_observation_date(population, sim) -> None:
    broken = replace(population, observation_date="")
    with pytest.raises(ValueError, match="observation_date"):
        build_customer_snapshot(broken, sim)


# ---------------------------------------------------------------------------
# Snapshot safety: features must not see past the cutoff
# ---------------------------------------------------------------------------


def test_features_identical_under_post_cutoff_censoring(population, sim, snapshot) -> None:
    """Every non-target, non-trap column must be reproducible from a sim
    result truncated at the cutoff — if deleting all post-cutoff events
    changes a feature, that feature leaks."""
    cutoff = population.observation_date
    censored = LifecycleSimulationResult(
        subscriptions=sim.subscriptions,
        subscription_events=[e for e in sim.subscription_events if e.event_timestamp <= cutoff],
        health_signals=[h for h in sim.health_signals if h.period_start <= cutoff],
        invoices=[i for i in sim.invoices if i.invoice_date <= cutoff],
    )
    censored_snap = build_customer_snapshot(population, censored)
    pd.testing.assert_frame_equal(snapshot[_FEATURE_COLS], censored_snap[_FEATURE_COLS])


# ---------------------------------------------------------------------------
# Feature derivations
# ---------------------------------------------------------------------------


def test_mrr_chain_consistency(snapshot) -> None:
    assert (
        snapshot["current_mrr"] - snapshot["initial_mrr"] == snapshot["mrr_change_at_snapshot"]
    ).all()
    no_expansion = snapshot["expansion_count"] == 0
    assert (snapshot.loc[no_expansion, "mrr_change_at_snapshot"] == 0).all()


def test_event_counts_match_event_table(population, sim, snapshot) -> None:
    cutoff = population.observation_date
    for _, row in snapshot.head(20).iterrows():
        events = [
            e
            for e in sim.subscription_events
            if e.customer_id == row["customer_id"] and e.event_timestamp <= cutoff
        ]
        by_type = {
            t: sum(1 for e in events if e.event_type == t) for t in {e.event_type for e in events}
        }
        assert row["expansion_count"] == by_type.get("expansion", 0)
        assert row["renewal_count"] == by_type.get("renewal", 0)
        assert row["payment_failure_count"] == by_type.get("payment_failure", 0)


def test_tenure_weeks(population, snapshot) -> None:
    cutoff = date.fromisoformat(population.observation_date)
    starts = {c.customer_id: date.fromisoformat(c.customer_start_at) for c in population.customers}
    for _, row in snapshot.iterrows():
        assert row["tenure_weeks"] == (cutoff - starts[row["customer_id"]]).days // 7


def test_weeks_to_next_renewal_lands_on_renewal_week(snapshot) -> None:
    for _, row in snapshot.iterrows():
        weeks_to = int(row["weeks_to_next_renewal"])
        assert weeks_to >= 1
        assert is_renewal_week(
            int(row["tenure_weeks"]) + weeks_to, int(row["contract_term_months"])
        )


def test_health_aggregates_match_signal_table(population, sim, snapshot) -> None:
    cutoff = date.fromisoformat(population.observation_date)
    window_start = cutoff - timedelta(weeks=12)
    for _, row in snapshot.head(10).iterrows():
        in_window = [
            h
            for h in sim.health_signals
            if h.customer_id == row["customer_id"]
            and window_start < date.fromisoformat(h.period_start) <= cutoff
        ]
        assert in_window, "active customer must have signals in the trailing window"
        assert row["avg_active_users_l12w"] == pytest.approx(
            sum(h.active_users for h in in_window) / len(in_window)
        )
        assert row["avg_feature_depth_l12w"] == pytest.approx(
            sum(h.feature_depth_score for h in in_window) / len(in_window)
        )
        assert row["support_ticket_count_l12w"] == sum(h.support_tickets for h in in_window)


def test_last_nps_is_latest_response_at_or_before_cutoff(population, sim, snapshot) -> None:
    cutoff = population.observation_date
    nps_history: dict[str, int] = {}
    for h in sim.health_signals:  # chronological per customer
        if h.nps_score is not None and h.period_start <= cutoff:
            nps_history[h.customer_id] = h.nps_score
    for _, row in snapshot.iterrows():
        expected = nps_history.get(row["customer_id"])
        if expected is None:
            assert pd.isna(row["last_nps_score"])
        else:
            assert row["last_nps_score"] == expected


def test_young_customers_have_null_nps(snapshot) -> None:
    young = snapshot[snapshot["tenure_weeks"] < 13]
    assert young["last_nps_score"].isna().all()


# ---------------------------------------------------------------------------
# Leakage trap
# ---------------------------------------------------------------------------


def test_trap_equals_terminal_minus_initial_mrr(sim, snapshot) -> None:
    terminal = {s.customer_id: s.current_mrr for s in sim.subscriptions}
    for _, row in snapshot.iterrows():
        assert row["mrr_change_full_period"] == terminal[row["customer_id"]] - row["initial_mrr"]


def test_trap_diverges_for_nontrivial_fraction(snapshot) -> None:
    """design.md §7: the trap must visibly differ from the valid counterpart
    (post-cutoff expansions inflate it) for the lesson to exist at all."""
    diverges = snapshot["mrr_change_full_period"] != snapshot["mrr_change_at_snapshot"]
    assert diverges.mean() > 0.10


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_ltv_targets_match_invoice_table(population, sim, snapshot) -> None:
    cutoff = date.fromisoformat(population.observation_date)
    for _, row in snapshot.iterrows():
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


def test_ltv_windows_are_monotone(snapshot) -> None:
    assert (snapshot["ltv_revenue_90d"] <= snapshot["ltv_revenue_365d"]).all()
    assert (snapshot["ltv_revenue_365d"] <= snapshot["ltv_revenue_730d"]).all()


def test_failed_and_written_off_invoices_excluded_from_revenue(population) -> None:
    """D7: only collected revenue counts.  payment_fragile worlds have enough
    write-offs that including them would inflate the targets measurably."""
    pf_pop = build_customer_population(_N, _POP_SEED, motif_family="payment_fragile")
    pf_sim = simulate_lifecycle(pf_pop, _SIM_SEED)
    cutoff = date.fromisoformat(pf_pop.observation_date)
    snap = build_customer_snapshot(pf_pop, pf_sim)
    bound = cutoff + timedelta(days=730)
    uncollected = sum(
        i.amount_usd
        for i in pf_sim.invoices
        if i.payment_status in ("failed", "written_off")
        and cutoff < date.fromisoformat(i.invoice_date) <= bound
        and i.customer_id in set(snap["customer_id"])
    )
    assert uncollected > 0, "fixture should produce post-cutoff uncollected invoices"
    gross_all = sum(
        i.amount_usd
        for i in pf_sim.invoices
        if cutoff < date.fromisoformat(i.invoice_date) <= bound
        and i.customer_id in set(snap["customer_id"])
    )
    assert float(snap["ltv_revenue_730d"].sum()) == float(gross_all - uncollected)


def test_churn_label_matches_churn_dates(population, sim, snapshot) -> None:
    cutoff = date.fromisoformat(population.observation_date)
    bound = cutoff + timedelta(days=CHURN_WINDOW_DAYS)
    churn_dates = {
        s.customer_id: date.fromisoformat(s.churn_at)
        for s in sim.subscriptions
        if s.churn_at is not None
    }
    for _, row in snapshot.iterrows():
        churned = churn_dates.get(row["customer_id"])
        assert row["churned_within_180d"] == (churned is not None and churned <= bound)


def test_target_distribution_is_ziln_shaped(snapshot) -> None:
    """Right-skewed with a heavy upper tail (the expansion world drives it)."""
    for window in FORWARD_WINDOWS_DAYS:
        col = snapshot[f"ltv_revenue_{window}d"]
        assert (col >= 0).all()
        assert col.mean() > col.median(), f"{window}d not right-skewed"
    long_window = snapshot["ltv_revenue_730d"]
    assert long_window.max() > 5 * long_window.median()


# ---------------------------------------------------------------------------
# Difficulty distortions
# ---------------------------------------------------------------------------


def test_distortions_perturb_float_features(population, sim, snapshot) -> None:
    distorted = build_customer_snapshot(
        population, sim, difficulty_params=_difficulty_params(), seed=7
    )
    assert not distorted["avg_active_users_l12w"].equals(snapshot["avg_active_users_l12w"])


def test_distortions_inject_missingness(population, sim) -> None:
    distorted = build_customer_snapshot(
        population, sim, difficulty_params=_difficulty_params(missing_rate=0.25), seed=7
    )
    numeric_feature_cols = [
        f.name
        for f in CUSTOMER_SNAPSHOT_FEATURES
        if f.dtype in ("Int64", "Float64") and not f.is_target and not f.leakage_risk
    ]
    assert distorted[numeric_feature_cols].isna().sum().sum() > 0


def test_distortions_never_touch_targets(population, sim, snapshot) -> None:
    distorted = build_customer_snapshot(
        population, sim, difficulty_params=_difficulty_params(), seed=7
    )
    pd.testing.assert_frame_equal(distorted[_TARGET_COLS], snapshot[_TARGET_COLS])


def test_trap_exempt_from_distortion(population, sim, snapshot) -> None:
    """Noise or missingness on the trap would hide the lesson it teaches."""
    distorted = build_customer_snapshot(
        population,
        sim,
        difficulty_params=_difficulty_params(missing_rate=0.5, noise_scale=1.0),
        seed=7,
    )
    pd.testing.assert_series_equal(
        distorted["mrr_change_full_period"], snapshot["mrr_change_full_period"]
    )


def test_distortions_deterministic_per_seed(population, sim) -> None:
    params = _difficulty_params()
    a = build_customer_snapshot(population, sim, difficulty_params=params, seed=7)
    b = build_customer_snapshot(population, sim, difficulty_params=params, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = build_customer_snapshot(population, sim, difficulty_params=params, seed=8)
    assert not a.equals(c)

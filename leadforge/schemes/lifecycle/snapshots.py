"""Customer snapshot builders — flatten the lifecycle simulation into ML-ready
pLTV tables, one per observation regime (design.md §3.1).

Two public entry points, both producing the same
:data:`~leadforge.schemes.lifecycle.features.CUSTOMER_SNAPSHOT_FEATURES`
columns from the same simulated world, differing only in the **cutoff** each
customer is anchored at:

- :func:`build_customer_snapshot` — **calendar-anchored** (standard) regime: a
  single absolute ``cutoff`` (the world ``observation_date``) shared by every
  customer.  Tenure at cutoff varies from cold to mature.
- :func:`build_early_pltv_snapshot` — **tenure-anchored** (early-pLTV) regime
  (D8): a per-customer relative cutoff at
  ``customer_start + early_tenure_weeks``.  Every row is observed at the same
  short tenure — the genuine cold-start case (only a few weeks of health
  signal exist at the cutoff).

Both delegate to one per-customer-cutoff core (:func:`_assemble_snapshot`), so
feature derivations, the leakage trap, target attribution, and difficulty
distortions are defined exactly once.

Snapshot-safety contract (design.md §5): every feature column is computed
exclusively from events at or before that row's cutoff — with one deliberate
exception, the ``mrr_change_full_period`` leakage trap (design.md §7), which
reads the end-of-simulation MRR.  The targets (``ltv_revenue_{90,365,730}d``,
``churned_within_180d``) are forward-window aggregates by construction and are
never published as features.

Cutoff coverage
---------------
Forward-window targets are only meaningful if the simulation ran long enough to
fill them.  The engine (D6) simulates each customer through
``max(observation_date, start + early_tenure_weeks) + forward_window_days`` and
records that horizon on the result; both builders refuse to run unless the
recorded horizon covers the 730d/180d target windows, rather than silently
emitting censored targets.

Revenue attribution (D7)
------------------------
``ltv_revenue_*`` sums ``amount_usd`` over invoices with
``cutoff < invoice_date <= cutoff + window`` whose **terminal** payment status
is ``paid`` or ``recovered`` (a recovered invoice is gross revenue collected
late; ``failed`` / ``written_off`` invoices never count).  Attribution is by
issuance date, so a window-edge invoice recovered after the window still
counts toward the window it was issued in.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from leadforge.render.distortions import apply_difficulty_distortions
from leadforge.schemes.lifecycle.features import CUSTOMER_SNAPSHOT_FEATURES
from leadforge.schemes.lifecycle.hazards import next_renewal_week

if TYPE_CHECKING:
    from leadforge.core.models import DifficultyParams
    from leadforge.schemes.lifecycle.engine import LifecycleSimulationResult
    from leadforge.schemes.lifecycle.entities import (
        CustomerLifecycleRow,
        SubscriptionLifecycleRow,
    )
    from leadforge.schemes.lifecycle.population import CustomerPopulationResult

__all__ = [
    "CHURN_WINDOW_DAYS",
    "DEFAULT_EARLY_TENURE_WEEKS",
    "FORWARD_WINDOWS_DAYS",
    "HEALTH_WINDOW_WEEKS",
    "build_customer_snapshot",
    "build_early_pltv_snapshot",
]

# pLTV forward windows (D6) and the secondary churn-label window (D9).
FORWARD_WINDOWS_DAYS: tuple[int, ...] = (90, 365, 730)
CHURN_WINDOW_DAYS = 180

# Look-back window for the health aggregates (*_l12w columns).
HEALTH_WINDOW_WEEKS = 12

# Default tenure anchor for the early-pLTV regime (design.md §3.1: "e.g. 4w").
DEFAULT_EARLY_TENURE_WEEKS = 4

# Invoice terminal statuses that count as collected gross revenue (D7).
_REVENUE_STATUSES = frozenset({"paid", "recovered"})

_SNAPSHOT_COLUMNS = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in CUSTOMER_SNAPSHOT_FEATURES}

# The trap must survive distortion intact (same policy as the lead-scoring
# total_touches_all trap): noise/missingness on it would muddy the lesson.
_DISTORTION_EXEMPT_COLS: frozenset[str] = frozenset({"mrr_change_full_period"})

# One eligible customer plus the cutoff its row is anchored at.
_Eligible = tuple["CustomerLifecycleRow", "SubscriptionLifecycleRow", date]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_customer_snapshot(
    population: CustomerPopulationResult,
    sim: LifecycleSimulationResult,
    *,
    cutoff: date | None = None,
    difficulty_params: DifficultyParams | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the **calendar-anchored** customer snapshot table.

    Every customer is anchored at the same absolute ``cutoff``.

    Args:
        population: Output of
            :func:`~leadforge.schemes.lifecycle.population.build_customer_population`.
        sim: Output of
            :func:`~leadforge.schemes.lifecycle.engine.simulate_lifecycle` for
            the same population.
        cutoff: Snapshot anchor date.  Defaults to the population's
            ``observation_date``.  Must not be later than it (the engine only
            guarantees complete forward windows up to that anchor).
        difficulty_params: Optional difficulty knobs; when given, noise /
            missingness / outliers are applied to the numeric feature columns
            (never to targets, identifiers, or the leakage trap).
        seed: Seed for the distortion RNG substream.

    Returns:
        One row per customer active at the cutoff (started at or before it,
        not yet churned), with columns in catalog order.

    Raises:
        ValueError: if the population lacks an ``observation_date``, the cutoff
            exceeds it, the sim horizon cannot cover the target windows, or the
            population and sim do not match.
    """
    obs_date, accounts, subscriptions = _validate_inputs(population, sim)
    if cutoff is None:
        cutoff = obs_date
    elif cutoff > obs_date:
        raise ValueError(
            f"cutoff {cutoff.isoformat()} is after observation_date "
            f"{population.observation_date}; forward-window targets would be censored"
        )

    eligible: list[_Eligible] = []
    for customer in population.customers:
        start = date.fromisoformat(customer.customer_start_at)
        if start > cutoff:
            continue
        sub = subscriptions[customer.customer_id]
        if sub.churn_at is not None and date.fromisoformat(sub.churn_at) <= cutoff:
            continue
        eligible.append((customer, sub, start))

    cutoffs = {customer.customer_id: cutoff for customer, _, _ in eligible}
    return _assemble_snapshot(sim, accounts, eligible, cutoffs, difficulty_params, seed)


def build_early_pltv_snapshot(
    population: CustomerPopulationResult,
    sim: LifecycleSimulationResult,
    *,
    early_tenure_weeks: int = DEFAULT_EARLY_TENURE_WEEKS,
    difficulty_params: DifficultyParams | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the **tenure-anchored** early-pLTV snapshot table (D8).

    Each customer is anchored at ``customer_start + early_tenure_weeks`` — a
    per-customer relative cutoff — so every row is observed at the same fixed,
    short tenure.  This is the cold-start regime: only a few weeks of health
    signal exist at the cutoff, and ``last_nps_score`` is null for the whole
    cohort when ``early_tenure_weeks`` precedes the first quarterly survey.

    Degenerate columns at a short anchor.  Several catalog columns are
    structurally constant/empty when ``early_tenure_weeks`` is short, because
    the events that would vary them have not happened yet (the cadence math,
    not the seed, makes them dead):

    - ``tenure_weeks`` — constant ``= early_tenure_weeks`` (the defining
      property of the regime, not a feature).
    - ``renewal_count`` — constant ``0`` for any anchor ``< 52`` weeks (the
      first contract anniversary is at week 52).
    - ``last_nps_score`` — entirely null for any anchor ``< 13`` weeks (the
      first quarterly survey lands at week 13).
    - ``weeks_since_last_payment_failure`` — near-degenerate (at most one
      distinct value, often all-null): only the week-0 invoice precedes a
      sub-month cutoff, so any failure shares the same recency.

    The catalog is shared with the calendar regime by design (design.md §8),
    so these columns are kept rather than dropped; the published-bundle
    no-zero-variance / no-all-null checks must **exempt them for this task
    family** (handled in the validation harness, LTV-Pp).  Whether to instead
    drop them from the early task's feature set is an open question for the
    bundle/task writer (LTV-Pn).

    Eligibility does **not** require the cutoff to fall on or before
    ``observation_date``: each customer's forward windows are fully simulated
    relative to its own start (the engine runs through
    ``max(obs, start + early_tenure_weeks) + forward_window_days``), so a
    late-starting customer whose tenure cutoff lands after ``observation_date``
    still has complete targets.  The cohort therefore differs from the
    calendar regime's (it drops onboarding churners but keeps late starters).

    Args:
        population: Customer population.
        sim: Simulation result for the same population.
        early_tenure_weeks: Tenure (whole weeks) at which every customer is
            observed.  Must not exceed the sim's recorded ``early_tenure_weeks``
            (otherwise the per-customer forward windows are not fully covered).
        difficulty_params: Optional difficulty knobs (see
            :func:`build_customer_snapshot`).
        seed: Seed for the distortion RNG substream.

    Returns:
        One row per customer that survived to ``early_tenure_weeks`` of tenure.

    Raises:
        ValueError: on the same input problems as
            :func:`build_customer_snapshot`, plus a non-positive
            ``early_tenure_weeks`` or one exceeding the sim's recorded anchor.
    """
    if early_tenure_weeks < 1:
        raise ValueError(f"early_tenure_weeks must be >= 1, got {early_tenure_weeks}")
    _obs_date, accounts, subscriptions = _validate_inputs(population, sim)
    if early_tenure_weeks > sim.early_tenure_weeks:
        raise ValueError(
            f"early_tenure_weeks={early_tenure_weeks} exceeds the sim's recorded "
            f"early_tenure_weeks={sim.early_tenure_weeks}; the per-customer forward "
            "windows would be censored"
        )

    eligible: list[_Eligible] = []
    cutoffs: dict[str, date] = {}
    for customer in population.customers:
        start = date.fromisoformat(customer.customer_start_at)
        cutoff = start + timedelta(weeks=early_tenure_weeks)
        sub = subscriptions[customer.customer_id]
        if sub.churn_at is not None and date.fromisoformat(sub.churn_at) <= cutoff:
            continue
        eligible.append((customer, sub, start))
        cutoffs[customer.customer_id] = cutoff

    return _assemble_snapshot(sim, accounts, eligible, cutoffs, difficulty_params, seed)


# ---------------------------------------------------------------------------
# Shared assembly (per-customer cutoff)
# ---------------------------------------------------------------------------


def _validate_inputs(
    population: CustomerPopulationResult, sim: LifecycleSimulationResult
) -> tuple[date, dict[str, Any], dict[str, Any]]:
    """Shared precondition checks for both regimes.

    Returns the parsed ``observation_date``, an ``account_id -> AccountRow``
    index, and a ``customer_id -> SubscriptionLifecycleRow`` index.
    """
    if not population.observation_date:
        raise ValueError("population.observation_date is not set")
    obs_date = date.fromisoformat(population.observation_date)

    required_days = max(*FORWARD_WINDOWS_DAYS, CHURN_WINDOW_DAYS)
    if sim.forward_window_days < required_days:
        raise ValueError(
            f"sim was run with forward_window_days={sim.forward_window_days}, which "
            f"cannot cover the {required_days}-day target windows; the ltv/churn "
            "targets would be silently censored"
        )

    accounts = {a.account_id: a for a in population.accounts}
    subscriptions = {s.customer_id: s for s in sim.subscriptions}
    missing = [c.customer_id for c in population.customers if c.customer_id not in subscriptions]
    if missing:
        raise ValueError(
            f"sim result lacks subscriptions for {len(missing)} of "
            f"{len(population.customers)} population customers (e.g. {missing[0]}); "
            "population/sim mismatch"
        )
    return obs_date, accounts, subscriptions


def _assemble_snapshot(
    sim: LifecycleSimulationResult,
    accounts: dict[str, Any],
    eligible: list[_Eligible],
    cutoffs: dict[str, date],
    difficulty_params: DifficultyParams | None,
    seed: int,
) -> pd.DataFrame:
    """Build the snapshot frame from a per-customer ``customer_id -> cutoff`` map."""
    if not eligible:
        return _empty_snapshot()

    events = _event_aggregates(sim, cutoffs)
    health = _health_aggregates(sim, cutoffs)
    revenue = _forward_revenue(sim, cutoffs)

    records: list[dict[str, object]] = []
    for customer, sub, start in eligible:
        cutoff = cutoffs[customer.customer_id]
        account = accounts[customer.account_id]
        tenure_weeks = (cutoff - start).days // 7
        ev: Mapping[str, Any] = events.get(customer.customer_id, _EMPTY_EVENT_AGG)
        hl: Mapping[str, Any] = health.get(customer.customer_id, _EMPTY_HEALTH_AGG)
        rv = revenue.get(customer.customer_id, {})

        churn_date = date.fromisoformat(sub.churn_at) if sub.churn_at else None
        records.append(
            {
                "customer_id": customer.customer_id,
                "account_id": customer.account_id,
                "industry": account.industry,
                "region": account.region,
                "employee_band": account.employee_band,
                "estimated_revenue_band": account.estimated_revenue_band,
                "tenure_weeks": tenure_weeks,
                "initial_plan": customer.initial_plan,
                "initial_mrr": customer.initial_mrr,
                "current_mrr": customer.initial_mrr + ev["mrr_delta"],
                "mrr_change_at_snapshot": ev["mrr_delta"],
                "renewal_count": ev["renewal_count"],
                "expansion_count": ev["expansion_count"],
                "contract_term_months": customer.contract_term_months,
                "weeks_to_next_renewal": (
                    next_renewal_week(tenure_weeks, customer.contract_term_months) - tenure_weeks
                ),
                "avg_active_users_l12w": hl["avg_active_users"],
                "active_user_trend_l12w": hl["trend"],
                "avg_feature_depth_l12w": hl["avg_depth"],
                "support_ticket_count_l12w": hl["tickets"],
                "last_nps_score": hl["last_nps"],
                "payment_failure_count": ev["payment_failure_count"],
                "weeks_since_last_payment_failure": (
                    (cutoff - ev["last_failure_date"]).days // 7
                    if ev["last_failure_date"] is not None
                    else None
                ),
                "mrr_change_full_period": sub.current_mrr - customer.initial_mrr,
                **{
                    f"ltv_revenue_{window}d": float(rv.get(window, 0))
                    for window in FORWARD_WINDOWS_DAYS
                },
                "churned_within_180d": (
                    churn_date is not None
                    and churn_date <= cutoff + timedelta(days=CHURN_WINDOW_DAYS)
                ),
            }
        )

    snapshot = pd.DataFrame.from_records(records)[_SNAPSHOT_COLUMNS]
    for col, dtype in _SNAPSHOT_DTYPES.items():
        snapshot[col] = snapshot[col].astype(dtype)

    if difficulty_params is not None:
        snapshot = apply_difficulty_distortions(
            snapshot,
            difficulty_params,
            seed,
            feature_specs=CUSTOMER_SNAPSHOT_FEATURES,
            exempt_cols=_DISTORTION_EXEMPT_COLS,
        )

    return snapshot


# ---------------------------------------------------------------------------
# Per-table aggregation helpers (per-customer cutoff)
# ---------------------------------------------------------------------------

# Frozen (MappingProxyType): these are handed out as shared fallbacks for
# customers with no events/signals — mutating one would corrupt every
# subsequent row, so accidental writes must raise.
_EMPTY_EVENT_AGG: Mapping[str, Any] = MappingProxyType(
    {
        "mrr_delta": 0,
        "renewal_count": 0,
        "expansion_count": 0,
        "payment_failure_count": 0,
        "last_failure_date": None,
    }
)

_EMPTY_HEALTH_AGG: Mapping[str, Any] = MappingProxyType(
    {
        "avg_active_users": None,
        "trend": None,
        "avg_depth": None,
        "tickets": 0,
        "last_nps": None,
    }
)


def _event_aggregates(sim: LifecycleSimulationResult, cutoffs: dict[str, date]) -> dict[str, dict]:
    """Aggregate each customer's subscription events at or before its cutoff."""
    cutoffs_iso = {cid: c.isoformat() for cid, c in cutoffs.items()}
    out: dict[str, dict] = {}
    for event in sim.subscription_events:
        cutoff_iso = cutoffs_iso.get(event.customer_id)
        # ISO dates compare correctly as strings — avoids per-event parsing.
        # A None cutoff means the customer is not eligible (skip entirely).
        if cutoff_iso is None or event.event_timestamp > cutoff_iso:
            continue
        agg = out.setdefault(event.customer_id, dict(_EMPTY_EVENT_AGG))
        if event.event_type == "expansion":
            agg["mrr_delta"] += event.mrr_after - event.mrr_before
            agg["expansion_count"] += 1
        elif event.event_type == "renewal":
            agg["renewal_count"] += 1
        elif event.event_type == "payment_failure":
            agg["payment_failure_count"] += 1
            agg["last_failure_date"] = date.fromisoformat(event.event_timestamp)
    return out


def _health_aggregates(sim: LifecycleSimulationResult, cutoffs: dict[str, date]) -> dict[str, dict]:
    """Aggregate health signals into the last-12-week window features.

    ``last_nps_score`` looks back over the customer's whole history (NPS is
    quarterly — a 12-week window would miss most customers' latest response
    purely by phase), while the ``*_l12w`` aggregates use each customer's
    ``(cutoff - 12w, cutoff]`` window.
    """
    cutoffs_iso = {cid: c.isoformat() for cid, c in cutoffs.items()}
    window_start_iso = {
        cid: (c - timedelta(weeks=HEALTH_WINDOW_WEEKS)).isoformat() for cid, c in cutoffs.items()
    }

    users: dict[str, list[tuple[str, int]]] = {}
    depths: dict[str, list[float]] = {}
    tickets: dict[str, int] = {}
    last_nps: dict[str, int] = {}
    for signal in sim.health_signals:
        cutoff_iso = cutoffs_iso.get(signal.customer_id)
        if cutoff_iso is None or signal.period_start > cutoff_iso:
            continue
        if signal.nps_score is not None:
            # Signals are chronological per customer — last write wins.
            last_nps[signal.customer_id] = signal.nps_score
        if signal.period_start <= window_start_iso[signal.customer_id]:
            continue
        users.setdefault(signal.customer_id, []).append((signal.period_start, signal.active_users))
        depths.setdefault(signal.customer_id, []).append(signal.feature_depth_score)
        tickets[signal.customer_id] = tickets.get(signal.customer_id, 0) + signal.support_tickets

    out: dict[str, dict] = {}
    for customer_id, points in users.items():
        cutoff = cutoffs[customer_id]
        weeks = [(date.fromisoformat(ts) - cutoff).days / 7.0 for ts, _ in points]
        counts = [n for _, n in points]
        if len(points) >= 2:
            trend = float(np.polyfit(weeks, counts, 1)[0])
        else:
            trend = None
        out[customer_id] = {
            "avg_active_users": float(np.mean(counts)),
            "trend": trend,
            "avg_depth": float(np.mean(depths[customer_id])),
            "tickets": tickets[customer_id],
            "last_nps": last_nps.get(customer_id),
        }
    # Customers with an NPS response but no in-window signals cannot occur for
    # an active customer (it always has a signal in the trailing window), but a
    # defensive merge keeps last_nps consistent if eligibility ever widens.
    for customer_id, nps in last_nps.items():
        out.setdefault(customer_id, dict(_EMPTY_HEALTH_AGG))["last_nps"] = nps
    return out


def _forward_revenue(
    sim: LifecycleSimulationResult, cutoffs: dict[str, date]
) -> dict[str, dict[int, int]]:
    """Sum collected gross revenue per customer per forward window (D7)."""
    cutoffs_iso = {cid: c.isoformat() for cid, c in cutoffs.items()}
    bounds_iso = {
        cid: {window: (c + timedelta(days=window)).isoformat() for window in FORWARD_WINDOWS_DAYS}
        for cid, c in cutoffs.items()
    }
    out: dict[str, dict[int, int]] = {}
    for invoice in sim.invoices:
        cutoff_iso = cutoffs_iso.get(invoice.customer_id)
        if cutoff_iso is None or invoice.payment_status not in _REVENUE_STATUSES:
            continue
        ts = invoice.invoice_date
        if ts <= cutoff_iso:
            continue
        sums = out.setdefault(invoice.customer_id, dict.fromkeys(FORWARD_WINDOWS_DAYS, 0))
        for window, bound in bounds_iso[invoice.customer_id].items():
            if ts <= bound:
                sums[window] += invoice.amount_usd
    return out


def _empty_snapshot() -> pd.DataFrame:
    df = pd.DataFrame({name: pd.Series(dtype=_SNAPSHOT_DTYPES[name]) for name in _SNAPSHOT_COLUMNS})
    return df[_SNAPSHOT_COLUMNS]

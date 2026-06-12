"""Customer snapshot builder — flatten the lifecycle simulation into an
ML-ready pLTV table.

:func:`build_customer_snapshot` produces one row per customer **active at the
cutoff**, containing the features defined in
:data:`~leadforge.schemes.lifecycle.features.CUSTOMER_SNAPSHOT_FEATURES`.

Snapshot-safety contract (design.md §5): every feature column is computed
exclusively from events at or before the cutoff — with one deliberate
exception, the ``mrr_change_full_period`` leakage trap (design.md §7), which
reads the end-of-simulation MRR.  The targets (``ltv_revenue_{90,365,730}d``,
``churned_within_180d``) are forward-window aggregates by construction and are
never published as features.

Cutoff semantics
----------------
The calendar-anchored regime (this PR, LTV-Pl) snapshots every customer at the
shared absolute ``observation_date``.  The tenure-anchored early-pLTV regime
(LTV-Pm) will reuse the same per-customer machinery with a relative cutoff.
The cutoff must not exceed the population's ``observation_date``: the engine
only guarantees full forward-window simulation up to
``observation_date + forward_window_days`` (D6), so a later cutoff would
silently censor the targets.

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
    from leadforge.schemes.lifecycle.population import CustomerPopulationResult

__all__ = [
    "CHURN_WINDOW_DAYS",
    "FORWARD_WINDOWS_DAYS",
    "HEALTH_WINDOW_WEEKS",
    "build_customer_snapshot",
]

# pLTV forward windows (D6) and the secondary churn-label window (D9).
FORWARD_WINDOWS_DAYS: tuple[int, ...] = (90, 365, 730)
CHURN_WINDOW_DAYS = 180

# Look-back window for the health aggregates (*_l12w columns).
HEALTH_WINDOW_WEEKS = 12

# Invoice terminal statuses that count as collected gross revenue (D7).
_REVENUE_STATUSES = frozenset({"paid", "recovered"})

_SNAPSHOT_COLUMNS = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in CUSTOMER_SNAPSHOT_FEATURES}

# The trap must survive distortion intact (same policy as the lead-scoring
# total_touches_all trap): noise/missingness on it would muddy the lesson.
_DISTORTION_EXEMPT_COLS: frozenset[str] = frozenset({"mrr_change_full_period"})


def build_customer_snapshot(
    population: CustomerPopulationResult,
    sim: LifecycleSimulationResult,
    *,
    cutoff: date | None = None,
    difficulty_params: DifficultyParams | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the calendar-anchored customer snapshot table.

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
        not yet churned), with columns in catalog order.  Customers who
        started after the cutoff or churned at/before it are excluded.

    Raises:
        ValueError: if the population lacks an ``observation_date`` or the
            cutoff exceeds it.
    """
    if not population.observation_date:
        raise ValueError("population.observation_date is not set")
    obs_date = date.fromisoformat(population.observation_date)
    if cutoff is None:
        cutoff = obs_date
    elif cutoff > obs_date:
        raise ValueError(
            f"cutoff {cutoff.isoformat()} is after observation_date "
            f"{population.observation_date}; forward-window targets would be censored"
        )

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

    # Eligibility: started at or before the cutoff, still active at it.
    eligible = []
    for customer in population.customers:
        start = date.fromisoformat(customer.customer_start_at)
        if start > cutoff:
            continue
        sub = subscriptions[customer.customer_id]
        if sub.churn_at is not None and date.fromisoformat(sub.churn_at) <= cutoff:
            continue
        eligible.append((customer, sub, start))

    if not eligible:
        return _empty_snapshot()

    events = _event_aggregates(sim, cutoff)
    health = _health_aggregates(sim, cutoff)
    revenue = _forward_revenue(sim, cutoff)

    records: list[dict[str, object]] = []
    for customer, sub, start in eligible:
        account = accounts[customer.account_id]
        tenure_weeks = (cutoff - start).days // 7
        ev: Mapping[str, Any] = events.get(customer.customer_id, _EMPTY_EVENT_AGG)
        hl: Mapping[str, Any] = health.get(customer.customer_id, _EMPTY_HEALTH_AGG)
        rv = revenue.get(customer.customer_id, {})

        current_mrr = customer.initial_mrr + ev["mrr_delta"]
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
                "current_mrr": current_mrr,
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
# Per-table aggregation helpers
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


def _event_aggregates(sim: LifecycleSimulationResult, cutoff: date) -> dict[str, dict]:
    """Aggregate subscription events at or before *cutoff*, per customer."""
    out: dict[str, dict] = {}
    cutoff_iso = cutoff.isoformat()
    for event in sim.subscription_events:
        # ISO dates compare correctly as strings — avoids per-event parsing.
        if event.event_timestamp > cutoff_iso:
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


def _health_aggregates(sim: LifecycleSimulationResult, cutoff: date) -> dict[str, dict]:
    """Aggregate health signals into the last-12-week window features.

    ``last_nps_score`` looks back over the customer's whole history (NPS is
    quarterly — a 12-week window would miss most customers' latest response
    purely by phase), while the ``*_l12w`` aggregates use the
    ``(cutoff - 12w, cutoff]`` window.
    """
    window_start_iso = (cutoff - timedelta(weeks=HEALTH_WINDOW_WEEKS)).isoformat()
    cutoff_iso = cutoff.isoformat()

    users: dict[str, list[tuple[str, int]]] = {}
    depths: dict[str, list[float]] = {}
    tickets: dict[str, int] = {}
    last_nps: dict[str, int] = {}
    for signal in sim.health_signals:
        ts = signal.period_start
        if ts > cutoff_iso:
            continue
        if signal.nps_score is not None:
            # Signals are chronological per customer — last write wins.
            last_nps[signal.customer_id] = signal.nps_score
        if ts <= window_start_iso:
            continue
        users.setdefault(signal.customer_id, []).append((ts, signal.active_users))
        depths.setdefault(signal.customer_id, []).append(signal.feature_depth_score)
        tickets[signal.customer_id] = tickets.get(signal.customer_id, 0) + signal.support_tickets

    out: dict[str, dict] = {}
    for customer_id, points in users.items():
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
    # Customers with an NPS response but no in-window signals cannot occur
    # (an active customer always has signals in the trailing window), but a
    # defensive merge keeps last_nps consistent if eligibility ever widens.
    for customer_id, nps in last_nps.items():
        out.setdefault(customer_id, dict(_EMPTY_HEALTH_AGG))["last_nps"] = nps
    return out


def _forward_revenue(sim: LifecycleSimulationResult, cutoff: date) -> dict[str, dict[int, int]]:
    """Sum collected gross revenue per customer per forward window (D7)."""
    bounds = {
        window: (cutoff + timedelta(days=window)).isoformat() for window in FORWARD_WINDOWS_DAYS
    }
    cutoff_iso = cutoff.isoformat()
    out: dict[str, dict[int, int]] = {}
    for invoice in sim.invoices:
        if invoice.payment_status not in _REVENUE_STATUSES:
            continue
        ts = invoice.invoice_date
        if ts <= cutoff_iso:
            continue
        sums = out.setdefault(invoice.customer_id, dict.fromkeys(FORWARD_WINDOWS_DAYS, 0))
        for window, bound in bounds.items():
            if ts <= bound:
                sums[window] += invoice.amount_usd
    return out


def _empty_snapshot() -> pd.DataFrame:
    df = pd.DataFrame({name: pd.Series(dtype=_SNAPSHOT_DTYPES[name]) for name in _SNAPSHOT_COLUMNS})
    return df[_SNAPSHOT_COLUMNS]

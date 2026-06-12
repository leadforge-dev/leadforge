"""Weekly lifecycle simulation engine (D2: weekly steps).

:func:`simulate_lifecycle` is the single public entry point.  It evolves each
customer of a :class:`~leadforge.schemes.lifecycle.population.CustomerPopulationResult`
week by week from their staggered start date, drawing against the pure hazard
functions in :mod:`leadforge.schemes.lifecycle.hazards`, and emits the three
lifecycle event tables (``subscription_events``, ``health_signals``,
``invoices``) plus one terminal-state subscription row per customer.

Simulation contract
-------------------
- **Fully simulated target windows (D6)** — every customer is simulated through
  ``max(observation_date, start + early_tenure_weeks) + forward_window_days``,
  so all pLTV forward-window targets (90/365/730d) are complete for **both**
  observation regimes (calendar-anchored and tenure-anchored).  A customer who
  reaches the end of their window still active is *censored for total LTV* but
  has complete forward-window revenue.
- **Per-customer RNG substreams** — every customer draws from its own named
  substream (``lifecycle_sim::<customer_id>``), so one customer's trajectory is
  invariant to the presence, ordering, or behaviour of every other customer.
  This is a stronger stability property than the lead-scoring engine's shared
  streams, and it makes per-customer regression tests exact.
- **Weekly step order** (fixed for determinism): health signal → invoice /
  dunning resolution → churn draw → renewal event → expansion draw.  A customer
  that churns in a week emits no further events after the churn event.
- **Churn reasons** are causal, not sampled: ``payment_failure`` when a
  written-off invoice forces the churn, ``non_renewal`` when the (spiked) churn
  draw fires on a contract-anniversary week, ``voluntary`` otherwise.
- **Dangling ``failed`` invoices are censoring, not bugs** — an invoice keeps
  terminal status ``failed`` only when the customer churned for another reason
  mid-dunning or the simulation window ended before the dunning period elapsed.
  Within a week, dunning resolution runs *before* invoice issuance, so a
  pending failure always resolves before the next invoice can fail.

Deliberately out of scope (tracked):
- ``downgrade`` events — no downgrade mechanism params exist in
  ``mechanisms.py`` yet; the snapshot's ``downgrade_count`` feature (design.md
  §8) would be zero-variance and must be revisited before LTV-M5 ships it.
- Difficulty-tier scaling — applied at the recipe/config layer in LTV-M6; this
  engine simulates the motif-calibrated intermediate-tier parameters as-is.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.core.rng import RNGRoot
from leadforge.schemes.lifecycle.entities import (
    HealthSignalRow,
    InvoiceRow,
    SubscriptionEventRow,
    SubscriptionLifecycleRow,
)
from leadforge.schemes.lifecycle.hazards import (
    churn_probability,
    expansion_probability,
    is_renewal_week,
    payment_failure_probability,
)
from leadforge.schemes.lifecycle.mechanisms import assign_lifecycle_mechanisms

if TYPE_CHECKING:
    from leadforge.schemes.lifecycle.entities import CustomerLifecycleRow
    from leadforge.schemes.lifecycle.mechanisms import LifecycleMechanismAssignment
    from leadforge.schemes.lifecycle.population import CustomerPopulationResult

__all__ = ["LifecycleSimulationResult", "simulate_lifecycle"]

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_WEEKS_PER_MONTH = 52.0 / 12.0

# NPS surveys go out quarterly; responses land on every 13th week of tenure
# (week 13, 26, …).  All other weeks carry a null nps_score.
_NPS_CADENCE_WEEKS = 13

# Health-signal generation: weekly active users by plan tier (seat-count
# proxy), modulated by adoption velocity and an onboarding usage ramp.
_ACTIVE_USERS_BASE_BY_PLAN: dict[str, int] = {
    "starter": 8,
    "growth": 25,
    "enterprise": 60,
}
_DEFAULT_ACTIVE_USERS_BASE = 20

# Usage ramps up over onboarding with this time-constant (weeks): customers
# reach ~63% of plateau usage by week 6, ~92% by week 15.
_USAGE_RAMP_WEEKS = 6.0

# Support-ticket Poisson intensity: lam = base + slope * (1 - product_fit).
_TICKET_LAM_BASE = 0.3
_TICKET_LAM_SLOPE = 1.2


# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------


@dataclass
class LifecycleSimulationResult:
    """Fully simulated lifecycle output, ready for the rendering layer.

    All lists are in insertion order: chronological within each customer,
    population order across customers.
    """

    subscriptions: list[SubscriptionLifecycleRow]
    subscription_events: list[SubscriptionEventRow] = field(default_factory=list)
    health_signals: list[HealthSignalRow] = field(default_factory=list)
    invoices: list[InvoiceRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-customer mutable state
# ---------------------------------------------------------------------------


@dataclass
class _CustomerSimState:
    current_mrr: int
    contract_term_months: int
    renewal_count: int = 0
    expansion_count: int = 0
    churned: bool = False
    churn_week: int | None = None
    churn_reason: str | None = None
    # Pending failed invoice: (invoice row, week it failed).
    pending_failure: tuple[InvoiceRow, int] | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate_lifecycle(
    population: CustomerPopulationResult,
    seed: int,
    *,
    forward_window_days: int = 730,
    early_tenure_weeks: int = 4,
) -> LifecycleSimulationResult:
    """Run the weekly lifecycle simulation for every customer in *population*.

    Args:
        population: Output of
            :func:`~leadforge.schemes.lifecycle.population.build_customer_population`.
            Its recorded ``motif_family`` selects the mechanism parameters and
            its ``observation_date`` anchors the simulation horizon.
        seed: Master RNG seed for the simulation (independent of the
            population seed; the same population can be re-simulated).
        forward_window_days: Longest pLTV forward-window target (D6).  Every
            customer is simulated through at least
            ``max(observation_date, start + early_tenure_weeks) +
            forward_window_days``.
        early_tenure_weeks: Tenure-anchored early-pLTV cutoff (D8) — extends
            the horizon for late-starting customers so the early regime's
            forward windows are also fully simulated.

    Returns:
        A :class:`LifecycleSimulationResult` with one subscription row per
        customer and the three event tables populated.

    Raises:
        ValueError: if *population* lacks an ``observation_date`` or
            ``motif_family`` (built by an outdated population builder), or if
            window arguments are not positive.
    """
    if not population.observation_date:
        raise ValueError("population.observation_date is not set")
    if not population.motif_family:
        raise ValueError("population.motif_family is not set")
    if forward_window_days < 1:
        raise ValueError(f"forward_window_days must be >= 1, got {forward_window_days}")
    if early_tenure_weeks < 0:
        raise ValueError(f"early_tenure_weeks must be >= 0, got {early_tenure_weeks}")

    obs_date = date.fromisoformat(population.observation_date)
    mechanisms = assign_lifecycle_mechanisms(population.motif_family)
    root = RNGRoot(seed)

    acct_latents = population.latent_state.account_latents
    cust_latents = population.latent_state.customer_latents

    result = LifecycleSimulationResult(subscriptions=[])
    # Event ID counters are global across customers (population order), so IDs
    # stay deterministic and dense.
    counters = {"subscription_event": 0, "health_signal": 0, "invoice": 0}

    for idx, customer in enumerate(population.customers, start=1):
        latents = _merge_latents(
            acct_latents.get(customer.account_id, {}),
            cust_latents.get(customer.customer_id, {}),
        )

        rng = root.child(f"lifecycle_sim::{customer.customer_id}")
        sub_id = make_id(ID_PREFIXES["subscription"], idx)
        start = date.fromisoformat(customer.customer_start_at)
        end_date = max(obs_date, start + timedelta(weeks=early_tenure_weeks)) + timedelta(
            days=forward_window_days
        )

        state = _simulate_customer(
            customer=customer,
            subscription_id=sub_id,
            latents=latents,
            mechanisms=mechanisms,
            rng=rng,
            start=start,
            end_date=end_date,
            counters=counters,
            result=result,
        )

        churn_at = (
            (start + timedelta(weeks=state.churn_week)).isoformat()
            if state.churned and state.churn_week is not None
            else None
        )
        result.subscriptions.append(
            SubscriptionLifecycleRow(
                subscription_id=sub_id,
                customer_id=customer.customer_id,
                plan_name=customer.initial_plan,
                subscription_status="churned" if state.churned else "active",
                subscription_start_at=customer.customer_start_at,
                current_mrr=state.current_mrr,
                contract_term_months=state.contract_term_months,
                renewal_count=state.renewal_count,
                expansion_count=state.expansion_count,
                subscription_end_at=churn_at,
                churn_at=churn_at,
                churn_reason=state.churn_reason,
            )
        )

    return result


def _merge_latents(
    account_latents: dict[str, float], customer_latents: dict[str, float]
) -> dict[str, float]:
    """Merge account- and customer-level latents into one effective trait dict.

    Traits present at **both** levels (``latent_budget_stability``,
    ``latent_organizational_stability``) are blended 50/50 rather than letting
    one level shadow the other: the account component is a shared random effect
    across all customers of the same account, so within-account churn and
    payment behaviour are *correlated* — the mixed-effects structure a B2B
    dataset should have.  Account-only or customer-only traits pass through.
    """
    merged = dict(customer_latents)
    for trait, account_value in account_latents.items():
        if trait in merged:
            merged[trait] = 0.5 * (account_value + merged[trait])
        else:
            merged[trait] = account_value
    return merged


# ---------------------------------------------------------------------------
# Per-customer weekly loop
# ---------------------------------------------------------------------------


def _simulate_customer(
    *,
    customer: CustomerLifecycleRow,
    subscription_id: str,
    latents: dict[str, float],
    mechanisms: LifecycleMechanismAssignment,
    rng: random.Random,
    start: date,
    end_date: date,
    counters: dict[str, int],
    result: LifecycleSimulationResult,
) -> _CustomerSimState:
    """Evolve one customer week by week; append events to *result*."""
    state = _CustomerSimState(
        current_mrr=customer.initial_mrr,
        contract_term_months=customer.contract_term_months,
    )
    churn_p = mechanisms.churn_hazard
    expansion_p = mechanisms.expansion_propensity
    payment_p = mechanisms.payment_failure

    week = 0
    last_month_index = -1
    while start + timedelta(weeks=week) <= end_date:
        week_date = start + timedelta(weeks=week)

        # -- 1. Health signal (weekly) --------------------------------------
        depth = _emit_health_signal(
            customer=customer,
            latents=latents,
            rng=rng,
            week=week,
            week_date=week_date,
            counters=counters,
            result=result,
        )

        # -- 2. Dunning resolution, then invoice on month boundary ----------
        # Resolution runs FIRST so a write-off churn cannot be preceded by a
        # fresh same-week invoice, and a newly failed invoice can always enter
        # dunning (pending slot is free by issuance time).
        if state.pending_failure is not None:
            pending_invoice, fail_week = state.pending_failure
            if week - fail_week >= payment_p.dunning_weeks:
                if rng.random() < payment_p.recovery_rate:
                    pending_invoice.payment_status = "recovered"
                    _emit_event(
                        result,
                        counters,
                        subscription_id=subscription_id,
                        customer_id=customer.customer_id,
                        week_date=week_date,
                        event_type="payment_recovered",
                        mrr_before=state.current_mrr,
                        mrr_after=state.current_mrr,
                    )
                    state.pending_failure = None
                else:
                    pending_invoice.payment_status = "written_off"
                    _churn(state, week, "payment_failure")
                    _emit_event(
                        result,
                        counters,
                        subscription_id=subscription_id,
                        customer_id=customer.customer_id,
                        week_date=week_date,
                        event_type="churn",
                        mrr_before=state.current_mrr,
                        mrr_after=0,
                    )
                    break

        month_index = int(week / _WEEKS_PER_MONTH)
        if month_index > last_month_index:
            last_month_index = month_index
            counters["invoice"] += 1
            failed = rng.random() < payment_failure_probability(payment_p, latents)
            invoice = InvoiceRow(
                invoice_id=make_id(ID_PREFIXES["invoice"], counters["invoice"]),
                customer_id=customer.customer_id,
                invoice_date=week_date.isoformat(),
                amount_usd=state.current_mrr,
                payment_status="failed" if failed else "paid",
            )
            result.invoices.append(invoice)
            if failed and state.pending_failure is None:
                state.pending_failure = (invoice, week)
                _emit_event(
                    result,
                    counters,
                    subscription_id=subscription_id,
                    customer_id=customer.customer_id,
                    week_date=week_date,
                    event_type="payment_failure",
                    mrr_before=state.current_mrr,
                    mrr_after=state.current_mrr,
                )

        # -- 3. Churn draw ---------------------------------------------------
        renewal_week = is_renewal_week(week, state.contract_term_months)
        p_churn = churn_probability(churn_p, latents, week, state.contract_term_months)
        if rng.random() < p_churn:
            _churn(state, week, "non_renewal" if renewal_week else "voluntary")
            _emit_event(
                result,
                counters,
                subscription_id=subscription_id,
                customer_id=customer.customer_id,
                week_date=week_date,
                event_type="churn",
                mrr_before=state.current_mrr,
                mrr_after=0,
            )
            break

        # -- 4. Renewal event (survived the anniversary) ---------------------
        if renewal_week:
            state.renewal_count += 1
            _emit_event(
                result,
                counters,
                subscription_id=subscription_id,
                customer_id=customer.customer_id,
                week_date=week_date,
                event_type="renewal",
                mrr_before=state.current_mrr,
                mrr_after=state.current_mrr,
                contract_term_months_new=state.contract_term_months,
            )

        # -- 5. Expansion draw ------------------------------------------------
        if rng.random() < expansion_probability(expansion_p, latents, depth):
            lo_frac, hi_frac = expansion_p.expansion_mrr_frac_range
            lo = max(1, int(lo_frac * state.current_mrr))
            hi = max(lo, int(hi_frac * state.current_mrr))
            delta = rng.randint(lo, hi)
            _emit_event(
                result,
                counters,
                subscription_id=subscription_id,
                customer_id=customer.customer_id,
                week_date=week_date,
                event_type="expansion",
                mrr_before=state.current_mrr,
                mrr_after=state.current_mrr + delta,
            )
            state.current_mrr += delta
            state.expansion_count += 1

        week += 1

    return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _churn(state: _CustomerSimState, week: int, reason: str) -> None:
    state.churned = True
    state.churn_week = week
    state.churn_reason = reason


def _emit_event(
    result: LifecycleSimulationResult,
    counters: dict[str, int],
    *,
    subscription_id: str,
    customer_id: str,
    week_date: date,
    event_type: str,
    mrr_before: int,
    mrr_after: int,
    contract_term_months_new: int | None = None,
) -> None:
    counters["subscription_event"] += 1
    result.subscription_events.append(
        SubscriptionEventRow(
            event_id=make_id(ID_PREFIXES["subscription_event"], counters["subscription_event"]),
            subscription_id=subscription_id,
            customer_id=customer_id,
            event_timestamp=week_date.isoformat(),
            event_type=event_type,
            mrr_before=mrr_before,
            mrr_after=mrr_after,
            contract_term_months_new=contract_term_months_new,
        )
    )


def _emit_health_signal(
    *,
    customer: CustomerLifecycleRow,
    latents: dict[str, float],
    rng: random.Random,
    week: int,
    week_date: date,
    counters: dict[str, int],
    result: LifecycleSimulationResult,
) -> float:
    """Emit this week's health signal; return the feature-depth score.

    The depth score feeds straight back into the expansion hazard for the same
    week, creating the causal link latents → health signals → expansion.
    """
    adoption = latents.get("latent_adoption_velocity", 0.5)
    fit = latents.get("latent_product_fit", 0.5)

    ramp = 1.0 - math.exp(-(week + 1) / _USAGE_RAMP_WEEKS)

    base_users = _ACTIVE_USERS_BASE_BY_PLAN.get(customer.initial_plan, _DEFAULT_ACTIVE_USERS_BASE)
    active_users = max(0, int(base_users * (0.5 + adoption) * ramp * rng.gauss(1.0, 0.10) + 0.5))

    target_depth = min(1.0, max(0.0, 0.20 + 0.40 * adoption + 0.30 * fit))
    depth = min(1.0, max(0.0, target_depth * ramp + rng.gauss(0.0, 0.04)))

    lam = _TICKET_LAM_BASE + _TICKET_LAM_SLOPE * (1.0 - fit)
    tickets = _poisson(rng, lam)

    nps: int | None = None
    if week > 0 and week % _NPS_CADENCE_WEEKS == 0:
        champion = latents.get("latent_champion_strength", 0.5)
        raw = 10.0 * (0.20 + 0.50 * fit + 0.30 * champion) + rng.gauss(0.0, 1.0)
        nps = max(0, min(10, round(raw)))

    # The recorded observable and the value fed to the expansion hazard must
    # be the SAME number — the data must fully explain the behaviour.
    depth = round(depth, 4)

    counters["health_signal"] += 1
    result.health_signals.append(
        HealthSignalRow(
            signal_id=make_id(ID_PREFIXES["health_signal"], counters["health_signal"]),
            customer_id=customer.customer_id,
            period_start=week_date.isoformat(),
            active_users=active_users,
            feature_depth_score=depth,
            support_tickets=tickets,
            nps_score=nps,
        )
    )
    return depth


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth Poisson sampler — fine for the small intensities used here."""
    threshold = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1

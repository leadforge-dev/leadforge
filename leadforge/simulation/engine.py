"""Discrete-time simulation engine — 90-day hybrid world evolution.

:func:`simulate_world` is the single public entry point.  It iterates daily
steps for every lead in the population, driven by the
:class:`~leadforge.mechanisms.base.MechanismAssignment` produced by
:func:`~leadforge.mechanisms.policies.assign_mechanisms`, and emits the full
set of relational event rows.

Simulation contract
-------------------
- All randomness is derived from named substreams of ``RNGRoot(config.seed)``,
  making every run fully deterministic given ``(config, population, world_graph)``.
- ``converted_within_90_days`` is **event-derived** — it becomes ``True`` only
  when a lead's simulated trajectory reaches the ``closed_won`` terminal stage.
- Stage advancement is driven by :class:`~leadforge.mechanisms.transitions.HazardTransition`
  (mql → … → negotiation); final conversion is driven by
  :class:`~leadforge.mechanisms.hazards.ConversionHazard` (negotiation → closed_won).
- A small daily churn probability independently moves any non-terminal lead to
  ``closed_lost``.

Post-simulation entity creation
--------------------------------
- An :class:`~leadforge.schema.entities.OpportunityRow` is created for every
  lead that reached ``sql`` or any deeper stage.
- :class:`~leadforge.schema.entities.CustomerRow` and
  :class:`~leadforge.schema.entities.SubscriptionRow` are created only for
  converted leads (``closed_won``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.core.models import GenerationConfig
from leadforge.core.rng import RNGRoot
from leadforge.mechanisms.base import MechanismContext
from leadforge.mechanisms.policies import assign_mechanisms
from leadforge.mechanisms.transitions import StageSequence
from leadforge.schema.entities import (
    CustomerRow,
    LeadRow,
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    SubscriptionRow,
    TouchRow,
)
from leadforge.simulation.population import PopulationResult
from leadforge.simulation.state import LeadSimState
from leadforge.structure.graph import WorldGraph

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Daily churn probability from any active stage.
_DAILY_CHURN_RATE = 0.004

# Funnel stages that imply meaningful sales engagement → opportunity creation.
_SQL_OR_DEEPER = frozenset(
    {
        "sql",
        "demo_scheduled",
        "demo_completed",
        "proposal_sent",
        "negotiation",
        "closed_won",
    }
)

# Stages where a sales rep is actively working the deal.
_SALES_ACTIVE_STAGES = frozenset(
    {
        "sal",
        "sql",
        "demo_scheduled",
        "demo_completed",
        "proposal_sent",
        "negotiation",
    }
)

# Touch / session / activity catalogues.
_TOUCH_TYPES = ("email", "call", "linkedin_message", "content_download", "webinar")
_SESSION_TYPES = ("website", "pricing_page", "demo_page")
_ACTIVITY_TYPES = ("call", "email", "meeting", "demo")
_ACTIVITY_OUTCOMES = (
    "connected",
    "no_answer",
    "left_voicemail",
    "meeting_set",
    "demo_completed",
)

# ACV range (lo, hi) in USD by account employee band.
_EMPLOYEE_ACV_RANGES: dict[str, tuple[int, int]] = {
    "200-499": (15_000, 50_000),
    "500-999": (30_000, 80_000),
    "1000-1999": (50_000, 120_000),
    "2000+": (80_000, 200_000),
}
_DEFAULT_ACV_RANGE: tuple[int, int] = (20_000, 60_000)


# ---------------------------------------------------------------------------
# Public output type
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Fully simulated world output, ready for the rendering layer.

    All lists are in insertion order (chronological within each lead,
    ascending lead-index across leads).

    Args:
        leads: Updated :class:`~leadforge.schema.entities.LeadRow` list
            with simulation outcomes filled in.
    """

    leads: list[LeadRow]
    touches: list[TouchRow] = field(default_factory=list)
    sessions: list[SessionRow] = field(default_factory=list)
    sales_activities: list[SalesActivityRow] = field(default_factory=list)
    opportunities: list[OpportunityRow] = field(default_factory=list)
    customers: list[CustomerRow] = field(default_factory=list)
    subscriptions: list[SubscriptionRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate_world(
    config: GenerationConfig,
    population: PopulationResult,
    world_graph: WorldGraph,
) -> SimulationResult:
    """Run the discrete-time simulation for all leads in *population*.

    Iterates ``config.horizon_days`` daily steps.  On each step, every
    non-terminal lead is processed in population order: churn check, stage
    advance or final-close check, then event emission (touches, sessions,
    sales activities).  After the main loop, post-conversion entities are
    created.

    Args:
        config: Fully resolved generation configuration (counts, seed,
            horizon).
        population: Output of
            :func:`~leadforge.simulation.population.build_population`.
        world_graph: The sampled hidden world graph; its ``motif_family``
            attribute selects the appropriate mechanism parameters.

    Returns:
        A :class:`SimulationResult` with all nine relational tables
        populated.
    """
    root = RNGRoot(config.seed)
    mech_rng = root.child("mechanisms")
    sim_rng = root.child("simulation")

    mechanisms = assign_mechanisms(world_graph.motif_family, mech_rng)
    stage_seq = StageSequence()

    # Build lookup indexes.
    account_by_id = {a.account_id: a for a in population.accounts}
    contact_by_id = {c.contact_id: c for c in population.contacts}

    # Merge latent traits per lead: account + contact + lead latents.
    lat = population.latent_state
    merged_latents: dict[str, dict[str, float]] = {}
    for lead in population.leads:
        contact = contact_by_id[lead.contact_id]
        merged: dict[str, float] = {}
        merged.update(lat.account_latents.get(lead.account_id, {}))
        merged.update(lat.contact_latents.get(contact.contact_id, {}))
        merged.update(lat.lead_latents.get(lead.lead_id, {}))
        merged_latents[lead.lead_id] = merged

    # Initialise per-lead mutable state.
    states: dict[str, LeadSimState] = {
        lead.lead_id: LeadSimState(
            lead_id=lead.lead_id,
            current_stage=lead.current_stage,
            # Track leads already at sql from population initialisation.
            sql_day=0 if lead.current_stage in _SQL_OR_DEEPER else None,
        )
        for lead in population.leads
    }

    # Event row buffers and counters.
    touches: list[TouchRow] = []
    sessions: list[SessionRow] = []
    sales_activities: list[SalesActivityRow] = []
    touch_ctr = 0
    session_ctr = 0
    activity_ctr = 0

    # -------------------------------------------------------------------
    # Main simulation loop: t = 0 … horizon_days-1
    # -------------------------------------------------------------------
    for t in range(config.horizon_days):
        for lead in population.leads:
            state = states[lead.lead_id]
            if state.is_terminal:
                continue

            latents = merged_latents[lead.lead_id]
            ctx = MechanismContext(
                latents=latents,
                stage=state.current_stage,
                t=t,
                extra={"dwell_days": state.dwell_days},
            )

            # -- 1. Churn check ------------------------------------------
            if sim_rng.random() < _DAILY_CHURN_RATE:
                state.mark_churned(t)
                continue  # no events emitted on churn day

            # -- 2. Stage advance or conversion check --------------------
            if state.current_stage == "negotiation":
                # Final close: ConversionHazard decides closed_won.
                if mechanisms.conversion_hazard.sample(ctx, sim_rng):
                    state.mark_converted(t)
                    # Fall through to emit events on conversion day.
            else:
                # Funnel advancement: HazardTransition advances the stage.
                if mechanisms.stage_transition.sample(ctx, sim_rng):
                    next_s = stage_seq.next_stage(state.current_stage)
                    if next_s is not None:
                        state.advance_stage(next_s, t)

            # -- 3. Touches ----------------------------------------------
            lead_date = date.fromisoformat(lead.lead_created_at)
            event_date = (lead_date + timedelta(days=t)).isoformat()

            n_touches = mechanisms.touch_intensity.sample(ctx, sim_rng)
            for _ in range(n_touches):
                touch_ctr += 1
                touches.append(
                    TouchRow(
                        touch_id=make_id(ID_PREFIXES["touch"], touch_ctr),
                        lead_id=lead.lead_id,
                        touch_timestamp=event_date,
                        touch_type=sim_rng.choice(_TOUCH_TYPES),
                        touch_channel=lead.first_touch_channel,
                        touch_direction="inbound"
                        if lead.first_touch_channel == "inbound_marketing"
                        else "outbound",
                        campaign_id=None,
                    )
                )

            # -- 4. Sessions (≈30 % of touch-days) -----------------------
            if n_touches > 0 and sim_rng.random() < 0.30:
                session_ctr += 1
                at_demo_stage = state.current_stage in {
                    "demo_scheduled",
                    "demo_completed",
                }
                at_late_stage = state.current_stage in _SQL_OR_DEEPER
                sessions.append(
                    SessionRow(
                        session_id=make_id(ID_PREFIXES["session"], session_ctr),
                        lead_id=lead.lead_id,
                        session_timestamp=event_date,
                        session_type=sim_rng.choice(_SESSION_TYPES),
                        page_views=sim_rng.randint(1, 10),
                        pricing_page_views=sim_rng.randint(0, 2) if at_late_stage else 0,
                        demo_page_views=sim_rng.randint(0, 2) if at_demo_stage else 0,
                        session_duration_seconds=sim_rng.randint(60, 600),
                    )
                )

            # -- 5. Sales activities (≈20 % of active-stage days) --------
            if state.current_stage in _SALES_ACTIVE_STAGES and sim_rng.random() < 0.20:
                activity_ctr += 1
                sales_activities.append(
                    SalesActivityRow(
                        activity_id=make_id(ID_PREFIXES["sales_activity"], activity_ctr),
                        lead_id=lead.lead_id,
                        rep_id=lead.owner_rep_id,
                        activity_timestamp=event_date,
                        activity_type=sim_rng.choice(_ACTIVITY_TYPES),
                        activity_outcome=sim_rng.choice(_ACTIVITY_OUTCOMES),
                    )
                )

            # -- 6. Advance dwell counter --------------------------------
            state.dwell_days += 1

    # -------------------------------------------------------------------
    # Post-simulation: build final entity rows
    # -------------------------------------------------------------------
    updated_leads: list[LeadRow] = []
    opportunities: list[OpportunityRow] = []
    customers: list[CustomerRow] = []
    subscriptions: list[SubscriptionRow] = []
    opp_ctr = 0
    cust_ctr = 0
    sub_ctr = 0

    for lead in population.leads:
        state = states[lead.lead_id]
        lead_date = date.fromisoformat(lead.lead_created_at)

        is_sql = state.sql_day is not None or state.current_stage in _SQL_OR_DEEPER
        conv_ts: str | None = None
        if state.converted and state.conversion_day is not None:
            conv_ts = (lead_date + timedelta(days=state.conversion_day)).isoformat()

        updated_leads.append(
            LeadRow(
                lead_id=lead.lead_id,
                contact_id=lead.contact_id,
                account_id=lead.account_id,
                lead_created_at=lead.lead_created_at,
                lead_source=lead.lead_source,
                first_touch_channel=lead.first_touch_channel,
                current_stage=state.current_stage,
                owner_rep_id=lead.owner_rep_id,
                is_mql=True,  # all leads start at mql
                is_sql=is_sql,
                converted_within_90_days=state.converted,
                conversion_timestamp=conv_ts,
            )
        )

        # Opportunity: created when lead first reached sql or deeper.
        if is_sql:
            opp_ctr += 1
            opp_id = make_id(ID_PREFIXES["opportunity"], opp_ctr)
            opp_day = state.sql_day if state.sql_day is not None else 0
            opp_created_at = (lead_date + timedelta(days=opp_day)).isoformat()

            close_outcome: str | None = None
            closed_at: str | None = None
            if state.converted:
                close_outcome = "closed_won"
                closed_at = conv_ts
            elif state.churned and state.churn_day is not None:
                close_outcome = "closed_lost"
                closed_at = (lead_date + timedelta(days=state.churn_day)).isoformat()

            acct = account_by_id.get(lead.account_id)
            emp_band = acct.employee_band if acct else ""
            acv = _sample_acv(sim_rng, emp_band)

            opp = OpportunityRow(
                opportunity_id=opp_id,
                lead_id=lead.lead_id,
                created_at=opp_created_at,
                stage=state.current_stage,
                estimated_acv=acv,
                close_outcome=close_outcome,
                closed_at=closed_at,
            )
            opportunities.append(opp)

            # Customer + subscription: converted leads only.
            if state.converted:
                cust_ctr += 1
                cust_id = make_id(ID_PREFIXES["customer"], cust_ctr)
                customers.append(
                    CustomerRow(
                        customer_id=cust_id,
                        opportunity_id=opp_id,
                        account_id=lead.account_id,
                        customer_start_at=conv_ts,  # type: ignore[arg-type]
                    )
                )

                sub_ctr += 1
                sub_id = make_id(ID_PREFIXES["subscription"], sub_ctr)
                subscriptions.append(
                    SubscriptionRow(
                        subscription_id=sub_id,
                        customer_id=cust_id,
                        plan_name=_plan_from_acv(acv),
                        subscription_start_at=conv_ts,  # type: ignore[arg-type]
                    subscription_status="active",
                    )
                )

    return SimulationResult(
        leads=updated_leads,
        touches=touches,
        sessions=sessions,
        sales_activities=sales_activities,
        opportunities=opportunities,
        customers=customers,
        subscriptions=subscriptions,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _sample_acv(rng: random.Random, employee_band: str) -> int:
    """Draw a random ACV (USD) appropriate for *employee_band*."""
    lo, hi = _EMPLOYEE_ACV_RANGES.get(employee_band, _DEFAULT_ACV_RANGE)
    return rng.randint(lo, hi)


def _plan_from_acv(acv: int) -> str:
    """Map ACV to a subscription plan tier name."""
    if acv < 30_000:
        return "starter"
    if acv < 80_000:
        return "growth"
    return "enterprise"

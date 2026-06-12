"""Lifecycle hazard functions — latent state + mechanism params → probabilities.

Three pure functions convert a customer's latent traits and the motif-family
mechanism parameters (from :mod:`leadforge.schemes.lifecycle.mechanisms`) into
per-step event probabilities:

- :func:`churn_probability` — weekly churn hazard with onboarding elevation and
  a contract-anniversary renewal spike.
- :func:`expansion_probability` — weekly upsell/seat-add propensity, optionally
  modulated by the current feature-depth health signal.
- :func:`payment_failure_probability` — monthly invoice-failure probability.

All three are **deterministic** — they take no RNG and perform no draws.  The
weekly simulation engine owns every Bernoulli sample; these functions only
compute the probability for the draw.  This keeps the hazard math directly
testable (exact values, monotonicity, spike shape) without seeding.

Latent modulation (proportional-hazards style)
----------------------------------------------
Each mechanism's ``latent_weights`` are applied as a Cox-style multiplicative
factor on the base rate::

    multiplier = exp( Σ_i  w_i · (latent_i − 0.5) )

Latents are centred at the neutral 0.5, so a customer with all-neutral traits
gets multiplier 1.0 (base rate unchanged).  Per the sign convention set in
``mechanisms.py``: a **negative** weight on a trait means a *high* trait value
*reduces* the probability (e.g. ``latent_product_fit: -2.0`` on churn).
A trait missing from the latent dict is treated as neutral (0.5) — it
contributes nothing, rather than raising or silently zeroing the hazard.

Tenure shape
------------
The churn hazard is elevated during onboarding (decreasing-hazard Weibull
behaviour, approximated by an exponential decay from
``_ONBOARDING_PEAK_MULTIPLIER`` toward 1.0 with time-constant
``_ONBOARDING_DECAY_WEEKS``) and spikes at each contract anniversary
(:func:`is_renewal_week`), where the ``renewal_hazard_multiplier`` and the
renewal-specific latent weights (champion-fights-for-renewal) apply.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from leadforge.schemes.lifecycle.mechanisms import (
        ChurnHazardParams,
        ExpansionPropensityParams,
        PaymentFailureParams,
    )

__all__ = [
    "churn_probability",
    "expansion_probability",
    "is_renewal_week",
    "payment_failure_probability",
]

# Probabilities are capped below 1.0 so extreme latent combinations never make
# an event *certain* — the simulation stays stochastic at the tails, and a
# mis-calibrated base rate degrades visibly instead of saturating silently.
_MAX_PROBABILITY = 0.95

# Neutral latent value: traits absent from the latent dict contribute nothing.
_NEUTRAL_LATENT = 0.5

# Onboarding churn elevation: hazard starts at peak × base in week 0 and decays
# exponentially toward 1× with this time-constant.  At week 12 the residual
# elevation is < 8% — effectively steady-state.
_ONBOARDING_PEAK_MULTIPLIER = 2.5
_ONBOARDING_DECAY_WEEKS = 4.0

# Weeks per month for contract-anniversary arithmetic (52-week year).
_WEEKS_PER_MONTH = 52.0 / 12.0


def _latent_multiplier(latents: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return the Cox-style multiplicative factor for *latents* under *weights*."""
    score = sum(
        weight * (latents.get(trait, _NEUTRAL_LATENT) - _NEUTRAL_LATENT)
        for trait, weight in weights.items()
    )
    return math.exp(score)


def _onboarding_multiplier(week_of_tenure: int) -> float:
    """Return the early-tenure churn elevation factor (≥ 1.0, → 1.0 with tenure)."""
    return 1.0 + (_ONBOARDING_PEAK_MULTIPLIER - 1.0) * math.exp(
        -week_of_tenure / _ONBOARDING_DECAY_WEEKS
    )


def is_renewal_week(week_of_tenure: int, contract_term_months: int) -> bool:
    """Return ``True`` iff *week_of_tenure* contains a contract anniversary.

    Anniversaries fall at ``round(k · contract_term_months · 52/12)`` weeks for
    ``k = 1, 2, …`` — e.g. a 12-month contract renews at weeks 52, 104, …; a
    24-month contract at weeks 104, 208, ….  Week 0 (signing week) is never a
    renewal week.

    Exposed publicly so the simulation engine can use the same boundary to emit
    ``renewal`` events that it uses for the churn spike.

    Raises:
        ValueError: if *week_of_tenure* is negative or *contract_term_months*
            is not a positive integer.
    """
    if week_of_tenure < 0:
        raise ValueError(f"week_of_tenure must be >= 0, got {week_of_tenure}")
    if contract_term_months < 1:
        raise ValueError(f"contract_term_months must be >= 1, got {contract_term_months}")
    if week_of_tenure == 0:
        return False
    term_weeks = contract_term_months * _WEEKS_PER_MONTH
    k = round(week_of_tenure / term_weeks)
    return k >= 1 and round(k * term_weeks) == week_of_tenure


def churn_probability(
    params: ChurnHazardParams,
    latents: Mapping[str, float],
    week_of_tenure: int,
    contract_term_months: int,
) -> float:
    """Return the weekly churn probability for one customer at one week.

    Composition: ``base_weekly_rate × latent multiplier × onboarding
    elevation``, and on a renewal week additionally ``×
    renewal_hazard_multiplier × renewal latent multiplier``.  Capped at
    ``_MAX_PROBABILITY``.

    Args:
        params: Motif-family churn parameters from
            :func:`~leadforge.schemes.lifecycle.mechanisms.assign_lifecycle_mechanisms`.
        latents: Merged customer + account latent traits in ``[0, 1]``.
            Missing traits are treated as neutral (0.5).
        week_of_tenure: Whole weeks since ``customer_start_at`` (0-based).
        contract_term_months: The customer's contract term, for the
            anniversary spike.

    Raises:
        ValueError: via :func:`is_renewal_week` on negative tenure or
            non-positive contract term.
    """
    p = params.base_weekly_rate
    p *= _latent_multiplier(latents, params.latent_weights)
    p *= _onboarding_multiplier(week_of_tenure)
    if is_renewal_week(week_of_tenure, contract_term_months):
        p *= params.renewal_hazard_multiplier
        p *= _latent_multiplier(latents, params.renewal_latent_weights)
    return min(p, _MAX_PROBABILITY)


def expansion_probability(
    params: ExpansionPropensityParams,
    latents: Mapping[str, float],
    feature_depth_score: float | None = None,
) -> float:
    """Return the weekly expansion (upsell / seat-add) probability.

    Composition: ``base_weekly_rate × latent multiplier``, optionally
    ``× (0.5 + feature_depth_score)`` when the current health signal is
    supplied — depth 0.5 is neutral (×1.0), full depth 1.0 raises the
    propensity by half, zero depth halves it.  Capped at ``_MAX_PROBABILITY``.

    Args:
        params: Motif-family expansion parameters.
        latents: Merged latent traits; missing traits are neutral.
        feature_depth_score: Optional current ``feature_depth_score`` health
            signal in ``[0, 1]``; ``None`` skips the health modulation (the
            engine passes it once health signals exist for the week).
    """
    p = params.base_weekly_rate * _latent_multiplier(latents, params.latent_weights)
    if feature_depth_score is not None:
        if not 0.0 <= feature_depth_score <= 1.0:
            raise ValueError(f"feature_depth_score must be in [0, 1], got {feature_depth_score}")
        p *= _NEUTRAL_LATENT + feature_depth_score
    return min(p, _MAX_PROBABILITY)


def payment_failure_probability(
    params: PaymentFailureParams,
    latents: Mapping[str, float],
) -> float:
    """Return the monthly invoice payment-failure probability.

    Composition: ``base_monthly_rate × latent multiplier`` (the dominant weight
    is on ``latent_budget_stability``, negative — stable budgets fail less).
    Capped at ``_MAX_PROBABILITY``.

    Args:
        params: Motif-family payment-failure parameters.
        latents: Merged latent traits; missing traits are neutral.
    """
    p = params.base_monthly_rate * _latent_multiplier(latents, params.latent_weights)
    return min(p, _MAX_PROBABILITY)

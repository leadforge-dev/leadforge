"""Lifecycle mechanism policies — parameter tables and motif dispatch.

:func:`assign_lifecycle_mechanisms` is the single public entry point.  It maps
a retention motif family to a :class:`LifecycleMechanismAssignment` carrying the
concrete parameter values the simulation engine uses on each weekly step.

The three mechanism types
-------------------------
**Churn hazard** — weekly probability a customer churns.  Two-component:

- *Background rate*: low constant hazard driven by ``latent_product_fit``
  (poor fit → higher churn) and ``latent_champion_strength``.
- *Renewal spike*: at contract-anniversary weeks the hazard multiplies by
  ``renewal_hazard_multiplier``; the exact spike is reduced by
  ``latent_champion_strength`` (a strong champion fights hard at renewal).

**Expansion propensity** — weekly probability of an upsell/seat-add event,
driven by ``latent_adoption_velocity`` and ``feature_depth_score`` health
signals.  The resulting MRR delta is drawn from
``expansion_mrr_frac_range = (lo, hi)`` × current MRR.

**Payment failure** — monthly billing event; probability of a failed invoice
driven by ``latent_budget_stability`` (low stability → higher failure rate).
Failed invoices enter a dunning window; unrecovered invoices escalate to churn.

Motif-family tuning
-------------------
Each of the five retention motif families tilts the base parameters so the DGP
is consistent with the population biases sampled in
:mod:`leadforge.schemes.lifecycle.population`:

- ``product_led_retention``  — low churn (strong product fit), moderate expansion.
- ``relationship_led_retention`` — moderate churn driven by champion strength at renewal.
- ``expansion_led_growth``   — very low churn, high expansion; pLTV variance from upsell.
- ``payment_fragile``        — moderate-to-high churn triggered by payment failure.
- ``churner_dominated``      — high background churn; strong early-warning signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

# ---------------------------------------------------------------------------
# Mechanism assignment dataclasses
# ---------------------------------------------------------------------------


__all__ = [
    "ChurnHazardParams",
    "ExpansionPropensityParams",
    "LifecycleMechanismAssignment",
    "PaymentFailureParams",
    "assign_lifecycle_mechanisms",
    "mechanism_params_for_motif",
]


@dataclass(frozen=True)
class ChurnHazardParams:
    """Parameters for the weekly churn hazard.

    Attributes:
        base_weekly_rate: Unconditional weekly churn probability before
            latent-score modulation.
        latent_weights: Read-only ``{trait: weight}`` mapping — positive weights
            increase churn, negative decrease it.  Wrapped in ``MappingProxyType``
            so the simulation engine cannot accidentally mutate the shared table.
        renewal_hazard_multiplier: Factor by which the hazard is amplified at a
            contract-anniversary week (e.g. ``10.0`` → 10× background).
        renewal_latent_weights: Trait weights used *only* at the renewal spike to
            model the champion fighting-for-renewal effect.  Also read-only.
    """

    base_weekly_rate: float
    latent_weights: MappingProxyType  # type: ignore[type-arg]
    renewal_hazard_multiplier: float
    renewal_latent_weights: MappingProxyType  # type: ignore[type-arg]


@dataclass(frozen=True)
class ExpansionPropensityParams:
    """Parameters for the weekly expansion (upsell / seat-add) propensity.

    Attributes:
        base_weekly_rate: Unconditional weekly probability of an expansion event.
        latent_weights: Read-only trait-weight mapping.
        expansion_mrr_frac_range: ``(lo, hi)`` — expansion MRR delta drawn
            uniformly from ``[lo * current_mrr, hi * current_mrr]``.
    """

    base_weekly_rate: float
    latent_weights: MappingProxyType  # type: ignore[type-arg]
    expansion_mrr_frac_range: tuple[float, float]


@dataclass(frozen=True)
class PaymentFailureParams:
    """Parameters for the monthly payment-failure event.

    Attributes:
        base_monthly_rate: Unconditional monthly probability of a payment failure.
        latent_weights: Read-only trait-weight mapping (negative
            ``latent_budget_stability`` increases failure probability).
        dunning_weeks: Weeks before a failed invoice is escalated — either
            recovered (``payment_recovered``) or written off and triggers churn.
        recovery_rate: Probability a failed invoice is recovered within the
            dunning window (vs. written off → churn).
    """

    base_monthly_rate: float
    latent_weights: MappingProxyType  # type: ignore[type-arg]
    dunning_weeks: int
    recovery_rate: float


@dataclass(frozen=True)
class LifecycleMechanismAssignment:
    """All mechanism parameters for one lifecycle simulation run.

    Produced by :func:`assign_lifecycle_mechanisms` and consumed by the
    weekly simulation engine.
    """

    motif_family: str
    churn_hazard: ChurnHazardParams
    expansion_propensity: ExpansionPropensityParams
    payment_failure: PaymentFailureParams


# ---------------------------------------------------------------------------
# Per-motif parameter tables
# ---------------------------------------------------------------------------

# Churn hazard base weekly rates.
# IMPORTANT — the per-motif "% annual" figures below are the BASE-RATE-ONLY
# equivalents (1 - (1-r)^52) at neutral latents.  The hazard functions in
# hazards.py add material churn mass on top: the onboarding elevation
# contributes ~6.8 x base_rate of extra first-year mass and each renewal spike
# adds (multiplier - 1) x base_rate, so true first-year churn runs roughly
# 5-14 points above these figures (e.g. churner_dominated ~52%, not 37.5%).
# Final calibration against the difficulty-profile bands
#   intro [0.10, 0.20] / intermediate [0.20, 0.35] / advanced [0.30, 0.50]
# happens in the engine tests (LTV-Pk), where these base rates are expected to
# be tuned DOWN to land inside the bands once the full tenure shape applies.
_CHURN_BASE_WEEKLY: dict[str, float] = {
    # Exact annual equivalent: 1 - (1-r)^52.
    "product_led_retention": 0.0042,  # 19.7% annual
    "relationship_led_retention": 0.0055,  # 24.9% annual
    "expansion_led_growth": 0.0028,  # 13.6% annual (lowest — high-fit customers)
    "payment_fragile": 0.0060,  # 26.9% annual (high base; mostly payment-driven)
    "churner_dominated": 0.0090,  # 37.5% annual
}

# Latent-trait weights for the background churn hazard.
# Positive weights *increase* churn probability (bad signal); negative *decrease* it.
_CHURN_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "product_led_retention": {
        "latent_product_fit": -2.0,  # strong fit → low churn
        "latent_adoption_velocity": -0.8,
        "latent_champion_strength": -0.5,
    },
    "relationship_led_retention": {
        "latent_champion_strength": -2.0,  # champion quality dominates
        "latent_product_fit": -0.8,
        "latent_organizational_stability": -0.6,
    },
    "expansion_led_growth": {
        "latent_adoption_velocity": -1.5,
        "latent_product_fit": -1.5,
        "latent_budget_stability": -0.5,
    },
    "payment_fragile": {
        "latent_budget_stability": -2.5,  # budget dominates
        "latent_organizational_stability": -1.0,
        "latent_product_fit": -0.5,
    },
    "churner_dominated": {
        "latent_product_fit": -1.8,
        "latent_champion_strength": -1.2,
        "latent_adoption_velocity": -0.8,
    },
}

# Renewal-date hazard spike multiplier.
_RENEWAL_HAZARD_MULTIPLIER: dict[str, float] = {
    "product_led_retention": 6.0,
    "relationship_led_retention": 12.0,  # renewal is the key decision point
    "expansion_led_growth": 4.0,
    "payment_fragile": 8.0,
    "churner_dominated": 10.0,
}

# Trait weights at the renewal spike (champion fighting for renewal).
_RENEWAL_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "product_led_retention": {
        "latent_champion_strength": -1.5,
        "latent_product_fit": -1.0,
    },
    "relationship_led_retention": {
        "latent_champion_strength": -2.5,  # champion strength matters most here
        "latent_organizational_stability": -1.0,
    },
    "expansion_led_growth": {
        "latent_adoption_velocity": -1.5,
        "latent_champion_strength": -1.0,
    },
    "payment_fragile": {
        "latent_budget_stability": -2.0,
        "latent_champion_strength": -1.0,
    },
    "churner_dominated": {
        "latent_champion_strength": -1.5,
        "latent_product_fit": -1.0,
    },
}

# Expansion propensity base weekly rates.
# Calibrated to yield ~10–30% annual expansion rates at neutral latents.
_EXPANSION_BASE_WEEKLY: dict[str, float] = {
    "product_led_retention": 0.0045,  # ~21% annual
    "relationship_led_retention": 0.0030,  # ~15% annual
    "expansion_led_growth": 0.0075,  # ~32% annual — the pLTV-variance driver
    "payment_fragile": 0.0020,  # ~10% annual (budget-constrained)
    "churner_dominated": 0.0018,  # ~9% annual (churners don't expand)
}

# Latent weights for expansion propensity.
_EXPANSION_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "product_led_retention": {
        "latent_adoption_velocity": 1.5,
        "latent_product_fit": 1.0,
    },
    "relationship_led_retention": {
        "latent_champion_strength": 1.5,
        "latent_adoption_velocity": 0.8,
    },
    "expansion_led_growth": {
        "latent_adoption_velocity": 2.0,
        "latent_product_fit": 1.0,
        "latent_budget_stability": 0.5,
    },
    "payment_fragile": {
        "latent_adoption_velocity": 1.0,
        "latent_budget_stability": 1.5,  # only expands when budget is stable
    },
    "churner_dominated": {
        "latent_adoption_velocity": 1.0,
        "latent_product_fit": 0.8,
    },
}

# MRR delta fraction range (lo, hi) for expansion events.
# Expansion MRR = randint(lo * current_mrr, hi * current_mrr).
_EXPANSION_MRR_FRAC: dict[str, tuple[float, float]] = {
    "product_led_retention": (0.20, 0.60),
    "relationship_led_retention": (0.15, 0.50),
    "expansion_led_growth": (0.30, 1.00),  # large expansions drive the tail
    "payment_fragile": (0.10, 0.30),
    "churner_dominated": (0.10, 0.25),
}

# Payment-failure base monthly rates.
_PAYMENT_FAILURE_BASE_MONTHLY: dict[str, float] = {
    "product_led_retention": 0.015,
    "relationship_led_retention": 0.020,
    "expansion_led_growth": 0.012,
    "payment_fragile": 0.080,  # high — financial fragility is the defining trait
    "churner_dominated": 0.030,
}

# Latent weights for payment failure.
# Negative latent_budget_stability increases failure probability.
_PAYMENT_FAILURE_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "product_led_retention": {
        "latent_budget_stability": -1.5,
    },
    "relationship_led_retention": {
        "latent_budget_stability": -1.5,
        "latent_organizational_stability": -0.5,
    },
    "expansion_led_growth": {
        "latent_budget_stability": -1.2,
    },
    "payment_fragile": {
        "latent_budget_stability": -3.0,  # dominant driver
        "latent_organizational_stability": -1.0,
    },
    "churner_dominated": {
        "latent_budget_stability": -2.0,
    },
}

# Dunning period (weeks) before a failed invoice is escalated.
_DUNNING_WEEKS: dict[str, int] = {
    "product_led_retention": 4,
    "relationship_led_retention": 4,
    "expansion_led_growth": 4,
    "payment_fragile": 3,  # shorter grace — fragile accounts have less runway
    "churner_dominated": 3,
}

# Probability a failed invoice is recovered within the dunning window
# (vs. written off → forced churn).
_RECOVERY_RATE: dict[str, float] = {
    "product_led_retention": 0.70,
    "relationship_led_retention": 0.65,
    "expansion_led_growth": 0.75,
    "payment_fragile": 0.40,  # low — these accounts are genuinely fragile
    "churner_dominated": 0.50,
}

# Fallback values for unknown motif families.
_DEFAULT_CHURN_BASE_WEEKLY: float = 0.0055
_DEFAULT_CHURN_LATENT_WEIGHTS: dict[str, float] = {
    "latent_product_fit": -1.5,
    "latent_champion_strength": -0.8,
}
_DEFAULT_RENEWAL_MULTIPLIER: float = 8.0
_DEFAULT_RENEWAL_LATENT_WEIGHTS: dict[str, float] = {
    "latent_champion_strength": -1.5,
}
_DEFAULT_EXPANSION_BASE_WEEKLY: float = 0.0035
_DEFAULT_EXPANSION_LATENT_WEIGHTS: dict[str, float] = {
    "latent_adoption_velocity": 1.2,
}
_DEFAULT_EXPANSION_MRR_FRAC: tuple[float, float] = (0.20, 0.60)
_DEFAULT_PAYMENT_FAILURE_BASE_MONTHLY: float = 0.025
_DEFAULT_PAYMENT_FAILURE_LATENT_WEIGHTS: dict[str, float] = {
    "latent_budget_stability": -1.5,
}
_DEFAULT_DUNNING_WEEKS: int = 4
_DEFAULT_RECOVERY_RATE: float = 0.60


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_lifecycle_mechanisms(motif_family: str) -> LifecycleMechanismAssignment:
    """Return a :class:`LifecycleMechanismAssignment` for *motif_family*.

    Looks up pre-calibrated parameter tables and constructs the three mechanism
    param objects consumed by the weekly simulation engine.  Unrecognised motif
    families fall back to sensible intermediate-tier defaults rather than
    raising, so a new motif family can be prototyped before its tables are
    calibrated.

    Args:
        motif_family: One of the five registered lifecycle retention motif
            families (see :data:`~leadforge.schemes.lifecycle.population.LIFECYCLE_MOTIF_FAMILIES`).

    Returns:
        A fully populated :class:`LifecycleMechanismAssignment`.
    """
    churn = ChurnHazardParams(
        base_weekly_rate=_CHURN_BASE_WEEKLY.get(motif_family, _DEFAULT_CHURN_BASE_WEEKLY),
        latent_weights=MappingProxyType(
            _CHURN_LATENT_WEIGHTS.get(motif_family, _DEFAULT_CHURN_LATENT_WEIGHTS)
        ),
        renewal_hazard_multiplier=_RENEWAL_HAZARD_MULTIPLIER.get(
            motif_family, _DEFAULT_RENEWAL_MULTIPLIER
        ),
        renewal_latent_weights=MappingProxyType(
            _RENEWAL_LATENT_WEIGHTS.get(motif_family, _DEFAULT_RENEWAL_LATENT_WEIGHTS)
        ),
    )

    expansion = ExpansionPropensityParams(
        base_weekly_rate=_EXPANSION_BASE_WEEKLY.get(motif_family, _DEFAULT_EXPANSION_BASE_WEEKLY),
        latent_weights=MappingProxyType(
            _EXPANSION_LATENT_WEIGHTS.get(motif_family, _DEFAULT_EXPANSION_LATENT_WEIGHTS)
        ),
        expansion_mrr_frac_range=_EXPANSION_MRR_FRAC.get(motif_family, _DEFAULT_EXPANSION_MRR_FRAC),
    )

    payment = PaymentFailureParams(
        base_monthly_rate=_PAYMENT_FAILURE_BASE_MONTHLY.get(
            motif_family, _DEFAULT_PAYMENT_FAILURE_BASE_MONTHLY
        ),
        latent_weights=MappingProxyType(
            _PAYMENT_FAILURE_LATENT_WEIGHTS.get(
                motif_family, _DEFAULT_PAYMENT_FAILURE_LATENT_WEIGHTS
            )
        ),
        dunning_weeks=_DUNNING_WEEKS.get(motif_family, _DEFAULT_DUNNING_WEEKS),
        recovery_rate=_RECOVERY_RATE.get(motif_family, _DEFAULT_RECOVERY_RATE),
    )

    return LifecycleMechanismAssignment(
        motif_family=motif_family,
        churn_hazard=churn,
        expansion_propensity=expansion,
        payment_failure=payment,
    )


def mechanism_params_for_motif(motif_family: str) -> dict[str, Any]:
    """Return a plain dict of the mechanism parameter tables for *motif_family*.

    Useful for inspection and testing without constructing mechanism objects.
    Derives directly from :func:`assign_lifecycle_mechanisms` so it is always
    consistent with the actual assignment — no duplicated lookup logic.
    """
    a = assign_lifecycle_mechanisms(motif_family)
    return {
        "motif_family": a.motif_family,
        "churn_base_weekly_rate": a.churn_hazard.base_weekly_rate,
        "churn_latent_weights": dict(a.churn_hazard.latent_weights),
        "renewal_hazard_multiplier": a.churn_hazard.renewal_hazard_multiplier,
        "renewal_latent_weights": dict(a.churn_hazard.renewal_latent_weights),
        "expansion_base_weekly_rate": a.expansion_propensity.base_weekly_rate,
        "expansion_latent_weights": dict(a.expansion_propensity.latent_weights),
        "expansion_mrr_frac_range": a.expansion_propensity.expansion_mrr_frac_range,
        "payment_failure_base_monthly_rate": a.payment_failure.base_monthly_rate,
        "payment_failure_latent_weights": dict(a.payment_failure.latent_weights),
        "dunning_weeks": a.payment_failure.dunning_weeks,
        "recovery_rate": a.payment_failure.recovery_rate,
    }

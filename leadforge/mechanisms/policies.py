"""Mechanism assignment policy — wires mechanism instances to a world.

:func:`assign_mechanisms` is the single entry point.  It inspects the active
motif family and constructs a :class:`~leadforge.mechanisms.base.MechanismAssignment`
whose parameters reflect the structural bias of that world.

Motif-family parameter tuning
------------------------------
Each motif family tilts the mechanism parameters so the DGP is consistent
with the hidden world structure selected by the sampler:

- **fit_dominant** — conversion hazard weighted heavily on account fit and
  budget readiness; stage transition also fit-driven.
- **intent_dominant** — conversion hazard weighted on engagement propensity
  and problem awareness.
- **sales_execution_sensitive** — stage transition heavily penalised by sales
  friction; low base conversion rate.
- **demo_trial_mediated** — touch intensity is higher; conversion gated on
  engagement.
- **buying_committee_friction** — very low base conversion rate; authority
  and friction interact.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import MechanismAssignment
from leadforge.mechanisms.counts import RecencyDecayIntensity
from leadforge.mechanisms.hazards import ConversionHazard
from leadforge.mechanisms.measurement import NoisyProxy
from leadforge.mechanisms.scores import LatentScore
from leadforge.mechanisms.transitions import HazardTransition

# ---------------------------------------------------------------------------
# Motif-family parameter tables
# ---------------------------------------------------------------------------

# Each entry: {latent_key: weight} for the LatentScore used by the
# ConversionHazard.  Positive = facilitates conversion; negative = inhibits.
_CONVERSION_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "fit_dominant": {
        "latent_account_fit": 2.5,
        "latent_budget_readiness": 1.5,
        "latent_problem_awareness": 0.5,
        "latent_engagement_propensity": 0.5,
        "latent_sales_friction": -0.5,
    },
    "intent_dominant": {
        "latent_engagement_propensity": 2.5,
        "latent_problem_awareness": 1.5,
        "latent_account_fit": 0.5,
        "latent_sales_friction": -0.5,
    },
    "sales_execution_sensitive": {
        "latent_account_fit": 1.0,
        "latent_engagement_propensity": 1.0,
        "latent_responsiveness": 1.5,
        "latent_sales_friction": -2.0,
    },
    "demo_trial_mediated": {
        "latent_engagement_propensity": 2.0,
        "latent_problem_awareness": 1.0,
        "latent_account_fit": 1.0,
        "latent_sales_friction": -0.5,
    },
    "buying_committee_friction": {
        "latent_account_fit": 1.5,
        "latent_contact_authority": 1.5,
        "latent_budget_readiness": 1.0,
        "latent_sales_friction": -2.5,
    },
}

# Conversion hazard base_rate and scale per motif family.
_HAZARD_PARAMS: dict[str, dict[str, float]] = {
    "fit_dominant": {"base_rate": 0.008, "scale": 0.06},
    "intent_dominant": {"base_rate": 0.010, "scale": 0.07},
    "sales_execution_sensitive": {"base_rate": 0.004, "scale": 0.05},
    "demo_trial_mediated": {"base_rate": 0.007, "scale": 0.06},
    "buying_committee_friction": {"base_rate": 0.003, "scale": 0.04},
}

# Stage-transition HazardTransition score weights per motif family.
_TRANSITION_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "fit_dominant": {
        "latent_account_fit": 2.0,
        "latent_problem_awareness": 1.0,
        "latent_responsiveness": 0.5,
    },
    "intent_dominant": {
        "latent_engagement_propensity": 2.0,
        "latent_problem_awareness": 1.5,
        "latent_responsiveness": 0.5,
    },
    "sales_execution_sensitive": {
        "latent_responsiveness": 2.0,
        "latent_engagement_propensity": 1.0,
        "latent_sales_friction": -1.5,
    },
    "demo_trial_mediated": {
        "latent_engagement_propensity": 2.5,
        "latent_account_fit": 0.5,
    },
    "buying_committee_friction": {
        "latent_contact_authority": 2.0,
        "latent_account_fit": 1.0,
        "latent_sales_friction": -2.0,
    },
}

# Touch intensity (RecencyDecayIntensity) base_rate per motif family.
_TOUCH_BASE_RATES: dict[str, float] = {
    "fit_dominant": 0.40,
    "intent_dominant": 0.55,
    "sales_execution_sensitive": 0.35,
    "demo_trial_mediated": 0.60,
    "buying_committee_friction": 0.30,
}

# Fallback weights/params for unknown motif families.
_DEFAULT_CONVERSION_WEIGHTS: dict[str, float] = {
    "latent_account_fit": 1.0,
    "latent_engagement_propensity": 1.0,
    "latent_sales_friction": -0.5,
}
_DEFAULT_HAZARD_PARAMS: dict[str, float] = {"base_rate": 0.006, "scale": 0.05}
_DEFAULT_TOUCH_BASE_RATE: float = 0.40


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_mechanisms(
    motif_family: str,
    rng: random.Random,
) -> MechanismAssignment:
    """Build a :class:`~leadforge.mechanisms.base.MechanismAssignment` for *motif_family*.

    Parameters are tuned to the structural bias of the motif family so the
    resulting simulation is consistent with the hidden world sampled by
    :func:`~leadforge.structure.sampler.sample_hidden_graph`.

    Args:
        motif_family: Name of the active motif family (e.g. ``"fit_dominant"``).
        rng: Seeded :class:`random.Random` instance for any stochastic
            parameter perturbation (currently unused but reserved for future
            use so the signature is stable).

    Returns:
        A fully populated :class:`~leadforge.mechanisms.base.MechanismAssignment`.
    """
    conv_weights = _CONVERSION_SCORE_WEIGHTS.get(motif_family, _DEFAULT_CONVERSION_WEIGHTS)
    hazard_p = _HAZARD_PARAMS.get(motif_family, _DEFAULT_HAZARD_PARAMS)
    trans_weights = _TRANSITION_SCORE_WEIGHTS.get(motif_family, _DEFAULT_CONVERSION_WEIGHTS)
    touch_rate = _TOUCH_BASE_RATES.get(motif_family, _DEFAULT_TOUCH_BASE_RATE)

    conversion_hazard = ConversionHazard(
        score_mech=LatentScore(weights=conv_weights, bias=-1.5),
        base_rate=hazard_p["base_rate"],
        scale=hazard_p["scale"],
    )

    stage_transition = HazardTransition(
        score_mech=LatentScore(weights=trans_weights, bias=-1.0),
        base_rate=0.05,
        scale=0.15,
        min_dwell_days=2,
    )

    touch_intensity = RecencyDecayIntensity(
        base_rate=touch_rate,
        decay_factor=0.97,
        floor_rate=0.02,
    )

    measurement = NoisyProxy(
        latent_key="latent_account_fit",
        noise_std=0.10,
        missing_rate=0.05,
    )

    return MechanismAssignment(
        motif_family=motif_family,
        conversion_hazard=conversion_hazard,
        stage_transition=stage_transition,
        touch_intensity=touch_intensity,
        measurement=measurement,
    )


def mechanism_params_for_motif(motif_family: str) -> dict[str, Any]:
    """Return a plain dict of the parameter tables for *motif_family*.

    Useful for inspection, testing, and mechanism summary rendering without
    constructing mechanism objects.
    """
    return {
        "motif_family": motif_family,
        "conversion_score_weights": _CONVERSION_SCORE_WEIGHTS.get(
            motif_family, _DEFAULT_CONVERSION_WEIGHTS
        ),
        "hazard_params": _HAZARD_PARAMS.get(motif_family, _DEFAULT_HAZARD_PARAMS),
        "transition_score_weights": _TRANSITION_SCORE_WEIGHTS.get(
            motif_family, _DEFAULT_CONVERSION_WEIGHTS
        ),
        "touch_base_rate": _TOUCH_BASE_RATES.get(motif_family, _DEFAULT_TOUCH_BASE_RATE),
    }

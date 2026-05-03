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
from typing import TYPE_CHECKING, Any

from leadforge.mechanisms.base import MechanismAssignment

if TYPE_CHECKING:
    from leadforge.core.models import DifficultyParams
from leadforge.mechanisms.counts import LatentDecayIntensity, RecencyDecayIntensity
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

# Latent weights for touch intensity (LatentDecayIntensity) per motif family.
# These make touch emission depend on the same latent traits that drive
# conversion, creating a causal link: latent → touches AND latent → conversion.
_TOUCH_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "fit_dominant": {
        "latent_account_fit": 1.5,
        "latent_engagement_propensity": 1.0,
        "latent_problem_awareness": 0.5,
    },
    "intent_dominant": {
        "latent_engagement_propensity": 2.0,
        "latent_problem_awareness": 1.0,
        "latent_account_fit": 0.5,
    },
    "sales_execution_sensitive": {
        "latent_engagement_propensity": 1.5,
        "latent_responsiveness": 1.0,
        "latent_account_fit": 0.5,
    },
    "demo_trial_mediated": {
        "latent_engagement_propensity": 2.0,
        "latent_problem_awareness": 1.0,
        "latent_account_fit": 0.5,
    },
    "buying_committee_friction": {
        "latent_contact_authority": 1.0,
        "latent_engagement_propensity": 1.0,
        "latent_account_fit": 0.5,
    },
}
_DEFAULT_TOUCH_LATENT_WEIGHTS: dict[str, float] = {
    "latent_engagement_propensity": 1.0,
    "latent_account_fit": 0.5,
}

# Follow-up latent weights: used AFTER the followup day (post-snapshot).
# These emphasise conversion-predictive latent dimensions that are WEAKLY
# represented in pre-snapshot features (budget_readiness, authority, process
# maturity).  This models sales teams learning which deals are real during
# qualification and adjusting follow-up intensity accordingly.
_FOLLOWUP_LATENT_WEIGHTS: dict[str, dict[str, float]] = {
    "fit_dominant": {
        "latent_budget_readiness": 2.5,
        "latent_account_fit": 1.5,
        "latent_process_maturity": 1.0,
    },
    "intent_dominant": {
        "latent_problem_awareness": 2.0,
        "latent_budget_readiness": 1.5,
        "latent_engagement_propensity": 1.0,
    },
    "sales_execution_sensitive": {
        "latent_responsiveness": 2.0,
        "latent_budget_readiness": 1.5,
        "latent_account_fit": 1.0,
    },
    "demo_trial_mediated": {
        "latent_problem_awareness": 2.0,
        "latent_budget_readiness": 1.5,
        "latent_account_fit": 1.0,
    },
    "buying_committee_friction": {
        "latent_contact_authority": 2.0,
        "latent_budget_readiness": 1.5,
        "latent_account_fit": 1.0,
    },
}
_DEFAULT_FOLLOWUP_LATENT_WEIGHTS: dict[str, float] = {
    "latent_budget_readiness": 2.0,
    "latent_account_fit": 1.0,
}

# Fallback weights/params for unknown motif families.
_DEFAULT_CONVERSION_WEIGHTS: dict[str, float] = {
    "latent_account_fit": 1.0,
    "latent_engagement_propensity": 1.0,
    "latent_sales_friction": -0.5,
}
_DEFAULT_HAZARD_PARAMS: dict[str, float] = {"base_rate": 0.006, "scale": 0.05}
_DEFAULT_TOUCH_BASE_RATE: float = 0.40

# Per-motif calibration constants for difficulty modulation.
# Each tuple is (reach_fraction, effective_days_at_negotiation):
#   - reach_fraction: approximate share of leads that reach negotiation stage
#     under baseline (no difficulty) parameters.
#   - effective_days_at_negotiation: approximate days a lead spends at
#     negotiation before converting or churning.
#
# Calibrated against v1.0.0 (2026-05-04) with 1000 leads × 20 seeds.
# Re-calibrate if stage transition rates, churn rate, or population
# initialisation logic changes.
_MOTIF_REACH_CALIBRATION: dict[str, tuple[float, float]] = {
    "fit_dominant": (0.85, 22.0),
    "intent_dominant": (0.85, 22.0),
    "sales_execution_sensitive": (0.40, 18.0),
    "demo_trial_mediated": (0.70, 20.0),
    "buying_committee_friction": (0.32, 16.0),
}
_DEFAULT_REACH_CALIBRATION: tuple[float, float] = (0.55, 20.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def assign_mechanisms(
    motif_family: str,
    rng: random.Random,
    *,
    latent_touch_intensity: bool = False,
    difficulty_params: DifficultyParams | None = None,
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
        latent_touch_intensity: When ``True``, use
            :class:`~leadforge.mechanisms.counts.LatentDecayIntensity` instead
            of :class:`~leadforge.mechanisms.counts.RecencyDecayIntensity` for
            touch emission, making touch intensity depend on the same latent
            traits that drive conversion.

    Returns:
        A fully populated :class:`~leadforge.mechanisms.base.MechanismAssignment`.
    """
    conv_weights = dict(_CONVERSION_SCORE_WEIGHTS.get(motif_family, _DEFAULT_CONVERSION_WEIGHTS))
    hazard_p = dict(_HAZARD_PARAMS.get(motif_family, _DEFAULT_HAZARD_PARAMS))
    trans_weights = dict(_TRANSITION_SCORE_WEIGHTS.get(motif_family, _DEFAULT_CONVERSION_WEIGHTS))
    touch_rate = _TOUCH_BASE_RATES.get(motif_family, _DEFAULT_TOUCH_BASE_RATE)

    # -- Difficulty modulation ------------------------------------------------
    signal = 1.0
    if difficulty_params is not None:
        signal = difficulty_params.signal_strength

        # Override conversion hazard params to produce the target conversion rate.
        #
        # The baseline conversion rate varies significantly by motif family due
        # to differences in how many leads reach negotiation and how latent
        # scores distribute.  We use per-motif calibration constants to compute
        # the daily hazard probability that produces the target overall rate.
        #
        # Model: P(convert) ≈ reach_frac × [1 - (1-daily_p)^N_days]
        target_mid = (
            difficulty_params.conversion_rate_lo + difficulty_params.conversion_rate_hi
        ) / 2

        reach_frac, days_at_negotiation = _MOTIF_REACH_CALIBRATION.get(
            motif_family, _DEFAULT_REACH_CALIBRATION
        )

        # Target P(convert | reached negotiation).
        p_convert_given_neg = min(0.92, target_mid / reach_frac)
        target_daily_p = 1.0 - (1.0 - p_convert_given_neg) ** (1.0 / days_at_negotiation)

        # Split into base_rate (score-independent) and scale (score-dependent).
        # Preserve the motif's original ratio between base_rate and scale.
        orig_sum = hazard_p["base_rate"] + hazard_p["scale"]
        if orig_sum > 0:
            base_frac = hazard_p["base_rate"] / orig_sum
        else:
            base_frac = 0.15
        hazard_p = {
            "base_rate": target_daily_p * base_frac,
            "scale": target_daily_p * (1.0 - base_frac),
        }

    # Apply signal_strength to LatentScore weights.
    # To reduce signal (lower signal_strength), we attenuate secondary weights
    # more than the primary one.  This reduces discriminability rather than just
    # shifting the sigmoid.  The strongest weight is scaled by `signal`, the
    # rest by `signal^1.5`, so intro (0.90) barely changes while advanced (0.50)
    # meaningfully weakens secondary signals.
    def _scale_weights(weights: dict[str, float], s: float) -> dict[str, float]:
        if not weights or s >= 1.0:
            return dict(weights)
        max_abs = max(abs(v) for v in weights.values())
        return {k: v * s if abs(v) >= max_abs - 1e-9 else v * (s**1.5) for k, v in weights.items()}

    scaled_conv_weights = _scale_weights(conv_weights, signal)
    scaled_trans_weights = _scale_weights(trans_weights, signal)

    conversion_hazard = ConversionHazard(
        score_mech=LatentScore(weights=scaled_conv_weights, bias=-1.5),
        base_rate=hazard_p["base_rate"],
        scale=hazard_p["scale"],
    )

    stage_transition = HazardTransition(
        score_mech=LatentScore(weights=scaled_trans_weights, bias=-1.0),
        base_rate=0.05,
        scale=0.15,
        min_dwell_days=2,
    )

    touch_intensity: RecencyDecayIntensity | LatentDecayIntensity
    if latent_touch_intensity:
        touch_latent_w = _TOUCH_LATENT_WEIGHTS.get(motif_family, _DEFAULT_TOUCH_LATENT_WEIGHTS)
        followup_latent_w = _FOLLOWUP_LATENT_WEIGHTS.get(
            motif_family, _DEFAULT_FOLLOWUP_LATENT_WEIGHTS
        )
        # Ramp dynamics are uniform across motif families: the follow-up
        # timing reflects a sales-process constant (assessment period = 20 days,
        # ramp-up over 10 days).  Per-motif differentiation comes entirely from
        # _FOLLOWUP_LATENT_WEIGHTS, which controls *what* latent signals drive
        # the post-assessment follow-up intensity for each motif family.
        touch_intensity = LatentDecayIntensity(
            base_rate=touch_rate,
            decay_factor=0.97,
            floor_rate=0.02,
            latent_weights=touch_latent_w,
            boost=1.2,
            followup_boost_after_day=20,
            followup_boost_factor=10.0,
            followup_ramp_days=10,
            followup_latent_weights=followup_latent_w,
        )
    else:
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

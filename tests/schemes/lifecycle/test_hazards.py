"""Tests for the lifecycle hazard functions (LTV-Pj)."""

import math

import pytest

from leadforge.schemes.lifecycle.hazards import (
    MAX_PROBABILITY,
    churn_probability,
    expansion_probability,
    is_renewal_week,
    next_renewal_week,
    payment_failure_probability,
)
from leadforge.schemes.lifecycle.mechanisms import (
    ChurnHazardParams,
    ExpansionPropensityParams,
    PaymentFailureParams,
    assign_lifecycle_mechanisms,
)
from leadforge.schemes.lifecycle.population import LIFECYCLE_MOTIF_FAMILIES

_NEUTRAL = {
    "latent_product_fit": 0.5,
    "latent_adoption_velocity": 0.5,
    "latent_budget_stability": 0.5,
    "latent_champion_strength": 0.5,
    "latent_organizational_stability": 0.5,
}

# A steady-state week well past onboarding and well before any renewal.
_STEADY_WEEK = 30
_TERM_12MO = 12


def _churn(motif: str = "product_led_retention") -> ChurnHazardParams:
    return assign_lifecycle_mechanisms(motif).churn_hazard


def _expansion(motif: str = "expansion_led_growth") -> ExpansionPropensityParams:
    return assign_lifecycle_mechanisms(motif).expansion_propensity


def _payment(motif: str = "payment_fragile") -> PaymentFailureParams:
    return assign_lifecycle_mechanisms(motif).payment_failure


# ---------------------------------------------------------------------------
# is_renewal_week
# ---------------------------------------------------------------------------


def test_renewal_week_12mo_first_anniversary() -> None:
    assert is_renewal_week(52, 12)


def test_renewal_week_12mo_second_anniversary() -> None:
    assert is_renewal_week(104, 12)


def test_renewal_week_24mo_first_anniversary() -> None:
    assert is_renewal_week(104, 24)


def test_week_52_is_not_renewal_for_24mo_term() -> None:
    # A 24-month contract has no anniversary at week 52 — the spike (and the
    # engine's renewal event) must not fire mid-contract.
    assert not is_renewal_week(52, 24)


def test_adjacent_weeks_are_not_renewal() -> None:
    assert not is_renewal_week(51, 12)
    assert not is_renewal_week(53, 12)


def test_week_zero_is_never_renewal() -> None:
    assert not is_renewal_week(0, 12)
    assert not is_renewal_week(0, 1)


def test_renewal_week_non_integer_term_weeks() -> None:
    # 13-month term → 56.33 weeks → anniversary at week 56.
    assert is_renewal_week(56, 13)
    assert not is_renewal_week(57, 13)


def test_renewal_week_rejects_negative_tenure() -> None:
    with pytest.raises(ValueError, match="week_of_tenure"):
        is_renewal_week(-1, 12)


def test_renewal_week_rejects_bad_term() -> None:
    with pytest.raises(ValueError, match="contract_term_months"):
        is_renewal_week(10, 0)


# ---------------------------------------------------------------------------
# next_renewal_week
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("week", "term", "expected"),
    [
        (0, 12, 52),
        (51, 12, 52),
        (52, 12, 104),
        (0, 24, 104),
        (104, 24, 208),
        (0, 13, 56),
        (56, 13, 113),
    ],
)
def test_next_renewal_week_known_anniversaries(week: int, term: int, expected: int) -> None:
    assert next_renewal_week(week, term) == expected


@pytest.mark.parametrize("term", [12, 13, 24])
def test_next_renewal_week_agrees_with_is_renewal_week(term: int) -> None:
    """The returned week is the FIRST week after the input that satisfies
    is_renewal_week — the two functions share one anniversary boundary."""
    for week in range(0, 160):
        nxt = next_renewal_week(week, term)
        assert nxt > week
        assert is_renewal_week(nxt, term)
        assert not any(is_renewal_week(w, term) for w in range(week + 1, nxt))


def test_next_renewal_week_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="week_of_tenure"):
        next_renewal_week(-1, 12)
    with pytest.raises(ValueError, match="contract_term_months"):
        next_renewal_week(10, 0)


# ---------------------------------------------------------------------------
# churn_probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_churn_in_unit_interval_at_extremes(motif: str) -> None:
    params = _churn(motif)
    lo = dict.fromkeys(_NEUTRAL, 0.0)
    hi = dict.fromkeys(_NEUTRAL, 1.0)
    for latents in (lo, hi, _NEUTRAL):
        for week in (0, 1, _STEADY_WEEK, 52):
            p = churn_probability(params, latents, week, _TERM_12MO)
            assert 0.0 < p <= MAX_PROBABILITY, f"{motif} week={week}: p={p}"


def test_churn_neutral_latents_steady_state_near_base_rate() -> None:
    params = _churn()
    p = churn_probability(params, _NEUTRAL, _STEADY_WEEK, _TERM_12MO)
    # At week 30 the onboarding elevation residual is exp(-30/4) ≈ 5.5e-4.
    assert math.isclose(p, params.base_weekly_rate, rel_tol=1e-2)


def test_churn_decreases_with_product_fit() -> None:
    # product_led_retention weights latent_product_fit at -2.0.
    params = _churn("product_led_retention")
    poor_fit = {**_NEUTRAL, "latent_product_fit": 0.1}
    good_fit = {**_NEUTRAL, "latent_product_fit": 0.9}
    p_poor = churn_probability(params, poor_fit, _STEADY_WEEK, _TERM_12MO)
    p_good = churn_probability(params, good_fit, _STEADY_WEEK, _TERM_12MO)
    assert p_poor > p_good


def test_churn_onboarding_elevated_vs_steady_state() -> None:
    params = _churn()
    p_week0 = churn_probability(params, _NEUTRAL, 0, _TERM_12MO)
    p_steady = churn_probability(params, _NEUTRAL, _STEADY_WEEK, _TERM_12MO)
    assert p_week0 > 2.0 * p_steady  # peak multiplier is 2.5


def test_churn_onboarding_decays_monotonically() -> None:
    params = _churn()
    probs = [churn_probability(params, _NEUTRAL, w, _TERM_12MO) for w in range(13)]
    assert probs == sorted(probs, reverse=True)


def test_churn_renewal_week_spikes() -> None:
    params = _churn("relationship_led_retention")  # multiplier 12.0
    p_renewal = churn_probability(params, _NEUTRAL, 52, _TERM_12MO)
    p_before = churn_probability(params, _NEUTRAL, 51, _TERM_12MO)
    p_after = churn_probability(params, _NEUTRAL, 53, _TERM_12MO)
    assert p_renewal > 5.0 * p_before
    assert p_renewal > 5.0 * p_after


def test_churn_strong_champion_dampens_renewal_spike() -> None:
    params = _churn("relationship_led_retention")
    weak = {**_NEUTRAL, "latent_champion_strength": 0.1}
    strong = {**_NEUTRAL, "latent_champion_strength": 0.9}
    p_weak = churn_probability(params, weak, 52, _TERM_12MO)
    p_strong = churn_probability(params, strong, 52, _TERM_12MO)
    assert p_weak > p_strong


def test_churn_missing_latents_treated_as_neutral() -> None:
    params = _churn()
    p_empty = churn_probability(params, {}, _STEADY_WEEK, _TERM_12MO)
    p_neutral = churn_probability(params, _NEUTRAL, _STEADY_WEEK, _TERM_12MO)
    assert p_empty == p_neutral


def test_churn_probability_is_capped() -> None:
    params = ChurnHazardParams(
        base_weekly_rate=0.5,
        latent_weights=_churn().latent_weights,
        renewal_hazard_multiplier=100.0,
        renewal_latent_weights=_churn().renewal_latent_weights,
    )
    p = churn_probability(params, _NEUTRAL, 52, _TERM_12MO)
    assert p == MAX_PROBABILITY


def test_churn_is_deterministic() -> None:
    params = _churn()
    a = churn_probability(params, _NEUTRAL, 7, _TERM_12MO)
    b = churn_probability(params, _NEUTRAL, 7, _TERM_12MO)
    assert a == b


# ---------------------------------------------------------------------------
# expansion_probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_expansion_in_unit_interval_at_extremes(motif: str) -> None:
    params = _expansion(motif)
    for latents in (dict.fromkeys(_NEUTRAL, 0.0), dict.fromkeys(_NEUTRAL, 1.0), _NEUTRAL):
        for depth in (None, 0.0, 0.5, 1.0):
            p = expansion_probability(params, latents, depth)
            assert 0.0 < p <= MAX_PROBABILITY


def test_expansion_increases_with_adoption_velocity() -> None:
    params = _expansion("expansion_led_growth")  # velocity weight +2.0
    slow = {**_NEUTRAL, "latent_adoption_velocity": 0.1}
    fast = {**_NEUTRAL, "latent_adoption_velocity": 0.9}
    assert expansion_probability(params, fast) > expansion_probability(params, slow)


def test_expansion_neutral_depth_is_no_op() -> None:
    params = _expansion()
    assert expansion_probability(params, _NEUTRAL, 0.5) == expansion_probability(
        params, _NEUTRAL, None
    )


def test_expansion_depth_modulates_monotonically() -> None:
    params = _expansion()
    p_lo = expansion_probability(params, _NEUTRAL, 0.0)
    p_mid = expansion_probability(params, _NEUTRAL, 0.5)
    p_hi = expansion_probability(params, _NEUTRAL, 1.0)
    assert p_lo < p_mid < p_hi


def test_expansion_rejects_out_of_range_depth() -> None:
    params = _expansion()
    with pytest.raises(ValueError, match="feature_depth_score"):
        expansion_probability(params, _NEUTRAL, 1.5)
    with pytest.raises(ValueError, match="feature_depth_score"):
        expansion_probability(params, _NEUTRAL, -0.1)


# ---------------------------------------------------------------------------
# payment_failure_probability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_payment_failure_in_unit_interval_at_extremes(motif: str) -> None:
    params = _payment(motif)
    for latents in (dict.fromkeys(_NEUTRAL, 0.0), dict.fromkeys(_NEUTRAL, 1.0), _NEUTRAL):
        p = payment_failure_probability(params, latents)
        assert 0.0 < p <= MAX_PROBABILITY


def test_payment_failure_decreases_with_budget_stability() -> None:
    params = _payment("payment_fragile")  # stability weight -3.0
    fragile = {**_NEUTRAL, "latent_budget_stability": 0.1}
    stable = {**_NEUTRAL, "latent_budget_stability": 0.9}
    p_fragile = payment_failure_probability(params, fragile)
    p_stable = payment_failure_probability(params, stable)
    assert p_fragile > p_stable
    # With weight -3.0 the spread between the tails is large (exp(2.4) ≈ 11×).
    assert p_fragile > 5.0 * p_stable


def test_payment_failure_neutral_equals_base_rate() -> None:
    params = _payment()
    p = payment_failure_probability(params, _NEUTRAL)
    assert math.isclose(p, params.base_monthly_rate, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Cross-mechanism sanity: motif identity flows through to probabilities
# ---------------------------------------------------------------------------


def test_fragile_world_fails_payments_more_than_product_led() -> None:
    p_fragile = payment_failure_probability(_payment("payment_fragile"), _NEUTRAL)
    p_plr = payment_failure_probability(_payment("product_led_retention"), _NEUTRAL)
    assert p_fragile > 2.0 * p_plr


def test_churner_world_churns_more_than_growth_world() -> None:
    p_churner = churn_probability(_churn("churner_dominated"), _NEUTRAL, _STEADY_WEEK, 12)
    p_growth = churn_probability(_churn("expansion_led_growth"), _NEUTRAL, _STEADY_WEEK, 12)
    assert p_churner > p_growth

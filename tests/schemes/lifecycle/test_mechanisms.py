"""Tests for the lifecycle mechanism policies (LTV-Pi)."""

import pytest

from leadforge.schemes.lifecycle.mechanisms import (
    ChurnHazardParams,
    ExpansionPropensityParams,
    LifecycleMechanismAssignment,
    PaymentFailureParams,
    assign_lifecycle_mechanisms,
    mechanism_params_for_motif,
)
from leadforge.schemes.lifecycle.population import LIFECYCLE_MOTIF_FAMILIES

# ---------------------------------------------------------------------------
# Basic dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_assignment_returned_for_each_motif(motif: str) -> None:
    assignment = assign_lifecycle_mechanisms(motif)
    assert isinstance(assignment, LifecycleMechanismAssignment)
    assert assignment.motif_family == motif


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_all_three_mechanisms_present(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    assert isinstance(a.churn_hazard, ChurnHazardParams)
    assert isinstance(a.expansion_propensity, ExpansionPropensityParams)
    assert isinstance(a.payment_failure, PaymentFailureParams)


def test_unknown_motif_falls_back_to_defaults() -> None:
    # Unknown families must not raise — they fall back to defaults.
    a = assign_lifecycle_mechanisms("nonexistent_motif")
    assert a.motif_family == "nonexistent_motif"
    assert a.churn_hazard.base_weekly_rate > 0
    assert a.expansion_propensity.base_weekly_rate > 0
    assert a.payment_failure.base_monthly_rate > 0


# ---------------------------------------------------------------------------
# Parameter value ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_churn_base_rate_is_positive_and_subunit(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    r = a.churn_hazard.base_weekly_rate
    assert 0.0 < r < 1.0, f"{motif}: base_weekly_rate={r} not in (0, 1)"


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_renewal_multiplier_gt_one(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    m = a.churn_hazard.renewal_hazard_multiplier
    assert m > 1.0, f"{motif}: renewal_hazard_multiplier={m} must be > 1"


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_expansion_base_rate_is_positive_and_subunit(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    r = a.expansion_propensity.base_weekly_rate
    assert 0.0 < r < 1.0, f"{motif}: expansion base_weekly_rate={r} not in (0, 1)"


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_expansion_mrr_frac_range_is_valid(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    lo, hi = a.expansion_propensity.expansion_mrr_frac_range
    assert 0.0 < lo < hi, f"{motif}: expansion_mrr_frac_range ({lo}, {hi}) invalid"
    assert hi <= 2.0, f"{motif}: expansion hi={hi} unrealistically large"


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_payment_failure_base_rate_is_positive_and_subunit(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    r = a.payment_failure.base_monthly_rate
    assert 0.0 < r < 1.0, f"{motif}: payment base_monthly_rate={r} not in (0, 1)"


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_dunning_weeks_is_positive(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    assert a.payment_failure.dunning_weeks >= 1


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_recovery_rate_in_unit_interval(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    r = a.payment_failure.recovery_rate
    assert 0.0 <= r <= 1.0, f"{motif}: recovery_rate={r}"


# ---------------------------------------------------------------------------
# Motif-family structural ordering
# ---------------------------------------------------------------------------


def test_expansion_led_growth_has_highest_expansion_rate() -> None:
    rates = {
        m: assign_lifecycle_mechanisms(m).expansion_propensity.base_weekly_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
    }
    assert rates["expansion_led_growth"] == max(rates.values()), (
        f"expansion_led_growth should have the highest expansion rate; got {rates}"
    )


def test_churner_dominated_has_highest_churn_rate() -> None:
    rates = {
        m: assign_lifecycle_mechanisms(m).churn_hazard.base_weekly_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
    }
    assert rates["churner_dominated"] == max(rates.values()), (
        f"churner_dominated should have the highest base churn rate; got {rates}"
    )


def test_expansion_led_growth_has_lowest_churn_rate() -> None:
    # expansion_led_growth is calibrated as the lowest-churn world — customers
    # that are growing fast are least likely to churn.
    rates = {
        m: assign_lifecycle_mechanisms(m).churn_hazard.base_weekly_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
    }
    assert rates["expansion_led_growth"] == min(rates.values()), (
        f"expansion_led_growth should have the lowest base churn rate; got {rates}"
    )


def test_payment_fragile_has_highest_payment_failure_rate() -> None:
    rates = {
        m: assign_lifecycle_mechanisms(m).payment_failure.base_monthly_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
    }
    assert rates["payment_fragile"] == max(rates.values()), (
        f"payment_fragile should have the highest payment failure rate; got {rates}"
    )


def test_payment_fragile_has_lowest_recovery_rate() -> None:
    rates = {
        m: assign_lifecycle_mechanisms(m).payment_failure.recovery_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
    }
    assert rates["payment_fragile"] == min(rates.values()), (
        f"payment_fragile should have the lowest recovery rate; got {rates}"
    )


# ---------------------------------------------------------------------------
# Latent weights structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_churn_latent_weights_non_empty(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    assert len(a.churn_hazard.latent_weights) >= 1


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_expansion_latent_weights_non_empty(motif: str) -> None:
    a = assign_lifecycle_mechanisms(motif)
    assert len(a.expansion_propensity.latent_weights) >= 1


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_churn_weights_reference_valid_lifecycle_traits(motif: str) -> None:
    valid_traits = {
        "latent_product_fit",
        "latent_adoption_velocity",
        "latent_budget_stability",
        "latent_champion_strength",
        "latent_organizational_stability",
    }
    a = assign_lifecycle_mechanisms(motif)
    for trait in a.churn_hazard.latent_weights:
        assert trait in valid_traits, f"{motif}: churn weight references unknown trait {trait!r}"


# ---------------------------------------------------------------------------
# Frozen dataclasses (immutability)
# ---------------------------------------------------------------------------


def test_assignment_is_frozen() -> None:
    a = assign_lifecycle_mechanisms("product_led_retention")
    with pytest.raises((AttributeError, TypeError)):
        a.motif_family = "other"  # type: ignore[misc]


def test_churn_params_are_frozen() -> None:
    a = assign_lifecycle_mechanisms("product_led_retention")
    with pytest.raises((AttributeError, TypeError)):
        a.churn_hazard.base_weekly_rate = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# mechanism_params_for_motif inspection helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_params_dict_covers_all_keys(motif: str) -> None:
    params = mechanism_params_for_motif(motif)
    expected_keys = {
        "motif_family",
        "churn_base_weekly_rate",
        "churn_latent_weights",
        "renewal_hazard_multiplier",
        "renewal_latent_weights",
        "expansion_base_weekly_rate",
        "expansion_latent_weights",
        "expansion_mrr_frac_range",
        "payment_failure_base_monthly_rate",
        "payment_failure_latent_weights",
        "dunning_weeks",
        "recovery_rate",
    }
    assert set(params.keys()) == expected_keys


def test_params_dict_consistent_with_assignment() -> None:
    motif = "expansion_led_growth"
    params = mechanism_params_for_motif(motif)
    a = assign_lifecycle_mechanisms(motif)
    assert params["churn_base_weekly_rate"] == a.churn_hazard.base_weekly_rate
    assert params["expansion_base_weekly_rate"] == a.expansion_propensity.base_weekly_rate
    assert params["payment_failure_base_monthly_rate"] == a.payment_failure.base_monthly_rate
    assert params["recovery_rate"] == a.payment_failure.recovery_rate

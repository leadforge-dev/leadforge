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


def test_expansion_led_growth_expansion_rate_above_churner_dominated() -> None:
    # Directional invariant: a growth-led world expands more than a churn-dominated one.
    # Uses a pair comparison rather than max() so future recalibration that changes
    # which other family has the highest rate won't break this test.
    elg = assign_lifecycle_mechanisms("expansion_led_growth").expansion_propensity.base_weekly_rate
    cd = assign_lifecycle_mechanisms("churner_dominated").expansion_propensity.base_weekly_rate
    assert elg > cd, (
        f"expansion_led_growth expansion rate ({elg}) should exceed churner_dominated ({cd})"
    )


def test_churner_dominated_churn_rate_above_product_led_retention() -> None:
    # Directional invariant: churner-dominated worlds churn more than product-led ones.
    cd = assign_lifecycle_mechanisms("churner_dominated").churn_hazard.base_weekly_rate
    plr = assign_lifecycle_mechanisms("product_led_retention").churn_hazard.base_weekly_rate
    assert cd > plr, f"churner_dominated churn ({cd}) should exceed product_led_retention ({plr})"


def test_expansion_led_growth_churn_rate_below_churner_dominated() -> None:
    # Directional invariant: fast-growing worlds churn less than churn-dominated ones.
    elg = assign_lifecycle_mechanisms("expansion_led_growth").churn_hazard.base_weekly_rate
    cd = assign_lifecycle_mechanisms("churner_dominated").churn_hazard.base_weekly_rate
    assert elg < cd, f"expansion_led_growth churn ({elg}) should be below churner_dominated ({cd})"


def test_payment_fragile_failure_rate_substantially_above_others() -> None:
    # Directional invariant: payment_fragile failure rate is materially higher
    # than any non-fragile world's.  Uses 2× threshold rather than max() so
    # a recalibration that raises another family's rate modestly won't fail.
    pf = assign_lifecycle_mechanisms("payment_fragile").payment_failure.base_monthly_rate
    others = [
        assign_lifecycle_mechanisms(m).payment_failure.base_monthly_rate
        for m in LIFECYCLE_MOTIF_FAMILIES
        if m != "payment_fragile"
    ]
    assert all(pf > 2 * r for r in others), (
        f"payment_fragile ({pf:.4f}) should be >2× all other families' rates: {others}"
    )


def test_payment_fragile_recovery_rate_below_product_led_retention() -> None:
    # Directional invariant: fragile accounts recover failed payments less often.
    pf = assign_lifecycle_mechanisms("payment_fragile").payment_failure.recovery_rate
    plr = assign_lifecycle_mechanisms("product_led_retention").payment_failure.recovery_rate
    assert pf < plr, (
        f"payment_fragile recovery ({pf}) should be below product_led_retention ({plr})"
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


def test_latent_weights_dicts_are_truly_immutable() -> None:
    # Regression: frozen=True on a dataclass prevents attribute reassignment
    # but NOT mutation of a plain dict field.  latent_weights are wrapped in
    # MappingProxyType so the simulation engine cannot corrupt shared state.
    a = assign_lifecycle_mechanisms("product_led_retention")
    with pytest.raises(TypeError):
        a.churn_hazard.latent_weights["latent_product_fit"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        a.expansion_propensity.latent_weights["latent_adoption_velocity"] = 999.0  # type: ignore[index]
    with pytest.raises(TypeError):
        a.payment_failure.latent_weights["latent_budget_stability"] = 999.0  # type: ignore[index]


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

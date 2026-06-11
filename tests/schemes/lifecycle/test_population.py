"""Tests for the lifecycle customer population builder (LTV-Ph)."""

from datetime import date, timedelta

import pytest

from leadforge.schemes.lifecycle.population import (
    LIFECYCLE_MOTIF_FAMILIES,
    CustomerLatentState,
    CustomerPopulationResult,
    build_customer_population,
)

_N_CUSTOMERS = 120
_N_ACCOUNTS = 40
_SEED = 7


# ---------------------------------------------------------------------------
# Basic shape
# ---------------------------------------------------------------------------


def test_returns_expected_types() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    assert isinstance(result, CustomerPopulationResult)
    assert isinstance(result.latent_state, CustomerLatentState)


def test_account_and_customer_counts() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    assert len(result.accounts) == _N_ACCOUNTS
    assert len(result.customers) == _N_CUSTOMERS


def test_default_n_accounts_is_n_customers_over_three() -> None:
    result = build_customer_population(90, _SEED)
    assert len(result.accounts) == 30


def test_default_n_accounts_minimum_one() -> None:
    result = build_customer_population(1, _SEED)
    assert len(result.accounts) == 1


def test_observation_date_recorded() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED, observation_date="2025-06-01")
    assert result.observation_date == "2025-06-01"


def test_observation_date_default_format() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    # Must be a valid ISO-8601 date string.
    parsed = date.fromisoformat(result.observation_date)
    assert isinstance(parsed, date)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_under_same_seed() -> None:
    a = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    b = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    assert [c.customer_id for c in a.customers] == [c.customer_id for c in b.customers]
    assert [c.customer_start_at for c in a.customers] == [c.customer_start_at for c in b.customers]
    assert a.latent_state.customer_latents == b.latent_state.customer_latents


def test_different_seeds_produce_different_results() -> None:
    a = build_customer_population(_N_CUSTOMERS, seed=1)
    b = build_customer_population(_N_CUSTOMERS, seed=2)
    assert [c.customer_start_at for c in a.customers] != [c.customer_start_at for c in b.customers]


# ---------------------------------------------------------------------------
# FK integrity
# ---------------------------------------------------------------------------


def test_customer_account_fk() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    account_ids = {a.account_id for a in result.accounts}
    for cust in result.customers:
        assert cust.account_id in account_ids, (
            f"customer {cust.customer_id} references unknown account {cust.account_id}"
        )


def test_latent_state_covers_all_customers() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    cust_ids = {c.customer_id for c in result.customers}
    assert set(result.latent_state.customer_latents.keys()) == cust_ids


def test_latent_state_covers_all_accounts() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED, n_accounts=_N_ACCOUNTS)
    acct_ids = {a.account_id for a in result.accounts}
    assert set(result.latent_state.account_latents.keys()) == acct_ids


# ---------------------------------------------------------------------------
# Staggered start dates + acquisition-window boundary
# ---------------------------------------------------------------------------


def test_all_starts_before_observation_date() -> None:
    obs = "2025-06-01"
    obs_date = date.fromisoformat(obs)
    result = build_customer_population(_N_CUSTOMERS, _SEED, observation_date=obs)
    for cust in result.customers:
        start = date.fromisoformat(cust.customer_start_at)
        assert start < obs_date, (
            f"customer {cust.customer_id} starts on or after observation date: {start}"
        )


def test_all_starts_within_acquisition_window() -> None:
    obs = "2025-06-01"
    obs_date = date.fromisoformat(obs)
    acq_weeks = 26
    acq_start = obs_date - timedelta(weeks=acq_weeks)
    result = build_customer_population(
        _N_CUSTOMERS, _SEED, observation_date=obs, acquisition_window_weeks=acq_weeks
    )
    for cust in result.customers:
        start = date.fromisoformat(cust.customer_start_at)
        assert start >= acq_start, (
            f"customer {cust.customer_id} starts before acquisition window: {start}"
        )


def test_start_dates_span_the_window() -> None:
    """With enough customers, start dates should cover both early and late in window."""
    obs = "2025-06-01"
    obs_date = date.fromisoformat(obs)
    acq_weeks = 52
    acq_start = obs_date - timedelta(weeks=acq_weeks)
    mid_point = acq_start + timedelta(weeks=acq_weeks // 2)

    result = build_customer_population(
        200, _SEED, observation_date=obs, acquisition_window_weeks=acq_weeks
    )
    starts = [date.fromisoformat(c.customer_start_at) for c in result.customers]
    early = sum(1 for s in starts if s < mid_point)
    late = sum(1 for s in starts if s >= mid_point)
    # With 200 customers and a uniform distribution, both halves should have
    # at least 25% of customers (expected ~50% each, allow wide tolerance).
    assert early >= 25, f"too few early-cohort customers: {early}"
    assert late >= 25, f"too few late-cohort customers: {late}"


def test_opportunity_id_is_none_independent_generation() -> None:
    """D3: independent generation leaves the chaining seam empty."""
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    assert all(c.opportunity_id is None for c in result.customers)


# ---------------------------------------------------------------------------
# Latent distributions
# ---------------------------------------------------------------------------


def test_customer_latents_in_unit_interval() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for cust_id, traits in result.latent_state.customer_latents.items():
        for trait, val in traits.items():
            assert 0.0 <= val <= 1.0, f"customer {cust_id} trait {trait}={val} outside [0,1]"


def test_customer_latents_has_five_traits() -> None:
    expected = {
        "latent_product_fit",
        "latent_adoption_velocity",
        "latent_budget_stability",
        "latent_champion_strength",
        "latent_organizational_stability",
    }
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for cust_id, traits in result.latent_state.customer_latents.items():
        assert set(traits.keys()) == expected, (
            f"customer {cust_id} has unexpected traits: {set(traits.keys())}"
        )


def test_account_latents_in_unit_interval() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for acct_id, traits in result.latent_state.account_latents.items():
        for trait, val in traits.items():
            assert 0.0 <= val <= 1.0, f"account {acct_id} trait {trait}={val} outside [0,1]"


def test_account_latents_use_lifecycle_not_lead_scoring_keys() -> None:
    # Regression: account latents must use lifecycle trait names (queried by the
    # lifecycle simulation engine), NOT lead-scoring names (latent_account_fit,
    # latent_budget_readiness, latent_process_maturity) which the engine never reads.
    expected = {"latent_budget_stability", "latent_organizational_stability"}
    lead_scoring_names = {
        "latent_account_fit",
        "latent_budget_readiness",
        "latent_process_maturity",
    }
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for acct_id, traits in result.latent_state.account_latents.items():
        assert set(traits.keys()) == expected, (
            f"account {acct_id} has wrong latent keys: {set(traits.keys())}"
        )
        assert not (set(traits.keys()) & lead_scoring_names), (
            f"account {acct_id} contains lead-scoring trait names: "
            f"{set(traits.keys()) & lead_scoring_names}"
        )


def test_motif_family_must_be_keyword_only() -> None:
    import inspect

    sig = inspect.signature(build_customer_population)
    p = sig.parameters["motif_family"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
        "motif_family must be keyword-only to prevent silent positional misuse"
    )


# ---------------------------------------------------------------------------
# Motif families
# ---------------------------------------------------------------------------


def test_all_motif_families_registered() -> None:
    expected = {
        "product_led_retention",
        "relationship_led_retention",
        "expansion_led_growth",
        "payment_fragile",
        "churner_dominated",
    }
    assert set(LIFECYCLE_MOTIF_FAMILIES) == expected


@pytest.mark.parametrize("motif", LIFECYCLE_MOTIF_FAMILIES)
def test_each_motif_family_runs(motif: str) -> None:
    result = build_customer_population(30, _SEED, motif_family=motif)
    assert len(result.customers) == 30


def test_motif_bias_shifts_latent_means() -> None:
    """product_led_retention biases latent_product_fit upward; churner_dominated downward."""
    n = 400
    plt_result = build_customer_population(n, _SEED, motif_family="product_led_retention")
    churn_result = build_customer_population(n, _SEED, motif_family="churner_dominated")

    def mean_fit(r: CustomerPopulationResult) -> float:
        vals = [traits["latent_product_fit"] for traits in r.latent_state.customer_latents.values()]
        return sum(vals) / len(vals)

    assert mean_fit(plt_result) > mean_fit(churn_result), (
        "product_led_retention should have higher mean latent_product_fit than churner_dominated"
    )


def test_unknown_motif_family_raises() -> None:
    with pytest.raises(ValueError, match="Unknown lifecycle motif family"):
        build_customer_population(10, _SEED, motif_family="does_not_exist")


# ---------------------------------------------------------------------------
# Input validation (COPILOT-3)
# ---------------------------------------------------------------------------


def test_zero_n_customers_raises() -> None:
    with pytest.raises(ValueError, match="n_customers"):
        build_customer_population(0, _SEED)


def test_negative_n_customers_raises() -> None:
    with pytest.raises(ValueError, match="n_customers"):
        build_customer_population(-1, _SEED)


def test_zero_n_accounts_explicit_raises() -> None:
    with pytest.raises(ValueError, match="n_accounts"):
        build_customer_population(10, _SEED, n_accounts=0)


def test_zero_acquisition_window_raises() -> None:
    # acquisition_window_weeks=0 would make every start == obs_date,
    # violating the < obs_date boundary invariant.
    with pytest.raises(ValueError, match="acquisition_window_weeks"):
        build_customer_population(10, _SEED, acquisition_window_weeks=0)


# ---------------------------------------------------------------------------
# Entity field values
# ---------------------------------------------------------------------------


def test_customer_fields_populated() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for cust in result.customers:
        assert cust.customer_id.startswith("cust_")
        assert cust.account_id.startswith("acct_")
        assert cust.initial_plan in ("starter", "growth", "enterprise")
        assert cust.initial_mrr > 0
        assert cust.contract_term_months in (12, 24)
        assert cust.csm_rep_id.startswith("rep_")


def test_account_fields_populated() -> None:
    result = build_customer_population(_N_CUSTOMERS, _SEED)
    for acct in result.accounts:
        assert acct.account_id.startswith("acct_")
        assert acct.industry in (
            "manufacturing",
            "logistics",
            "professional_services",
            "healthcare_non_clinical",
        )
        assert acct.region in ("US", "UK")
        assert acct.employee_band in ("200-499", "500-999", "1000-1999", "2000+")

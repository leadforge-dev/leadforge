"""Tests for leadforge.simulation.population — build_population."""

from __future__ import annotations

import pytest

from leadforge.api.generator import Generator
from leadforge.core.exceptions import InvalidConfigError
from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.core.models import GenerationConfig
from leadforge.narrative.spec import NarrativeSpec
from leadforge.simulation.population import (
    _N_REPS,
    PopulationResult,
    _channel_weights,
    build_population,
)
from leadforge.structure.sampler import sample_hidden_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEED = 42
_N_ACCOUNTS = 50
_N_CONTACTS = 120
_N_LEADS = 200


def _make_result(seed: int = _SEED, motif: str | None = None) -> PopulationResult:
    config = GenerationConfig(
        seed=seed,
        n_accounts=_N_ACCOUNTS,
        n_contacts=_N_CONTACTS,
        n_leads=_N_LEADS,
    )
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed)
    narrative = gen.world_spec.narrative
    assert narrative is not None
    graph = sample_hidden_graph(seed=seed, motif_family_name=motif)
    return build_population(config, narrative, graph)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_population_counts() -> None:
    result = _make_result()
    assert len(result.accounts) == _N_ACCOUNTS
    assert len(result.contacts) == _N_CONTACTS
    assert len(result.leads) == _N_LEADS


def test_latent_state_counts() -> None:
    result = _make_result()
    assert len(result.latent_state.account_latents) == _N_ACCOUNTS
    assert len(result.latent_state.contact_latents) == _N_CONTACTS
    assert len(result.latent_state.lead_latents) == _N_LEADS


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_build_population_is_deterministic() -> None:
    r1 = _make_result(seed=7)
    r2 = _make_result(seed=7)
    assert [a.account_id for a in r1.accounts] == [a.account_id for a in r2.accounts]
    assert [c.contact_id for c in r1.contacts] == [c.contact_id for c in r2.contacts]
    assert [lead.lead_id for lead in r1.leads] == [lead.lead_id for lead in r2.leads]
    assert r1.latent_state.account_latents == r2.latent_state.account_latents
    assert r1.latent_state.contact_latents == r2.latent_state.contact_latents
    assert r1.latent_state.lead_latents == r2.latent_state.lead_latents


def test_different_seeds_give_different_results() -> None:
    r1 = _make_result(seed=1)
    r2 = _make_result(seed=2)
    assert r1.latent_state.account_latents != r2.latent_state.account_latents


# ---------------------------------------------------------------------------
# Entity IDs
# ---------------------------------------------------------------------------


def test_account_ids_are_sequential_and_unique() -> None:
    result = _make_result()
    ids = [a.account_id for a in result.accounts]
    expected = [make_id(ID_PREFIXES["account"], i) for i in range(1, _N_ACCOUNTS + 1)]
    assert ids == expected


def test_contact_ids_are_sequential_and_unique() -> None:
    result = _make_result()
    ids = [c.contact_id for c in result.contacts]
    expected = [make_id(ID_PREFIXES["contact"], i) for i in range(1, _N_CONTACTS + 1)]
    assert ids == expected


def test_lead_ids_are_sequential_and_unique() -> None:
    result = _make_result()
    ids = [lead.lead_id for lead in result.leads]
    expected = [make_id(ID_PREFIXES["lead"], i) for i in range(1, _N_LEADS + 1)]
    assert ids == expected


# ---------------------------------------------------------------------------
# FK integrity
# ---------------------------------------------------------------------------


def test_contact_account_ids_are_valid() -> None:
    result = _make_result()
    valid_acct_ids = {a.account_id for a in result.accounts}
    for c in result.contacts:
        assert c.account_id in valid_acct_ids, f"contact {c.contact_id} → unknown account"


def test_lead_contact_ids_are_valid() -> None:
    result = _make_result()
    valid_cnt_ids = {c.contact_id for c in result.contacts}
    for lead in result.leads:
        assert lead.contact_id in valid_cnt_ids, f"lead {lead.lead_id} → unknown contact"


def test_lead_account_ids_are_valid() -> None:
    result = _make_result()
    valid_acct_ids = {a.account_id for a in result.accounts}
    for lead in result.leads:
        assert lead.account_id in valid_acct_ids, f"lead {lead.lead_id} → unknown account"


def test_lead_contact_account_consistency() -> None:
    """lead.account_id must match the account_id of lead.contact_id."""
    result = _make_result()
    contact_to_account = {c.contact_id: c.account_id for c in result.contacts}
    for lead in result.leads:
        assert lead.account_id == contact_to_account[lead.contact_id]


# ---------------------------------------------------------------------------
# Latent value ranges and completeness
# ---------------------------------------------------------------------------

_EXPECTED_ACCOUNT_TRAITS = {
    "latent_account_fit",
    "latent_budget_readiness",
    "latent_process_maturity",
}
_EXPECTED_CONTACT_TRAITS = {
    "latent_problem_awareness",
    "latent_contact_authority",
    "latent_responsiveness",
    "latent_engagement_propensity",
}
_EXPECTED_LEAD_TRAITS = {"latent_sales_friction"}


def test_account_latent_traits_present() -> None:
    result = _make_result()
    for acct_id, traits in result.latent_state.account_latents.items():
        assert traits.keys() == _EXPECTED_ACCOUNT_TRAITS, f"account {acct_id}"


def test_contact_latent_traits_present() -> None:
    result = _make_result()
    for cnt_id, traits in result.latent_state.contact_latents.items():
        assert traits.keys() == _EXPECTED_CONTACT_TRAITS, f"contact {cnt_id}"


def test_lead_latent_traits_present() -> None:
    result = _make_result()
    for lead_id, traits in result.latent_state.lead_latents.items():
        assert traits.keys() == _EXPECTED_LEAD_TRAITS, f"lead {lead_id}"


def test_all_latent_values_in_unit_interval() -> None:
    result = _make_result()
    for store in (
        result.latent_state.account_latents,
        result.latent_state.contact_latents,
        result.latent_state.lead_latents,
    ):
        for entity_id, traits in store.items():
            for trait, val in traits.items():
                assert 0.0 <= val <= 1.0, f"{entity_id}.{trait} = {val}"


# ---------------------------------------------------------------------------
# Lead observable fields
# ---------------------------------------------------------------------------


def test_lead_initial_stage_is_mql() -> None:
    result = _make_result()
    for lead in result.leads:
        assert lead.current_stage == "mql"
        assert lead.is_mql is True
        assert lead.is_sql is False
        assert lead.converted_within_90_days is False
        assert lead.conversion_timestamp is None


def test_lead_source_is_valid_channel() -> None:
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=_SEED)
    narrative = gen.world_spec.narrative
    assert narrative is not None
    valid_channels = set(narrative.gtm_motion.channels)
    result = _make_result()
    for lead in result.leads:
        assert lead.lead_source in valid_channels
        assert lead.first_touch_channel == lead.lead_source


def test_lead_owner_rep_id_is_valid() -> None:
    result = _make_result()
    valid_rep_ids = {make_id(ID_PREFIXES["rep"], i) for i in range(1, _N_REPS + 1)}
    for lead in result.leads:
        assert lead.owner_rep_id in valid_rep_ids


def test_lead_created_at_within_base_window() -> None:
    from datetime import date

    result = _make_result()
    base = date(2024, 1, 1)
    end = date(2024, 1, 30)
    for lead in result.leads:
        d = date.fromisoformat(lead.lead_created_at)
        assert base <= d <= end, f"lead {lead.lead_id} created_at {d} out of window"


# ---------------------------------------------------------------------------
# Account observable fields
# ---------------------------------------------------------------------------


def test_account_industry_is_valid() -> None:
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=_SEED)
    narrative = gen.world_spec.narrative
    assert narrative is not None
    valid = set(narrative.market.icp_industries)
    result = _make_result()
    for a in result.accounts:
        assert a.industry in valid


def test_account_region_is_valid() -> None:
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=_SEED)
    narrative = gen.world_spec.narrative
    assert narrative is not None
    valid = set(narrative.market.geographies)
    result = _make_result()
    for a in result.accounts:
        assert a.region in valid


# ---------------------------------------------------------------------------
# Motif latent bias (property test across seeds)
# ---------------------------------------------------------------------------


def test_fit_dominant_raises_account_fit_mean() -> None:
    """fit_dominant worlds should have higher mean latent_account_fit than
    buying_committee_friction worlds across a range of seeds."""
    fit_means = []
    friction_means = []
    for seed in range(15):
        config = GenerationConfig(seed=seed, n_accounts=200, n_contacts=400, n_leads=600)
        gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed)
        narrative = gen.world_spec.narrative
        assert narrative is not None

        g_fit = sample_hidden_graph(seed=seed, motif_family_name="fit_dominant")
        r_fit = build_population(config, narrative, g_fit)
        fit_means.append(
            sum(t["latent_account_fit"] for t in r_fit.latent_state.account_latents.values())
            / config.n_accounts
        )

        g_fric = sample_hidden_graph(seed=seed, motif_family_name="buying_committee_friction")
        r_fric = build_population(config, narrative, g_fric)
        friction_means.append(
            sum(t["latent_account_fit"] for t in r_fric.latent_state.account_latents.values())
            / config.n_accounts
        )

    avg_fit = sum(fit_means) / len(fit_means)
    avg_fric = sum(friction_means) / len(friction_means)
    assert avg_fit > avg_fric, (
        f"Expected fit_dominant mean ({avg_fit:.3f}) > buying_committee_friction ({avg_fric:.3f})"
    )


def test_buying_committee_friction_lowers_contact_authority() -> None:
    """buying_committee_friction worlds should have lower mean latent_contact_authority."""
    bc_means = []
    fd_means = []
    for seed in range(15):
        config = GenerationConfig(seed=seed, n_accounts=100, n_contacts=300, n_leads=400)
        gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed)
        narrative = gen.world_spec.narrative
        assert narrative is not None

        g_bc = sample_hidden_graph(seed=seed, motif_family_name="buying_committee_friction")
        r_bc = build_population(config, narrative, g_bc)
        bc_means.append(
            sum(t["latent_contact_authority"] for t in r_bc.latent_state.contact_latents.values())
            / config.n_contacts
        )

        g_fd = sample_hidden_graph(seed=seed, motif_family_name="fit_dominant")
        r_fd = build_population(config, narrative, g_fd)
        fd_means.append(
            sum(t["latent_contact_authority"] for t in r_fd.latent_state.contact_latents.values())
            / config.n_contacts
        )

    avg_bc = sum(bc_means) / len(bc_means)
    avg_fd = sum(fd_means) / len(fd_means)
    assert avg_bc < avg_fd, (
        f"Expected buying_committee_friction mean ({avg_bc:.3f}) < fit_dominant ({avg_fd:.3f})"
    )


# ---------------------------------------------------------------------------
# Latent state entity-ID alignment
# ---------------------------------------------------------------------------


def test_latent_state_account_ids_match_rows() -> None:
    result = _make_result()
    row_ids = {a.account_id for a in result.accounts}
    assert set(result.latent_state.account_latents.keys()) == row_ids


def test_latent_state_contact_ids_match_rows() -> None:
    result = _make_result()
    row_ids = {c.contact_id for c in result.contacts}
    assert set(result.latent_state.contact_latents.keys()) == row_ids


def test_latent_state_lead_ids_match_rows() -> None:
    result = _make_result()
    row_ids = {lead.lead_id for lead in result.leads}
    assert set(result.latent_state.lead_latents.keys()) == row_ids


# ---------------------------------------------------------------------------
# Narrative validation (COPILOT-2 / COPILOT-3)
# ---------------------------------------------------------------------------


def _base_narrative() -> NarrativeSpec:
    """Return a minimal valid NarrativeSpec for mutation tests."""
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=0)
    narrative = gen.world_spec.narrative
    assert narrative is not None
    return narrative


def _build_with_narrative(narrative: NarrativeSpec) -> PopulationResult:
    config = GenerationConfig(seed=0, n_accounts=10, n_contacts=20, n_leads=30)
    graph = sample_hidden_graph(seed=0)
    return build_population(config, narrative, graph)


def test_empty_industries_raises() -> None:
    import dataclasses

    narrative = _base_narrative()
    bad_market = dataclasses.replace(narrative.market, icp_industries=())
    bad_narrative = dataclasses.replace(narrative, market=bad_market)
    with pytest.raises(InvalidConfigError, match="icp_industries"):
        _build_with_narrative(bad_narrative)


def test_empty_geographies_raises() -> None:
    import dataclasses

    narrative = _base_narrative()
    bad_market = dataclasses.replace(narrative.market, geographies=())
    bad_narrative = dataclasses.replace(narrative, market=bad_market)
    with pytest.raises(InvalidConfigError, match="geographies"):
        _build_with_narrative(bad_narrative)


def test_empty_personas_raises() -> None:
    import dataclasses

    narrative = _base_narrative()
    bad_narrative = dataclasses.replace(narrative, personas=())
    with pytest.raises(InvalidConfigError, match="personas"):
        _build_with_narrative(bad_narrative)


def test_empty_channels_raises() -> None:
    import dataclasses

    narrative = _base_narrative()
    bad_gtm = dataclasses.replace(narrative.gtm_motion, channels=())
    bad_narrative = dataclasses.replace(narrative, gtm_motion=bad_gtm)
    with pytest.raises(InvalidConfigError, match="channels"):
        _build_with_narrative(bad_narrative)


def test_channel_weights_zero_shares_falls_back_to_uniform() -> None:
    """If all GTM shares are 0, _channel_weights should return uniform weights."""
    narrative = _base_narrative()
    import dataclasses

    bad_gtm = dataclasses.replace(
        narrative.gtm_motion,
        inbound_share=0.0,
        outbound_share=0.0,
        partner_share=0.0,
    )
    bad_narrative = dataclasses.replace(narrative, gtm_motion=bad_gtm)
    channels, weights = _channel_weights(bad_narrative)
    assert len(channels) == len(weights)
    assert all(w > 0 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-9
    expected = 1.0 / len(channels)
    assert all(abs(w - expected) < 1e-9 for w in weights)

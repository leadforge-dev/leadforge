"""Tests for the generation-scheme registry and Generator dispatch (LTV-Pd)."""

import pytest

from leadforge.api.generator import Generator
from leadforge.api.recipes import Recipe
from leadforge.core.exceptions import InvalidRecipeError
from leadforge.core.models import DEFAULT_SCHEME, WorldSpec
from leadforge.schemes import (
    GenerationScheme,
    UnknownSchemeError,
    available_schemes,
    get_scheme,
    register_scheme,
)
from leadforge.schemes.lead_scoring import LEAD_SCORING_SCHEME, LeadScoringScheme

_SMALL = {"n_accounts": 20, "n_contacts": 40, "n_leads": 60, "difficulty": "intro"}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_lead_scoring_registered() -> None:
    assert "lead_scoring" in available_schemes()
    assert get_scheme("lead_scoring") is LEAD_SCORING_SCHEME


def test_lifecycle_scheme_registered() -> None:
    from leadforge.schemes.lifecycle import LIFECYCLE_SCHEME

    assert "lifecycle" in available_schemes()
    assert get_scheme("lifecycle") is LIFECYCLE_SCHEME
    assert LIFECYCLE_SCHEME.name == "lifecycle"


def test_lifecycle_scheme_is_a_stub() -> None:
    # Pipeline not implemented yet (built across LTV-M3…M6); calling it must
    # fail loudly rather than silently doing nothing.
    sch = get_scheme("lifecycle")
    with pytest.raises(NotImplementedError):
        sch.build_world(None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        sch.write_bundle(None, "out")  # type: ignore[arg-type]


def test_lead_scoring_scheme_name() -> None:
    assert LEAD_SCORING_SCHEME.name == "lead_scoring"


def test_lead_scoring_name_matches_default_scheme() -> None:
    # DEFAULT_SCHEME (core) and LeadScoringScheme.name (schemes) are declared in
    # separate layers; guard against drift.
    assert LeadScoringScheme.name == DEFAULT_SCHEME


def test_lead_scoring_satisfies_protocol() -> None:
    # runtime_checkable Protocol checks attribute *names* only (name,
    # build_world), not signatures — a weak structural check, not full
    # conformance. End-to-end behaviour is covered by the generate() tests.
    assert isinstance(LEAD_SCORING_SCHEME, GenerationScheme)
    assert callable(LEAD_SCORING_SCHEME.build_world)


def test_get_unknown_scheme_raises() -> None:
    with pytest.raises(UnknownSchemeError, match="does_not_exist"):
        get_scheme("does_not_exist")


def test_register_same_instance_is_idempotent() -> None:
    register_scheme(LEAD_SCORING_SCHEME)  # already registered; must not raise
    assert get_scheme("lead_scoring") is LEAD_SCORING_SCHEME


def test_register_conflicting_name_raises() -> None:
    clash = LeadScoringScheme()  # same name, different instance
    with pytest.raises(ValueError, match="already registered"):
        register_scheme(clash)


def test_available_schemes_sorted_tuple() -> None:
    names = available_schemes()
    assert isinstance(names, tuple)
    assert list(names) == sorted(names)


# ---------------------------------------------------------------------------
# Recipe.scheme field
# ---------------------------------------------------------------------------


def _minimal_recipe_dict(**extra: object) -> dict:
    base = {
        "id": "test_recipe",
        "title": "Test",
        "vertical": "test",
        "description": "test recipe",
        "primary_task": "converted_within_90_days",
        "supported_modes": ["student_public"],
        "supported_difficulty": ["intro"],
        "default_population": {"n_accounts": 10, "n_contacts": 20, "n_leads": 30},
        "horizon_days": 90,
    }
    base.update(extra)
    return base


def test_recipe_scheme_defaults_to_lead_scoring() -> None:
    recipe = Recipe.from_dict(_minimal_recipe_dict())
    assert recipe.scheme == "lead_scoring"


def test_recipe_scheme_parsed_when_present() -> None:
    recipe = Recipe.from_dict(_minimal_recipe_dict(scheme="lifecycle"))
    assert recipe.scheme == "lifecycle"


def test_recipe_scheme_rejects_empty() -> None:
    with pytest.raises(InvalidRecipeError, match="scheme"):
        Recipe.from_dict(_minimal_recipe_dict(scheme=""))


def test_recipe_scheme_rejects_non_string() -> None:
    with pytest.raises(InvalidRecipeError, match="scheme"):
        Recipe.from_dict(_minimal_recipe_dict(scheme=123))


# ---------------------------------------------------------------------------
# WorldSpec + Generator threading
# ---------------------------------------------------------------------------


def test_world_spec_scheme_defaults_to_lead_scoring() -> None:
    assert WorldSpec().scheme == "lead_scoring"


def test_from_recipe_sets_scheme_on_world_spec() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert gen.world_spec.scheme == "lead_scoring"


def test_generate_runs_through_registered_scheme() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    bundle = gen.generate(**_SMALL)
    assert bundle.artifacts.population is not None
    assert bundle.artifacts.simulation_result is not None
    assert len(bundle.artifacts.population.leads) == 60


def test_generate_records_scheme_on_bundle_spec() -> None:
    # Regression: generate() must thread the scheme through to the returned
    # bundle's spec (an earlier revision rebuilt WorldSpec without it, so
    # bundle.spec.scheme silently fell back to the default).
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    bundle = gen.generate(**_SMALL)
    assert bundle.spec.scheme == "lead_scoring"


def test_generate_is_deterministic_through_scheme() -> None:
    # Locks the byte-identity intent of LTV-Pd: the scheme path is deterministic
    # given (recipe, config, seed).
    a = Generator.from_recipe("b2b_saas_procurement_v1", seed=42).generate(**_SMALL)
    b = Generator.from_recipe("b2b_saas_procurement_v1", seed=42).generate(**_SMALL)
    assert a.artifacts.simulation_result is not None
    assert b.artifacts.simulation_result is not None
    lead_outcomes_a = {
        lead.lead_id: lead.converted_within_90_days for lead in a.artifacts.simulation_result.leads
    }
    lead_outcomes_b = {
        lead.lead_id: lead.converted_within_90_days for lead in b.artifacts.simulation_result.leads
    }
    assert lead_outcomes_a == lead_outcomes_b
    assert len(a.artifacts.simulation_result.touches) == len(b.artifacts.simulation_result.touches)


def test_generate_unknown_scheme_raises() -> None:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    # Force an unregistered scheme onto the world spec to prove dispatch is live.
    gen.world_spec.scheme = "nope"
    with pytest.raises(UnknownSchemeError):
        gen.generate(n_accounts=10, n_contacts=20, n_leads=30, difficulty="intro")

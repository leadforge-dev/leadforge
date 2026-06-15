"""Tests for LifecycleScheme.build_world + relational export (LTV-Pn.4a)."""

from __future__ import annotations

import pytest

from leadforge.core.models import GenerationConfig, WorldBundle
from leadforge.schemes import get_scheme
from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts
from leadforge.schemes.lifecycle.population import LIFECYCLE_MOTIF_FAMILIES
from leadforge.schemes.lifecycle.relationships import LIFECYCLE_CONSTRAINTS
from leadforge.schemes.lifecycle.render.relational import to_dataframes

_N = 120


def _build(seed: int = 42, n_customers: int = _N) -> WorldBundle:
    return get_scheme("lifecycle").build_world(
        GenerationConfig(seed=seed, n_customers=n_customers), narrative=None
    )


# ---------------------------------------------------------------------------
# build_world
# ---------------------------------------------------------------------------


def test_returns_bundle_with_lifecycle_artifacts() -> None:
    bundle = _build()
    assert isinstance(bundle, WorldBundle)
    assert bundle.spec.scheme == "lifecycle"
    assert isinstance(bundle.artifacts, LifecycleArtifacts)
    arts = bundle.artifacts
    assert len(arts.population.customers) == _N
    assert len(arts.simulation_result.subscriptions) == _N
    assert arts.motif_family in LIFECYCLE_MOTIF_FAMILIES
    # The sampled motif is recorded on the population too (engine reads it there).
    assert arts.population.motif_family == arts.motif_family


def test_consumes_config_lifecycle_fields() -> None:
    bundle = get_scheme("lifecycle").build_world(
        GenerationConfig(seed=1, n_customers=50, early_tenure_weeks=6), narrative=None
    )
    arts = bundle.artifacts
    assert len(arts.population.customers) == 50
    # forward_window_days = max(forward_windows_days) = 730 → full coverage.
    assert arts.simulation_result.forward_window_days == 730
    assert arts.simulation_result.early_tenure_weeks == 6


def test_deterministic_given_seed() -> None:
    a = _build(seed=7)
    b = _build(seed=7)
    assert a.artifacts.motif_family == b.artifacts.motif_family
    assert [s.to_dict() for s in a.artifacts.simulation_result.subscriptions] == [
        s.to_dict() for s in b.artifacts.simulation_result.subscriptions
    ]


def test_motif_varies_across_seeds() -> None:
    motifs = {_build(seed=s, n_customers=60).artifacts.motif_family for s in range(15)}
    # The invariant we care about: not a single fixed motif across seeds.
    assert len(motifs) >= 3
    assert motifs <= set(LIFECYCLE_MOTIF_FAMILIES)


def test_difficulty_not_yet_differentiating() -> None:
    """Tracked-gap guard (LTV-Pn.4a): build_world does not yet consume
    config.difficulty, so every tier yields the same world.  When Pn.4b wires
    difficulty in, this test must be updated to assert the tiers DIFFER —
    flipping it is the reminder that the gap is closed.
    """
    intro = get_scheme("lifecycle").build_world(
        GenerationConfig(seed=5, n_customers=60, difficulty="intro"), narrative=None
    )
    advanced = get_scheme("lifecycle").build_world(
        GenerationConfig(seed=5, n_customers=60, difficulty="advanced"), narrative=None
    )
    assert intro.artifacts.motif_family == advanced.artifacts.motif_family
    assert [s.to_dict() for s in intro.artifacts.simulation_result.subscriptions] == [
        s.to_dict() for s in advanced.artifacts.simulation_result.subscriptions
    ]


def test_rejects_unsupported_forward_windows_override() -> None:
    """COPILOT-1: config-driven forward windows aren't threaded into the
    snapshot builder yet, so an override must be rejected early and clearly
    rather than produce a manifest that disagrees with the task dirs (or
    under-simulate and fail opaquely downstream)."""
    from leadforge.core.exceptions import InvalidConfigError

    cfg = GenerationConfig(seed=5, n_customers=40, forward_windows_days=(30, 90))
    with pytest.raises(InvalidConfigError, match="forward_windows_days"):
        get_scheme("lifecycle").build_world(cfg, narrative=None)


def test_narrative_is_optional() -> None:
    # The lifecycle population builder generates its own firmographics; build_world
    # accepts narrative for protocol parity but must not require it.
    bundle = get_scheme("lifecycle").build_world(GenerationConfig(seed=3), narrative=None)
    assert isinstance(bundle.artifacts, LifecycleArtifacts)


# ---------------------------------------------------------------------------
# relational export
# ---------------------------------------------------------------------------


def test_relational_tables_present_and_typed() -> None:
    arts = _build().artifacts
    dfs = to_dataframes(arts.simulation_result, arts.population)
    assert set(dfs) == {
        "accounts",
        "customers",
        "subscriptions",
        "subscription_events",
        "health_signals",
        "invoices",
    }
    assert len(dfs["customers"]) == _N
    assert len(dfs["subscriptions"]) == _N
    assert len(dfs["accounts"]) == len(arts.population.accounts)
    # Non-empty event tables for a 120-customer world.
    assert len(dfs["health_signals"]) > 0
    assert len(dfs["invoices"]) > 0


def test_relational_fk_integrity() -> None:
    arts = _build().artifacts
    dfs = to_dataframes(arts.simulation_result, arts.population)
    for fk in LIFECYCLE_CONSTRAINTS:
        child = dfs[fk.child_table]
        parent = dfs[fk.parent_table]
        if child.empty:
            continue
        parent_keys = set(parent[fk.parent_column])
        orphans = set(child[fk.child_column]) - parent_keys
        assert not orphans, f"{fk.child_table}.{fk.child_column} has orphans: {list(orphans)[:3]}"


def test_relational_deterministic() -> None:
    a_arts = _build(seed=9).artifacts
    a = to_dataframes(a_arts.simulation_result, a_arts.population)
    b_arts = _build(seed=9).artifacts
    b = to_dataframes(b_arts.simulation_result, b_arts.population)
    for name in a:
        assert a[name].equals(b[name]), name


def test_empty_population_yields_typed_empty_tables() -> None:
    # A degenerate-but-valid tiny world still produces correctly-typed tables.
    arts = _build(seed=2, n_customers=1).artifacts
    dfs = to_dataframes(arts.simulation_result, arts.population)
    assert len(dfs["customers"]) == 1
    for name, df in dfs.items():
        assert list(df.columns), f"{name} has no columns"

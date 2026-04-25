"""Tests for leadforge.structure.sampler — sample_hidden_graph."""

import pytest

from leadforge.structure.graph import WorldGraph
from leadforge.structure.motifs import MOTIF_FAMILY_NAMES
from leadforge.structure.node_types import NodeType
from leadforge.structure.sampler import sample_hidden_graph

# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------


def test_returns_world_graph() -> None:
    g = sample_hidden_graph(seed=0)
    assert isinstance(g, WorldGraph)


def test_sampled_graph_has_outcome_node() -> None:
    g = sample_hidden_graph(seed=0)
    outcome_nodes = [n for n in g.graph.nodes if g.node_type(n) == NodeType.OUTCOME]
    assert len(outcome_nodes) >= 1


def test_sampled_graph_is_dag() -> None:
    import networkx as nx

    g = sample_hidden_graph(seed=0)
    assert nx.is_directed_acyclic_graph(g.graph)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_graph() -> None:
    g1 = sample_hidden_graph(seed=42)
    g2 = sample_hidden_graph(seed=42)
    assert g1.motif_family == g2.motif_family
    assert sorted(g1.graph.nodes) == sorted(g2.graph.nodes)
    assert sorted(g1.graph.edges) == sorted(g2.graph.edges)
    # Edge weights must also be identical — catches regressions in weight jitter.
    weights1 = {(u, v): d["weight"] for u, v, d in g1.graph.edges(data=True)}
    weights2 = {(u, v): d["weight"] for u, v, d in g2.graph.edges(data=True)}
    assert weights1 == weights2


def test_different_seeds_can_differ() -> None:
    graphs = [sample_hidden_graph(seed=s) for s in range(20)]
    families = {g.motif_family for g in graphs}
    # With 5 families and 20 seeds, we expect more than one family.
    assert len(families) > 1


# ---------------------------------------------------------------------------
# Pinned motif family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MOTIF_FAMILY_NAMES)
def test_pinned_motif_family(name: str) -> None:
    g = sample_hidden_graph(seed=7, motif_family_name=name)
    assert g.motif_family == name


def test_unknown_motif_family_raises() -> None:
    with pytest.raises(KeyError, match="bad_family"):
        sample_hidden_graph(seed=0, motif_family_name="bad_family")


def test_bool_seed_raises() -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        sample_hidden_graph(seed=True)  # type: ignore[arg-type]


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError, match="non-negative int"):
        sample_hidden_graph(seed=-1)


# ---------------------------------------------------------------------------
# Graph properties across many seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_all_sampled_graphs_are_valid(seed: int) -> None:
    """Property test: no seed should produce an invalid graph."""
    g = sample_hidden_graph(seed=seed)
    # If we got here without GraphValidationError, the graph is valid.
    assert g.graph.number_of_nodes() >= 2
    assert g.graph.number_of_edges() >= 1


@pytest.mark.parametrize("name", MOTIF_FAMILY_NAMES)
def test_pinned_family_graphs_are_valid_across_seeds(name: str) -> None:
    for seed in range(10):
        g = sample_hidden_graph(seed=seed, motif_family_name=name)
        assert g.graph.number_of_nodes() >= 2


# ---------------------------------------------------------------------------
# Exports smoke tests
# ---------------------------------------------------------------------------


def test_to_json_is_parseable() -> None:
    import json

    g = sample_hidden_graph(seed=1)
    data = json.loads(g.to_json())
    assert "nodes" in data
    assert "edges" in data


def test_to_graphml_contains_graph_tag() -> None:
    g = sample_hidden_graph(seed=1)
    assert "<graph" in g.to_graphml()

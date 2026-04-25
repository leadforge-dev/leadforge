"""Tests for leadforge.structure.rewiring — stochastic rewiring rules."""

import numpy as np
import pytest

from leadforge.structure.graph import WorldGraph
from leadforge.structure.motifs import ALL_MOTIF_FAMILIES, MotifFamily
from leadforge.structure.rewiring import rewire

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_rewired_graph_passes_validation(motif: MotifFamily) -> None:
    """Every rewired graph must satisfy WorldGraph structural invariants."""
    for seed in range(20):
        nodes, edges = rewire(motif, _rng(seed))
        # Should not raise.
        WorldGraph(nodes=nodes, edges=edges, motif_family=motif.name)


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_rewired_graph_has_at_least_two_nodes(motif: MotifFamily) -> None:
    for seed in range(10):
        nodes, _ = rewire(motif, _rng(seed))
        assert len(nodes) >= 2


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_rewired_graph_has_at_least_one_edge(motif: MotifFamily) -> None:
    for seed in range(10):
        _, edges = rewire(motif, _rng(seed))
        assert len(edges) >= 1


# ---------------------------------------------------------------------------
# Edge weight bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_rewired_edge_weights_in_range(motif: MotifFamily) -> None:
    for seed in range(10):
        _, edges = rewire(motif, _rng(seed))
        for e in edges:
            assert -1.0 <= e.weight <= 1.0, (
                f"Weight {e.weight} out of range for {e.source}→{e.target}"
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_rewire_is_deterministic(motif: MotifFamily) -> None:
    nodes_a, edges_a = rewire(motif, _rng(42))
    nodes_b, edges_b = rewire(motif, _rng(42))
    assert [n.node_id for n in nodes_a] == [n.node_id for n in nodes_b]
    assert [(e.source, e.target, round(e.weight, 10)) for e in edges_a] == [
        (e.source, e.target, round(e.weight, 10)) for e in edges_b
    ]


# ---------------------------------------------------------------------------
# Variability across seeds
# ---------------------------------------------------------------------------


def test_different_seeds_produce_different_graphs() -> None:
    """At least some seeds should yield structurally different graphs."""
    from leadforge.structure.motifs import FIT_DOMINANT

    structures: set[tuple[str, ...]] = set()
    for seed in range(40):
        nodes, _ = rewire(FIT_DOMINANT, _rng(seed))
        structures.add(tuple(sorted(n.node_id for n in nodes)))
    # With _DROP_PROB=0.4 and two optional nodes we expect variation.
    assert len(structures) > 1


# ---------------------------------------------------------------------------
# Optional node dropping
# ---------------------------------------------------------------------------


def test_required_nodes_never_dropped() -> None:
    """Non-optional nodes must always be present after rewiring."""
    from leadforge.structure.motifs import FIT_DOMINANT

    required = {
        n.node_id
        for n in FIT_DOMINANT.canonical_nodes
        if n.node_id not in FIT_DOMINANT.optional_node_ids
    }
    for seed in range(30):
        nodes, _ = rewire(FIT_DOMINANT, _rng(seed))
        present = {n.node_id for n in nodes}
        assert required <= present, (
            f"Seed {seed}: required node(s) {required - present} were dropped"
        )

"""Tests for leadforge.structure.motifs — motif family definitions."""

import pytest

from leadforge.structure.motifs import (
    ALL_MOTIF_FAMILIES,
    MOTIF_FAMILY_NAMES,
    MotifFamily,
    get_motif_family,
)
from leadforge.structure.node_types import NodeType

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_five_motif_families_defined() -> None:
    assert len(ALL_MOTIF_FAMILIES) == 5


def test_motif_family_names_match_registry() -> None:
    assert set(MOTIF_FAMILY_NAMES) == {m.name for m in ALL_MOTIF_FAMILIES}


def test_all_five_expected_names_present() -> None:
    expected = {
        "fit_dominant",
        "intent_dominant",
        "sales_execution_sensitive",
        "demo_trial_mediated",
        "buying_committee_friction",
    }
    assert set(MOTIF_FAMILY_NAMES) == expected


def test_get_motif_family_returns_correct_instance() -> None:
    for motif in ALL_MOTIF_FAMILIES:
        assert get_motif_family(motif.name) is motif


def test_get_motif_family_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown_family"):
        get_motif_family("unknown_family")


# ---------------------------------------------------------------------------
# Structural invariants per motif
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_has_at_least_one_outcome_node(motif: MotifFamily) -> None:
    outcomes = [n for n in motif.canonical_nodes if n.node_type == NodeType.OUTCOME]
    assert len(outcomes) >= 1, f"{motif.name} has no OUTCOME node"


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_node_ids_unique(motif: MotifFamily) -> None:
    ids = [n.node_id for n in motif.canonical_nodes]
    assert len(ids) == len(set(ids)), f"{motif.name} has duplicate node IDs"


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_edge_endpoints_exist(motif: MotifFamily) -> None:
    node_ids = {n.node_id for n in motif.canonical_nodes}
    for e in motif.canonical_edges:
        assert e.source in node_ids, f"{motif.name}: edge source {e.source!r} not in node set"
        assert e.target in node_ids, f"{motif.name}: edge target {e.target!r} not in node set"


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_optional_nodes_exist(motif: MotifFamily) -> None:
    node_ids = {n.node_id for n in motif.canonical_nodes}
    for opt_id in motif.optional_node_ids:
        assert opt_id in node_ids, (
            f"{motif.name}: optional node {opt_id!r} not in canonical node set"
        )


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_edge_weights_in_range(motif: MotifFamily) -> None:
    for e in motif.canonical_edges:
        assert -1.0 <= e.weight <= 1.0, (
            f"{motif.name}: edge {e.source}→{e.target} weight {e.weight} out of [-1, 1]"
        )


@pytest.mark.parametrize("motif", ALL_MOTIF_FAMILIES)
def test_motif_canonical_skeleton_builds_valid_graph(motif: MotifFamily) -> None:
    """The canonical (non-rewired) skeleton must pass WorldGraph validation."""
    from leadforge.structure.graph import WorldGraph

    g = WorldGraph(
        nodes=list(motif.canonical_nodes),
        edges=list(motif.canonical_edges),
        motif_family=motif.name,
    )
    assert g.graph.number_of_nodes() == len(motif.canonical_nodes)

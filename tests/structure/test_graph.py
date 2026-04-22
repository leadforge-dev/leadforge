"""Tests for leadforge.structure.graph — WorldGraph validation and exports."""

import json

import pytest

from leadforge.structure.graph import (
    EdgeSpec,
    GraphValidationError,
    NodeSpec,
    WorldGraph,
)
from leadforge.structure.node_types import NodeType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_valid_graph() -> WorldGraph:
    """Smallest possible valid WorldGraph (root → outcome)."""
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT, label="Root"),
        NodeSpec("lead", NodeType.LEAD_STATE, label="Lead state"),
        NodeSpec("outcome", NodeType.OUTCOME, label="Outcome"),
    ]
    edges = [
        EdgeSpec("root", "lead", weight=0.8),
        EdgeSpec("lead", "outcome", weight=0.7),
    ]
    return WorldGraph(nodes=nodes, edges=edges, motif_family="test")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_valid_graph_constructs_without_error() -> None:
    g = _minimal_valid_graph()
    assert g.graph.number_of_nodes() == 3
    assert g.graph.number_of_edges() == 2


def test_motif_family_stored() -> None:
    g = _minimal_valid_graph()
    assert g.motif_family == "test"


def test_node_type_accessor() -> None:
    g = _minimal_valid_graph()
    assert g.node_type("root") == NodeType.ACCOUNT_LATENT
    assert g.node_type("outcome") == NodeType.OUTCOME


def test_topological_order_root_first() -> None:
    g = _minimal_valid_graph()
    order = g.topological_order()
    assert order.index("root") < order.index("outcome")


# ---------------------------------------------------------------------------
# Validation — acyclicity
# ---------------------------------------------------------------------------


def test_cycle_raises_graph_validation_error() -> None:
    nodes = [
        NodeSpec("a", NodeType.ACCOUNT_LATENT),
        NodeSpec("b", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [
        EdgeSpec("a", "b"),
        EdgeSpec("b", "a"),  # creates cycle
        EdgeSpec("b", "out"),
    ]
    with pytest.raises(GraphValidationError, match="cycle"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


# ---------------------------------------------------------------------------
# Validation — node type legality
# ---------------------------------------------------------------------------


def test_outcome_with_child_raises() -> None:
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT),
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
        NodeSpec("post", NodeType.LEAD_STATE),  # outcome → post is forbidden
    ]
    edges = [
        EdgeSpec("root", "lead"),
        EdgeSpec("lead", "out"),
        EdgeSpec("out", "post"),
    ]
    with pytest.raises(GraphValidationError, match="leaf"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


def test_lead_state_without_parent_raises() -> None:
    nodes = [
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [EdgeSpec("lead", "out")]
    with pytest.raises(GraphValidationError, match="requires at least one parent"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


# ---------------------------------------------------------------------------
# Validation — nondegeneracy
# ---------------------------------------------------------------------------


def test_no_edges_raises() -> None:
    # Two root-eligible nodes with no edges hit the nondegeneracy check.
    nodes = [
        NodeSpec("a", NodeType.ACCOUNT_LATENT),
        NodeSpec("b", NodeType.ACCOUNT_LATENT),
    ]
    with pytest.raises(GraphValidationError, match="no edges"):
        WorldGraph(nodes=nodes, edges=[], motif_family="test")


def test_single_node_raises() -> None:
    nodes = [NodeSpec("a", NodeType.ACCOUNT_LATENT)]
    with pytest.raises(GraphValidationError, match="only 1 node"):
        WorldGraph(nodes=nodes, edges=[], motif_family="test")


# ---------------------------------------------------------------------------
# Validation — outcome reachability
# ---------------------------------------------------------------------------


def test_no_outcome_node_raises() -> None:
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT),
        NodeSpec("lead", NodeType.LEAD_STATE),
    ]
    edges = [EdgeSpec("root", "lead")]
    with pytest.raises(GraphValidationError, match="no OUTCOME node"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


def test_outcome_reachable_from_different_root_passes() -> None:
    # 'out' is reachable from 'root2', even though 'lead' has no path to it.
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT),
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("root2", NodeType.ACCOUNT_LATENT),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [
        EdgeSpec("root", "lead"),
        EdgeSpec("root2", "out"),
    ]
    g = WorldGraph(nodes=nodes, edges=edges, motif_family="test")
    assert g.graph.number_of_nodes() == 4


# ---------------------------------------------------------------------------
# Duplicate node IDs
# ---------------------------------------------------------------------------


def test_reserved_node_metadata_key_raises() -> None:
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT, metadata={"node_type": "bad"}),
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [EdgeSpec("root", "lead"), EdgeSpec("lead", "out")]
    with pytest.raises(GraphValidationError, match="reserved key"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


def test_reserved_edge_weight_key_raises() -> None:
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT),
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [
        EdgeSpec("root", "lead", metadata={"weight": 0.5}),
        EdgeSpec("lead", "out"),
    ]
    with pytest.raises(GraphValidationError, match="reserved key 'weight'"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


def test_edge_weight_out_of_range_raises() -> None:
    nodes = [
        NodeSpec("root", NodeType.ACCOUNT_LATENT),
        NodeSpec("lead", NodeType.LEAD_STATE),
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [EdgeSpec("root", "lead", weight=1.5), EdgeSpec("lead", "out")]
    with pytest.raises(GraphValidationError, match="outside \\[-1, 1\\]"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


def test_duplicate_node_id_raises() -> None:
    nodes = [
        NodeSpec("a", NodeType.ACCOUNT_LATENT),
        NodeSpec("a", NodeType.LEAD_STATE),  # duplicate
        NodeSpec("out", NodeType.OUTCOME),
    ]
    edges = [EdgeSpec("a", "out")]
    with pytest.raises(GraphValidationError, match="Duplicate"):
        WorldGraph(nodes=nodes, edges=edges, motif_family="test")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_to_dict_keys() -> None:
    g = _minimal_valid_graph()
    d = g.to_dict()
    assert set(d.keys()) == {"motif_family", "nodes", "edges"}


def test_to_json_round_trips() -> None:
    g = _minimal_valid_graph()
    data = json.loads(g.to_json())
    assert data["motif_family"] == "test"
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2


def test_to_graphml_returns_string() -> None:
    g = _minimal_valid_graph()
    gml = g.to_graphml()
    assert "graphml" in gml.lower()

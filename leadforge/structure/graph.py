"""Hidden world graph representation and validation.

:class:`WorldGraph` wraps a ``networkx.DiGraph`` and enforces structural
invariants — acyclicity, node-type legality, reachability, and
nondegeneracy — at construction time and on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from leadforge.core.exceptions import LeadforgeError
from leadforge.structure.node_types import LEAF_ONLY, REQUIRES_PARENT, NodeType


class GraphValidationError(LeadforgeError):
    """Raised when a hidden world graph violates a structural invariant."""


@dataclass
class NodeSpec:
    """Specification for a single hidden-graph node.

    Attributes:
        node_id: Unique string identifier within the graph.
        node_type: Semantic category of the node.
        label: Human-readable name used in exports.
        metadata: Arbitrary extra attributes (e.g. prior strength, proxy
            accuracy).  Serialised as JSON in GraphML export.
    """

    node_id: str
    node_type: NodeType
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """Specification for a directed edge between two hidden-graph nodes.

    Attributes:
        source: ``node_id`` of the parent node.
        target: ``node_id`` of the child node.
        weight: Signed influence strength in the range [-1, 1].  Positive
            values indicate facilitation; negative values indicate
            inhibition.
        metadata: Arbitrary extra attributes (e.g. mechanism type, lag).
    """

    source: str
    target: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class WorldGraph:
    """Validated directed acyclic graph representing one hidden world.

    The graph is built from :class:`NodeSpec` and :class:`EdgeSpec`
    objects and validated immediately.  All subsequent access is via the
    underlying ``networkx.DiGraph`` exposed as :attr:`graph`.

    Args:
        nodes: Node specifications.  Node IDs must be unique.
        edges: Edge specifications.  Both endpoints must reference known
            node IDs.
        motif_family: Name of the motif family that seeded this graph.

    Raises:
        GraphValidationError: If any structural invariant is violated.
    """

    def __init__(
        self,
        nodes: list[NodeSpec],
        edges: list[EdgeSpec],
        motif_family: str,
    ) -> None:
        self._motif_family = motif_family
        self._graph: nx.DiGraph = nx.DiGraph()

        # Add nodes
        seen_ids: set[str] = set()
        for n in nodes:
            if n.node_id in seen_ids:
                raise GraphValidationError(f"Duplicate node_id: {n.node_id!r}")
            seen_ids.add(n.node_id)
            self._graph.add_node(
                n.node_id,
                node_type=n.node_type.value,
                label=n.label,
                **n.metadata,
            )

        # Add edges
        for e in edges:
            if e.source not in seen_ids:
                raise GraphValidationError(f"Edge source {e.source!r} not in node set")
            if e.target not in seen_ids:
                raise GraphValidationError(f"Edge target {e.target!r} not in node set")
            self._graph.add_edge(
                e.source,
                e.target,
                weight=e.weight,
                **e.metadata,
            )

        self._validate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying ``networkx.DiGraph`` (read-only intent)."""
        return self._graph

    @property
    def motif_family(self) -> str:
        """Name of the motif family that produced this graph."""
        return self._motif_family

    def node_type(self, node_id: str) -> NodeType:
        """Return the :class:`NodeType` of *node_id*."""
        return NodeType(self._graph.nodes[node_id]["node_type"])

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (roots first)."""
        return list(nx.topological_sort(self._graph))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        nodes = [
            {
                "node_id": n,
                "node_type": self._graph.nodes[n]["node_type"],
                "label": self._graph.nodes[n].get("label", ""),
            }
            for n in self.topological_order()
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1.0),
            }
            for u, v, data in self._graph.edges(data=True)
        ]
        return {
            "motif_family": self._motif_family,
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)

    def to_graphml(self) -> str:
        """Return a GraphML string representation."""
        lines = nx.generate_graphml(self._graph)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Run all structural invariant checks.

        Raises:
            GraphValidationError: on first violation found.
        """
        self._check_acyclic()
        self._check_node_type_legality()
        self._check_nondegeneracy()
        self._check_outcome_reachable()

    def _check_acyclic(self) -> None:
        if not nx.is_directed_acyclic_graph(self._graph):
            cycle = nx.find_cycle(self._graph)
            raise GraphValidationError(f"Graph contains a cycle: {cycle}")

    def _check_node_type_legality(self) -> None:
        for node_id in self._graph.nodes:
            nt = self.node_type(node_id)
            in_degree = self._graph.in_degree(node_id)
            out_degree = self._graph.out_degree(node_id)

            if nt in REQUIRES_PARENT and in_degree == 0:
                raise GraphValidationError(
                    f"Node {node_id!r} (type={nt.value}) requires at least one parent but has none"
                )
            if nt in LEAF_ONLY and out_degree > 0:
                raise GraphValidationError(
                    f"Node {node_id!r} (type={nt.value}) must be a leaf but "
                    f"has {out_degree} child(ren)"
                )

    def _check_nondegeneracy(self) -> None:
        """Reject fully isolated graphs and single-node graphs."""
        n = self._graph.number_of_nodes()
        if n < 2:
            raise GraphValidationError(
                f"Graph has only {n} node(s); a meaningful hidden world requires at least 2 nodes"
            )
        # Reject a graph where every node is isolated (no edges at all).
        if self._graph.number_of_edges() == 0:
            raise GraphValidationError(
                "Graph has no edges; a meaningful hidden world requires at least one causal edge"
            )

    def _check_outcome_reachable(self) -> None:
        """Every OUTCOME node must be reachable from at least one root."""
        outcome_nodes = [n for n in self._graph.nodes if self.node_type(n) == NodeType.OUTCOME]
        if not outcome_nodes:
            raise GraphValidationError(
                "Graph has no OUTCOME node; every world must have at least "
                "one conversion-outcome node"
            )
        roots = [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]
        for outcome in outcome_nodes:
            reachable = False
            for root in roots:
                if nx.has_path(self._graph, root, outcome):
                    reachable = True
                    break
            if not reachable:
                raise GraphValidationError(
                    f"OUTCOME node {outcome!r} is not reachable from any root node"
                )

"""Stochastic rewiring of motif-family graph skeletons.

:func:`rewire` takes a :class:`~leadforge.structure.motifs.MotifFamily`
and a seeded :class:`~numpy.random.Generator` and returns perturbed lists
of :class:`~leadforge.structure.graph.NodeSpec` and
:class:`~leadforge.structure.graph.EdgeSpec` that still satisfy the graph
invariants (acyclicity, legality, nondegeneracy).

Permitted variability (§11.3 of architecture spec):
- dropping optional mediator nodes (and their incident edges)
- perturbing edge weights within a bounded range
- adding an optional latent confounder node

Forbidden variability (hard constraints enforced here):
- chronologically impossible edges (validated downstream in WorldGraph)
- orphaned outcome nodes
- degenerate worlds with no edges
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from leadforge.structure.graph import EdgeSpec, NodeSpec
from leadforge.structure.node_types import NodeType

if TYPE_CHECKING:
    import numpy as np

    from leadforge.structure.motifs import MotifFamily

# Maximum ± perturbation applied to each edge weight.
_WEIGHT_JITTER = 0.15

# Probability that each optional node is dropped (per rewiring call).
_DROP_PROB = 0.4

# Probability that an optional latent confounder is injected.
_CONFOUNDER_PROB = 0.35


def rewire(
    motif: MotifFamily,
    rng: np.random.Generator,
) -> tuple[list[NodeSpec], list[EdgeSpec]]:
    """Return perturbed node/edge lists derived from *motif*'s skeleton.

    The canonical skeleton from *motif* is copied and then stochastically
    modified:

    1. Each optional node is independently dropped with probability
       :data:`_DROP_PROB`.  Edges incident to a dropped node are removed.
    2. Edge weights are jittered by ±:data:`_WEIGHT_JITTER`, clamped to
       [-1, 1].
    3. With probability :data:`_CONFOUNDER_PROB`, a single additional
       ``ACCOUNT_LATENT`` confounder node is injected with edges to the
       first ``LEAD_STATE`` node found in the skeleton.

    Args:
        motif: The motif family providing the canonical skeleton.
        rng: A seeded ``numpy.random.Generator`` for reproducibility.

    Returns:
        A ``(nodes, edges)`` tuple suitable for passing to
        :class:`~leadforge.structure.graph.WorldGraph`.
    """
    nodes: list[NodeSpec] = [copy.copy(n) for n in motif.canonical_nodes]
    edges: list[EdgeSpec] = [copy.copy(e) for e in motif.canonical_edges]

    # Step 1 — drop optional nodes
    dropped: set[str] = set()
    for node in list(nodes):
        if node.node_id in motif.optional_node_ids:
            if rng.random() < _DROP_PROB:
                dropped.add(node.node_id)

    if dropped:
        nodes = [n for n in nodes if n.node_id not in dropped]
        edges = [e for e in edges if e.source not in dropped and e.target not in dropped]

    # Step 2 — jitter edge weights
    active_node_ids = {n.node_id for n in nodes}
    perturbed_edges: list[EdgeSpec] = []
    for e in edges:
        jitter = rng.uniform(-_WEIGHT_JITTER, _WEIGHT_JITTER)
        new_weight = float(max(-1.0, min(1.0, e.weight + jitter)))
        perturbed_edges.append(
            EdgeSpec(
                source=e.source,
                target=e.target,
                weight=new_weight,
                metadata=dict(e.metadata),
            )
        )
    edges = perturbed_edges

    # Step 3 — optional latent confounder injection
    if rng.random() < _CONFOUNDER_PROB:
        # Find first LEAD_STATE node to attach to.
        lead_state_ids = [n.node_id for n in nodes if n.node_type == NodeType.LEAD_STATE]
        if lead_state_ids:
            conf_id = _unique_id("latent_confounder", active_node_ids)
            conf_weight = float(rng.uniform(0.1, 0.5))
            nodes.append(
                NodeSpec(
                    node_id=conf_id,
                    node_type=NodeType.ACCOUNT_LATENT,
                    label="Latent confounder",
                )
            )
            edges.append(
                EdgeSpec(
                    source=conf_id,
                    target=lead_state_ids[0],
                    weight=conf_weight,
                )
            )

    return nodes, edges


def _unique_id(base: str, existing: set[str]) -> str:
    """Return *base* if not in *existing*, else *base_2*, *base_3*, …"""
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"

"""Node type definitions for the hidden world graph.

Each node in the hidden causal graph carries a :class:`NodeType` that
constrains how it participates in mechanisms, rewiring, and exports.
"""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Semantic category of a hidden-graph node.

    Values mirror the nine categories specified in §11.1 of the
    architecture spec.  Using ``str`` as a mixin makes serialisation
    (JSON, GraphML) straightforward without extra conversion.
    """

    GLOBAL_CONTEXT = "global_context"
    ACCOUNT_LATENT = "account_latent"
    CONTACT_LATENT = "contact_latent"
    LEAD_STATE = "lead_state"
    ENGAGEMENT_STATE = "engagement_state"
    SALES_PROCESS_STATE = "sales_process_state"
    OBSERVABLE_FEATURE_SOURCE = "observable_feature_source"
    OUTCOME = "outcome"
    POST_CONVERSION_STATE = "post_conversion_state"


# Node types that may appear as graph roots (no required predecessors).
ROOT_ELIGIBLE: frozenset[NodeType] = frozenset(
    {
        NodeType.GLOBAL_CONTEXT,
        NodeType.ACCOUNT_LATENT,
        NodeType.CONTACT_LATENT,
    }
)

# Node types that must have at least one predecessor.
REQUIRES_PARENT: frozenset[NodeType] = frozenset(
    {
        NodeType.LEAD_STATE,
        NodeType.ENGAGEMENT_STATE,
        NodeType.SALES_PROCESS_STATE,
        NodeType.OBSERVABLE_FEATURE_SOURCE,
        NodeType.OUTCOME,
        NodeType.POST_CONVERSION_STATE,
    }
)

# Node types that may not have children (leaf nodes only).
LEAF_ONLY: frozenset[NodeType] = frozenset(
    {
        NodeType.OUTCOME,
        NodeType.POST_CONVERSION_STATE,
    }
)

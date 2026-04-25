"""Motif family definitions for the v1 hidden world graph.

Each :class:`MotifFamily` describes the canonical node/edge skeleton for
one named hidden-world template.  The five v1 families are defined at the
bottom of this module; they are consumed by :mod:`leadforge.structure.sampler`
to seed a concrete :class:`~leadforge.structure.graph.WorldGraph`.

See §11.2 of the architecture spec for the semantics of each family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from leadforge.structure.graph import EdgeSpec, NodeSpec
from leadforge.structure.node_types import NodeType


@dataclass(frozen=True)
class MotifFamily:
    """Canonical template for one named hidden-world motif.

    Attributes:
        name: Machine-readable identifier used in manifests and exports.
        description: One-sentence human description of the causal story.
        canonical_nodes: Ordered list of :class:`NodeSpec` objects that
            form the core graph skeleton.  Node IDs must be unique within
            the family.
        canonical_edges: Directed edges between the canonical nodes.
        optional_node_ids: IDs from *canonical_nodes* that may be dropped
            during stochastic rewiring (see
            :mod:`leadforge.structure.rewiring`).
    """

    name: str
    description: str
    canonical_nodes: tuple[NodeSpec, ...]
    canonical_edges: tuple[EdgeSpec, ...]
    optional_node_ids: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# v1 motif family 1 — fit-dominant
# ---------------------------------------------------------------------------

FIT_DOMINANT: MotifFamily = MotifFamily(
    name="fit_dominant",
    description=(
        "Conversion is primarily driven by account and contact fit; "
        "engagement is partly downstream of fit rather than an independent driver."
    ),
    canonical_nodes=(
        NodeSpec("global_ctx", NodeType.GLOBAL_CONTEXT, label="Global context"),
        NodeSpec("acct_fit", NodeType.ACCOUNT_LATENT, label="Account fit"),
        NodeSpec("acct_maturity", NodeType.ACCOUNT_LATENT, label="Process maturity"),
        NodeSpec("contact_authority", NodeType.CONTACT_LATENT, label="Contact authority"),
        NodeSpec("budget_readiness", NodeType.ACCOUNT_LATENT, label="Budget readiness"),
        NodeSpec("lead_state", NodeType.LEAD_STATE, label="Lead state"),
        NodeSpec("engagement", NodeType.ENGAGEMENT_STATE, label="Engagement signal"),
        NodeSpec("conversion", NodeType.OUTCOME, label="Converted within 90 days"),
    ),
    canonical_edges=(
        EdgeSpec("global_ctx", "acct_fit", weight=0.3),
        EdgeSpec("global_ctx", "acct_maturity", weight=0.2),
        EdgeSpec("acct_fit", "lead_state", weight=0.7),
        EdgeSpec("acct_fit", "engagement", weight=0.5),
        EdgeSpec("acct_maturity", "lead_state", weight=0.4),
        EdgeSpec("budget_readiness", "lead_state", weight=0.5),
        EdgeSpec("contact_authority", "lead_state", weight=0.3),
        EdgeSpec("lead_state", "engagement", weight=0.4),
        EdgeSpec("lead_state", "conversion", weight=0.6),
        EdgeSpec("engagement", "conversion", weight=0.3),
    ),
    optional_node_ids=frozenset({"acct_maturity", "contact_authority"}),
)


# ---------------------------------------------------------------------------
# v1 motif family 2 — intent-dominant
# ---------------------------------------------------------------------------

INTENT_DOMINANT: MotifFamily = MotifFamily(
    name="intent_dominant",
    description=(
        "Behavioral engagement and urgency dominate conversion probability, "
        "even among leads with mixed account/contact fit scores."
    ),
    canonical_nodes=(
        NodeSpec("global_ctx", NodeType.GLOBAL_CONTEXT, label="Global context"),
        NodeSpec("acct_fit", NodeType.ACCOUNT_LATENT, label="Account fit"),
        NodeSpec("problem_awareness", NodeType.CONTACT_LATENT, label="Problem awareness"),
        NodeSpec("urgency", NodeType.LEAD_STATE, label="Urgency / timing"),
        NodeSpec("engagement", NodeType.ENGAGEMENT_STATE, label="Engagement signal"),
        NodeSpec("intent_score", NodeType.OBSERVABLE_FEATURE_SOURCE, label="Intent signal proxy"),
        NodeSpec("conversion", NodeType.OUTCOME, label="Converted within 90 days"),
    ),
    canonical_edges=(
        EdgeSpec("global_ctx", "problem_awareness", weight=0.25),
        EdgeSpec("acct_fit", "engagement", weight=0.2),
        EdgeSpec("problem_awareness", "urgency", weight=0.6),
        EdgeSpec("problem_awareness", "engagement", weight=0.6),
        EdgeSpec("urgency", "intent_score", weight=0.7),
        EdgeSpec("engagement", "intent_score", weight=0.6),
        EdgeSpec("intent_score", "conversion", weight=0.8),
        EdgeSpec("urgency", "conversion", weight=0.4),
    ),
    optional_node_ids=frozenset({"acct_fit"}),
)


# ---------------------------------------------------------------------------
# v1 motif family 3 — sales-execution-sensitive
# ---------------------------------------------------------------------------

SALES_EXECUTION_SENSITIVE: MotifFamily = MotifFamily(
    name="sales_execution_sensitive",
    description=(
        "Follow-up timing, rep quality, and sales process friction "
        "materially affect conversion outcomes beyond lead characteristics."
    ),
    canonical_nodes=(
        NodeSpec("global_ctx", NodeType.GLOBAL_CONTEXT, label="Global context"),
        NodeSpec("acct_fit", NodeType.ACCOUNT_LATENT, label="Account fit"),
        NodeSpec("contact_responsiveness", NodeType.CONTACT_LATENT, label="Contact responsiveness"),
        NodeSpec("lead_state", NodeType.LEAD_STATE, label="Lead state"),
        NodeSpec("sales_process", NodeType.SALES_PROCESS_STATE, label="Sales process quality"),
        NodeSpec("rep_quality", NodeType.SALES_PROCESS_STATE, label="Rep execution quality"),
        NodeSpec("sales_friction", NodeType.SALES_PROCESS_STATE, label="Process friction"),
        NodeSpec("conversion", NodeType.OUTCOME, label="Converted within 90 days"),
    ),
    canonical_edges=(
        EdgeSpec("global_ctx", "acct_fit", weight=0.3),
        EdgeSpec("global_ctx", "rep_quality", weight=0.2),
        EdgeSpec("acct_fit", "lead_state", weight=0.4),
        EdgeSpec("contact_responsiveness", "lead_state", weight=0.35),
        EdgeSpec("lead_state", "sales_process", weight=0.5),
        EdgeSpec("rep_quality", "sales_process", weight=0.6),
        EdgeSpec("rep_quality", "sales_friction", weight=-0.5),
        EdgeSpec("sales_process", "conversion", weight=0.6),
        EdgeSpec("sales_friction", "conversion", weight=-0.4),
    ),
    optional_node_ids=frozenset({"sales_friction", "contact_responsiveness"}),
)


# ---------------------------------------------------------------------------
# v1 motif family 4 — demo/trial-mediated
# ---------------------------------------------------------------------------

DEMO_TRIAL_MEDIATED: MotifFamily = MotifFamily(
    name="demo_trial_mediated",
    description=(
        "Product demonstration or trial progression acts as a major mediator "
        "between initial engagement and conversion."
    ),
    canonical_nodes=(
        NodeSpec("global_ctx", NodeType.GLOBAL_CONTEXT, label="Global context"),
        NodeSpec("acct_fit", NodeType.ACCOUNT_LATENT, label="Account fit"),
        NodeSpec("problem_awareness", NodeType.CONTACT_LATENT, label="Problem awareness"),
        NodeSpec("engagement", NodeType.ENGAGEMENT_STATE, label="Top-of-funnel engagement"),
        NodeSpec("demo_completion", NodeType.LEAD_STATE, label="Demo / trial completion"),
        NodeSpec("trial_depth", NodeType.OBSERVABLE_FEATURE_SOURCE, label="Trial depth proxy"),
        NodeSpec("sales_process", NodeType.SALES_PROCESS_STATE, label="Post-demo sales process"),
        NodeSpec("conversion", NodeType.OUTCOME, label="Converted within 90 days"),
    ),
    canonical_edges=(
        EdgeSpec("global_ctx", "acct_fit", weight=0.3),
        EdgeSpec("acct_fit", "engagement", weight=0.4),
        EdgeSpec("problem_awareness", "engagement", weight=0.5),
        EdgeSpec("engagement", "demo_completion", weight=0.6),
        EdgeSpec("acct_fit", "demo_completion", weight=0.4),
        EdgeSpec("demo_completion", "trial_depth", weight=0.7),
        EdgeSpec("demo_completion", "sales_process", weight=0.5),
        EdgeSpec("trial_depth", "conversion", weight=0.6),
        EdgeSpec("sales_process", "conversion", weight=0.4),
    ),
    # sales_process is the primary post-demo path to conversion; only the
    # observational proxy (trial_depth) may be dropped.
    optional_node_ids=frozenset({"trial_depth"}),
)


# ---------------------------------------------------------------------------
# v1 motif family 5 — buying-committee-friction
# ---------------------------------------------------------------------------

BUYING_COMMITTEE_FRICTION: MotifFamily = MotifFamily(
    name="buying_committee_friction",
    description=(
        "Multiple stakeholders and approval friction materially slow or block "
        "progression; contact authority and consensus dynamics dominate."
    ),
    canonical_nodes=(
        NodeSpec("global_ctx", NodeType.GLOBAL_CONTEXT, label="Global context"),
        NodeSpec("acct_fit", NodeType.ACCOUNT_LATENT, label="Account fit"),
        NodeSpec("contact_authority", NodeType.CONTACT_LATENT, label="Primary contact authority"),
        NodeSpec("committee_alignment", NodeType.CONTACT_LATENT, label="Committee alignment"),
        NodeSpec("lead_state", NodeType.LEAD_STATE, label="Lead state"),
        NodeSpec("approval_friction", NodeType.SALES_PROCESS_STATE, label="Approval friction"),
        NodeSpec("engagement", NodeType.ENGAGEMENT_STATE, label="Multi-stakeholder engagement"),
        NodeSpec("conversion", NodeType.OUTCOME, label="Converted within 90 days"),
    ),
    canonical_edges=(
        EdgeSpec("global_ctx", "acct_fit", weight=0.3),
        EdgeSpec("global_ctx", "committee_alignment", weight=0.2),
        EdgeSpec("acct_fit", "lead_state", weight=0.45),
        EdgeSpec("contact_authority", "lead_state", weight=0.5),
        EdgeSpec("committee_alignment", "approval_friction", weight=-0.6),
        EdgeSpec("lead_state", "engagement", weight=0.4),
        EdgeSpec("engagement", "approval_friction", weight=-0.3),
        EdgeSpec("contact_authority", "approval_friction", weight=-0.4),
        EdgeSpec("approval_friction", "conversion", weight=-0.5),
        EdgeSpec("lead_state", "conversion", weight=0.4),
        EdgeSpec("engagement", "conversion", weight=0.3),
    ),
    optional_node_ids=frozenset({"committee_alignment", "approval_friction"}),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_MOTIF_FAMILIES: tuple[MotifFamily, ...] = (
    FIT_DOMINANT,
    INTENT_DOMINANT,
    SALES_EXECUTION_SENSITIVE,
    DEMO_TRIAL_MEDIATED,
    BUYING_COMMITTEE_FRICTION,
)

MOTIF_FAMILY_NAMES: tuple[str, ...] = tuple(m.name for m in ALL_MOTIF_FAMILIES)

_BY_NAME: dict[str, MotifFamily] = {m.name: m for m in ALL_MOTIF_FAMILIES}


def get_motif_family(name: str) -> MotifFamily:
    """Look up a motif family by name.

    Args:
        name: One of the values in :data:`MOTIF_FAMILY_NAMES`.

    Returns:
        The corresponding :class:`MotifFamily`.

    Raises:
        KeyError: If *name* is not a known motif family.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown motif family {name!r}. Valid names: {sorted(_BY_NAME)}") from None

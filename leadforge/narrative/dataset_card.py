"""Dataset card renderer.

Produces the ``dataset_card.md`` artifact from a :class:`WorldSpec`.
The card follows the structure required by the architecture spec (§14.3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leadforge.core.models import WorldSpec


def render_dataset_card(world_spec: WorldSpec) -> str:
    """Return a Markdown dataset card string for *world_spec*.

    Sections present at all milestones:
    - Header (recipe id, version, seed, exposure mode)
    - Narrative summary (company, product, market, GTM)
    - Primary task and label definition
    - Suggested use cases
    - Caveats

    Sections populated in later milestones (rendered as stubs here):
    - Table inventory
    - Feature categories
    """
    cfg = world_spec.config
    narrative = world_spec.narrative

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    lines += [
        "# leadforge dataset card",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Recipe | `{cfg.recipe_id}` |",
        f"| Package version | `{cfg.package_version}` |",
        f"| Seed | `{cfg.seed}` |",
        f"| Exposure mode | `{cfg.exposure_mode}` |",
        f"| Difficulty | `{cfg.difficulty}` |",
        f"| Horizon | {cfg.horizon_days} days |",
        "",
    ]

    # ------------------------------------------------------------------
    # Narrative summary
    # ------------------------------------------------------------------
    lines.append("## Narrative summary")
    lines.append("")
    if narrative is not None:
        c = narrative.company
        p = narrative.product
        m = narrative.market
        gtm = narrative.gtm_motion
        lines += [
            f"**Vendor:** {c.name} ({c.stage}, founded {c.founded_year},"
            f" {c.hq_city}, {c.hq_country})",
            "",
            f"**Product:** {p.name} — {p.category}. "
            f"Deployment: {p.deployment}. "
            f"Pricing: {p.pricing_model}. "
            f"ACV range: ${p.acv_range_usd[0]:,}–${p.acv_range_usd[1]:,}.",
            "",
            f"**Target market:** {m.icp_employee_range[0]}–{m.icp_employee_range[1]}-employee"
            f" firms in {', '.join(m.geographies)}. "
            f"Key industries: {', '.join(m.icp_industries)}. "
            f"Average deal size: ${m.avg_deal_size_usd:,}. "
            f"Average sales cycle: {m.avg_sales_cycle_days} days.",
            "",
            f"**GTM motion:** {', '.join(gtm.channels)} "
            f"({gtm.inbound_share:.0%} inbound / "
            f"{gtm.outbound_share:.0%} outbound / "
            f"{gtm.partner_share:.0%} partner).",
            "",
            "**Buyer personas:**",
            "",
        ]
        for persona in narrative.personas:
            ellipsis = "…" if len(persona.title_variants) > 2 else ""
            lines.append(
                f"- **{persona.role}** ({persona.decision_authority}) — "
                f"{', '.join(persona.title_variants[:2])}{ellipsis}"
            )
        lines.append("")
    else:
        lines += ["*Narrative not available for this exposure mode.*", ""]

    # ------------------------------------------------------------------
    # Primary task
    # ------------------------------------------------------------------
    lines += [
        "## Primary task",
        "",
        "**Task:** `converted_within_90_days`",
        "",
        "**Label definition:** A lead is considered converted if a `closed_won` event "
        "is recorded within 90 days of the lead's snapshot anchor date. "
        "The label is derived from simulated events — it is never sampled directly.",
        "",
    ]

    # ------------------------------------------------------------------
    # Table inventory (stub — populated in later milestones)
    # ------------------------------------------------------------------
    lines += [
        "## Table inventory",
        "",
        "*Table counts will appear here once the simulation layer is implemented (v0.3.0+).*",
        "",
    ]

    # ------------------------------------------------------------------
    # Feature categories (stub)
    # ------------------------------------------------------------------
    lines += [
        "## Feature categories",
        "",
        "*Feature dictionary will appear here once the schema layer is implemented (v0.3.0+).*",
        "",
    ]

    # ------------------------------------------------------------------
    # Suggested use cases
    # ------------------------------------------------------------------
    lines += [
        "## Suggested use cases",
        "",
        "- Teaching binary classification on realistic CRM data",
        "- Portfolio projects demonstrating end-to-end ML pipelines",
        "- Benchmarking lead-scoring models under controlled signal/noise conditions",
        "- Research on causal structure in funnel conversion data",
        "",
    ]

    # ------------------------------------------------------------------
    # Caveats
    # ------------------------------------------------------------------
    lines += [
        "## Caveats",
        "",
        "- This is **synthetic** data. It does not represent any real company, product, or market.",
        "- The hidden world structure varies by motif family and stochastic rewiring; "
        "no two seeds produce the same DGP.",
        "- Features are anchored at the snapshot date. No post-anchor data is "
        "included (leakage-free by construction).",
        "- In `student_public` mode, the latent world graph, mechanism summary, "
        "and full world spec are withheld.",
        "",
    ]

    return "\n".join(lines)

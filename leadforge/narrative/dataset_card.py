"""Dataset card renderer.

Produces the ``dataset_card.md`` artifact from a :class:`WorldSpec`.
The card follows the structure required by the architecture spec (§14.3).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, FeatureSpec

if TYPE_CHECKING:
    from leadforge.core.models import WorldSpec
    from leadforge.schema.tasks import TaskManifest


def render_dataset_card(
    world_spec: WorldSpec,
    task_manifest: TaskManifest | None = None,
    table_counts: dict[str, int] | None = None,
    features: tuple[FeatureSpec, ...] = LEAD_SNAPSHOT_FEATURES,
) -> str:
    """Return a Markdown dataset card string for *world_spec*.

    Args:
        world_spec: The world specification containing config and narrative.
        task_manifest: Optional task manifest whose ``description`` is used
            as the label definition prose.  When ``None`` or when
            ``description`` is empty, a generic fallback is rendered.
        table_counts: Optional mapping of table name → row count.  When
            provided, the table inventory section renders actual counts
            instead of a placeholder.
        features: Feature spec tuple to render in the categories / leakage
            sections.  Defaults to the canonical list; pass the redacted
            tuple when rendering an exposure-filtered bundle so the card
            describes only what is actually present.

    Sections:
    - Header (recipe id, version, seed, exposure mode)
    - Narrative summary (company, product, market, GTM)
    - Primary task and label definition
    - Table inventory
    - Feature categories
    - Suggested use cases
    - Caveats
    """
    cfg = world_spec.config
    narrative = world_spec.narrative

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    snapshot_label = (
        f"{cfg.snapshot_day} days (windowed)"
        if cfg.snapshot_day is not None and cfg.snapshot_day < cfg.horizon_days
        else f"{cfg.horizon_days} days (full horizon)"
    )
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
        f"| Label window | {cfg.label_window_days} days |",
        f"| Feature snapshot window | {snapshot_label} |",
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
        lines += ["*Narrative unavailable for this dataset.*", ""]

    # ------------------------------------------------------------------
    # Primary task
    # ------------------------------------------------------------------
    if task_manifest is not None and task_manifest.description:
        label_def = task_manifest.description
    else:
        label_def = (
            f"Binary label evaluated over a {cfg.label_window_days}-day window "
            f"from the snapshot anchor date. The label is event-derived — never "
            f"sampled directly."
        )
    lines += [
        "## Primary task",
        "",
        f"**Task:** `{cfg.primary_task}`",
        "",
        f"**Label definition:** {label_def}",
        "",
    ]

    # ------------------------------------------------------------------
    # Table inventory
    # ------------------------------------------------------------------
    lines += ["## Table inventory", ""]
    if table_counts is not None:
        lines += [
            "| Table | Rows |",
            "|---|---:|",
        ]
        for tbl, count in table_counts.items():
            lines.append(f"| {tbl} | {count:,} |")
        lines.append("")
    else:
        lines += [
            "*Table counts not available (pass ``table_counts`` to populate).*",
            "",
        ]

    # ------------------------------------------------------------------
    # Feature categories
    # ------------------------------------------------------------------
    lines += ["## Feature categories", ""]
    category_counts: Counter[str] = Counter()
    for feat in features:
        category_counts[feat.category] += 1
    lines += [
        "| Category | Count | Examples |",
        "|---|---:|---|",
    ]
    for cat, count in category_counts.items():
        examples = [f.name for f in features if f.category == cat and not f.is_target][:3]
        lines.append(f"| {cat} | {count} | {', '.join(examples)} |")
    leakage_cols = [f.name for f in features if f.leakage_risk]
    if leakage_cols:
        lines += [
            "",
            f"**Leakage-flagged columns:** {', '.join(f'`{c}`' for c in leakage_cols)}. "
            "See `feature_dictionary.csv` for details.",
        ]
    lines.append("")

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
    if cfg.snapshot_day is not None and cfg.snapshot_day < cfg.horizon_days:
        feature_window_caveat = (
            f"- Event-aggregate features (e.g. `touch_count`, `session_count`, "
            f"`expected_acv`) are computed over the first {cfg.snapshot_day} days "
            f"after lead creation; the label resolves over the next "
            f"{cfg.label_window_days - cfg.snapshot_day} days. The deliberate "
            f"exception is `total_touches_all`, which counts touches over the "
            f"full {cfg.horizon_days}-day horizon as a pedagogical leakage trap."
        )
    else:
        feature_window_caveat = (
            "- Features are anchored at the snapshot date. No post-anchor data is "
            "included (leakage-free by construction)."
        )
    lines += [
        "## Caveats",
        "",
        "- This is **synthetic** data. It does not represent any real company, product, or market.",
        "- The hidden world structure varies by motif family and stochastic rewiring; "
        "no two seeds produce the same DGP.",
        feature_window_caveat,
        "- In `student_public` mode, the latent world graph, mechanism summary, "
        "and full world spec are withheld.",
        "",
    ]

    return "\n".join(lines)

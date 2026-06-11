"""Dataset card renderer.

Produces the ``dataset_card.md`` artifact from a :class:`WorldSpec`.
The card is written for a data scientist with no prior leadforge knowledge —
it opens with a plain-English explanation of what the dataset is, what the
prediction task is, and what the difficulty tier means before getting into
technical metadata.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from leadforge.schemes.lead_scoring.features import LEAD_SNAPSHOT_FEATURES, FeatureSpec

if TYPE_CHECKING:
    from leadforge.core.models import WorldSpec
    from leadforge.schema.tasks import TaskManifest


# Tier descriptions for the plain-English "this tier" callout.
# Keys match the ``difficulty`` values used in difficulty_profiles.yaml.
_TIER_DESCRIPTIONS: dict[str, str] = {
    "intro": (
        "The **intro tier is the easiest version of this task.** Signal is strong, "
        "conversion rate is high, and missing values are minimal. A simple logistic "
        "regression is competitive. Use this tier to prototype your pipeline and "
        "sanity-check your approach before scaling up difficulty."
    ),
    "intermediate": (
        "The **intermediate tier is the default benchmark.** Conversion rate is more "
        "realistic for B2B SaaS than the intro tier, and noise is moderate enough "
        "that feature engineering starts to matter. GBM does not consistently beat "
        "logistic regression here (the snapshot is dominated by near-linear features). "
        "Calibration becomes important at this prevalence level."
    ),
    "advanced": (
        "The **advanced tier is a calibration and rare-event exercise.** Conversion "
        "rate is low (~8%) and noise is heavy. AUC barely moves across tiers by "
        "design; here you will want average precision, P@K, and value-weighted "
        "ranking (``expected_acv × P(convert)``) to measure what matters. "
        "Calibration is harder in this tier: a miscalibrated model can rank "
        "correctly but still predict systematically wrong probabilities."
    ),
}

_TIER_DESCRIPTION_FALLBACK = (
    "See the difficulty profile YAML for signal_strength, noise_scale, "
    "and missing_rate knobs for this tier."
)


def render_dataset_card(
    world_spec: WorldSpec,
    task_manifest: TaskManifest | None = None,
    table_counts: dict[str, int] | None = None,
    features: tuple[FeatureSpec, ...] = LEAD_SNAPSHOT_FEATURES,
) -> str:
    """Return a Markdown dataset card string for *world_spec*.

    The card is structured for a zero-prior-knowledge data scientist:
    it opens with what the dataset is and what you are predicting, then
    has a per-tier callout, then the technical inventory.

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
    - Title (tier-aware)
    - What is this / what you are predicting
    - This tier at a glance (difficulty callout)
    - Table inventory
    - Feature categories
    - Simulated world (company / product / market)
    - How to load (Python snippet)
    - Reproducibility (recipe, seed, version)
    - Caveats
    """
    cfg = world_spec.config
    narrative = world_spec.narrative
    difficulty = str(cfg.difficulty) if cfg.difficulty else "unknown"

    lines: list[str] = []

    # ------------------------------------------------------------------
    # Title — tier-aware
    # ------------------------------------------------------------------
    tier_title = difficulty.capitalize()
    lines += [
        f"# B2B Lead Scoring Dataset — {tier_title} Tier",
        "",
    ]

    # ------------------------------------------------------------------
    # What is this / prediction task intro
    # ------------------------------------------------------------------
    snapshot_label = (
        f"{cfg.snapshot_day} days"
        if cfg.snapshot_day is not None and cfg.snapshot_day < cfg.horizon_days
        else f"{cfg.horizon_days} days (full horizon)"
    )
    if task_manifest is not None and task_manifest.description:
        label_def = task_manifest.description
    else:
        label_def = (
            f"Binary label: did this lead close as a paid deal within "
            f"{cfg.label_window_days} days? The label is event-derived — never "
            f"sampled directly."
        )
    lines += [
        "**This is a synthetic dataset** for practicing B2B lead scoring. It was "
        "generated by [leadforge](https://github.com/leadforge-dev/leadforge), an "
        "open-source Python framework for producing realistic CRM/funnel training "
        "data. No real company, customer, or transaction is represented.",
        "",
        "**What you are predicting:** Each row is a sales lead at a fictional B2B "
        "SaaS company. The task is binary classification:",
        "",
        f"> `{cfg.primary_task}` — {label_def}",
        "",
        f"Features capture the first {snapshot_label} of CRM activity per lead "
        "(email/call touches, product sessions, deal stage, account firmographics). "
        "The label is derived from simulated events — never directly sampled — so "
        "there is genuine causal structure behind the signal.",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # This tier at a glance
    # ------------------------------------------------------------------
    tier_desc = _TIER_DESCRIPTIONS.get(difficulty, _TIER_DESCRIPTION_FALLBACK)
    lines += [
        f"## This tier: {difficulty}",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Signal strength | {cfg.signal_strength} / 1.0 |"
        if hasattr(cfg, "signal_strength")
        else "| Signal strength | see difficulty_profiles.yaml |",
        "",
        tier_desc,
        "",
        "This dataset ships in three tiers — **intro → intermediate → advanced** — "
        "with decreasing signal, lower conversion rates, and heavier noise and "
        "missingness. All three tiers share the same schema and simulate the same "
        "fictional B2B world.",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Table inventory
    # ------------------------------------------------------------------
    lines += ["## Table inventory", ""]
    if table_counts is not None:
        lines += [
            "| Table | Rows | Description |",
            "|---|---:|---|",
        ]
        _table_descriptions = {
            "accounts": "One row per company",
            "contacts": "One row per buyer-side individual (multiple per account)",
            "leads": "One row per lead — the prediction unit",
            "touches": f"Marketing / SDR outreach events (first {snapshot_label} per lead)",
            "sessions": f"Product demo or trial sessions (first {snapshot_label} per lead)",
            "sales_activities": (
                f"CRM activities: calls, emails, meetings (first {snapshot_label} per lead)"
            ),
            "opportunities": f"Deal records opened before the {snapshot_label} snapshot",
        }
        for tbl, count in table_counts.items():
            desc = _table_descriptions.get(tbl, "")
            lines.append(f"| {tbl} | {count:,} | {desc} |")
        lines += [
            "",
            "**Snapshot-safe:** event tables contain only rows with timestamps "
            f"≤ {snapshot_label} from lead creation. Outcome columns "
            "(`converted_within_90_days`, `conversion_timestamp`, `close_outcome`) "
            "are excluded from the public relational tables — they appear only in the "
            "task splits.",
        ]
    else:
        lines += [
            "*Table counts not available (pass `table_counts` to populate).*",
        ]
    lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # Feature categories
    # ------------------------------------------------------------------
    _category_labels: dict[str, str] = {
        "account": "Account",
        "contact": "Contact",
        "lead_meta": "Lead metadata",
        "engagement": "Engagement",
        "sales": "Sales",
        "target": "Target",
    }
    lines += ["## Features", ""]
    category_counts: Counter[str] = Counter()
    for feat in features:
        category_counts[feat.category] += 1
    lines += [
        "| Category | Count | Examples |",
        "|---|---:|---|",
    ]
    for cat, count in category_counts.items():
        label = _category_labels.get(cat, cat)
        examples = [f"`{f.name}`" for f in features if f.category == cat and not f.is_target][:3]
        lines.append(f"| {label} | {count} | {', '.join(examples)} |")
    leakage_cols = [f.name for f in features if f.leakage_risk]
    if leakage_cols:
        lines += [
            "",
            f"**Leakage-flagged columns:** "
            f"{', '.join(f'`{c}`' for c in leakage_cols)}. "
            f"{'This column aggregates' if len(leakage_cols) == 1 else 'These columns aggregate'} "
            f"events over the full {cfg.horizon_days}-day window (not just the "
            f"{snapshot_label} feature window) and "
            f"{'is' if len(leakage_cols) == 1 else 'are'} deliberately retained as "
            f"a leakage-detection teaching exercise. Drop "
            f"{'it' if len(leakage_cols) == 1 else 'them'} from your feature set "
            f"unless you are studying leakage. "
            "See `feature_dictionary.csv` for details.",
        ]
    lines += [
        "",
        "See `feature_dictionary.csv` for the full column-by-column specification.",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Simulated world (company / product / market)
    # ------------------------------------------------------------------
    lines.append("## The simulated world")
    lines.append("")
    if narrative is not None:
        c = narrative.company
        p = narrative.product
        m = narrative.market
        gtm = narrative.gtm_motion
        lines += [
            "The dataset simulates a fictional company — "
            f"**{c.name}** — "
            f"a {c.stage} startup ({c.hq_city}, {c.hq_country}, "
            f"founded {c.founded_year}) selling **{p.name}**, a "
            f"{p.deployment.replace('_', ' ')} {p.category}. "
            "Everything below is invented:",
            "",
            f"- **Target customers:** {m.icp_employee_range[0]}–{m.icp_employee_range[1]}"
            f"-employee firms in {', '.join(m.geographies)} "
            f"({', '.join(m.icp_industries)})",
            f"- **Deal range:** ${p.acv_range_usd[0]:,}–${p.acv_range_usd[1]:,} ACV; "
            f"average deal ${m.avg_deal_size_usd:,}; "
            f"average sales cycle {m.avg_sales_cycle_days} days",
            f"- **Go-to-market:** {gtm.inbound_share:.0%} inbound marketing, "
            f"{gtm.outbound_share:.0%} SDR outbound, "
            f"{gtm.partner_share:.0%} partner referrals",
        ]
        if narrative.personas:
            persona_strs = []
            for persona in narrative.personas:
                title = persona.title_variants[0] if persona.title_variants else persona.role
                # Include the internal role key (e.g. vp_finance) as a machine-readable
                # anchor alongside the human-readable title.
                persona_strs.append(
                    f"{title} / {persona.role} ({persona.decision_authority.replace('_', ' ')})"
                )
            lines.append(f"- **Buyer personas:** {', '.join(persona_strs)}")
        lines += [
            "",
            "In this public version, the hidden causal graph, latent trait scores, "
            "and mechanism parameters are withheld. The instructor companion bundle "
            "includes them.",
        ]
    else:
        lines += ["*Narrative unavailable for this dataset.*"]
    lines += ["", "---", ""]

    # ------------------------------------------------------------------
    # How to load
    # ------------------------------------------------------------------
    lines += [
        "## How to load",
        "",
        "```python",
        "import pandas as pd",
        "",
        "# Flat CSV — all leads, all splits combined (convenient for exploration)",
        'df = pd.read_csv("lead_scoring.csv")',
        f'X = df.drop(columns=["{cfg.primary_task}"])',
        f'y = df["{cfg.primary_task}"]',
        "",
        "# Parquet task splits — recommended for model training",
        f'train = pd.read_parquet("tasks/{cfg.primary_task}/train.parquet")',
        f'valid = pd.read_parquet("tasks/{cfg.primary_task}/valid.parquet")',
        f'test  = pd.read_parquet("tasks/{cfg.primary_task}/test.parquet")',
        "",
        "# Relational tables — for feature engineering",
        'leads   = pd.read_parquet("tables/leads.parquet")',
        'touches = pd.read_parquet("tables/touches.parquet")',
        "```",
        "",
        "Splits are 70 / 15 / 15 (train / valid / test), stratified on the target, "
        f"deterministic given seed {cfg.seed}.",
        "",
        "**Note on account overlap:** ~93% of test-set accounts also appear in the "
        "training set (splits are keyed on `lead_id`). Headline AUC overstates "
        "generalisation to *unseen* accounts. For a faithful out-of-sample estimate, "
        'use `GroupKFold(groups=df["account_id"])`.',
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    lines += [
        "## Reproducibility",
        "",
        f"Generated with **leadforge v{cfg.package_version}**, "
        f"recipe `{cfg.recipe_id}`, seed {cfg.seed}, "
        f"difficulty `{difficulty}`. To reproduce:",
        "",
        "```bash",
        "pip install leadforge",
        f"leadforge generate --recipe {cfg.recipe_id} --seed {cfg.seed} \\",
        f"                   --mode student_public --difficulty {difficulty} --out my_bundle",
        "```",
        "",
        "Every file in this bundle is SHA-256 hashed in `manifest.json`. Run "
        "`leadforge validate my_bundle` to verify integrity.",
        "",
        "**Author:** [Shay Palachy Affek](https://huggingface.co/shaypal5) "
        "· [Kaggle](https://www.kaggle.com/derelictpanda) "
        "· [GitHub](https://github.com/shaypalachy)",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Intended uses
    # ------------------------------------------------------------------
    lines += [
        "## Intended uses",
        "",
        "- Teaching binary classification on realistic B2B CRM data",
        "- Portfolio projects demonstrating end-to-end lead-scoring pipelines",
        "- Benchmarking model families under controlled signal / noise / prevalence conditions",
        "- Teaching leakage detection, calibration, lift, P@K, and value-weighted ranking",
        "- Research on causal structure in funnel conversion data",
        "",
        "**Out of scope:** production lead scoring (the company is fictional), vendor "
        "benchmarking / paper baselines, or causal-inference research that requires "
        "recovery of the true DGP (use the instructor bundle for that).",
        "",
        "---",
        "",
    ]

    # ------------------------------------------------------------------
    # Caveats
    # ------------------------------------------------------------------
    if cfg.snapshot_day is not None and cfg.snapshot_day < cfg.horizon_days:
        window_caveat = (
            f"- **Snapshot window.** Engagement features cover days 0–{cfg.snapshot_day} "
            f"per lead; the label resolves at day {cfg.horizon_days}. "
            f"`total_touches_all` is the intentional exception — it aggregates over the "
            f"full {cfg.horizon_days}-day window and is a leakage trap."
        )
    else:
        window_caveat = (
            "- **Snapshot window.** Features are anchored at the snapshot date. "
            "No post-anchor data is included (leakage-free by construction), "
            "except `total_touches_all` which is the intentional leakage trap."
        )

    lines += [
        "## Caveats",
        "",
        "- **Synthetic data only.** No real company, customer, or market is represented.",
        "- **AUC does not distinguish tiers.** LR AUC is similar across all three tiers "
        "by design. The tiers differ in conversion rate, noise, and missing values — not "
        "in rank discrimination. Use average precision, P@K, and calibration metrics to "
        "see the difficulty gradient.",
        "- **~93% train/test account overlap.** Splits are keyed on `lead_id`; most test "
        "accounts also appear in train. Headline metrics overstate generalisation to "
        "unseen accounts.",
        window_caveat,
        "- **Public version.** The hidden causal graph, latent trait scores, and "
        "mechanism parameters are withheld. The instructor companion bundle includes them.",
        "",
    ]

    return "\n".join(lines)

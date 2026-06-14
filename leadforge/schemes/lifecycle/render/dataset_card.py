"""Dataset-card renderer for the lifecycle (pLTV) scheme.

The lead-scoring card (:func:`leadforge.narrative.dataset_card.render_dataset_card`)
is hard-coupled to the lead-scoring framing (binary conversion label, single
task, narrative-driven firmographics), so the lifecycle scheme renders its own.
Kept deliberately concise for LTV-Pn.4b; richer prose can follow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leadforge.core.models import WorldSpec
    from leadforge.schema.tasks import TaskManifest

__all__ = ["render_lifecycle_dataset_card"]


def render_lifecycle_dataset_card(
    world_spec: WorldSpec,
    *,
    table_counts: dict[str, int],
    tasks: tuple[TaskManifest, ...],
    observation_date: str,
) -> str:
    """Return a Markdown dataset card for a lifecycle (pLTV) bundle."""
    cfg = world_spec.config
    tier = (str(cfg.difficulty) if cfg.difficulty else "unknown").capitalize()

    lines: list[str] = [
        f"# B2B SaaS pLTV Dataset — {tier} Tier",
        "",
        "## What this is",
        "",
        "A synthetic B2B SaaS customer base simulated week by week from "
        "acquisition through retention, expansion, and churn.  The prediction "
        "task is **predicted lifetime value (pLTV)**: a continuous, "
        "zero-inflated, right-skewed regression target — forecast each "
        "customer's future gross revenue over a fixed forward window.  Customer "
        "churn is provided as a secondary classification label.",
        "",
        "## Two observation regimes",
        "",
        "- **Calendar-anchored (standard)** — every customer observed at the "
        f"fixed observation date (`{observation_date}`); tenure varies from "
        "cold to mature.  Task ids: `pltv_revenue_*`, `churned_within_180d`.",
        "- **Tenure-anchored (early-pLTV)** — every customer observed at a "
        f"fixed short tenure (`customer_start + {cfg.early_tenure_weeks}w`); the "
        "genuine cold-start case.  Task ids prefixed `early_`.",
        "",
        "## Tasks",
        "",
        "| task_id | type | target | window (days) |",
        "|---|---|---|---|",
    ]
    for t in tasks:
        lines.append(
            f"| `{t.task_id}` | {t.task_type} | `{t.label_column}` | {t.label_window_days} |"
        )

    lines += [
        "",
        "## Relational tables",
        "",
        "| table | rows |",
        "|---|---|",
    ]
    for name, count in table_counts.items():
        lines.append(f"| `{name}` | {count} |")

    lines += [
        "",
        "## Leakage trap",
        "",
        "`mrr_change_full_period` is a deliberate trap: it is computed through "
        "the end of simulation, so post-cutoff expansions inflate it.  Use "
        "`mrr_change_at_snapshot` (computed strictly at the cutoff) instead.",
        "",
        "## Reproducibility",
        "",
        f"- Recipe: `{cfg.recipe_id}`",
        f"- Seed: `{cfg.seed}`",
        f"- Scheme: `{world_spec.scheme}`",
        "",
        "Deterministic given (recipe, config, seed, package version).",
        "",
    ]
    return "\n".join(lines)

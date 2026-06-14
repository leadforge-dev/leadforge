"""Lifecycle (``b2b_saas_ltv_v1``) task definitions — pLTV regression + churn.

Each observation regime (design.md §3.1) exports one task family:

- **calendar-anchored** (standard): task ids ``pltv_revenue_{90,365,730}d`` +
  ``churned_within_180d``.
- **tenure-anchored** (early-pLTV, D8): the same set prefixed ``early_`` so the
  two families occupy separate task directories within one bundle.

The three ``pltv_revenue_*`` tasks are **regression** (continuous, ZILN-shaped
gross-revenue targets, D1); ``churned_within_180d`` is the secondary
**binary_classification** task (D9).  Target columns and windows mirror the
snapshot catalog (:data:`~leadforge.schemes.lifecycle.snapshots.FORWARD_WINDOWS_DAYS`
/ :data:`~leadforge.schemes.lifecycle.snapshots.CHURN_WINDOW_DAYS`) so the task
specs and the snapshot columns can never drift.

These are data definitions only; wiring them into the bundle writer is LTV-Pn.4.
"""

from __future__ import annotations

from leadforge.schema.tasks import SplitSpec, TaskManifest
from leadforge.schemes.lifecycle.snapshots import CHURN_WINDOW_DAYS, FORWARD_WINDOWS_DAYS

__all__ = [
    "CALENDAR_REGIME",
    "EARLY_REGIME",
    "lifecycle_task_manifests",
]

CALENDAR_REGIME = "calendar"
EARLY_REGIME = "early"

# Shared split ratios across all lifecycle tasks (matches the lead-scoring task).
_SPLIT = SplitSpec(train=0.7, valid=0.15, test=0.15)

# Per-regime task-id prefix.  The calendar (standard) regime is unprefixed; the
# early-pLTV regime is ``early_`` so both families coexist in one bundle.
_REGIME_PREFIX = {CALENDAR_REGIME: "", EARLY_REGIME: "early_"}

_PRIMARY_TABLE = "customers"


def lifecycle_task_manifests(regime: str) -> tuple[TaskManifest, ...]:
    """Return the pLTV regression + churn task manifests for *regime*.

    Args:
        regime: :data:`CALENDAR_REGIME` or :data:`EARLY_REGIME`.

    Returns:
        One :class:`~leadforge.schema.tasks.TaskManifest` per forward window
        (regression) plus the secondary churn classification task.

    Raises:
        ValueError: if *regime* is not a known regime.
    """
    if regime not in _REGIME_PREFIX:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_REGIME_PREFIX)}")
    prefix = _REGIME_PREFIX[regime]
    anchor = (
        "the fixed observation date"
        if regime == CALENDAR_REGIME
        else "each customer's tenure anchor (customer_start + early_tenure_weeks)"
    )

    tasks: list[TaskManifest] = []
    for window in FORWARD_WINDOWS_DAYS:
        tasks.append(
            TaskManifest(
                task_id=f"{prefix}pltv_revenue_{window}d",
                label_column=f"ltv_revenue_{window}d",
                label_window_days=window,
                primary_table=_PRIMARY_TABLE,
                split=_SPLIT,
                task_type="regression",
                description=(
                    f"Predict gross revenue (paid + recovered invoices) in the "
                    f"{window} days after {anchor}.  Continuous, zero-inflated, "
                    f"right-skewed pLTV regression target.  All features are "
                    f"computed at or before the cutoff (leakage-free by "
                    f"construction, except the documented mrr_change_full_period "
                    f"trap)."
                ),
            )
        )

    tasks.append(
        TaskManifest(
            task_id=f"{prefix}churned_within_180d",
            label_column="churned_within_180d",
            label_window_days=CHURN_WINDOW_DAYS,
            primary_table=_PRIMARY_TABLE,
            split=_SPLIT,
            task_type="binary_classification",
            description=(
                f"Secondary task: whether the customer churns within "
                f"{CHURN_WINDOW_DAYS} days after {anchor}.  Doubles as the "
                f"ZILN zero-inflation indicator for the pLTV targets."
            ),
        )
    )
    return tuple(tasks)

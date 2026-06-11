"""Lead-scoring (``b2b_saas_procurement_v1``) task definitions.

:data:`CONVERTED_WITHIN_90_DAYS` and :func:`task_manifest_for_config` live here.
The shared primitives (:class:`~leadforge.schema.tasks.SplitSpec`,
:class:`~leadforge.schema.tasks.TaskManifest`) stay in
:mod:`leadforge.schema.tasks`.
"""

from __future__ import annotations

from dataclasses import replace

from leadforge.schema.tasks import SplitSpec, TaskManifest

CONVERTED_WITHIN_90_DAYS: TaskManifest = TaskManifest(
    task_id="converted_within_90_days",
    label_column="converted_within_90_days",
    label_window_days=90,
    primary_table="leads",
    split=SplitSpec(train=0.7, valid=0.15, test=0.15),
    task_type="binary_classification",
    description=(
        "A lead is considered converted if a `closed_won` event is recorded "
        "within 90 days of the lead's snapshot anchor date. The label is "
        "event-derived — never sampled directly. All features are pre-anchor "
        "(leakage-free by construction)."
    ),
)


def task_manifest_for_config(
    primary_task: str = CONVERTED_WITHIN_90_DAYS.task_id,
    label_window_days: int = CONVERTED_WITHIN_90_DAYS.label_window_days,
) -> TaskManifest:
    """Build a :class:`~leadforge.schema.tasks.TaskManifest` from config fields.

    Derives from :data:`CONVERTED_WITHIN_90_DAYS` via ``dataclasses.replace``,
    overriding only the fields that vary.

    Args:
        primary_task: Task identifier — used as the task directory name.
        label_window_days: Label observation window in days.
    """
    if primary_task == CONVERTED_WITHIN_90_DAYS.task_id:
        description = (
            f"A lead is considered converted if a `closed_won` event is recorded "
            f"within {label_window_days} days of the lead's snapshot anchor date. "
            f"The label is event-derived — never sampled directly. All features "
            f"are pre-anchor (leakage-free by construction)."
        )
    else:
        description = (
            f"Binary label `{primary_task}` evaluated over a "
            f"{label_window_days}-day window from the snapshot anchor date. "
            f"The label is event-derived — never sampled directly. All features "
            f"are pre-anchor (leakage-free by construction)."
        )
    return replace(
        CONVERTED_WITHIN_90_DAYS,
        task_id=primary_task,
        label_window_days=label_window_days,
        description=description,
    )

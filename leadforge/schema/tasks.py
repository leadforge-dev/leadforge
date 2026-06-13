"""Shared task-manifest primitives.

:class:`SplitSpec` and :class:`TaskManifest` are scheme-agnostic types used by
every scheme's task definition.  The lead-scoring task definition
(:data:`~leadforge.schemes.lead_scoring.tasks.CONVERTED_WITHIN_90_DAYS`,
:func:`~leadforge.schemes.lead_scoring.tasks.task_manifest_for_config`) lives
in :mod:`leadforge.schemes.lead_scoring.tasks`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSpec:
    """Train / validation / test proportions for a task.

    Attributes:
        train: Fraction of rows allocated to the training split.
        valid: Fraction allocated to validation.
        test: Fraction allocated to test.

    Raises:
        ValueError: if the three fractions do not sum to 1.0 (within 1e-6).
    """

    train: float
    valid: float
    test: float

    def __post_init__(self) -> None:
        import math

        for name, value in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))  # noqa: UP038
                or math.isnan(value)
                or math.isinf(value)
            ):
                raise ValueError(f"SplitSpec.{name} must be a finite number, got {value!r}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"SplitSpec.{name} must be in [0, 1], got {value}")
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"SplitSpec fractions must sum to 1.0, got {total:.6f}")


#: ML task types a :class:`TaskManifest` may declare.  ``binary_classification``
#: covers the lead-scoring ``converted_within_90_days`` label and the lifecycle
#: secondary churn label; ``regression`` covers the continuous pLTV
#: ``ltv_revenue_*`` targets (D1).
VALID_TASK_TYPES: frozenset[str] = frozenset({"binary_classification", "regression"})


@dataclass(frozen=True)
class TaskManifest:
    """Immutable descriptor for one ML task exported from a bundle.

    Serves both classification and regression tasks; ``task_type`` distinguishes
    them and ``label_column`` names the target either way.

    Attributes:
        task_id: Machine-readable task identifier (also the task directory name,
            so it must be unique within a bundle).
        label_column: Column in the task Parquet files holding the target — a
            binary label for ``binary_classification`` or a continuous value
            for ``regression``.
        label_window_days: Forward window in days that defines the target — the
            positive-event window for a classification label, or the
            revenue-accumulation horizon for a pLTV regression target.
        primary_table: The relational table the snapshot rows are derived
            from (e.g. ``"leads"`` / ``"customers"``).
        split: Train/valid/test proportions.
        task_type: One of :data:`VALID_TASK_TYPES`.
        description: Human-readable description of the task, suitable for
            display in dataset cards and documentation.

    Raises:
        ValueError: if ``task_type`` is not in :data:`VALID_TASK_TYPES`.
    """

    task_id: str
    label_column: str
    label_window_days: int
    primary_table: str
    split: SplitSpec
    task_type: str = "binary_classification"
    description: str = ""

    def __post_init__(self) -> None:
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {sorted(VALID_TASK_TYPES)}, got {self.task_type!r}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "label_column": self.label_column,
            "label_window_days": self.label_window_days,
            "primary_table": self.primary_table,
            "split": {
                "train": self.split.train,
                "valid": self.split.valid,
                "test": self.split.test,
            },
            "description": self.description,
        }

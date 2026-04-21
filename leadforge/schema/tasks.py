"""Task manifest definition for the primary v1 classification task.

A :class:`TaskManifest` describes everything needed to reconstruct the task
from the output bundle: the label column, the time window, the split ratios,
and the table it lives in.
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
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):  # noqa: UP038
                raise ValueError(f"SplitSpec.{name} must be a finite number, got {value!r}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"SplitSpec.{name} must be in [0, 1], got {value}")
        total = self.train + self.valid + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"SplitSpec fractions must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True)
class TaskManifest:
    """Immutable descriptor for one ML task exported from a bundle.

    Attributes:
        task_id: Machine-readable task identifier.
        label_column: Column name in the task Parquet files that holds the
            binary label.
        label_window_days: Number of days after the snapshot anchor date
            within which a conversion event counts as positive.
        primary_table: The relational table the snapshot rows are derived
            from (usually ``"leads"``).
        split: Train/valid/test proportions.
        task_type: ML task type string (``"binary_classification"`` for v1).
        description: Human-readable description of the task.
    """

    task_id: str
    label_column: str
    label_window_days: int
    primary_table: str
    split: SplitSpec
    task_type: str = "binary_classification"
    description: str = ""

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


# ---------------------------------------------------------------------------
# v1 task definition
# ---------------------------------------------------------------------------

CONVERTED_WITHIN_90_DAYS: TaskManifest = TaskManifest(
    task_id="converted_within_90_days",
    label_column="converted_within_90_days",
    label_window_days=90,
    primary_table="leads",
    split=SplitSpec(train=0.7, valid=0.15, test=0.15),
    task_type="binary_classification",
    description=(
        "Predict whether a lead converts (closed_won event) within 90 days "
        "of the snapshot anchor date. Label is event-derived — never sampled "
        "directly. All features are pre-anchor (leakage-free by construction)."
    ),
)

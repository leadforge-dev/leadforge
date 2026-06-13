"""Tests for the lifecycle task manifests + regression task model (LTV-Pn.3)."""

from __future__ import annotations

import pytest

from leadforge.schema.tasks import VALID_TASK_TYPES, SplitSpec, TaskManifest
from leadforge.schemes.lifecycle.snapshots import CHURN_WINDOW_DAYS, FORWARD_WINDOWS_DAYS
from leadforge.schemes.lifecycle.tasks import (
    CALENDAR_REGIME,
    EARLY_REGIME,
    lifecycle_task_manifests,
)

# ---------------------------------------------------------------------------
# TaskManifest regression support
# ---------------------------------------------------------------------------


def test_regression_is_a_valid_task_type() -> None:
    assert "regression" in VALID_TASK_TYPES
    t = TaskManifest(
        task_id="x",
        label_column="y",
        label_window_days=365,
        primary_table="customers",
        split=SplitSpec(0.7, 0.15, 0.15),
        task_type="regression",
    )
    assert t.task_type == "regression"
    assert t.to_dict()["task_type"] == "regression"


def test_invalid_task_type_rejected() -> None:
    with pytest.raises(ValueError, match="task_type must be one of"):
        TaskManifest(
            task_id="x",
            label_column="y",
            label_window_days=1,
            primary_table="t",
            split=SplitSpec(0.7, 0.15, 0.15),
            task_type="ranking",
        )


# ---------------------------------------------------------------------------
# Lifecycle task families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regime", [CALENDAR_REGIME, EARLY_REGIME])
def test_family_shape(regime: str) -> None:
    tasks = lifecycle_task_manifests(regime)
    # One regression task per forward window + one churn classification.
    assert len(tasks) == len(FORWARD_WINDOWS_DAYS) + 1
    regression = [t for t in tasks if t.task_type == "regression"]
    classification = [t for t in tasks if t.task_type == "binary_classification"]
    assert len(regression) == len(FORWARD_WINDOWS_DAYS)
    assert len(classification) == 1


@pytest.mark.parametrize("regime", [CALENDAR_REGIME, EARLY_REGIME])
def test_targets_match_snapshot_columns(regime: str) -> None:
    tasks = {t.task_id: t for t in lifecycle_task_manifests(regime)}
    for window in FORWARD_WINDOWS_DAYS:
        reg = next(
            t
            for t in tasks.values()
            if t.label_window_days == window and t.task_type == "regression"
        )
        assert reg.label_column == f"ltv_revenue_{window}d"
    churn = next(t for t in tasks.values() if t.task_type == "binary_classification")
    assert churn.label_column == "churned_within_180d"
    assert churn.label_window_days == CHURN_WINDOW_DAYS


def test_all_target_customers_table() -> None:
    for regime in (CALENDAR_REGIME, EARLY_REGIME):
        for t in lifecycle_task_manifests(regime):
            assert t.primary_table == "customers"


def test_task_ids_unique_across_regimes() -> None:
    ids = [
        t.task_id
        for regime in (CALENDAR_REGIME, EARLY_REGIME)
        for t in lifecycle_task_manifests(regime)
    ]
    assert len(ids) == len(set(ids)), "task ids collide across regimes (would share a task dir)"


def test_early_regime_is_prefixed() -> None:
    for t in lifecycle_task_manifests(EARLY_REGIME):
        assert t.task_id.startswith("early_")
    for t in lifecycle_task_manifests(CALENDAR_REGIME):
        assert not t.task_id.startswith("early_")


def test_unknown_regime_raises() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        lifecycle_task_manifests("monthly")

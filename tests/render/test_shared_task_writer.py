"""Tests for the scheme-agnostic task-split writer (LTV-Pn.3)."""

from __future__ import annotations

import json

import pandas as pd

from leadforge.render.tasks import write_task_splits
from leadforge.schema.tasks import SplitSpec, TaskManifest


def _regression_task() -> TaskManifest:
    return TaskManifest(
        task_id="pltv_revenue_365d",
        label_column="ltv_revenue_365d",
        label_window_days=365,
        primary_table="customers",
        split=SplitSpec(0.7, 0.15, 0.15),
        task_type="regression",
        description="continuous target",
    )


def test_writes_splits_and_manifest_for_continuous_target(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "customer_id": [f"cust_{i:03d}" for i in range(100)],
            "ltv_revenue_365d": [float(i) * 1.5 for i in range(100)],
        }
    )
    task = _regression_task()
    counts = write_task_splits(df, tmp_path, seed=42, task=task)

    task_dir = tmp_path / "pltv_revenue_365d"
    for split in ("train", "valid", "test"):
        assert (task_dir / f"{split}.parquet").exists()
    assert sum(counts.values()) == 100
    assert counts["train"] == 70

    manifest = json.loads((task_dir / "task_manifest.json").read_text())
    assert manifest["task_type"] == "regression"
    assert manifest["label_column"] == "ltv_revenue_365d"


def test_continuous_target_values_preserved(tmp_path) -> None:
    # The writer is target-agnostic: it must not coerce/round the continuous
    # target — the union of split values equals the input set.
    df = pd.DataFrame({"id": range(50), "ltv_revenue_365d": [i + 0.25 for i in range(50)]})
    write_task_splits(df, tmp_path, seed=7, task=_regression_task())
    task_dir = tmp_path / "pltv_revenue_365d"
    recombined = pd.concat(
        [pd.read_parquet(task_dir / f"{s}.parquet") for s in ("train", "valid", "test")]
    )
    assert set(recombined["ltv_revenue_365d"]) == set(df["ltv_revenue_365d"])


def test_deterministic_given_seed(tmp_path) -> None:
    df = pd.DataFrame({"id": range(40), "ltv_revenue_365d": [float(i) for i in range(40)]})
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_task_splits(df, a, seed=11, task=_regression_task())
    write_task_splits(df, b, seed=11, task=_regression_task())
    for split in ("train", "valid", "test"):
        left = pd.read_parquet(a / "pltv_revenue_365d" / f"{split}.parquet")
        right = pd.read_parquet(b / "pltv_revenue_365d" / f"{split}.parquet")
        pd.testing.assert_frame_equal(left, right)

"""Scheme-agnostic task export — deterministic train/valid/test split + Parquet.

:func:`write_task_splits` shuffles a snapshot DataFrame deterministically,
splits it by the task manifest's ratios, and writes ``train``/``valid``/``test``
Parquet files plus ``task_manifest.json`` into the task directory.

The split logic is target-agnostic: it never inspects the label/target column,
so it serves both classification labels (lead-scoring ``converted_within_90_days``,
lifecycle secondary churn) and continuous regression targets (lifecycle pLTV
``ltv_revenue_*``).  Each scheme passes its own :class:`~leadforge.schema.tasks.TaskManifest`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from leadforge.core.rng import RNGRoot

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from leadforge.schema.tasks import TaskManifest

__all__ = ["write_task_splits"]


def write_task_splits(
    snapshot: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    task: TaskManifest,
) -> dict[str, int]:
    """Shuffle, split, and write snapshot Parquet files for *task*.

    Files written under ``out_dir / task.task_id /``::

        train.parquet
        valid.parquet
        test.parquet
        task_manifest.json

    Args:
        snapshot: The task's source snapshot DataFrame.
        out_dir: Parent directory for task outputs (typically
            ``bundle_root / "tasks"``).
        seed: Seed used for the deterministic row shuffle.
        task: Task manifest describing the split ratios, target column, and
            task type.

    Returns:
        Dict mapping split name (``"train"``, ``"valid"``, ``"test"``) to the
        number of rows written.
    """
    task_dir = out_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic shuffle via the project's RNG substream system.
    rng = RNGRoot(seed).child("task_split_shuffle")
    indices = list(range(len(snapshot)))
    rng.shuffle(indices)
    shuffled = snapshot.iloc[indices].reset_index(drop=True)

    n = len(shuffled)
    n_train = int(n * task.split.train)
    n_valid = int(n * task.split.valid)

    splits: dict[str, pd.DataFrame] = {
        "train": shuffled.iloc[:n_train],
        "valid": shuffled.iloc[n_train : n_train + n_valid],
        "test": shuffled.iloc[n_train + n_valid :],  # remainder avoids rounding off-by-one
    }

    row_counts: dict[str, int] = {}
    for split_name, df in splits.items():
        path = task_dir / f"{split_name}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        row_counts[split_name] = len(df)

    # Write task_manifest.json alongside the Parquet files.
    manifest_path = task_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(task.to_dict(), indent=2))

    return row_counts

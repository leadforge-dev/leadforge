"""Task export — deterministic train/valid/test split and Parquet output.

:func:`write_task_splits` takes the lead snapshot DataFrame, shuffles it
deterministically, splits it according to the task manifest ratios, and
writes the three Parquet files plus a ``task_manifest.json`` into the
tasks directory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from leadforge.schema.tasks import CONVERTED_WITHIN_90_DAYS, TaskManifest


def write_task_splits(
    snapshot: pd.DataFrame,
    out_dir: Path,
    *,
    seed: int,
    task: TaskManifest = CONVERTED_WITHIN_90_DAYS,
) -> dict[str, int]:
    """Shuffle, split, and write snapshot Parquet files for *task*.

    Files written under ``out_dir / task.task_id /``::

        train.parquet
        valid.parquet
        test.parquet
        task_manifest.json

    Args:
        snapshot: Lead snapshot DataFrame from
            :func:`~leadforge.render.snapshots.build_snapshot`.
        out_dir: Parent directory for task outputs (typically
            ``bundle_root / "tasks"``).
        seed: Seed used for deterministic row shuffle.
        task: Task manifest describing the split ratios and label column.

    Returns:
        Dict mapping split name (``"train"``, ``"valid"``, ``"test"``) to
        the number of rows written.
    """
    task_dir = out_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic shuffle via seeded RNG (index permutation).
    rng = random.Random(seed)  # noqa: S311
    indices = list(range(len(snapshot)))
    rng.shuffle(indices)
    shuffled = snapshot.iloc[indices].reset_index(drop=True)

    n = len(shuffled)
    n_train = int(n * task.split.train)
    n_valid = int(n * task.split.valid)
    # Test gets the remainder to avoid off-by-one from integer rounding.

    splits: dict[str, pd.DataFrame] = {
        "train": shuffled.iloc[:n_train],
        "valid": shuffled.iloc[n_train : n_train + n_valid],
        "test": shuffled.iloc[n_train + n_valid :],
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

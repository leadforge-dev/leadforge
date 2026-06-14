"""Lead-scoring task export — thin wrapper over the shared split writer.

The deterministic shuffle/split/write logic is scheme-agnostic and lives in
:func:`leadforge.render.tasks.write_task_splits` (lifted there in LTV-Pn.3,
byte-identical for this scheme).  This wrapper preserves the lead-scoring
default task so existing call sites are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from leadforge.render.tasks import write_task_splits as _write_task_splits
from leadforge.schemes.lead_scoring.tasks import CONVERTED_WITHIN_90_DAYS

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
    task: TaskManifest = CONVERTED_WITHIN_90_DAYS,
) -> dict[str, int]:
    """Write lead-scoring task splits (see :func:`leadforge.render.tasks.write_task_splits`).

    Defaults ``task`` to :data:`CONVERTED_WITHIN_90_DAYS` for this scheme.
    """
    return _write_task_splits(snapshot, out_dir, seed=seed, task=task)

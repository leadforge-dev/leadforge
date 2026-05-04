"""Pipeline functions for building the v7 lead scoring intro CSVs.

v7 produces TWO exports:
- **Student-safe**: no leakage columns.
- **Instructor**: identical rows + one ``__leakage__touches_post_snapshot_21_90``
  column computed purely from the simulator's actual event timeline (days 21..90).

Key v7 changes over v6:
- Purely causal leakage trap: NO label-conditioned Poisson boost.
  Post-snapshot touches are correlated with conversion only through shared
  latent drivers (fit, intent, engagement), not through injected label noise.
- Documentation alignment: all docs match actual generated data exactly.
- Canonical validation pipeline used consistently throughout.
"""

from __future__ import annotations

import pandas as pd

from leadforge.pipelines.common import (
    ACV_CAP,
    ACV_FLOOR,
    FINAL_COLUMNS_INSTRUCTOR,
    FINAL_COLUMNS_STUDENT,
    INSTRUCTOR_TRAP_COL,
    RENAME_MAP,
    SUBSAMPLE_N,
    TARGET_RATE,
    assign_acquisition_wave,
    compute_post_snapshot_touches,
    derive_features,
    softcap_expected_acv,
    subsample,
)
from leadforge.pipelines.common import (
    inject_missingness_v6 as inject_missingness,
)
from leadforge.pipelines.common import (
    rename_and_select as _rename_and_select_generic,
)

__all__ = [
    "ACV_CAP",
    "ACV_FLOOR",
    "FINAL_COLUMNS_INSTRUCTOR",
    "FINAL_COLUMNS_STUDENT",
    "INSTRUCTOR_TRAP_COL",
    "N_LEADS",
    "RENAME_MAP",
    "SEED",
    "SNAPSHOT_DAY",
    "SUBSAMPLE_N",
    "TARGET_RATE",
    "assign_acquisition_wave",
    "compute_post_snapshot_touches",
    "derive_features",
    "inject_missingness",
    "rename_and_select",
    "softcap_expected_acv",
    "subsample",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_LEADS = 5000
SNAPSHOT_DAY = 20


# ---------------------------------------------------------------------------
# Version-specific pipeline steps
# ---------------------------------------------------------------------------


def rename_and_select(
    df: pd.DataFrame,
    *,
    instructor: bool = False,
    label_column: str = "converted_within_90_days",
) -> pd.DataFrame:
    """Rename snapshot columns to v7 names and select final column set.

    Args:
        df: Snapshot DataFrame.
        instructor: If True, include the instructor leakage trap column.
        label_column: Source column for the binary label. Defaults to
            ``"converted_within_90_days"`` for backward compatibility.
    """
    return _rename_and_select_generic(
        df,
        rename_map=RENAME_MAP,
        final_columns=FINAL_COLUMNS_STUDENT,
        instructor=instructor,
        instructor_columns=FINAL_COLUMNS_INSTRUCTOR,
        label_column=label_column,
    )

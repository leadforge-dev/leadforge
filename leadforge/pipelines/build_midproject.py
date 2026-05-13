"""Pipeline functions for building the mid-project lead scoring dataset.

Produces a single student-safe CSV with 1,200 rows at ~30% conversion rate.
No leakage trap column — this dataset is published directly to students.

Key parameters vs v7:
- SEED = 100  (different seed → different rows from v7's seed=42)
- SUBSAMPLE_N = 1200 (slightly larger than v7's 1000)
- No instructor/trap variant
- Same schema, narrative, missingness patterns as v7
"""

from __future__ import annotations

import pandas as pd

from leadforge.pipelines.common import (
    ACV_CAP,
    ACV_FLOOR,
    FINAL_COLUMNS_STUDENT,
    RENAME_MAP,
    TARGET_RATE,
    assign_acquisition_wave,
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
    "FINAL_COLUMNS_STUDENT",
    "N_LEADS",
    "RENAME_MAP",
    "SEED",
    "SNAPSHOT_DAY",
    "SUBSAMPLE_N",
    "TARGET_RATE",
    "assign_acquisition_wave",
    "derive_features",
    "inject_missingness",
    "rename_and_select",
    "softcap_expected_acv",
    "subsample",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 100
N_LEADS = 5000
SNAPSHOT_DAY = 20
SUBSAMPLE_N = 1200


# ---------------------------------------------------------------------------
# Version-specific pipeline steps
# ---------------------------------------------------------------------------


def rename_and_select(
    df: pd.DataFrame,
    *,
    label_column: str = "converted_within_90_days",
) -> pd.DataFrame:
    """Rename snapshot columns to midproject names and select final column set."""
    return _rename_and_select_generic(
        df,
        rename_map=RENAME_MAP,
        final_columns=FINAL_COLUMNS_STUDENT,
        instructor=False,
        label_column=label_column,
    )

"""Pipeline functions for building the v6 lead scoring intro CSVs.

v6 produces TWO exports:
- **Student-safe**: no leakage columns.
- **Instructor**: identical rows + one ``__leakage__touches_post_snapshot_21_90``
  column computed from the simulator's actual event timeline (days 15..90).

Key v6 changes over v5:
- Snapshot day 20 (shifted from 14 after engine changes weakened day-14 signal).
- Causally-grounded leakage trap (post-snapshot touches from sim events).
- Poisson(1) boost on trap column for converted leads (restores signal post-engine-changes).
- ``touches_last_7_days`` momentum feature.
- ``acquisition_wave`` cohort feature for distribution-shift lecture.
- Nonlinear interaction: opportunity_created x touches_last_7_days.
"""

from __future__ import annotations

import pandas as pd

from leadforge.core.rng import RNGRoot
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
    "boost_leakage_trap",
    "cap_expected_acv",
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


def cap_expected_acv(df: pd.DataFrame) -> pd.DataFrame:
    """Hard clip expected_acv to narrative-consistent range [ACV_FLOOR, ACV_CAP].

    Kept for backward compatibility; prefer ``softcap_expected_acv`` for v6.
    """
    df = df.copy()
    df["expected_acv"] = df["expected_acv"].clip(lower=ACV_FLOOR, upper=ACV_CAP)
    return df


def boost_leakage_trap(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Amplify the causal trap signal with target-correlated Poisson noise.

    Converted leads get an extra Poisson(3) count added to the trap column,
    making it a stronger leakage signal for teaching purposes.
    """
    rng = RNGRoot(seed).numpy_child("leakage_trap_boost")
    df = df.copy()
    n = len(df)
    converted = df["converted"].values
    boost = converted * rng.poisson(3, size=n)
    df[INSTRUCTOR_TRAP_COL] = df[INSTRUCTOR_TRAP_COL] + boost
    return df


def rename_and_select(
    df: pd.DataFrame,
    *,
    instructor: bool = False,
    label_column: str = "converted_within_90_days",
) -> pd.DataFrame:
    """Rename snapshot columns to v6 names and select final column set.

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

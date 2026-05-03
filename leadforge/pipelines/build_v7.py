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
    SUBSAMPLE_N,
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

INSTRUCTOR_TRAP_COL = "__leakage__touches_post_snapshot_21_90"

# v7 student column set: 19 features + 1 target = 20 columns.
FINAL_COLUMNS_STUDENT = [
    "industry",
    "region",
    "company_size",
    "company_revenue",
    "contact_role",
    "seniority",
    "lead_source",
    "opportunity_created",
    "demo_completed",
    "expected_acv",
    "inbound_touches",
    "outbound_touches",
    "touches_week_1",
    "touches_last_7_days",
    "days_since_first_touch",
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
    "acquisition_wave",
    "converted",
]

# Instructor adds the trap column at the end.
FINAL_COLUMNS_INSTRUCTOR = FINAL_COLUMNS_STUDENT + [INSTRUCTOR_TRAP_COL]

# Snapshot column -> v7 column renaming.
RENAME_MAP = {
    "employee_band": "company_size",
    "estimated_revenue_band": "company_revenue",
    "role_function": "contact_role",
    "inbound_touch_count": "inbound_touches",
    "outbound_touch_count": "outbound_touches",
    "session_count": "web_sessions",
    "activity_count": "sales_activities",
    "converted_within_90_days": "converted",
}


# ---------------------------------------------------------------------------
# Version-specific pipeline steps
# ---------------------------------------------------------------------------


def compute_post_snapshot_touches(
    snapshot_df: pd.DataFrame,
    all_touches: list,
    lead_dates: dict[str, str],
    snapshot_day: int = SNAPSHOT_DAY,
    horizon_day: int = 90,
) -> pd.Series:
    """Count touches in (snapshot_day, horizon_day] per lead from event data.

    This is the purely causal leakage trap: it counts actual simulated
    touches that occur after the snapshot cutoff. No label-conditioned
    boost is applied. The trap is predictive because future engagement
    and conversion share latent causal drivers (fit, intent, engagement
    propensity), not because the target was injected.
    """
    if not all_touches:
        return pd.Series(0, index=snapshot_df.index, name=INSTRUCTOR_TRAP_COL)

    td = pd.DataFrame([t.to_dict() for t in all_touches])
    td["_ts"] = pd.to_datetime(td["touch_timestamp"])
    td["_lead_date"] = td["lead_id"].map({lid: pd.Timestamp(d) for lid, d in lead_dates.items()})
    td["_day"] = (td["_ts"] - td["_lead_date"]).dt.days

    # Filter: days in (snapshot_day, horizon_day]
    post = td[(td["_day"] > snapshot_day) & (td["_day"] <= horizon_day)]
    counts = post.groupby("lead_id").size().reset_index(name=INSTRUCTOR_TRAP_COL)

    # Merge back onto snapshot
    result = snapshot_df[["lead_id"]].merge(counts, on="lead_id", how="left")
    result[INSTRUCTOR_TRAP_COL] = result[INSTRUCTOR_TRAP_COL].fillna(0).astype(int)
    return result[INSTRUCTOR_TRAP_COL]


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

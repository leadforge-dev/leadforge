"""Pipeline functions for building the v5 lead scoring intro CSV.

This module contains the reusable data transformation steps. The CLI
orchestration (bundle generation, file I/O) lives in
``scripts/build_v5_snapshot.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadforge.core.rng import RNGRoot
from leadforge.pipelines.common import (
    ACV_CAP,
    ACV_FLOOR,
    SUBSAMPLE_N,
    TARGET_RATE,
    subsample,
)
from leadforge.pipelines.common import (
    derive_features as _derive_features,
)
from leadforge.pipelines.common import (
    rename_and_select as _rename_and_select_generic,
)

__all__ = [
    "ACV_CAP",
    "ACV_FLOOR",
    "FINAL_COLUMNS",
    "N_LEADS",
    "RENAME_MAP",
    "SEED",
    "SNAPSHOT_DAY",
    "SUBSAMPLE_N",
    "TARGET_RATE",
    "boost_leakage_trap",
    "cap_expected_acv",
    "derive_binary_features",
    "inject_missingness",
    "rename_and_select",
    "subsample",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_LEADS = 5000
SNAPSHOT_DAY = 10

# v5 column set: 18 features + 1 target = 19 columns.
FINAL_COLUMNS = [
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
    "days_since_first_touch",
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
    "__leakage__total_touches_90d",
    "converted",
]

# Snapshot column → v5 column renaming.
RENAME_MAP = {
    "employee_band": "company_size",
    "estimated_revenue_band": "company_revenue",
    "role_function": "contact_role",
    "inbound_touch_count": "inbound_touches",
    "outbound_touch_count": "outbound_touches",
    "session_count": "web_sessions",
    "activity_count": "sales_activities",
    "converted_within_90_days": "converted",
    "total_touches_all": "__leakage__total_touches_90d",
    # touches_days_0_7 renamed back to touches_week_1 for v5 CSV format compatibility.
    "touches_days_0_7": "touches_week_1",
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def derive_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary features for the v5 column set."""
    return _derive_features(df)


def cap_expected_acv(df: pd.DataFrame) -> pd.DataFrame:
    """Clip expected_acv to narrative-consistent range [ACV_FLOOR, ACV_CAP]."""
    df = df.copy()
    df["expected_acv"] = df["expected_acv"].clip(lower=ACV_FLOOR, upper=ACV_CAP)
    return df


def rename_and_select(
    df: pd.DataFrame, *, label_column: str = "converted_within_90_days"
) -> pd.DataFrame:
    """Rename snapshot columns to v5 names and select final column set.

    Args:
        df: Snapshot DataFrame.
        label_column: Source column for the binary label. Defaults to
            ``"converted_within_90_days"`` for backward compatibility.
    """
    return _rename_and_select_generic(
        df,
        rename_map=RENAME_MAP,
        final_columns=FINAL_COLUMNS,
        label_column=label_column,
    )


def inject_missingness(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply structured missingness per the v5 contract.

    Conditional rates per source (overall per-column rate stays <10%):
    - web_sessions: SDR outbound 15%, inbound marketing 2%, partner referral 5%
    - seniority: partner referral 8%, others 1%
    - days_since_last_touch: structural NaN (no touches) + 3% MCAR
    - days_since_first_touch: structural NaN (no touches) + 2% MCAR
    """
    rng = RNGRoot(seed).numpy_child("missingness")
    df = df.copy()
    n = len(df)

    # web_sessions: source-conditional missingness
    for source, rate in [
        ("sdr_outbound", 0.15),
        ("inbound_marketing", 0.02),
        ("partner_referral", 0.05),
    ]:
        mask = (df["lead_source"] == source) & (rng.random(n) < rate)
        df.loc[mask, "web_sessions"] = np.nan

    # seniority: source-conditional missingness
    partner_mask = (df["lead_source"] == "partner_referral") & (rng.random(n) < 0.08)
    other_mask = (df["lead_source"] != "partner_referral") & (rng.random(n) < 0.01)
    df.loc[partner_mask | other_mask, "seniority"] = np.nan

    # days_since_last_touch: additional 3% MCAR on top of structural NaN
    dslt_mask = rng.random(n) < 0.03
    df.loc[dslt_mask, "days_since_last_touch"] = np.nan

    # days_since_first_touch: additional 2% MCAR on top of structural NaN
    dsft_mask = rng.random(n) < 0.02
    df.loc[dsft_mask, "days_since_first_touch"] = np.nan

    return df


def boost_leakage_trap(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Amplify the leakage trap signal to ensure robust detectability.

    Adds target-correlated noise to ``__leakage__total_touches_90d`` so
    that converted leads accumulate extra post-snapshot touches.  This
    simulates a realistic scenario where the feature aggregates engagement
    activity that occurs *after* the conversion decision is made.
    """
    rng = RNGRoot(seed).numpy_child("leakage_trap")
    df = df.copy()
    trap_col = "__leakage__total_touches_90d"
    n = len(df)
    converted = df["converted"].values
    # Converted leads: add a Poisson(1)-distributed number of extra
    # "post-conversion" touches (typically small, but unbounded)
    boost = converted * rng.poisson(1, size=n)
    df[trap_col] = df[trap_col] + boost
    return df

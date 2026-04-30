"""Pipeline functions for building the v5 lead scoring intro CSV.

This module contains the reusable data transformation steps. The CLI
orchestration (bundle generation, file I/O) lives in
``scripts/build_v5_snapshot.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

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
SUBSAMPLE_N = 1000
TARGET_RATE = 0.30

# Narrative-consistent ACV bounds (from narrative.yaml: $18k–$120k).
ACV_FLOOR = 18_000.0
ACV_CAP = 120_000.0

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
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def derive_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary features for the v5 column set."""
    df = df.copy()
    df["opportunity_created"] = df["opportunity_created"].astype(int)
    df["demo_completed"] = (df["demo_page_views"] > 0).astype(int)
    return df


def cap_expected_acv(df: pd.DataFrame) -> pd.DataFrame:
    """Clip expected_acv to narrative-consistent range [ACV_FLOOR, ACV_CAP]."""
    df = df.copy()
    df["expected_acv"] = df["expected_acv"].clip(lower=ACV_FLOOR, upper=ACV_CAP)
    return df


def rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snapshot columns to v5 names and select final column set."""
    df = df.rename(columns=RENAME_MAP)
    df["converted"] = df["converted"].astype(int)
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}. Available: {sorted(df.columns)}"
        )
    return df[FINAL_COLUMNS]


def subsample(
    df: pd.DataFrame,
    rng: np.random.RandomState,
    n: int = SUBSAMPLE_N,
    target_rate: float = TARGET_RATE,
) -> pd.DataFrame:
    """Stratified subsample to n rows at target_rate conversion."""
    positives = df[df["converted"] == 1]
    negatives = df[df["converted"] == 0]
    n_pos = int(n * target_rate)
    n_neg = n - n_pos

    if len(positives) < n_pos:
        print(f"WARNING: only {len(positives)} positives, need {n_pos}", file=sys.stderr)
        n_pos = len(positives)
        n_neg = n - n_pos
    if len(negatives) < n_neg:
        print(f"WARNING: only {len(negatives)} negatives, need {n_neg}", file=sys.stderr)
        n_neg = len(negatives)

    pos_sample = positives.sample(n=n_pos, random_state=rng)
    neg_sample = negatives.sample(n=n_neg, random_state=rng)
    return (
        pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=rng).reset_index(drop=True)
    )


def inject_missingness(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Apply structured missingness per the v5 contract.

    Patterns (all <10% per column):
    - web_sessions: SDR outbound 15%, inbound marketing 2%, partner referral 5%
    - seniority: partner referral 8%, others 1%
    - days_since_last_touch: structural NaN (no touches) + 3% MCAR
    - days_since_first_touch: structural NaN (no touches) + 2% MCAR
    """
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


def boost_leakage_trap(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Amplify the leakage trap signal to ensure robust detectability.

    Adds target-correlated noise to ``__leakage__total_touches_90d`` so
    that converted leads accumulate extra post-snapshot touches.  This
    simulates a realistic scenario where the feature aggregates engagement
    activity that occurs *after* the conversion decision is made.
    """
    df = df.copy()
    trap_col = "__leakage__total_touches_90d"
    n = len(df)
    converted = df["converted"].values
    # Converted leads: add a Poisson(1)-distributed number of extra
    # "post-conversion" touches (typically small, but unbounded)
    boost = converted * rng.poisson(1, size=n)
    df[trap_col] = df[trap_col] + boost
    return df

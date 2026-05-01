"""Pipeline functions for building the v6 lead scoring intro CSVs.

v6 produces TWO exports:
- **Student-safe**: no leakage columns.
- **Instructor**: identical rows + one ``__leakage__touches_post_snapshot_15_90``
  column computed from the simulator's actual event timeline (days 15..90).

Key v6 changes over v5:
- Snapshot day 14 (was 10).
- Causally-grounded leakage trap (post-snapshot touches from sim events).
- No boost/noise injection -- trap signal is purely causal.
- ``touches_last_7_days`` momentum feature.
- ``acquisition_wave`` cohort feature for distribution-shift lecture.
- Nonlinear interaction: opportunity_created x touches_last_7_days.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from leadforge.core.rng import RNGRoot

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
SNAPSHOT_DAY = 14
SUBSAMPLE_N = 1000
TARGET_RATE = 0.30

# Narrative-consistent ACV bounds (from narrative.yaml: $18k-$120k).
ACV_FLOOR = 18_000.0
ACV_CAP = 120_000.0

INSTRUCTOR_TRAP_COL = "__leakage__touches_post_snapshot_15_90"

# v6 student column set: 19 features + 1 target = 20 columns.
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

# Snapshot column -> v6 column renaming.
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
# Pipeline steps
# ---------------------------------------------------------------------------


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary and momentum features for the v6 column set."""
    df = df.copy()
    df["opportunity_created"] = df["opportunity_created"].astype(int)
    df["demo_completed"] = (df["demo_page_views"] > 0).astype(int)
    return df


def softcap_expected_acv(
    df: pd.DataFrame,
    seed: int,
    floor: float = ACV_FLOOR,
    cap: float = ACV_CAP,
) -> pd.DataFrame:
    """Soft winsorize expected_acv to avoid hard-clipping ties at the cap.

    Values below floor are clipped. Values above cap are pulled toward cap
    with additive noise so they cluster near the cap rather than pile at it.
    """
    rng = RNGRoot(seed).numpy_child("softcap_acv")
    df = df.copy()
    acv = df["expected_acv"].copy()

    # Floor: hard clip
    acv = acv.clip(lower=floor)

    # Cap: soft winsorize -- draw values in [cap - 5k, cap] for outliers
    over_mask = acv > cap
    n_over = int(over_mask.sum())
    if n_over > 0:
        acv.loc[over_mask] = cap - rng.uniform(0, 5000, size=n_over)

    df["expected_acv"] = acv.round(0)
    return df


def cap_expected_acv(df: pd.DataFrame) -> pd.DataFrame:
    """Hard clip expected_acv to narrative-consistent range [ACV_FLOOR, ACV_CAP].

    Kept for backward compatibility; prefer ``softcap_expected_acv`` for v6.
    """
    df = df.copy()
    df["expected_acv"] = df["expected_acv"].clip(lower=ACV_FLOOR, upper=ACV_CAP)
    return df


def assign_acquisition_wave(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign acquisition_wave (A, B, C) based on lead index position.

    Waves A/B/C are roughly chronological: first third = A, middle = B,
    last third = C. A small amount of noise is added at the boundaries.
    """
    rng = RNGRoot(seed).numpy_child("acquisition_wave")
    df = df.copy()
    n = len(df)
    waves = np.empty(n, dtype=object)
    third = n // 3

    # Base assignment by position (stable across seeds)
    waves[:third] = "A"
    waves[third : 2 * third] = "B"
    waves[2 * third :] = "C"

    # Add ~5% boundary noise so it's not perfectly deterministic
    noise_mask = rng.random(n) < 0.05
    noise_vals = rng.choice(["A", "B", "C"], size=n)
    waves[noise_mask] = noise_vals[noise_mask]

    df["acquisition_wave"] = waves
    return df


def compute_post_snapshot_touches(
    snapshot_df: pd.DataFrame,
    all_touches: list,
    lead_dates: dict[str, str],
    snapshot_day: int = SNAPSHOT_DAY,
    horizon_day: int = 90,
) -> pd.Series:
    """Count touches in (snapshot_day, horizon_day] per lead from event data.

    This is the causally-grounded leakage trap: it counts actual simulated
    touches that occur after the snapshot cutoff.
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
    """Rename snapshot columns to v6 names and select final column set.

    Args:
        df: Snapshot DataFrame.
        instructor: If True, include the instructor leakage trap column.
        label_column: Source column for the binary label. Defaults to
            ``"converted_within_90_days"`` for backward compatibility.
    """
    if label_column == "converted_within_90_days":
        rename_map = RENAME_MAP
    else:
        rename_map = {k: v for k, v in RENAME_MAP.items() if v != "converted"}
        rename_map[label_column] = "converted"
    df = df.rename(columns=rename_map)
    df["converted"] = df["converted"].astype(int)
    columns = FINAL_COLUMNS_INSTRUCTOR if instructor else FINAL_COLUMNS_STUDENT
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}. Available: {sorted(df.columns)}"
        )
    return df[columns]


def subsample(
    df: pd.DataFrame,
    seed: int,
    n: int = SUBSAMPLE_N,
    target_rate: float = TARGET_RATE,
) -> pd.DataFrame:
    """Stratified subsample to n rows at target_rate conversion."""
    rng = RNGRoot(seed).numpy_child("subsample")
    positives = df[df["converted"] == 1]
    negatives = df[df["converted"] == 0]
    n_pos = int(n * target_rate)
    n_neg = n - n_pos

    if len(positives) < n_pos:
        warnings.warn(
            f"only {len(positives)} positives available, need {n_pos}",
            stacklevel=2,
        )
        n_pos = len(positives)
        n_neg = n - n_pos
    if len(negatives) < n_neg:
        warnings.warn(
            f"only {len(negatives)} negatives available, need {n_neg}",
            stacklevel=2,
        )
        n_neg = len(negatives)

    pos_sample = positives.sample(n=n_pos, random_state=rng)
    neg_sample = negatives.sample(n=n_neg, random_state=rng)
    return (
        pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=rng).reset_index(drop=True)
    )


def inject_missingness(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply structured missingness per the v6 contract.

    Patterns:
    1. Structural: days_since_last_touch is NaN when touch_count=0 (from snapshot)
    2. MAR: web_sessions — SDR outbound 15%, inbound marketing 2%, partner 5%
    3. MAR: seniority — partner referral 8%, others 1%
    4. MCAR: expected_acv — 2% uniform
    5. Structural + MCAR: days_since_first_touch — NaN when no touches + 2% MCAR
    6. MCAR: days_since_last_touch — additional 3% on top of structural
    """
    rng = RNGRoot(seed).numpy_child("missingness")
    df = df.copy()
    n = len(df)

    # (1) Structural for days_since_last_touch is already NaN from snapshot builder
    # Note: also structural for days_since_first_touch when no touches

    # (2) MAR: web_sessions by lead_source
    for source, rate in [
        ("sdr_outbound", 0.15),
        ("inbound_marketing", 0.02),
        ("partner_referral", 0.05),
    ]:
        mask = (df["lead_source"] == source) & (rng.random(n) < rate)
        df.loc[mask, "web_sessions"] = np.nan

    # (3) MAR: seniority by lead_source
    partner_mask = (df["lead_source"] == "partner_referral") & (rng.random(n) < 0.08)
    other_mask = (df["lead_source"] != "partner_referral") & (rng.random(n) < 0.01)
    df.loc[partner_mask | other_mask, "seniority"] = np.nan

    # (4) MCAR: expected_acv 2%
    acv_mcar = rng.random(n) < 0.02
    df.loc[acv_mcar, "expected_acv"] = np.nan

    # (5) MCAR: days_since_first_touch 2% on top of structural
    dsft_mask = rng.random(n) < 0.02
    df.loc[dsft_mask, "days_since_first_touch"] = np.nan

    # (6) MCAR: days_since_last_touch 3% on top of structural
    dslt_mask = rng.random(n) < 0.03
    df.loc[dslt_mask, "days_since_last_touch"] = np.nan

    return df

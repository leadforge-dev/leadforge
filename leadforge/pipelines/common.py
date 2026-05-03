"""Shared pipeline functions used across v5/v6/v7 build pipelines.

This module contains the canonical implementations of functions that are
identical (or nearly so) across pipeline versions. Version-specific modules
import from here and override only what differs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from leadforge.core.rng import RNGRoot

__all__ = [
    "ACV_CAP",
    "ACV_FLOOR",
    "BINARY_FEATURES",
    "CAT_FEATURES",
    "NUM_FEATURES",
    "SUBSAMPLE_N",
    "TARGET",
    "TARGET_RATE",
    "assign_acquisition_wave",
    "derive_features",
    "inject_missingness_v6",
    "rename_and_select",
    "softcap_expected_acv",
    "subsample",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBSAMPLE_N = 1000
TARGET_RATE = 0.30
TARGET = "converted"

# Narrative-consistent ACV bounds (from narrative.yaml: $18k-$120k).
ACV_FLOOR = 18_000.0
ACV_CAP = 120_000.0

# Canonical feature lists for v6/v7 datasets.
CAT_FEATURES = [
    "industry",
    "region",
    "company_size",
    "company_revenue",
    "contact_role",
    "seniority",
    "lead_source",
    "acquisition_wave",
]

NUM_FEATURES = [
    "expected_acv",
    "inbound_touches",
    "outbound_touches",
    "touches_week_1",
    "touches_last_7_days",
    "days_since_first_touch",
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
]

BINARY_FEATURES = [
    "opportunity_created",
    "demo_completed",
]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary and momentum features from raw snapshot columns."""
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


def rename_and_select(
    df: pd.DataFrame,
    *,
    rename_map: dict[str, str],
    final_columns: list[str],
    instructor: bool = False,
    instructor_columns: list[str] | None = None,
    label_column: str = "converted_within_90_days",
) -> pd.DataFrame:
    """Rename snapshot columns and select final column set.

    Args:
        df: Snapshot DataFrame.
        rename_map: Mapping from snapshot column names to output names.
        final_columns: Student column set (used when instructor=False).
        instructor: If True, use instructor_columns instead.
        instructor_columns: Instructor column set (superset of final_columns).
        label_column: Source column for the binary label.
    """
    if label_column not in df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found. Available: {sorted(df.columns)}"
        )
    if label_column == "converted_within_90_days":
        rmap = rename_map
    else:
        rmap = {k: v for k, v in rename_map.items() if v != "converted"}
        rmap[label_column] = "converted"
    df = df.rename(columns=rmap)
    df["converted"] = df["converted"].astype(int)
    columns = instructor_columns if instructor and instructor_columns else final_columns
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
    """Stratified subsample to n rows at target_rate conversion.

    Raises:
        ValueError: If there are not enough negative samples to reach *n* rows.
    """
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
        raise ValueError(
            f"only {len(negatives)} negatives available, need {n_neg}; "
            f"cannot produce {n} rows at target_rate={target_rate}"
        )

    pos_sample = positives.sample(n=n_pos, random_state=rng)
    neg_sample = negatives.sample(n=n_neg, random_state=rng)
    return (
        pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=rng).reset_index(drop=True)
    )


def inject_missingness_v6(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply structured missingness (v6/v7 contract — 6 patterns).

    Patterns:
    1. Structural: days_since_last_touch is NaN when touch_count=0 (from snapshot)
    2. MAR: web_sessions -- SDR outbound 15%, inbound marketing 2%, partner 5%
    3. MAR: seniority -- partner referral 8%, others 1%
    4. MCAR: expected_acv -- 2% uniform
    5. Structural + MCAR: days_since_first_touch -- NaN when no touches + 2% MCAR
    6. MCAR: days_since_last_touch -- additional 3% on top of structural
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

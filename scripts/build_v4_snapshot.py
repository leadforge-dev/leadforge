#!/usr/bin/env python3
"""Build the v4 lead scoring intro CSV (generates the bundle internally).

Usage:
    python scripts/build_v4_snapshot.py OUTPUT_CSV

Produces a 1000-row × 18-column CSV at ~30% conversion rate with:
- Day-10 windowed features
- Structured missingness (MAR for web_sessions, seniority)
- Leakage trap (total_touches_all using full 90-day data)
- Stratified subsampling
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from leadforge.api.generator import Generator
from leadforge.render.snapshots import build_snapshot

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_LEADS = 5000
SNAPSHOT_DAY = 10
SUBSAMPLE_N = 1000
TARGET_RATE = 0.30

# v4 column set: 17 features + 1 target = 18 columns.
_FINAL_COLUMNS = [
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
    "web_sessions",
    "sales_activities",
    "days_since_last_touch",
    "total_touches_all",
    "converted",
]

# Snapshot column → v4 column renaming.
_RENAME_MAP = {
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


def generate_bundle(seed: int = SEED, n_leads: int = N_LEADS) -> pd.DataFrame:
    """Generate a full bundle and return the day-10 snapshot."""
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=seed,
        exposure_mode="research_instructor",
        n_leads=n_leads,
        difficulty="intro",
    )
    bundle = gen.generate()
    return build_snapshot(
        bundle.simulation_result,
        bundle.population,
        snapshot_day=SNAPSHOT_DAY,
    )


def derive_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary features for the v4 column set."""
    df = df.copy()
    # opportunity_created comes from the snapshot (any opp, open or closed).
    df["opportunity_created"] = df["opportunity_created"].astype(int)
    df["demo_completed"] = (df["demo_page_views"] > 0).astype(int)
    return df


def rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Rename snapshot columns to v4 names and select final column set."""
    df = df.rename(columns=_RENAME_MAP)
    # Convert boolean converted to int 0/1
    df["converted"] = df["converted"].astype(int)
    missing = [c for c in _FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after renaming: {missing}. Available: {list(df.columns)}"
        )
    return df[_FINAL_COLUMNS]


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
    """Apply structured missingness per the v4 contract."""
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

    return df


def build_v4_dataset(seed: int = SEED) -> pd.DataFrame:
    """Full pipeline: generate → snapshot → derive → rename → subsample → missingness."""
    rng = np.random.RandomState(seed)

    print("Generating bundle...", file=sys.stderr)
    snapshot = generate_bundle(seed=seed)
    conv = snapshot["converted_within_90_days"].mean()
    print(
        f"  Raw snapshot: {len(snapshot)} rows, conversion={conv:.1%}",
        file=sys.stderr,
    )

    df = derive_binary_features(snapshot)
    df = rename_and_select(df)

    print("Subsampling...", file=sys.stderr)
    df = subsample(df, rng)
    print(f"  Subsampled: {len(df)} rows, conversion={df['converted'].mean():.1%}", file=sys.stderr)

    print("Injecting missingness...", file=sys.stderr)
    df = inject_missingness(df, rng)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_CSV", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[1])
    df = build_v4_dataset()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} columns to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

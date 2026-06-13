#!/usr/bin/env python3
"""Build the v5 lead scoring intro CSV (generates the bundle internally).

Usage:
    python scripts/build_v5_snapshot.py OUTPUT_CSV

Produces a 1000-row × 19-column CSV at ~30% conversion rate with:
- Day-10 windowed features
- Structured missingness (MAR for web_sessions, seniority; MCAR on days_since_last_touch)
- Leakage trap (__leakage__total_touches_90d using full 90-day data)
- Expected ACV capped to narrative range [18k, 120k]
- Momentum features (touches_week_1, days_since_first_touch)
- Stratified subsampling
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from leadforge.api.generator import Generator
from leadforge.pipelines.build_v5 import (
    N_LEADS,
    SEED,
    SNAPSHOT_DAY,
    boost_leakage_trap,
    cap_expected_acv,
    derive_binary_features,
    inject_missingness,
    rename_and_select,
    subsample,
)
from leadforge.schemes.lead_scoring.render.snapshots import build_snapshot

# ---------------------------------------------------------------------------
# Orchestration (stays in script — depends on Generator)
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
        bundle.artifacts.simulation_result,
        bundle.artifacts.population,
        snapshot_day=SNAPSHOT_DAY,
    )


def build_v5_dataset(seed: int = SEED) -> pd.DataFrame:
    """Full pipeline: generate → derive → cap ACV → rename → subsample → boost → missingness."""
    print("Generating bundle...", file=sys.stderr)
    snapshot = generate_bundle(seed=seed)
    conv = snapshot["converted_within_90_days"].mean()
    print(
        f"  Raw snapshot: {len(snapshot)} rows, conversion={conv:.1%}",
        file=sys.stderr,
    )

    df = derive_binary_features(snapshot)
    df = cap_expected_acv(df)
    df = rename_and_select(df)

    print("Subsampling...", file=sys.stderr)
    df = subsample(df, seed)
    print(f"  Subsampled: {len(df)} rows, conversion={df['converted'].mean():.1%}", file=sys.stderr)

    print("Boosting leakage trap...", file=sys.stderr)
    df = boost_leakage_trap(df, seed)

    print("Injecting missingness...", file=sys.stderr)
    df = inject_missingness(df, seed)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_CSV", file=sys.stderr)
        sys.exit(1)

    output_path = Path(sys.argv[1])
    df = build_v5_dataset()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows × {len(df.columns)} columns to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

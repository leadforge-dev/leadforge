#!/usr/bin/env python3
"""Build the mid-project lead scoring dataset.

Usage:
    python scripts/build_midproject_lead_scoring.py OUTPUT_DIR

Produces one file in OUTPUT_DIR:
    lead_scoring_midproject.csv   (student-safe, no leakage columns)

1,200 rows at ~30% conversion rate, snapshot day 20.
Seed: 100.  Schema identical to lead_scoring_intro_v7.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from leadforge.api.generator import Generator
from leadforge.pipelines.build_midproject import (
    N_LEADS,
    SEED,
    SNAPSHOT_DAY,
    SUBSAMPLE_N,
    assign_acquisition_wave,
    derive_features,
    inject_missingness,
    rename_and_select,
    softcap_expected_acv,
    subsample,
)
from leadforge.schemes.lead_scoring.render.snapshots import build_snapshot

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_bundle(seed: int = SEED, n_leads: int = N_LEADS):
    """Generate a full bundle and return (snapshot, bundle)."""
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=seed,
        exposure_mode="research_instructor",
        n_leads=n_leads,
        difficulty="intro",
    )
    bundle = gen.generate(latent_touch_intensity=True)
    snapshot = build_snapshot(
        bundle.artifacts.simulation_result,
        bundle.artifacts.population,
        snapshot_day=SNAPSHOT_DAY,
    )
    return snapshot, bundle


def build_midproject_dataset(seed: int = SEED) -> pd.DataFrame:
    """Full pipeline: generate → derive → process → subsample → missingness."""
    print("Generating bundle...", file=sys.stderr)
    snapshot, _bundle = generate_bundle(seed=seed)
    conv = snapshot["converted_within_90_days"].mean()
    print(
        f"  Raw snapshot: {len(snapshot)} rows, conversion={conv:.1%}",
        file=sys.stderr,
    )

    df = derive_features(snapshot)
    df = softcap_expected_acv(df, seed)
    df = assign_acquisition_wave(df, seed)
    df = rename_and_select(df)

    print(f"Subsampling to {SUBSAMPLE_N} rows...", file=sys.stderr)
    df = subsample(df, seed, n=SUBSAMPLE_N)
    print(
        f"  Subsampled: {len(df)} rows, conversion={df['converted'].mean():.1%}",
        file=sys.stderr,
    )

    print("Injecting missingness...", file=sys.stderr)
    df = inject_missingness(df, seed)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_DIR", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_midproject_dataset()

    out_path = output_dir / "lead_scoring_midproject.csv"
    df.to_csv(out_path, index=False)
    print(
        f"Midproject: {len(df)} rows x {len(df.columns)} cols → {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

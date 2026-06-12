#!/usr/bin/env python3
"""Build the v6 lead scoring intro CSVs (generates the bundle internally).

Usage:
    python scripts/build_v6_snapshot.py OUTPUT_DIR

Produces two CSV files in OUTPUT_DIR:
- lead_scoring_intro_v6.csv          (student-safe, no leakage columns)
- lead_scoring_intro_v6_instructor.csv (same rows + __leakage__ trap column)

Both are 1000-row files at ~30% conversion rate with:
- Day-20 windowed features
- Structured missingness (MAR + structural + MCAR)
- Leakage trap: causal post-snapshot touches + Poisson(3) boost for converted leads
- Expected ACV with soft winsorization
- Momentum features (touches_week_1, touches_last_7_days, days_since_first_touch)
- Acquisition wave cohort feature (A/B/C)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from leadforge.api.generator import Generator
from leadforge.pipelines.build_v6 import (
    INSTRUCTOR_TRAP_COL,
    N_LEADS,
    SEED,
    SNAPSHOT_DAY,
    assign_acquisition_wave,
    boost_leakage_trap,
    compute_post_snapshot_touches,
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
    """Generate a full bundle and return (snapshot, bundle) for event access."""
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


def build_v6_datasets(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: generate -> derive -> process -> split into student + instructor."""
    print("Generating bundle (with latent touch intensity)...", file=sys.stderr)
    snapshot, bundle = generate_bundle(seed=seed)
    conv = snapshot["converted_within_90_days"].mean()
    print(
        f"  Raw snapshot: {len(snapshot)} rows, conversion={conv:.1%}",
        file=sys.stderr,
    )

    # Compute post-snapshot touches from event timeline (boosted in next step)
    lead_dates = {lead.lead_id: lead.lead_created_at for lead in bundle.artifacts.population.leads}
    trap_series = compute_post_snapshot_touches(
        snapshot,
        bundle.artifacts.simulation_result.touches,
        lead_dates,
        snapshot_day=SNAPSHOT_DAY,
    )
    snapshot[INSTRUCTOR_TRAP_COL] = trap_series.values

    df = derive_features(snapshot)
    df = softcap_expected_acv(df, seed)
    df = assign_acquisition_wave(df, seed)

    # Rename and select (instructor first to keep trap column)
    df_instructor = rename_and_select(df, instructor=True)

    # Boost trap signal with target-correlated Poisson noise
    df_instructor = boost_leakage_trap(df_instructor, seed)

    print("Subsampling...", file=sys.stderr)
    df_instructor = subsample(df_instructor, seed)
    print(
        f"  Subsampled: {len(df_instructor)} rows, "
        f"conversion={df_instructor['converted'].mean():.1%}",
        file=sys.stderr,
    )

    print("Injecting missingness...", file=sys.stderr)
    df_instructor = inject_missingness(df_instructor, seed)

    # Student version: drop the trap column
    student_cols = [c for c in df_instructor.columns if not c.startswith("__leakage__")]
    df_student = df_instructor[student_cols].copy()

    return df_student, df_instructor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} OUTPUT_DIR", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_student, df_instructor = build_v6_datasets()

    student_path = output_dir / "lead_scoring_intro_v6.csv"
    instructor_path = output_dir / "lead_scoring_intro_v6_instructor.csv"

    df_student.to_csv(student_path, index=False)
    print(
        f"Student:    {len(df_student)} rows x {len(df_student.columns)} cols -> {student_path}",
        file=sys.stderr,
    )

    df_instructor.to_csv(instructor_path, index=False)
    n_r, n_c = len(df_instructor), len(df_instructor.columns)
    print(f"Instructor: {n_r} rows x {n_c} cols -> {instructor_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Cross-seed stability checks.

Verifies that different seeds produce statistically similar distributions,
catching degenerate parameter regimes where one seed produces reasonable
output but another collapses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def check_cross_seed_stability(bundles: dict[int, Path]) -> list[str]:
    """Compare bundles generated with different seeds.

    Args:
        bundles: Mapping of seed → bundle path.  Must contain at least 2
            entries to perform any checks.

    Returns:
        Error strings for any instabilities detected.
    """
    if len(bundles) < 2:
        return []

    errors: list[str] = []
    rates: dict[int, float] = {}
    stage_counts: dict[int, int] = {}

    for seed, bundle_path in bundles.items():
        train_path = bundle_path / "tasks/converted_within_90_days/train.parquet"
        if not train_path.exists():
            errors.append(f"Seed {seed}: missing tasks/converted_within_90_days/train.parquet")
            continue
        df = pd.read_parquet(train_path, columns=["converted_within_90_days"])
        if len(df) > 0:
            rates[seed] = float(df["converted_within_90_days"].mean())

        leads_path = bundle_path / "tables/leads.parquet"
        if leads_path.exists():
            leads = pd.read_parquet(leads_path, columns=["current_stage"])
            stage_counts[seed] = int(leads["current_stage"].nunique())

    # Check conversion rate spread — if one seed's rate is 5x another's, that's suspicious
    if len(rates) >= 2:
        min_rate = min(rates.values())
        max_rate = max(rates.values())
        if min_rate > 0 and max_rate / min_rate > 5.0:
            errors.append(
                f"Conversion rate spread too wide across seeds: "
                f"min={min_rate:.4f}, max={max_rate:.4f} (ratio {max_rate / min_rate:.1f}x)"
            )
        # Also flag if any seed produces near-0% or near-100% conversion
        eps = 1e-9
        for seed, rate in rates.items():
            if rate < eps:
                errors.append(f"Seed {seed}: 0% conversion rate — simulation degenerate")
            elif rate > 1.0 - eps:
                errors.append(f"Seed {seed}: 100% conversion rate — simulation degenerate")

    # Check stage diversity — all seeds should produce multiple stages
    for seed, n_stages in stage_counts.items():
        if n_stages < 2:
            errors.append(f"Seed {seed}: only {n_stages} funnel stage(s) — degenerate")

    return errors

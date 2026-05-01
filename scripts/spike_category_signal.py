#!/usr/bin/env python3
"""Spike experiment: measure category → conversion signal under different settings.

Tests:
1. Baseline (current engine) — expect near-zero category signal
2. Correlated observables (1x boost) — seniority/revenue/source → latent traits
3. Correlated observables (1.8x boost) — stronger correlation

Reports category spread (max - min conversion rate) per categorical feature
and logistic regression AUC at day-21 snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Ensure the package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leadforge.api.generator import Generator
from leadforge.core.rng import RNGRoot
from leadforge.render.snapshots import build_snapshot
from leadforge.simulation.engine import simulate_world
from leadforge.simulation.population import PopulationResult, build_population
from leadforge.structure.sampler import sample_hidden_graph

SEED = 42
N_LEADS = 5000
SUBSAMPLE_N = 1000
TARGET_RATE = 0.30
CAT_FEATURES = [
    "industry",
    "region",
    "estimated_revenue_band",
    "role_function",
    "seniority",
    "lead_source",
]

# Base boosts (scale=1.0)
SENIORITY_BOOST = {
    "individual_contributor": -0.15,
    "manager": -0.05,
    "director": 0.05,
    "vp": 0.12,
    "c_suite": 0.20,
}
REVENUE_BOOST = {
    "$1M-$10M": -0.10,
    "$10M-$50M": 0.0,
    "$50M-$200M": 0.10,
    "$200M+": 0.18,
}
SOURCE_BOOST = {
    "partner_referral": 0.12,
    "inbound_marketing": 0.05,
    "sdr_outbound": -0.08,
}


def subsample(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Stratified subsample to SUBSAMPLE_N rows at TARGET_RATE conversion."""
    positives = df[df["converted_within_90_days"]]
    negatives = df[~df["converted_within_90_days"]]
    n_pos = int(SUBSAMPLE_N * TARGET_RATE)
    n_neg = SUBSAMPLE_N - n_pos

    if len(positives) < n_pos:
        print(f"  WARNING: only {len(positives)} positives, need {n_pos}")
        n_pos = len(positives)
    if len(negatives) < n_neg:
        print(f"  WARNING: only {len(negatives)} negatives, need {n_neg}")
        n_neg = len(negatives)

    pos_sample = positives.sample(n=n_pos, random_state=rng)
    neg_sample = negatives.sample(n=n_neg, random_state=rng)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=rng)


def measure_category_spread(df: pd.DataFrame) -> dict[str, dict]:
    """Conversion rate spread for groups with n >= 50, plus per-value detail."""
    results = {}
    for col in CAT_FEATURES:
        if col not in df.columns:
            continue
        stats = df.groupby(col)["converted_within_90_days"].agg(["mean", "count"])
        large = stats[stats["count"] >= 50]
        spread = float(large["mean"].max() - large["mean"].min()) if len(large) >= 2 else 0.0
        # Show per-value rates for groups with n >= 30
        detail = stats[stats["count"] >= 30].sort_values("mean", ascending=False)
        results[col] = {
            "spread": spread,
            "detail": {str(v): (f"{r['mean']:.1%}", int(r["count"])) for v, r in detail.iterrows()},
        }
    return results


def measure_auc(df: pd.DataFrame) -> float:
    """Logistic regression AUC using all snapshot features."""
    feature_cols = [c for c in df.columns if c != "converted_within_90_days"]
    x_df = df[feature_cols].copy()
    y = df["converted_within_90_days"].astype(int)

    for col in x_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        x_df[col] = le.fit_transform(x_df[col].astype(str))

    x_df = x_df.select_dtypes(include=[np.number])
    x_df = x_df.fillna(x_df.median())

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(x_df, y)
    probs = lr.predict_proba(x_df)[:, 1]
    return float(roc_auc_score(y, probs))


def patch_population(pop: PopulationResult, scale: float = 1.0) -> None:
    """Correlate observable categories with latent traits."""
    # Seniority → latent_contact_authority
    for contact in pop.contacts:
        cid = contact.contact_id
        if cid in pop.latent_state.contact_latents:
            boost = SENIORITY_BOOST.get(contact.seniority, 0.0) * scale
            traits = pop.latent_state.contact_latents[cid]
            traits["latent_contact_authority"] = max(
                0.0, min(1.0, traits["latent_contact_authority"] + boost)
            )

    # Revenue band → latent_account_fit
    for account in pop.accounts:
        aid = account.account_id
        if aid in pop.latent_state.account_latents:
            boost = REVENUE_BOOST.get(account.estimated_revenue_band, 0.0) * scale
            traits = pop.latent_state.account_latents[aid]
            traits["latent_account_fit"] = max(0.0, min(1.0, traits["latent_account_fit"] + boost))

    # Lead source → latent_engagement_propensity
    for lead in pop.leads:
        cid = lead.contact_id
        if cid in pop.latent_state.contact_latents:
            boost = SOURCE_BOOST.get(lead.lead_source, 0.0) * scale
            traits = pop.latent_state.contact_latents[cid]
            traits["latent_engagement_propensity"] = max(
                0.0, min(1.0, traits["latent_engagement_propensity"] + boost)
            )


def run_pipeline(label: str, gen: Generator, scale: float | None = None) -> None:
    """Generate, optionally patch, simulate, snapshot, subsample, measure."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    config = gen._world_spec.config
    narrative = gen._world_spec.narrative
    if narrative is None:
        raise RuntimeError("No narrative loaded")

    rng_root = RNGRoot(config.seed)
    world_graph = sample_hidden_graph(rng_root)
    print(f"  Motif family: {world_graph.motif_family}")

    pop = build_population(config, narrative, world_graph)

    if scale is not None:
        patch_population(pop, scale=scale)

    sim = simulate_world(config, pop, world_graph)
    snapshot = build_snapshot(sim, pop)

    raw_rate = snapshot["converted_within_90_days"].mean()
    print(f"  Raw conversion rate: {raw_rate:.1%} (n={len(snapshot)})")

    rng = np.random.RandomState(SEED)
    df = subsample(snapshot, rng)
    actual_rate = df["converted_within_90_days"].mean()
    print(f"  Subsampled: n={len(df)}, conversion={actual_rate:.1%}")

    results = measure_category_spread(df)
    print("\n  Category spreads (groups n>=50):")
    for feat in CAT_FEATURES:
        if feat not in results:
            continue
        info = results[feat]
        print(f"\n    {feat}: spread={info['spread']:.1%}")
        for val, (rate, n) in info["detail"].items():
            marker = "*" if n >= 50 else " "
            print(f"      {marker} {val:30s} rate={rate} n={n}")

    auc = measure_auc(df)
    print(f"\n  Logistic regression AUC (train): {auc:.3f}")
    return auc


def main() -> None:
    results = {}

    # Experiment 1: Baseline
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=SEED,
        exposure_mode="research_instructor",
        n_leads=N_LEADS,
        difficulty="intro",
    )
    results["baseline"] = run_pipeline("BASELINE (current engine)", gen, scale=None)

    # Experiment 2: Scale 1.0
    gen2 = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=SEED,
        exposure_mode="research_instructor",
        n_leads=N_LEADS,
        difficulty="intro",
    )
    results["scale_1.0"] = run_pipeline("PATCHED scale=1.0", gen2, scale=1.0)

    # Experiment 3: Scale 1.8
    gen3 = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=SEED,
        exposure_mode="research_instructor",
        n_leads=N_LEADS,
        difficulty="intro",
    )
    results["scale_1.8"] = run_pipeline("PATCHED scale=1.8", gen3, scale=1.8)

    # Experiment 4: Scale 2.5
    gen4 = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=SEED,
        exposure_mode="research_instructor",
        n_leads=N_LEADS,
        difficulty="intro",
    )
    results["scale_2.5"] = run_pipeline("PATCHED scale=2.5", gen4, scale=2.5)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for label, auc in results.items():
        print(f"  {label:<30s} AUC={auc:.3f}")

    print()
    print("  KEY FINDING: The spec's approach of scaling CategoricalInfluence")
    print("  weights in LatentScore is incorrect — CategoricalInfluence is")
    print("  not used in the conversion score. The correct approach is to")
    print("  correlate observable categories with latent traits during")
    print("  population generation.")


if __name__ == "__main__":
    main()

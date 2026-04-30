"""Shared test fixtures and helpers for leadforge tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical v5 column set (post-rename)
# ---------------------------------------------------------------------------

V5_COLUMNS = [
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

# Category value pools shared across all synthetic data builders.
INDUSTRIES = ["manufacturing", "logistics", "services", "healthcare"]
REGIONS = ["US", "UK"]
COMPANY_SIZES = ["200-499", "500-999", "1000-1999", "2000+"]
COMPANY_REVENUES = ["$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+"]
CONTACT_ROLES = ["finance", "ap_manager", "it_director", "procurement"]
SENIORITIES = ["individual_contributor", "manager", "director", "vp", "c_suite"]
LEAD_SOURCES = ["inbound_marketing", "sdr_outbound", "partner_referral"]


def make_v5_dataset(
    n: int = 200,
    conversion_rate: float = 0.30,
    include_leakage: bool = True,
    deterministic_col: bool = False,
    inject_signal: bool = False,
    seed: int = 99,
) -> pd.DataFrame:
    """Build a synthetic dataset in v5 column format.

    Parameters
    ----------
    n : int
        Number of rows.
    conversion_rate : float
        Approximate target conversion rate (exact when inject_signal=False).
    include_leakage : bool
        Whether to include __leakage__total_touches_90d column.
    deterministic_col : bool
        If True, add a "bad_feature" column that perfectly predicts conversion
        for a subgroup (useful for testing group determinism checks).
    inject_signal : bool
        If True, generate features that are correlated with the target so
        that a logistic regression baseline can achieve AUC >= 0.62.  The
        conversion rate is approximate (~30%) rather than exact.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)

    if inject_signal:
        # Latent score drives both features and outcome.
        # Bias of -0.85 targets ~30% base rate under the sigmoid.
        latent = rng.normal(0, 1, size=n)
        prob = 1 / (1 + np.exp(-(1.5 * latent - 0.85)))
        converted = (rng.random(n) < prob).astype(int)

        inbound = np.clip(rng.poisson(3, size=n) + (latent * 1.5).astype(int), 0, None)
        web_sessions = np.clip(rng.poisson(4, size=n) + (latent * 1.0).astype(int), 0, None).astype(
            float
        )
        demo_completed = (latent + rng.normal(0, 0.8, size=n) > 0.3).astype(int)
        opp_created = (latent + rng.normal(0, 0.8, size=n) > 0.0).astype(int)
    else:
        n_pos = int(n * conversion_rate)
        n_neg = n - n_pos
        converted = np.array([1] * n_pos + [0] * n_neg)
        rng.shuffle(converted)

        inbound = rng.poisson(3, size=n)
        web_sessions = rng.poisson(4, size=n).astype(float)
        demo_completed = rng.randint(0, 2, size=n)
        opp_created = rng.randint(0, 2, size=n)

    df = pd.DataFrame(
        {
            "industry": rng.choice(INDUSTRIES, size=n),
            "region": rng.choice(REGIONS, size=n),
            "company_size": rng.choice(COMPANY_SIZES, size=n),
            "company_revenue": rng.choice(COMPANY_REVENUES, size=n),
            "contact_role": rng.choice(CONTACT_ROLES, size=n),
            "seniority": rng.choice(SENIORITIES, size=n),
            "lead_source": rng.choice(LEAD_SOURCES, size=n),
            "opportunity_created": opp_created,
            "demo_completed": demo_completed,
            "expected_acv": rng.uniform(18_000, 120_000, size=n).round(0),
            "inbound_touches": inbound,
            "outbound_touches": rng.poisson(2, size=n),
            "touches_week_1": rng.poisson(2, size=n),
            "days_since_first_touch": rng.uniform(0, 14, size=n).round(1),
            "web_sessions": web_sessions,
            "sales_activities": rng.poisson(3, size=n),
            "days_since_last_touch": rng.uniform(0, 14, size=n).round(1),
            "converted": converted,
        }
    )

    # Inject some missingness
    miss_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[miss_idx, "web_sessions"] = np.nan

    if include_leakage:
        noise = rng.poisson(3, size=n)
        df["__leakage__total_touches_90d"] = converted * rng.poisson(8, size=n) + noise

    if deterministic_col:
        df["bad_feature"] = "normal"
        df.loc[:59, "bad_feature"] = "leaked"
        df.loc[:59, "converted"] = 1

    return df


def save_csv(df: pd.DataFrame, tmp_path: Path, name: str = "data.csv") -> Path:
    """Write a DataFrame to CSV and return the path."""
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path

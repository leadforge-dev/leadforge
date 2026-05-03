"""Lead snapshot builder — flatten the simulated world into an ML-ready table.

:func:`build_snapshot` produces one row per lead, containing the features
defined in :data:`~leadforge.schema.features.LEAD_SNAPSHOT_FEATURES`.  All
columns are anchored at or before the snapshot date (lead creation + horizon),
preserving the leakage-free guarantee.

The snapshot is the source table for the primary task export
(``converted_within_90_days``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from leadforge.core.rng import RNGRoot
from leadforge.schema.entities import (
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    TouchRow,
)
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES
from leadforge.simulation.population import REVENUE_BAND_MIDPOINTS

if TYPE_CHECKING:
    from leadforge.core.models import DifficultyParams
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult

# Ordered column list and dtypes derived from the canonical feature spec.
_SNAPSHOT_COLUMNS = [f.name for f in LEAD_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in LEAD_SNAPSHOT_FEATURES}

# Join columns derived from the feature spec — single source of truth.
# Adding a new account/contact feature to LEAD_SNAPSHOT_FEATURES automatically
# includes it here without any manual list maintenance.
_ACCOUNT_JOIN_COLS = [f.name for f in LEAD_SNAPSHOT_FEATURES if f.category == "account"]
_CONTACT_JOIN_COLS = [f.name for f in LEAD_SNAPSHOT_FEATURES if f.category == "contact"]

# Aggregated count columns that need zero-filling after left-merge.
_INT_AGG_COLS = [
    "touch_count",
    "inbound_touch_count",
    "outbound_touch_count",
    "session_count",
    "pricing_page_views",
    "demo_page_views",
    "total_session_duration_seconds",
    "activity_count",
]


def build_snapshot(
    result: SimulationResult,
    population: PopulationResult,
    horizon_days: int = 90,
    snapshot_day: int | None = None,
    difficulty_params: DifficultyParams | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the lead snapshot DataFrame from simulation output.

    One row is produced per lead.  Features are computed by aggregating
    touches, sessions, and sales activities that occurred during the
    simulation horizon.  The snapshot anchor date is
    ``lead_created_at + timedelta(days=horizon_days)``.

    When *snapshot_day* is set, event aggregations are filtered to events
    within ``[lead_created_at, lead_created_at + snapshot_day]``.  This
    enables windowed feature computation (e.g. day-21 snapshots for v4
    datasets).  The default ``None`` preserves existing behavior (full
    horizon).

    Args:
        result: Output of :func:`~leadforge.simulation.engine.simulate_world`.
        population: Output of
            :func:`~leadforge.simulation.population.build_population`.
        horizon_days: Simulation horizon length.  Defaults to 90.
        snapshot_day: Optional windowed snapshot day.  When set, only events
            with timestamps ``<= lead_created_at + timedelta(days=snapshot_day)``
            are included (midnight-exclusive by construction, since the
            simulation engine uses daily steps).  Default ``None`` means use
            *horizon_days*.

    Returns:
        A ``pd.DataFrame`` with the columns specified in
        :data:`~leadforge.schema.features.LEAD_SNAPSHOT_FEATURES` and dtypes
        matching the feature spec.  Row order matches ``result.leads``.
    """
    # Note: when label_window_days < horizon_days, the label
    # (converted_within_90_days on LeadRow) is derived using the shorter
    # window, but features here are aggregated over the full horizon.  This
    # is intentional — the simulation produces rich event histories over the
    # full horizon, and feature aggregation follows suit.  Callers that need
    # windowed features should use the snapshot_day parameter.
    effective_window = snapshot_day if snapshot_day is not None else horizon_days

    # -------------------------------------------------------------------
    # Build base lead DataFrame first (needed for per-lead date filtering).
    # -------------------------------------------------------------------
    lead_df = pd.DataFrame([lead.to_dict() for lead in result.leads])
    lead_dates = pd.to_datetime(lead_df["lead_created_at"])
    lead_df["anchor_date"] = lead_dates + pd.Timedelta(days=horizon_days)
    lead_df["snapshot_cutoff"] = lead_dates + pd.Timedelta(days=effective_window)

    # Build a lead_id → snapshot_cutoff lookup for event filtering.
    cutoff_map = dict(zip(lead_df["lead_id"], lead_df["snapshot_cutoff"], strict=False))

    # -------------------------------------------------------------------
    # Aggregate event tables by lead_id using pandas for efficiency.
    # Empty event lists fall back to the entity's canonical empty DataFrame
    # so groupby always produces the correct output column names.
    # -------------------------------------------------------------------

    # Touch aggregates
    td = (
        pd.DataFrame([t.to_dict() for t in result.touches])
        if result.touches
        else TouchRow.empty_dataframe()
    )

    # Apply snapshot window filter to touches.
    if len(td) > 0 and snapshot_day is not None:
        td["_ts"] = pd.to_datetime(td["touch_timestamp"])
        td["_cutoff"] = td["lead_id"].map(cutoff_map)
        td_windowed = td[td["_ts"] <= td["_cutoff"]].copy()
        td_full = td  # Keep full set for total_touches_all
    else:
        td_windowed = td
        td_full = td

    touch_agg = (
        td_windowed.groupby("lead_id")
        .agg(
            touch_count=("touch_id", "count"),
            inbound_touch_count=(
                "touch_direction",
                lambda s: int((s == "inbound").sum()),
            ),
            outbound_touch_count=(
                "touch_direction",
                lambda s: int((s == "outbound").sum()),
            ),
            last_touch_timestamp=("touch_timestamp", "max"),
        )
        .reset_index()
    )

    # touches_week_1: count touches within first 7 days of lead creation.
    # touches_last_7_days: count touches within last 7 days of snapshot window.
    if len(td_windowed) > 0:
        if "_ts" not in td_windowed.columns:
            td_windowed = td_windowed.copy()
            td_windowed["_ts"] = pd.to_datetime(td_windowed["touch_timestamp"])
        td_windowed_copy = td_windowed.copy()
        td_windowed_copy["_lead_date"] = td_windowed_copy["lead_id"].map(
            dict(zip(lead_df["lead_id"], lead_dates, strict=False))
        )
        td_windowed_copy["_day"] = (
            td_windowed_copy["_ts"] - td_windowed_copy["_lead_date"]
        ).dt.days
        week1 = td_windowed_copy[td_windowed_copy["_day"] <= 7]
        touches_week_1 = week1.groupby("lead_id").size().reset_index(name="touches_week_1")

        # touches_last_7_days: touches in [effective_window - 7, effective_window]
        last7 = td_windowed_copy[td_windowed_copy["_day"] > (effective_window - 7)]
        touches_last_7_days = (
            last7.groupby("lead_id").size().reset_index(name="touches_last_7_days")
        )

        # days_since_first_touch: snapshot_day - first_touch_day
        first_touch_day = (
            td_windowed_copy.groupby("lead_id")["_day"]
            .min()
            .reset_index()
            .rename(columns={"_day": "_first_touch_day"})
        )
    else:
        touches_week_1 = pd.DataFrame(columns=["lead_id", "touches_week_1"])
        touches_last_7_days = pd.DataFrame(columns=["lead_id", "touches_last_7_days"])
        first_touch_day = pd.DataFrame(columns=["lead_id", "_first_touch_day"])

    # total_touches_all: count over full horizon (leakage trap when windowed,
    # equals touch_count when using full horizon).
    if len(td_full) > 0:
        total_touches_all = (
            td_full.groupby("lead_id")["touch_id"]
            .count()
            .reset_index()
            .rename(columns={"touch_id": "total_touches_all"})
        )
    else:
        total_touches_all = pd.DataFrame(columns=["lead_id", "total_touches_all"])

    # Session aggregates
    sd = (
        pd.DataFrame([s.to_dict() for s in result.sessions])
        if result.sessions
        else SessionRow.empty_dataframe()
    )
    if len(sd) > 0 and snapshot_day is not None:
        sd["_ts"] = pd.to_datetime(sd["session_timestamp"])
        sd["_cutoff"] = sd["lead_id"].map(cutoff_map)
        sd = sd[sd["_ts"] <= sd["_cutoff"]]

    sess_agg = (
        sd.groupby("lead_id")
        .agg(
            session_count=("session_id", "count"),
            pricing_page_views=("pricing_page_views", "sum"),
            demo_page_views=("demo_page_views", "sum"),
            total_session_duration_seconds=("session_duration_seconds", "sum"),
        )
        .reset_index()
    )

    # Sales activity aggregates
    ad = (
        pd.DataFrame([a.to_dict() for a in result.sales_activities])
        if result.sales_activities
        else SalesActivityRow.empty_dataframe()
    )
    if len(ad) > 0 and snapshot_day is not None:
        ad["_ts"] = pd.to_datetime(ad["activity_timestamp"])
        ad["_cutoff"] = ad["lead_id"].map(cutoff_map)
        ad = ad[ad["_ts"] <= ad["_cutoff"]]

    act_agg = ad.groupby("lead_id").agg(activity_count=("activity_id", "count")).reset_index()

    # Opportunity join: find opportunity created by snapshot cutoff.
    od = (
        pd.DataFrame([o.to_dict() for o in result.opportunities])
        if result.opportunities
        else OpportunityRow.empty_dataframe()
    )
    if len(od) > 0 and snapshot_day is not None:
        od["_created"] = pd.to_datetime(od["created_at"])
        od["_cutoff"] = od["lead_id"].map(cutoff_map)
        od = od[od["_created"] <= od["_cutoff"]]

    # Track ANY opportunity created (regardless of close outcome) for opportunity_created flag.
    any_opps = od[["lead_id"]].drop_duplicates()
    any_opps["opportunity_created"] = True

    open_opps = od[od["close_outcome"].isna()][["lead_id", "estimated_acv"]]
    open_opps = open_opps.groupby("lead_id").first().reset_index()
    open_opps = open_opps.rename(columns={"estimated_acv": "opportunity_estimated_acv"})
    open_opps["has_open_opportunity"] = True

    # -------------------------------------------------------------------
    # Join aggregates (left join preserves all leads).
    # -------------------------------------------------------------------
    lead_df = lead_df.merge(touch_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(sess_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(act_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(any_opps, on="lead_id", how="left")
    lead_df = lead_df.merge(open_opps, on="lead_id", how="left")
    lead_df = lead_df.merge(touches_week_1, on="lead_id", how="left")
    lead_df = lead_df.merge(touches_last_7_days, on="lead_id", how="left")
    lead_df = lead_df.merge(first_touch_day, on="lead_id", how="left")
    lead_df = lead_df.merge(total_touches_all, on="lead_id", how="left")

    # Fill missing event aggregate counts with zero; has_open_opportunity with False.
    # opportunity_estimated_acv and days_since_last_touch intentionally stay NaN.
    int_agg_cols = [c for c in _INT_AGG_COLS if c in lead_df.columns]
    lead_df[int_agg_cols] = lead_df[int_agg_cols].fillna(0)
    lead_df["touches_week_1"] = lead_df["touches_week_1"].fillna(0)
    lead_df["touches_last_7_days"] = lead_df["touches_last_7_days"].fillna(0)
    if "total_touches_all" in lead_df.columns:
        lead_df["total_touches_all"] = pd.to_numeric(
            lead_df["total_touches_all"], errors="coerce"
        ).fillna(0)
    opp_created_mask = lead_df["opportunity_created"].notna()
    lead_df["opportunity_created"] = lead_df["opportunity_created"].where(
        opp_created_mask, other=False
    )
    opp_mask = lead_df["has_open_opportunity"].notna()
    lead_df["has_open_opportunity"] = lead_df["has_open_opportunity"].where(opp_mask, other=False)

    # Compute days_since_last_touch fully vectorised.
    # pd.to_datetime returns NaT for nulls; (Timestamp - NaT) yields NaN naturally.
    last_ts = pd.to_datetime(lead_df["last_touch_timestamp"])
    lead_df["days_since_last_touch"] = (lead_df["snapshot_cutoff"] - last_ts).dt.days

    # Compute days_since_first_touch: snapshot_day - first_touch_day.
    if "_first_touch_day" in lead_df.columns:
        lead_df["days_since_first_touch"] = effective_window - lead_df["_first_touch_day"]
    else:
        lead_df["days_since_first_touch"] = pd.NA

    # Compute expected_acv: opportunity ACV if available, else revenue band midpoint.

    # -------------------------------------------------------------------
    # Join account and contact features via vectorised merge (not apply).
    # Columns are derived from LEAD_SNAPSHOT_FEATURES categories so this
    # list stays in sync automatically when the feature spec changes.
    # -------------------------------------------------------------------
    acct_df = pd.DataFrame([a.to_dict() for a in population.accounts])[_ACCOUNT_JOIN_COLS]
    cont_df = pd.DataFrame([c.to_dict() for c in population.contacts])[_CONTACT_JOIN_COLS]
    lead_df = lead_df.merge(acct_df, on="account_id", how="left")
    lead_df = lead_df.merge(cont_df, on="contact_id", how="left")

    # expected_acv: opportunity ACV where available, else revenue band midpoint.
    band_midpoint = lead_df["estimated_revenue_band"].map(REVENUE_BAND_MIDPOINTS)
    lead_df["expected_acv"] = lead_df["opportunity_estimated_acv"].fillna(band_midpoint)

    # -------------------------------------------------------------------
    # Select, order, and cast columns — single authoritative dtype pass.
    # -------------------------------------------------------------------
    snapshot = lead_df[_SNAPSHOT_COLUMNS].copy()
    for col, dtype in _SNAPSHOT_DTYPES.items():
        if col in snapshot.columns:
            snapshot[col] = snapshot[col].astype(dtype)

    # -------------------------------------------------------------------
    # Difficulty distortions: noise, missingness, outliers.
    # -------------------------------------------------------------------
    if difficulty_params is not None:
        snapshot = _apply_difficulty_distortions(snapshot, difficulty_params, seed)

    return snapshot


# ---------------------------------------------------------------------------
# Difficulty distortion helpers
# ---------------------------------------------------------------------------

# Columns that must never be distorted (labels, IDs, categoricals).
_DISTORTION_EXCLUDE = frozenset({"converted_within_90_days", "lead_id", "account_id", "contact_id"})


def _apply_difficulty_distortions(
    df: pd.DataFrame,
    params: DifficultyParams,
    seed: int,
) -> pd.DataFrame:
    """Apply noise, missingness, and outliers to numeric snapshot features."""
    rng_root = RNGRoot(seed)
    np_rng = rng_root.numpy_child("snapshot_distortions")

    # Identify float columns eligible for noise/outlier distortion.
    float_cols = [
        c
        for c in df.columns
        if c not in _DISTORTION_EXCLUDE and df[c].dtype in ("float64", "Float64")
    ]
    # All numeric columns (int + float) eligible for missingness.
    all_numeric_cols = [
        c
        for c in df.columns
        if c not in _DISTORTION_EXCLUDE and df[c].dtype in ("float64", "Float64", "int64", "Int64")
    ]

    # 1. Gaussian noise on float features only (avoids int casting issues).
    if params.noise_scale > 0:
        for col in float_cols:
            valid_mask = df[col].notna()
            if valid_mask.sum() == 0:
                continue
            col_std = float(df.loc[valid_mask, col].std())
            if col_std == 0 or np.isnan(col_std):
                continue
            noise = np_rng.normal(0, params.noise_scale * col_std, size=len(df))
            # Add noise only where values are valid.
            values = df[col].copy()
            values[valid_mask] = values[valid_mask] + noise[valid_mask.values]
            df[col] = values

    # 2. MCAR missingness injection (all numeric columns).
    if params.missing_rate > 0:
        mask = np_rng.random(size=(len(df), len(all_numeric_cols))) < params.missing_rate
        for i, col in enumerate(all_numeric_cols):
            col_mask = mask[:, i]
            if col_mask.any():
                # Convert int columns to float to support NaN.
                if df[col].dtype in ("int64", "Int64"):
                    df[col] = df[col].astype("Float64")
                df.loc[col_mask, col] = np.nan

    # 3. Outlier injection (float columns only).
    if params.outlier_rate > 0:
        for col in float_cols:
            valid_mask = df[col].notna()
            col_std = float(df.loc[valid_mask, col].std())
            if col_std == 0 or np.isnan(col_std):
                continue
            col_median = float(df[col].median())
            outlier_mask = np_rng.random(size=len(df)) < params.outlier_rate
            signs = np_rng.choice([-1, 1], size=len(df)).astype(float)
            outlier_values = col_median + signs * 3 * col_std
            combined = outlier_mask & valid_mask.values
            if combined.any():
                df.loc[combined, col] = outlier_values[combined]

    return df

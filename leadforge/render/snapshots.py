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

import pandas as pd

from leadforge.schema.entities import (
    OpportunityRow,
    SalesActivityRow,
    SessionRow,
    TouchRow,
)
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES

if TYPE_CHECKING:
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult

# Ordered column list derived from the canonical feature spec.
_SNAPSHOT_COLUMNS = [f.name for f in LEAD_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in LEAD_SNAPSHOT_FEATURES}

# Account and contact columns needed in the snapshot (subset of their full DTYPE_MAP).
_ACCOUNT_JOIN_COLS = [
    "account_id",
    "industry",
    "region",
    "employee_band",
    "estimated_revenue_band",
    "process_maturity_band",
]
_CONTACT_JOIN_COLS = ["contact_id", "role_function", "seniority", "buyer_role"]


def build_snapshot(
    result: SimulationResult,
    population: PopulationResult,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """Build the lead snapshot DataFrame from simulation output.

    One row is produced per lead.  Features are computed by aggregating
    touches, sessions, and sales activities that occurred during the
    simulation horizon.  The snapshot anchor date is
    ``lead_created_at + timedelta(days=horizon_days)``.

    Args:
        result: Output of :func:`~leadforge.simulation.engine.simulate_world`.
        population: Output of
            :func:`~leadforge.simulation.population.build_population`.
        horizon_days: Simulation horizon length.  Defaults to 90.

    Returns:
        A ``pd.DataFrame`` with the columns specified in
        :data:`~leadforge.schema.features.LEAD_SNAPSHOT_FEATURES` and dtypes
        matching the feature spec.  Row order matches ``result.leads``.
    """
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
    touch_agg = (
        td.groupby("lead_id")
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

    # Session aggregates
    sd = (
        pd.DataFrame([s.to_dict() for s in result.sessions])
        if result.sessions
        else SessionRow.empty_dataframe()
    )
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
    act_agg = ad.groupby("lead_id").agg(activity_count=("activity_id", "count")).reset_index()

    # Opportunity join: find open (unclosed) opportunity per lead.
    od = (
        pd.DataFrame([o.to_dict() for o in result.opportunities])
        if result.opportunities
        else OpportunityRow.empty_dataframe()
    )
    open_opps = od[od["close_outcome"].isna()][["lead_id", "estimated_acv"]]
    open_opps = open_opps.groupby("lead_id").first().reset_index()
    open_opps = open_opps.rename(columns={"estimated_acv": "opportunity_estimated_acv"})
    open_opps["has_open_opportunity"] = True

    # -------------------------------------------------------------------
    # Build base lead DataFrame and join aggregates.
    # -------------------------------------------------------------------
    lead_df = pd.DataFrame([lead.to_dict() for lead in result.leads])

    # Compute snapshot anchor date (per lead, vectorised).
    lead_df["anchor_date"] = pd.to_datetime(lead_df["lead_created_at"]) + pd.Timedelta(
        days=horizon_days
    )

    # Join aggregates (left join preserves all leads).
    lead_df = lead_df.merge(touch_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(sess_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(act_agg, on="lead_id", how="left")
    lead_df = lead_df.merge(open_opps, on="lead_id", how="left")

    # Fill missing aggregates with zero / False.
    lead_df["touch_count"] = lead_df["touch_count"].fillna(0).astype("Int64")
    lead_df["inbound_touch_count"] = lead_df["inbound_touch_count"].fillna(0).astype("Int64")
    lead_df["outbound_touch_count"] = lead_df["outbound_touch_count"].fillna(0).astype("Int64")
    lead_df["session_count"] = lead_df["session_count"].fillna(0).astype("Int64")
    lead_df["pricing_page_views"] = lead_df["pricing_page_views"].fillna(0).astype("Int64")
    lead_df["demo_page_views"] = lead_df["demo_page_views"].fillna(0).astype("Int64")
    lead_df["total_session_duration_seconds"] = (
        lead_df["total_session_duration_seconds"].fillna(0).astype("Int64")
    )
    lead_df["activity_count"] = lead_df["activity_count"].fillna(0).astype("Int64")
    mask = lead_df["has_open_opportunity"].notna()
    lead_df["has_open_opportunity"] = (
        lead_df["has_open_opportunity"].where(mask, other=False).astype("boolean")
    )
    lead_df["opportunity_estimated_acv"] = lead_df["opportunity_estimated_acv"].astype("Float64")

    # Compute days_since_last_touch (Float64, NaN when no touches).
    has_touch = lead_df["last_touch_timestamp"].notna()
    lead_df["days_since_last_touch"] = pd.NA
    if has_touch.any():
        last_ts = pd.to_datetime(lead_df.loc[has_touch, "last_touch_timestamp"])
        lead_df.loc[has_touch, "days_since_last_touch"] = (
            lead_df.loc[has_touch, "anchor_date"] - last_ts
        ).dt.days
    lead_df["days_since_last_touch"] = lead_df["days_since_last_touch"].astype("Float64")

    # -------------------------------------------------------------------
    # Join account and contact features via vectorised merge (not apply).
    # -------------------------------------------------------------------
    acct_df = pd.DataFrame([a.to_dict() for a in population.accounts])[_ACCOUNT_JOIN_COLS]
    cont_df = pd.DataFrame([c.to_dict() for c in population.contacts])[_CONTACT_JOIN_COLS]
    lead_df = lead_df.merge(acct_df, on="account_id", how="left")
    lead_df = lead_df.merge(cont_df, on="contact_id", how="left")

    # -------------------------------------------------------------------
    # Select and order columns per canonical feature spec; apply dtypes.
    # -------------------------------------------------------------------
    snapshot = lead_df[_SNAPSHOT_COLUMNS].copy()
    for col, dtype in _SNAPSHOT_DTYPES.items():
        if col in snapshot.columns:
            snapshot[col] = snapshot[col].astype(dtype)

    return snapshot

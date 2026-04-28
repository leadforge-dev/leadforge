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

from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES

if TYPE_CHECKING:
    from leadforge.simulation.engine import SimulationResult
    from leadforge.simulation.population import PopulationResult

# Ordered column list derived from the canonical feature spec.
_SNAPSHOT_COLUMNS = [f.name for f in LEAD_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in LEAD_SNAPSHOT_FEATURES}


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
    account_by_id = {a.account_id: a for a in population.accounts}
    contact_by_id = {c.contact_id: c for c in population.contacts}

    # -------------------------------------------------------------------
    # Aggregate event tables by lead_id using pandas for efficiency.
    # -------------------------------------------------------------------

    # Touch aggregates
    if result.touches:
        td = pd.DataFrame([t.to_dict() for t in result.touches])
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
    else:
        touch_agg = pd.DataFrame(
            columns=[
                "lead_id",
                "touch_count",
                "inbound_touch_count",
                "outbound_touch_count",
                "last_touch_timestamp",
            ]
        )

    # Session aggregates
    if result.sessions:
        sd = pd.DataFrame([s.to_dict() for s in result.sessions])
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
    else:
        sess_agg = pd.DataFrame(
            columns=[
                "lead_id",
                "session_count",
                "pricing_page_views",
                "demo_page_views",
                "total_session_duration_seconds",
            ]
        )

    # Sales activity aggregates
    if result.sales_activities:
        ad = pd.DataFrame([a.to_dict() for a in result.sales_activities])
        act_agg = ad.groupby("lead_id").agg(activity_count=("activity_id", "count")).reset_index()
    else:
        act_agg = pd.DataFrame(columns=["lead_id", "activity_count"])

    # Opportunity join: find open (unclosed) opportunity per lead.
    if result.opportunities:
        od = pd.DataFrame([o.to_dict() for o in result.opportunities])
        open_opps = od[od["close_outcome"].isna()][["lead_id", "estimated_acv"]]
        # One open opp per lead (first if multiple, which shouldn't happen in v1).
        open_opps = open_opps.groupby("lead_id").first().reset_index()
        open_opps = open_opps.rename(columns={"estimated_acv": "opportunity_estimated_acv"})
        open_opps["has_open_opportunity"] = True
    else:
        open_opps = pd.DataFrame(
            columns=["lead_id", "has_open_opportunity", "opportunity_estimated_acv"]
        )

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
    # Join account and contact features.
    # -------------------------------------------------------------------
    def _account_field(row: pd.Series, field: str) -> object:
        acct = account_by_id.get(row["account_id"])
        return getattr(acct, field, pd.NA) if acct else pd.NA

    def _contact_field(row: pd.Series, field: str) -> object:
        cont = contact_by_id.get(row["contact_id"])
        return getattr(cont, field, pd.NA) if cont else pd.NA

    for field in (
        "industry",
        "region",
        "employee_band",
        "estimated_revenue_band",
        "process_maturity_band",
    ):
        lead_df[field] = lead_df.apply(_account_field, axis=1, field=field)

    for field in ("role_function", "seniority", "buyer_role"):
        lead_df[field] = lead_df.apply(_contact_field, axis=1, field=field)

    # -------------------------------------------------------------------
    # Select and order columns per canonical feature spec; apply dtypes.
    # -------------------------------------------------------------------
    snapshot = lead_df[_SNAPSHOT_COLUMNS].copy()
    for col, dtype in _SNAPSHOT_DTYPES.items():
        if col in snapshot.columns:
            try:
                snapshot[col] = snapshot[col].astype(dtype)
            except (ValueError, TypeError):
                pass  # column already has compatible dtype

    return snapshot

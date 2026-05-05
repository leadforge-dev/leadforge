"""Snapshot-safe relational export for ``student_public`` bundles.

:func:`to_dataframes_snapshot_safe` projects the full-horizon dict
returned by :func:`leadforge.render.relational.to_dataframes` onto the
shape published in public bundles.  The transformation strips every
known channel through which ``converted_within_90_days`` is
reconstructible from joins (see
``docs/release/v1_current_state_audit.md``):

* ``leads``: drops :data:`BANNED_LEAD_COLUMNS`.
* ``opportunities``: drops :data:`BANNED_OPP_COLUMNS` and filters rows
  per-lead to ``created_at <= lead_created_at + snapshot_day``.
* ``touches`` / ``sessions`` / ``sales_activities``: filtered per-lead
  on their respective timestamp column to the same boundary.  This is
  defence-in-depth — the alpha bundles already pass G4.4 because the
  simulation horizon bounds event timestamps, but the public contract
  is the snapshot window, not the horizon.
* ``customers`` / ``subscriptions`` (:data:`BANNED_TABLES`): omitted
  entirely from the output dict; they exist only for converted leads,
  so their presence is the leak.
* ``accounts`` / ``contacts``: passed through unchanged (firmographic
  / personographic, time-invariant).

The ``research_instructor`` mode keeps using
:func:`leadforge.render.relational.to_dataframes` for the full-horizon
export.  The contract constants live in
:mod:`leadforge.validation.leakage_probes` (validator owns the
definition of "leakage"); this module re-exports them for ergonomics.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    SNAPSHOT_FILTERED_TABLES,
)

__all__ = [
    "BANNED_LEAD_COLUMNS",
    "BANNED_OPP_COLUMNS",
    "BANNED_TABLES",
    "SNAPSHOT_FILTERED_TABLES",
    "to_dataframes_snapshot_safe",
]

_ANCHOR_COL = "_lead_anchor_ts"


def to_dataframes_snapshot_safe(
    dfs: Mapping[str, pd.DataFrame],
    *,
    snapshot_day: int,
) -> dict[str, pd.DataFrame]:
    """Project the full-horizon relational dict onto the snapshot-safe form.

    Args:
        dfs: Output of :func:`leadforge.render.relational.to_dataframes`.
            Must contain ``leads``; other tables are optional and
            missing keys are silently skipped.  Input frames are never
            mutated.
        snapshot_day: Number of days after ``lead_created_at`` beyond
            which event rows are dropped.  This is independent of
            ``label_window_days`` (which gates the task splits).

    Returns:
        A new dict containing — in canonical order — ``accounts``,
        ``contacts``, ``leads``, ``touches``, ``sessions``,
        ``sales_activities``, ``opportunities``.  ``customers`` and
        ``subscriptions`` are absent.

    Raises:
        ValueError: if ``snapshot_day`` is negative, ``leads`` is
            absent, ``leads`` lacks the anchor columns, or
            ``leads.lead_id`` is not unique.
    """
    if snapshot_day < 0:
        raise ValueError(f"snapshot_day must be non-negative, got {snapshot_day}")
    if "leads" not in dfs:
        raise ValueError("dfs must contain a 'leads' frame")

    out: dict[str, pd.DataFrame] = {}

    for name in ("accounts", "contacts"):
        if name in dfs:
            out[name] = dfs[name]

    leads = _drop_columns(dfs["leads"], BANNED_LEAD_COLUMNS)
    out["leads"] = leads
    anchor = _build_anchor(leads)
    horizon = pd.Timedelta(days=snapshot_day)

    for name, ts_col in SNAPSHOT_FILTERED_TABLES:
        if name not in dfs:
            continue
        df = dfs[name]
        if name == "opportunities":
            df = _drop_columns(df, BANNED_OPP_COLUMNS)
        out[name] = _filter_to_snapshot_window(df, anchor, ts_col, horizon)

    return out


def _drop_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    cols_to_drop = [c for c in columns if c in df.columns]
    if not cols_to_drop:
        return df
    return df.drop(columns=cols_to_drop)


def _build_anchor(leads: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ("lead_id", "lead_created_at") if c not in leads.columns]
    if missing:
        raise ValueError(f"leads is missing required columns: {missing}")
    # Duplicate lead_ids would broadcast in the per-lead merge below and
    # silently inflate event-table row counts.  Match the same invariant
    # asserted by ``deterministic_relational_reconstruction``.
    if not leads["lead_id"].is_unique:
        raise ValueError("leads.lead_id must be unique")
    anchor = leads[["lead_id", "lead_created_at"]].rename(columns={"lead_created_at": _ANCHOR_COL})
    anchor[_ANCHOR_COL] = pd.to_datetime(anchor[_ANCHOR_COL], errors="coerce")
    # NaT here would silently drop every event for the affected leads via
    # the ``ts <= NaT`` -> NaN -> fillna(False) path downstream — exactly
    # the kind of silent data-quality erosion a public-bundle exporter
    # must refuse to ship.
    nat_mask = anchor[_ANCHOR_COL].isna()
    if nat_mask.any():
        sample = anchor.loc[nat_mask, "lead_id"].head(5).tolist()
        raise ValueError(
            f"leads.lead_created_at has {int(nat_mask.sum())} unparseable / null "
            f"value(s); offending lead_id sample: {sample}"
        )
    return anchor


def _filter_to_snapshot_window(
    events: pd.DataFrame,
    anchor: pd.DataFrame,
    ts_col: str,
    horizon: pd.Timedelta,
) -> pd.DataFrame:
    if len(events) == 0:
        return events
    merged = events.merge(anchor, on="lead_id", how="left")
    ts = pd.to_datetime(merged[ts_col])
    cutoff = merged[_ANCHOR_COL] + horizon
    keep = (ts <= cutoff).fillna(False).to_numpy()
    return events.loc[keep].reset_index(drop=True)

"""Tests for ``leadforge/render/relational_snapshot_safe.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from leadforge.schemes.lead_scoring.render.relational_snapshot_safe import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    SNAPSHOT_FILTERED_TABLES,
    to_dataframes_snapshot_safe,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
#
# Two leads, anchored on the same date.  Events span 0d, 5d, 12d, 35d after
# anchor — a snapshot_day=10 must keep day-0 / day-5 and drop day-12 / day-35.
# ---------------------------------------------------------------------------

ANCHOR = pd.Timestamp("2026-01-01")


def _ts(offset_days: int) -> str:
    return (ANCHOR + pd.Timedelta(days=offset_days)).isoformat()


def _full_horizon_dict() -> dict[str, pd.DataFrame]:
    accounts = pd.DataFrame(
        [
            {"account_id": "acct_1", "industry": "saas"},
            {"account_id": "acct_2", "industry": "saas"},
        ]
    )
    contacts = pd.DataFrame(
        [
            {"contact_id": "c_1", "account_id": "acct_1", "seniority": "vp"},
            {"contact_id": "c_2", "account_id": "acct_2", "seniority": "manager"},
        ]
    )
    leads = pd.DataFrame(
        [
            {
                "lead_id": "lead_1",
                "account_id": "acct_1",
                "contact_id": "c_1",
                "lead_created_at": ANCHOR.isoformat(),
                "current_stage": "closed_won",
                "is_sql": True,
                "converted_within_90_days": True,
                "conversion_timestamp": _ts(40),
            },
            {
                "lead_id": "lead_2",
                "account_id": "acct_2",
                "contact_id": "c_2",
                "lead_created_at": ANCHOR.isoformat(),
                "current_stage": "discovery",
                "is_sql": False,
                "converted_within_90_days": False,
                "conversion_timestamp": None,
            },
        ]
    )
    touches = pd.DataFrame(
        [
            {"touch_id": "t_1", "lead_id": "lead_1", "touch_timestamp": _ts(0)},
            {"touch_id": "t_2", "lead_id": "lead_1", "touch_timestamp": _ts(5)},
            {"touch_id": "t_3", "lead_id": "lead_1", "touch_timestamp": _ts(12)},
            {"touch_id": "t_4", "lead_id": "lead_2", "touch_timestamp": _ts(35)},
        ]
    )
    sessions = pd.DataFrame(
        [
            {"session_id": "s_1", "lead_id": "lead_1", "session_timestamp": _ts(0)},
            {"session_id": "s_2", "lead_id": "lead_1", "session_timestamp": _ts(20)},
        ]
    )
    sales_activities = pd.DataFrame(
        [
            {"activity_id": "a_1", "lead_id": "lead_1", "activity_timestamp": _ts(7)},
            {"activity_id": "a_2", "lead_id": "lead_2", "activity_timestamp": _ts(15)},
        ]
    )
    opportunities = pd.DataFrame(
        [
            {
                "opportunity_id": "opp_1",
                "lead_id": "lead_1",
                "created_at": _ts(8),
                "stage": "closed_won",
                "estimated_acv": 50_000,
                "close_outcome": "closed_won",
                "closed_at": _ts(40),
            },
            {
                "opportunity_id": "opp_2",
                "lead_id": "lead_1",
                "created_at": _ts(30),
                "stage": "negotiation",
                "estimated_acv": 60_000,
                "close_outcome": None,
                "closed_at": None,
            },
        ]
    )
    customers = pd.DataFrame(
        [
            {
                "customer_id": "cu_1",
                "opportunity_id": "opp_1",
                "account_id": "acct_1",
                "customer_start_at": _ts(40),
            }
        ]
    )
    subscriptions = pd.DataFrame(
        [
            {
                "subscription_id": "sub_1",
                "customer_id": "cu_1",
                "plan_name": "starter",
                "subscription_start_at": _ts(40),
                "subscription_status": "active",
            }
        ]
    )
    return {
        "accounts": accounts,
        "contacts": contacts,
        "leads": leads,
        "touches": touches,
        "sessions": sessions,
        "sales_activities": sales_activities,
        "opportunities": opportunities,
        "customers": customers,
        "subscriptions": subscriptions,
    }


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


def test_drops_banned_columns_from_leads_and_opportunities() -> None:
    out = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    for col in BANNED_LEAD_COLUMNS:
        assert col not in out["leads"].columns, f"leads should not contain banned column {col}"
    for col in BANNED_OPP_COLUMNS:
        assert col not in out["opportunities"].columns, (
            f"opportunities should not contain banned column {col}"
        )


def test_omits_banned_tables_entirely() -> None:
    out = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    for name in BANNED_TABLES:
        assert name not in out, f"output dict should not contain banned table {name}"


def test_event_tables_filtered_to_snapshot_window() -> None:
    out = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)

    # touches: kept day-0 & day-5 (lead_1); dropped day-12 (lead_1) and day-35 (lead_2).
    assert sorted(out["touches"]["touch_id"]) == ["t_1", "t_2"]
    # sessions: kept day-0; dropped day-20.
    assert sorted(out["sessions"]["session_id"]) == ["s_1"]
    # sales_activities: kept day-7 (lead_1); dropped day-15 (lead_2).
    assert sorted(out["sales_activities"]["activity_id"]) == ["a_1"]
    # opportunities: kept day-8 (lead_1); dropped day-30 (lead_1).
    assert sorted(out["opportunities"]["opportunity_id"]) == ["opp_1"]


def test_snapshot_window_invariant_holds_per_lead() -> None:
    out = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    leads_anchor = out["leads"].set_index("lead_id")["lead_created_at"].apply(pd.Timestamp)
    horizon = pd.Timedelta(days=10)

    for name, ts_col in SNAPSHOT_FILTERED_TABLES:
        df = out[name]
        if df.empty:
            continue
        for _, row in df.iterrows():
            anchor = leads_anchor[row["lead_id"]]
            assert pd.Timestamp(row[ts_col]) <= anchor + horizon, (
                f"{name}.{ts_col} for lead {row['lead_id']} exceeds window"
            )


def test_accounts_and_contacts_pass_through_unchanged() -> None:
    src = _full_horizon_dict()
    out = to_dataframes_snapshot_safe(src, snapshot_day=10)
    pd.testing.assert_frame_equal(out["accounts"], src["accounts"])
    pd.testing.assert_frame_equal(out["contacts"], src["contacts"])


def test_idempotent_on_already_safe_input() -> None:
    once = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    twice = to_dataframes_snapshot_safe(once, snapshot_day=10)
    assert set(once.keys()) == set(twice.keys())
    for name in once:
        pd.testing.assert_frame_equal(once[name], twice[name])


def test_deterministic_across_two_calls() -> None:
    a = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    b = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    assert set(a.keys()) == set(b.keys())
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_does_not_mutate_input_frames() -> None:
    src = _full_horizon_dict()
    leads_before = src["leads"].copy(deep=True)
    opps_before = src["opportunities"].copy(deep=True)
    touches_before = src["touches"].copy(deep=True)

    to_dataframes_snapshot_safe(src, snapshot_day=10)

    pd.testing.assert_frame_equal(src["leads"], leads_before)
    pd.testing.assert_frame_equal(src["opportunities"], opps_before)
    pd.testing.assert_frame_equal(src["touches"], touches_before)


def test_canonical_output_table_order() -> None:
    out = to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=10)
    assert list(out.keys()) == [
        "accounts",
        "contacts",
        "leads",
        "touches",
        "sessions",
        "sales_activities",
        "opportunities",
    ]


def test_handles_missing_optional_tables() -> None:
    src = _full_horizon_dict()
    minimal = {"leads": src["leads"], "opportunities": src["opportunities"]}
    out = to_dataframes_snapshot_safe(minimal, snapshot_day=10)
    assert "leads" in out
    assert "opportunities" in out
    assert "touches" not in out
    assert "accounts" not in out


def test_zero_snapshot_day_keeps_only_anchor_day_events() -> None:
    src = _full_horizon_dict()
    out = to_dataframes_snapshot_safe(src, snapshot_day=0)
    # Only the day-0 touches and sessions survive.
    assert sorted(out["touches"]["touch_id"]) == ["t_1"]
    assert sorted(out["sessions"]["session_id"]) == ["s_1"]
    # No sales_activities or opportunities at day-0.
    assert out["sales_activities"].empty
    assert out["opportunities"].empty


def test_negative_snapshot_day_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        to_dataframes_snapshot_safe(_full_horizon_dict(), snapshot_day=-1)


def test_missing_leads_raises() -> None:
    with pytest.raises(ValueError, match="must contain a 'leads'"):
        to_dataframes_snapshot_safe({}, snapshot_day=10)


def test_leads_missing_anchor_columns_raises() -> None:
    bad = {"leads": pd.DataFrame([{"lead_id": "lead_1"}])}
    with pytest.raises(ValueError, match="missing required columns"):
        to_dataframes_snapshot_safe(bad, snapshot_day=10)


def test_duplicate_lead_id_raises() -> None:
    """Per-lead snapshot filter would broadcast on duplicates; matches the
    same invariant asserted by ``deterministic_relational_reconstruction``."""
    src = _full_horizon_dict()
    src["leads"] = pd.concat([src["leads"], src["leads"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="lead_id must be unique"):
        to_dataframes_snapshot_safe(src, snapshot_day=10)


def test_nat_lead_created_at_raises() -> None:
    """A NaT/null anchor would silently drop every event for the affected
    lead via ``ts <= NaT`` -> NaN -> fillna(False).  Must raise instead."""
    src = _full_horizon_dict()
    src["leads"] = src["leads"].copy()
    src["leads"].loc[0, "lead_created_at"] = None
    with pytest.raises(ValueError, match="unparseable / null"):
        to_dataframes_snapshot_safe(src, snapshot_day=10)


def test_unparseable_lead_created_at_raises() -> None:
    """Garbage anchor strings would coerce to NaT and silently misbehave."""
    src = _full_horizon_dict()
    src["leads"] = src["leads"].copy()
    src["leads"].loc[0, "lead_created_at"] = "not-a-date"
    with pytest.raises(ValueError, match="unparseable / null"):
        to_dataframes_snapshot_safe(src, snapshot_day=10)

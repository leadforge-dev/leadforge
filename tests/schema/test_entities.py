"""Tests for leadforge.schema.entities — row contracts and empty DataFrames."""

from pathlib import Path

import pandas as pd
import pytest

from leadforge.schema.entities import (
    ALL_ROW_TYPES,
    TABLE_NAMES,
    AccountRow,
    ContactRow,
    LeadRow,
    SessionRow,
    TouchRow,
)
from leadforge.schema.tables import read_parquet, write_parquet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account() -> AccountRow:
    return AccountRow(
        account_id="acct_000001",
        company_name="Acme Corp",
        industry="manufacturing",
        region="US-East",
        employee_band="100-250",
        estimated_revenue_band="10M-50M",
        process_maturity_band="medium",
        created_at="2024-01-15",
    )


def _make_contact() -> ContactRow:
    return ContactRow(
        contact_id="cnt_000001",
        account_id="acct_000001",
        job_title="VP Finance",
        role_function="finance",
        seniority="vp",
        buyer_role="economic_buyer",
        email_domain_type="corporate",
        created_at="2024-01-20",
    )


def _make_lead() -> LeadRow:
    return LeadRow(
        lead_id="lead_000001",
        contact_id="cnt_000001",
        account_id="acct_000001",
        lead_created_at="2024-02-01",
        lead_source="inbound_form",
        first_touch_channel="organic_search",
        current_stage="mql",
        owner_rep_id="rep_000001",
        is_mql=True,
        is_sql=False,
        converted_within_90_days=False,
        conversion_timestamp=None,
    )


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_account_to_dict_contains_all_columns() -> None:
    row = _make_account()
    d = row.to_dict()
    assert set(d.keys()) == set(AccountRow.DTYPE_MAP.keys())


def test_lead_to_dict_nullable_is_none() -> None:
    row = _make_lead()
    assert row.to_dict()["conversion_timestamp"] is None


def test_touch_to_dict_nullable_campaign_id() -> None:
    row = TouchRow(
        touch_id="touch_000001",
        lead_id="lead_000001",
        touch_timestamp="2024-02-05T10:00:00",
        touch_type="email",
        touch_channel="outbound",
        touch_direction="outbound",
        campaign_id=None,
    )
    assert row.to_dict()["campaign_id"] is None


# ---------------------------------------------------------------------------
# empty_dataframe — columns and dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_ROW_TYPES)
def test_empty_dataframe_has_correct_columns(cls: type) -> None:
    df = cls.empty_dataframe()  # type: ignore[attr-defined]
    assert list(df.columns) == list(cls.DTYPE_MAP.keys())  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", ALL_ROW_TYPES)
def test_empty_dataframe_has_zero_rows(cls: type) -> None:
    assert len(cls.empty_dataframe()) == 0  # type: ignore[attr-defined]


def test_lead_empty_dataframe_boolean_columns() -> None:
    df = LeadRow.empty_dataframe()
    assert str(df["is_mql"].dtype) == "boolean"
    assert str(df["converted_within_90_days"].dtype) == "boolean"


def test_session_empty_dataframe_int_columns() -> None:
    df = SessionRow.empty_dataframe()
    assert str(df["page_views"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_ROW_TYPES)
def test_empty_dataframe_parquet_roundtrip(cls: type, tmp_path: Path) -> None:
    df = cls.empty_dataframe()  # type: ignore[attr-defined]
    path = tmp_path / f"{cls.TABLE_NAME}.parquet"  # type: ignore[attr-defined]
    write_parquet(df, path)
    restored = read_parquet(path)
    assert list(restored.columns) == list(df.columns)
    assert len(restored) == 0


def test_lead_rows_parquet_roundtrip(tmp_path: Path) -> None:
    lead = _make_lead()
    df = pd.DataFrame([lead.to_dict()])
    # cast to declared dtypes
    for col, dtype in LeadRow.DTYPE_MAP.items():
        df[col] = df[col].astype(dtype)
    path = tmp_path / "leads.parquet"
    write_parquet(df, path)
    restored = read_parquet(path)
    assert restored["lead_id"].iloc[0] == "lead_000001"
    assert bool(restored["is_mql"].iloc[0]) is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_row_types_covers_nine_tables() -> None:
    assert len(ALL_ROW_TYPES) == 9


def test_table_names_unique() -> None:
    assert len(set(TABLE_NAMES)) == len(TABLE_NAMES)


def test_table_names_expected_values() -> None:
    expected = {
        "accounts",
        "contacts",
        "leads",
        "touches",
        "sessions",
        "sales_activities",
        "opportunities",
        "customers",
        "subscriptions",
    }
    assert set(TABLE_NAMES) == expected

"""Tests for the lifecycle (b2b_saas_ltv_v1) entity rows and registries.

Covers the new lifecycle entity contracts added in LTV-Pb and asserts that the
lead-scoring catalog (ALL_ROW_TYPES / TABLE_NAMES / ALL_CONSTRAINTS) is left
completely unchanged.
"""

from pathlib import Path

import pandas as pd
import pytest

from leadforge.core.ids import ID_PREFIXES, make_id
from leadforge.schema.tables import read_parquet, write_parquet
from leadforge.schemes.lead_scoring.entities import ALL_ROW_TYPES, TABLE_NAMES, AccountRow
from leadforge.schemes.lead_scoring.relationships import ALL_CONSTRAINTS, FKConstraint
from leadforge.schemes.lifecycle.entities import (
    LIFECYCLE_ROW_TYPES,
    LIFECYCLE_TABLE_NAMES,
    CustomerLifecycleRow,
    HealthSignalRow,
    InvoiceRow,
    SubscriptionEventRow,
    SubscriptionLifecycleRow,
)
from leadforge.schemes.lifecycle.relationships import LIFECYCLE_CONSTRAINTS

# ---------------------------------------------------------------------------
# Row factories
# ---------------------------------------------------------------------------


def _make_customer() -> CustomerLifecycleRow:
    return CustomerLifecycleRow(
        customer_id="cust_000001",
        account_id="acct_000001",
        customer_start_at="2024-03-01",
        initial_plan="growth",
        initial_mrr=4000,
        contract_term_months=12,
        csm_rep_id="rep_000003",
    )


def _make_subscription() -> SubscriptionLifecycleRow:
    return SubscriptionLifecycleRow(
        subscription_id="sub_000001",
        customer_id="cust_000001",
        plan_name="growth",
        subscription_status="active",
        subscription_start_at="2024-03-01",
        current_mrr=6000,
        contract_term_months=12,
        renewal_count=1,
        expansion_count=1,
    )


def _make_subscription_event() -> SubscriptionEventRow:
    return SubscriptionEventRow(
        event_id="subev_000001",
        subscription_id="sub_000001",
        customer_id="cust_000001",
        event_timestamp="2024-09-01",
        event_type="expansion",
        mrr_before=4000,
        mrr_after=6000,
    )


def _make_health_signal() -> HealthSignalRow:
    return HealthSignalRow(
        signal_id="hsig_000001",
        customer_id="cust_000001",
        period_start="2024-09-02",
        active_users=42,
        feature_depth_score=0.61,
        support_tickets=2,
    )


def _make_invoice() -> InvoiceRow:
    return InvoiceRow(
        invoice_id="inv_000001",
        customer_id="cust_000001",
        invoice_date="2024-09-01",
        amount_usd=6000,
        payment_status="paid",
    )


_FACTORIES = {
    CustomerLifecycleRow: _make_customer,
    SubscriptionLifecycleRow: _make_subscription,
    SubscriptionEventRow: _make_subscription_event,
    HealthSignalRow: _make_health_signal,
    InvoiceRow: _make_invoice,
}

# Concrete lifecycle row classes (AccountRow is shared/tested elsewhere).
_LIFECYCLE_ONLY = tuple(_FACTORIES.keys())


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_to_dict_contains_all_columns(cls: type) -> None:
    row = _FACTORIES[cls]()
    assert set(row.to_dict().keys()) == set(cls.DTYPE_MAP.keys())  # type: ignore[attr-defined]


def test_customer_opportunity_id_defaults_none() -> None:
    assert _make_customer().to_dict()["opportunity_id"] is None


def test_subscription_terminal_fields_default_none() -> None:
    d = _make_subscription().to_dict()
    assert d["subscription_end_at"] is None
    assert d["churn_at"] is None
    assert d["churn_reason"] is None


def test_event_contract_term_new_defaults_none() -> None:
    assert _make_subscription_event().to_dict()["contract_term_months_new"] is None


def test_health_signal_nps_defaults_none() -> None:
    assert _make_health_signal().to_dict()["nps_score"] is None


# ---------------------------------------------------------------------------
# empty_dataframe — columns, dtypes, round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_empty_dataframe_has_correct_columns(cls: type) -> None:
    df = cls.empty_dataframe()  # type: ignore[attr-defined]
    assert list(df.columns) == list(cls.DTYPE_MAP.keys())  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_empty_dataframe_has_zero_rows(cls: type) -> None:
    assert len(cls.empty_dataframe()) == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_empty_dataframe_dtypes_match_spec(cls: type) -> None:
    df = cls.empty_dataframe()  # type: ignore[attr-defined]
    for col, dtype in cls.DTYPE_MAP.items():  # type: ignore[attr-defined]
        assert str(df[col].dtype) == dtype


def test_health_signal_feature_depth_is_float() -> None:
    assert str(HealthSignalRow.empty_dataframe()["feature_depth_score"].dtype) == "Float64"


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_empty_dataframe_parquet_roundtrip(cls: type, tmp_path: Path) -> None:
    df = cls.empty_dataframe()  # type: ignore[attr-defined]
    path = tmp_path / f"{cls.TABLE_NAME}.parquet"  # type: ignore[attr-defined]
    write_parquet(df, path)
    restored = read_parquet(path)
    assert list(restored.columns) == list(df.columns)
    assert len(restored) == 0


@pytest.mark.parametrize("cls", _LIFECYCLE_ONLY)
def test_populated_row_parquet_roundtrip(cls: type, tmp_path: Path) -> None:
    row = _FACTORIES[cls]()
    df = pd.DataFrame([row.to_dict()])
    for col, dtype in cls.DTYPE_MAP.items():  # type: ignore[attr-defined]
        df[col] = df[col].astype(dtype)
    path = tmp_path / f"{cls.TABLE_NAME}.parquet"  # type: ignore[attr-defined]
    write_parquet(df, path)
    restored = read_parquet(path)
    assert list(restored.columns) == list(cls.DTYPE_MAP.keys())  # type: ignore[attr-defined]
    assert len(restored) == 1


# ---------------------------------------------------------------------------
# Lifecycle registry
# ---------------------------------------------------------------------------


def test_lifecycle_row_types_count() -> None:
    # accounts + customers + subscriptions + subscription_events
    # + health_signals + invoices
    assert len(LIFECYCLE_ROW_TYPES) == 6


def test_lifecycle_table_names_expected() -> None:
    assert set(LIFECYCLE_TABLE_NAMES) == {
        "accounts",
        "customers",
        "subscriptions",
        "subscription_events",
        "health_signals",
        "invoices",
    }


def test_lifecycle_table_names_unique() -> None:
    assert len(set(LIFECYCLE_TABLE_NAMES)) == len(LIFECYCLE_TABLE_NAMES)


def test_lifecycle_shares_account_row() -> None:
    assert AccountRow in LIFECYCLE_ROW_TYPES


# ---------------------------------------------------------------------------
# Lead-scoring catalog is unchanged (guard)
# ---------------------------------------------------------------------------


def test_lead_scoring_catalog_unchanged() -> None:
    assert len(ALL_ROW_TYPES) == 9
    assert set(TABLE_NAMES) == {
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


def test_lifecycle_only_rows_absent_from_lead_scoring_catalog() -> None:
    for cls in _LIFECYCLE_ONLY:
        assert cls not in ALL_ROW_TYPES


# ---------------------------------------------------------------------------
# Lifecycle FK constraints
# ---------------------------------------------------------------------------


def test_lifecycle_constraints_count() -> None:
    assert len(LIFECYCLE_CONSTRAINTS) == 6


def test_lifecycle_constraints_are_fk_instances() -> None:
    for c in LIFECYCLE_CONSTRAINTS:
        assert isinstance(c, FKConstraint)


def test_lifecycle_constraints_reference_known_tables() -> None:
    names = set(LIFECYCLE_TABLE_NAMES)
    for c in LIFECYCLE_CONSTRAINTS:
        assert c.child_table in names, c
        assert c.parent_table in names, c


def test_lifecycle_has_no_customer_opportunity_fk() -> None:
    # Independent generation: lifecycle customers link to accounts, not opps.
    assert not any(
        c.child_table == "customers" and c.parent_table == "opportunities"
        for c in LIFECYCLE_CONSTRAINTS
    )


def test_lead_scoring_constraints_unchanged() -> None:
    assert len(ALL_CONSTRAINTS) == 10


# ---------------------------------------------------------------------------
# ID prefixes
# ---------------------------------------------------------------------------


def test_lifecycle_id_prefixes_present() -> None:
    assert ID_PREFIXES["subscription_event"] == "subev"
    assert ID_PREFIXES["health_signal"] == "hsig"
    assert ID_PREFIXES["invoice"] == "inv"


def test_lifecycle_id_format() -> None:
    assert make_id(ID_PREFIXES["subscription_event"], 1) == "subev_000001"
    assert make_id(ID_PREFIXES["health_signal"], 42) == "hsig_000042"
    assert make_id(ID_PREFIXES["invoice"], 7) == "inv_000007"

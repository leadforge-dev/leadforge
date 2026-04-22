"""Typed row contracts for all v1 relational tables.

Each class represents one row in a Parquet table.  Fields map directly to
the column specifications in §16 of the architecture spec.  Optional columns
(nullable in the output) use ``... | None`` typing.

All row classes expose:

- ``TABLE_NAME`` — the canonical Parquet table name (no extension).
- ``DTYPE_MAP`` — ``{column: pandas-dtype-string}`` used to build empty
  DataFrames with the right schema.
- ``to_dict()`` — returns a plain ``dict`` suitable for ``pd.DataFrame([...])``
  or JSON serialization.
- ``empty_dataframe()`` — class method returning a zero-row ``pd.DataFrame``
  with the correct columns and nullable dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import pandas as pd


def _empty_df(dtype_map: dict[str, str]) -> pd.DataFrame:
    """Return a zero-row DataFrame with columns ordered as *dtype_map*."""
    return pd.DataFrame({col: pd.array([], dtype=dtype) for col, dtype in dtype_map.items()})


# ---------------------------------------------------------------------------
# accounts
# ---------------------------------------------------------------------------


@dataclass
class AccountRow:
    """One row in the ``accounts`` table."""

    TABLE_NAME: ClassVar[str] = "accounts"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "account_id": "string",
        "company_name": "string",
        "industry": "string",
        "region": "string",
        "employee_band": "string",
        "estimated_revenue_band": "string",
        "process_maturity_band": "string",
        "created_at": "string",
    }

    account_id: str
    company_name: str
    industry: str
    region: str
    employee_band: str
    estimated_revenue_band: str
    process_maturity_band: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------------


@dataclass
class ContactRow:
    """One row in the ``contacts`` table."""

    TABLE_NAME: ClassVar[str] = "contacts"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "contact_id": "string",
        "account_id": "string",
        "job_title": "string",
        "role_function": "string",
        "seniority": "string",
        "buyer_role": "string",
        "email_domain_type": "string",
        "created_at": "string",
    }

    contact_id: str
    account_id: str
    job_title: str
    role_function: str
    seniority: str
    buyer_role: str
    email_domain_type: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# leads
# ---------------------------------------------------------------------------


@dataclass
class LeadRow:
    """One row in the ``leads`` table."""

    TABLE_NAME: ClassVar[str] = "leads"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "lead_id": "string",
        "contact_id": "string",
        "account_id": "string",
        "lead_created_at": "string",
        "lead_source": "string",
        "first_touch_channel": "string",
        "current_stage": "string",
        "owner_rep_id": "string",
        "is_mql": "boolean",
        "is_sql": "boolean",
        "converted_within_90_days": "boolean",
        "conversion_timestamp": "string",
    }

    lead_id: str
    contact_id: str
    account_id: str
    lead_created_at: str
    lead_source: str
    first_touch_channel: str
    current_stage: str
    owner_rep_id: str
    is_mql: bool
    is_sql: bool
    converted_within_90_days: bool
    conversion_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# touches
# ---------------------------------------------------------------------------


@dataclass
class TouchRow:
    """One row in the ``touches`` table."""

    TABLE_NAME: ClassVar[str] = "touches"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "touch_id": "string",
        "lead_id": "string",
        "touch_timestamp": "string",
        "touch_type": "string",
        "touch_channel": "string",
        "touch_direction": "string",
        "campaign_id": "string",
    }

    touch_id: str
    lead_id: str
    touch_timestamp: str
    touch_type: str
    touch_channel: str
    touch_direction: str
    campaign_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


@dataclass
class SessionRow:
    """One row in the ``sessions`` table."""

    TABLE_NAME: ClassVar[str] = "sessions"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "session_id": "string",
        "lead_id": "string",
        "session_timestamp": "string",
        "session_type": "string",
        "page_views": "Int64",
        "pricing_page_views": "Int64",
        "demo_page_views": "Int64",
        "session_duration_seconds": "Int64",
    }

    session_id: str
    lead_id: str
    session_timestamp: str
    session_type: str
    page_views: int
    pricing_page_views: int
    demo_page_views: int
    session_duration_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# sales_activities
# ---------------------------------------------------------------------------


@dataclass
class SalesActivityRow:
    """One row in the ``sales_activities`` table."""

    TABLE_NAME: ClassVar[str] = "sales_activities"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "activity_id": "string",
        "lead_id": "string",
        "rep_id": "string",
        "activity_timestamp": "string",
        "activity_type": "string",
        "activity_outcome": "string",
    }

    activity_id: str
    lead_id: str
    rep_id: str
    activity_timestamp: str
    activity_type: str
    activity_outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# opportunities
# ---------------------------------------------------------------------------


@dataclass
class OpportunityRow:
    """One row in the ``opportunities`` table."""

    TABLE_NAME: ClassVar[str] = "opportunities"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "opportunity_id": "string",
        "lead_id": "string",
        "created_at": "string",
        "stage": "string",
        "estimated_acv": "Int64",
        "close_outcome": "string",
        "closed_at": "string",
    }

    opportunity_id: str
    lead_id: str
    created_at: str
    stage: str
    estimated_acv: int
    close_outcome: str | None = None
    closed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# customers
# ---------------------------------------------------------------------------


@dataclass
class CustomerRow:
    """One row in the ``customers`` table."""

    TABLE_NAME: ClassVar[str] = "customers"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "customer_id": "string",
        "opportunity_id": "string",
        "account_id": "string",
        "customer_start_at": "string",
    }

    customer_id: str
    opportunity_id: str
    account_id: str
    customer_start_at: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# subscriptions
# ---------------------------------------------------------------------------


@dataclass
class SubscriptionRow:
    """One row in the ``subscriptions`` table."""

    TABLE_NAME: ClassVar[str] = "subscriptions"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "subscription_id": "string",
        "customer_id": "string",
        "plan_name": "string",
        "subscription_start_at": "string",
        "subscription_status": "string",
    }

    subscription_id: str
    customer_id: str
    plan_name: str
    subscription_start_at: str
    subscription_status: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_ROW_TYPES: tuple[type, ...] = (
    AccountRow,
    ContactRow,
    LeadRow,
    TouchRow,
    SessionRow,
    SalesActivityRow,
    OpportunityRow,
    CustomerRow,
    SubscriptionRow,
)

TABLE_NAMES: tuple[str, ...] = tuple(cls.TABLE_NAME for cls in ALL_ROW_TYPES)  # type: ignore[attr-defined]

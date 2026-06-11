"""Lead-scoring (``b2b_saas_procurement_v1``) entity row contracts.

The 8 lead-scoring-specific entity row classes and the ``ALL_ROW_TYPES`` /
``TABLE_NAMES`` catalog live here.  They are distinct from the lifecycle rows
in :mod:`leadforge.schemes.lifecycle.entities`.

``AccountRow`` and the shared primitives (``EntityRowProtocol``,
``make_empty_dataframe``) remain in :mod:`leadforge.schema.entities`; both
schemes import them from there.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import pandas as pd

from leadforge.schema.entities import (
    AccountRow,
    EntityRowProtocol,
    make_empty_dataframe,
)

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
        return make_empty_dataframe(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# leads
# ---------------------------------------------------------------------------


@dataclass
class LeadRow:
    """One row in the ``leads`` table.

    .. note:: The ``converted_within_90_days`` field name is retained for
       schema stability, but its value is derived using
       ``GenerationConfig.label_window_days`` (which defaults to 90).  A
       lead is marked ``True`` only if its conversion event occurred before
       ``label_window_days`` from lead creation — **not** necessarily within
       the full ``horizon_days`` simulation window.

       Consequently, ``conversion_timestamp`` may be set (non-``None``)
       while ``converted_within_90_days`` is ``False``, indicating the lead
       converted after the label observation window closed.
    """

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
        "is_sql": "boolean",
        "converted_within_90_days": "boolean",
        "conversion_timestamp": "string",
    }

    # ``is_mql`` was removed in bundle schema v3 (issue #57).  Every lead
    # is initialised at MQL stage in ``simulation/population.py``, so the
    # field was constant ``True`` and zero-variance across all bundles.

    lead_id: str
    contact_id: str
    account_id: str
    lead_created_at: str
    lead_source: str
    first_touch_channel: str
    current_stage: str
    owner_rep_id: str
    is_sql: bool
    converted_within_90_days: bool
    conversion_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


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
        return make_empty_dataframe(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# Lead-scoring catalog
# ---------------------------------------------------------------------------

ALL_ROW_TYPES: tuple[type[EntityRowProtocol], ...] = (
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

TABLE_NAMES: tuple[str, ...] = tuple(cls.TABLE_NAME for cls in ALL_ROW_TYPES)

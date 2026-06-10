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
from typing import Any, ClassVar, Protocol

import pandas as pd


class EntityRowProtocol(Protocol):
    """Structural protocol shared by all entity row dataclasses.

    Allows typed dispatch in render code without coupling to concrete classes.
    """

    TABLE_NAME: ClassVar[str]
    DTYPE_MAP: ClassVar[dict[str, str]]

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame: ...


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


# ===========================================================================
# Lifecycle entity rows (b2b_saas_ltv_v1 — see docs/ltv/design.md)
# ---------------------------------------------------------------------------
# These rows belong to the *lifecycle* bundle shape only.  They are kept in a
# separate registry (:data:`LIFECYCLE_ROW_TYPES`) and are NOT added to
# :data:`ALL_ROW_TYPES`, so the lead-scoring bundle's table inventory and
# column schemas are completely unchanged.
#
# The lifecycle bundle's ``customers`` and ``subscriptions`` tables are richer
# than the thin lead-scoring :class:`CustomerRow` / :class:`SubscriptionRow`
# (which exist only to record conversion in the procurement world).  Rather
# than extend those classes in place — which would change the lead-scoring
# instructor bundle's parquet schema, since ``to_dict()`` emits every field —
# the lifecycle bundle uses the dedicated :class:`CustomerLifecycleRow` /
# :class:`SubscriptionLifecycleRow` classes below.  Both deliberately reuse the
# logical table names ``customers`` / ``subscriptions``; the two shapes never
# co-occur in one bundle, and the registries that hold them are disjoint.
# ===========================================================================


@dataclass
class CustomerLifecycleRow:
    """One row in the lifecycle ``customers`` table.

    Static, set-at-acquisition attributes of a customer.  ``opportunity_id`` is
    nullable because the lifecycle recipe generates customers **independently**
    (no upstream opportunities table); it is reserved for future chained
    generation from a lead-scoring bundle's converted leads.
    """

    TABLE_NAME: ClassVar[str] = "customers"
    # Column order matches the dataclass field order below; ``opportunity_id``
    # carries a default (nullable) so it must come last in both.
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "customer_id": "string",
        "account_id": "string",
        "customer_start_at": "string",
        "initial_plan": "string",
        "initial_mrr": "Int64",
        "contract_term_months": "Int64",
        "csm_rep_id": "string",
        "opportunity_id": "string",
    }

    customer_id: str
    account_id: str
    customer_start_at: str
    initial_plan: str
    initial_mrr: int
    contract_term_months: int
    csm_rep_id: str
    opportunity_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


@dataclass
class SubscriptionLifecycleRow:
    """One row in the lifecycle ``subscriptions`` table.

    Carries the subscription's terminal/dynamic state as of the end of the
    simulation.  Terminal fields (``subscription_end_at``, ``churn_at``,
    ``churn_reason``) are redacted from ``student_public`` bundles per the
    lifecycle snapshot-safety contract (see ``docs/ltv/design.md`` §5).
    """

    TABLE_NAME: ClassVar[str] = "subscriptions"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "subscription_id": "string",
        "customer_id": "string",
        "plan_name": "string",
        "subscription_status": "string",
        "subscription_start_at": "string",
        "current_mrr": "Int64",
        "contract_term_months": "Int64",
        "renewal_count": "Int64",
        "expansion_count": "Int64",
        "subscription_end_at": "string",
        "churn_at": "string",
        "churn_reason": "string",
    }

    subscription_id: str
    customer_id: str
    plan_name: str
    subscription_status: str
    subscription_start_at: str
    current_mrr: int
    contract_term_months: int
    renewal_count: int
    expansion_count: int
    subscription_end_at: str | None = None
    churn_at: str | None = None
    churn_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


@dataclass
class SubscriptionEventRow:
    """One row in the ``subscription_events`` table — a lifecycle state change."""

    TABLE_NAME: ClassVar[str] = "subscription_events"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "event_id": "string",
        "subscription_id": "string",
        "customer_id": "string",
        "event_timestamp": "string",
        "event_type": "string",
        "mrr_before": "Int64",
        "mrr_after": "Int64",
        "contract_term_months_new": "Int64",
    }

    event_id: str
    subscription_id: str
    customer_id: str
    event_timestamp: str
    event_type: str
    mrr_before: int
    mrr_after: int
    contract_term_months_new: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


@dataclass
class HealthSignalRow:
    """One row in the ``health_signals`` table — weekly product-usage telemetry."""

    TABLE_NAME: ClassVar[str] = "health_signals"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "signal_id": "string",
        "customer_id": "string",
        "period_start": "string",
        "active_users": "Int64",
        "feature_depth_score": "Float64",
        "support_tickets": "Int64",
        "nps_score": "Int64",
    }

    signal_id: str
    customer_id: str
    period_start: str
    active_users: int
    feature_depth_score: float
    support_tickets: int
    nps_score: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


@dataclass
class InvoiceRow:
    """One row in the ``invoices`` table — monthly billing; the unit of pLTV value."""

    TABLE_NAME: ClassVar[str] = "invoices"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "invoice_id": "string",
        "customer_id": "string",
        "invoice_date": "string",
        "amount_usd": "Int64",
        "payment_status": "string",
    }

    invoice_id: str
    customer_id: str
    invoice_date: str
    amount_usd: int
    payment_status: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return _empty_df(cls.DTYPE_MAP)


# ---------------------------------------------------------------------------
# Registry
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

# Lifecycle (b2b_saas_ltv_v1) bundle table inventory.  Kept separate from
# ALL_ROW_TYPES so the lead-scoring bundle is unaffected.  AccountRow is shared
# (reused unchanged); customers/subscriptions use the richer lifecycle classes.
LIFECYCLE_ROW_TYPES: tuple[type[EntityRowProtocol], ...] = (
    AccountRow,
    CustomerLifecycleRow,
    SubscriptionLifecycleRow,
    SubscriptionEventRow,
    HealthSignalRow,
    InvoiceRow,
)

LIFECYCLE_TABLE_NAMES: tuple[str, ...] = tuple(cls.TABLE_NAME for cls in LIFECYCLE_ROW_TYPES)

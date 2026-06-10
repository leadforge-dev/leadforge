"""Lifecycle (``b2b_saas_ltv_v1``) entity rows.

These rows belong to the *lifecycle* bundle shape only.  They are kept in this
scheme package (registry :data:`LIFECYCLE_ROW_TYPES`) and are entirely separate
from the lead-scoring catalog, so the lead-scoring bundle's table inventory and
column schemas are unaffected.

The lifecycle bundle's ``customers`` and ``subscriptions`` tables are richer
than the thin lead-scoring :class:`~leadforge.schema.entities.CustomerRow` /
:class:`~leadforge.schema.entities.SubscriptionRow` (which exist only to record
conversion in the procurement world).  Rather than extend those classes in
place — which would change the lead-scoring instructor bundle's parquet schema,
since ``to_dict()`` emits every field — the lifecycle bundle uses the dedicated
:class:`CustomerLifecycleRow` / :class:`SubscriptionLifecycleRow` classes below.
Both deliberately reuse the logical table names ``customers`` / ``subscriptions``;
the two shapes never co-occur in one bundle.

``AccountRow`` is shared across schemes and is imported from the shared schema
package (``accounts`` is the same entity in both the lead-scoring and lifecycle
worlds).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar

import pandas as pd

from leadforge.schema.entities import AccountRow, EntityRowProtocol, _empty_df


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
# Lifecycle bundle table inventory.  AccountRow is shared (reused unchanged);
# customers/subscriptions use the richer lifecycle classes above.
# ---------------------------------------------------------------------------
LIFECYCLE_ROW_TYPES: tuple[type[EntityRowProtocol], ...] = (
    AccountRow,
    CustomerLifecycleRow,
    SubscriptionLifecycleRow,
    SubscriptionEventRow,
    HealthSignalRow,
    InvoiceRow,
)

LIFECYCLE_TABLE_NAMES: tuple[str, ...] = tuple(cls.TABLE_NAME for cls in LIFECYCLE_ROW_TYPES)

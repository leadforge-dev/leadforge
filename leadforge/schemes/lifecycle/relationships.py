"""Lifecycle (``b2b_saas_ltv_v1``) foreign-key constraints.

Kept in the lifecycle scheme package, separate from the lead-scoring
``ALL_CONSTRAINTS``.  Reuses the shared :class:`~leadforge.schema.relationships.FKConstraint`
primitive.  The lifecycle ``customers`` table links only to ``accounts``
(independent generation, no ``opportunities`` table), so there is no
customer→opportunity FK despite the nullable ``opportunity_id`` column being
reserved for future chained generation.
"""

from __future__ import annotations

from leadforge.schema.relationships import FKConstraint

LIFECYCLE_CONSTRAINTS: tuple[FKConstraint, ...] = (
    FKConstraint("customers", "account_id", "accounts", "account_id"),
    FKConstraint("subscriptions", "customer_id", "customers", "customer_id"),
    FKConstraint("subscription_events", "subscription_id", "subscriptions", "subscription_id"),
    FKConstraint("subscription_events", "customer_id", "customers", "customer_id"),
    FKConstraint("health_signals", "customer_id", "customers", "customer_id"),
    FKConstraint("invoices", "customer_id", "customers", "customer_id"),
)

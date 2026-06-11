"""Lead-scoring (``b2b_saas_procurement_v1``) foreign-key constraints.

``ALL_CONSTRAINTS`` lives here; the shared primitives (:class:`FKConstraint`,
:func:`validate_fk`, :class:`FKViolationError`) remain in
:mod:`leadforge.schema.relationships`.
"""

from __future__ import annotations

from leadforge.schema.relationships import FKConstraint

ALL_CONSTRAINTS: tuple[FKConstraint, ...] = (
    FKConstraint("contacts", "account_id", "accounts", "account_id"),
    FKConstraint("leads", "account_id", "accounts", "account_id"),
    FKConstraint("leads", "contact_id", "contacts", "contact_id"),
    FKConstraint("touches", "lead_id", "leads", "lead_id"),
    FKConstraint("sessions", "lead_id", "leads", "lead_id"),
    FKConstraint("sales_activities", "lead_id", "leads", "lead_id"),
    FKConstraint("opportunities", "lead_id", "leads", "lead_id"),
    FKConstraint("customers", "opportunity_id", "opportunities", "opportunity_id"),
    FKConstraint("customers", "account_id", "accounts", "account_id"),
    FKConstraint("subscriptions", "customer_id", "customers", "customer_id"),
)

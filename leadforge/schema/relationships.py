"""Foreign-key relationship definitions and validation helpers.

Describes the canonical FK graph for the v1 relational model and provides
:func:`validate_fk` to assert referential integrity on a collection of rows
before they are written to Parquet.
"""

from __future__ import annotations

from dataclasses import dataclass

from leadforge.core.exceptions import LeadforgeError


class FKViolationError(LeadforgeError):
    """Raised when a foreign-key constraint is violated in synthetic data."""


@dataclass(frozen=True)
class FKConstraint:
    """Describes one foreign-key relationship between two tables."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


# All v1 FK constraints, derived from §9.2 of the architecture spec.
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


def validate_fk(
    child_values: list[str],
    parent_values: set[str],
    constraint: FKConstraint,
) -> None:
    """Assert every value in *child_values* exists in *parent_values*.

    Args:
        child_values: All FK column values from the child table.
        parent_values: All PK values from the parent table.
        constraint: The :class:`FKConstraint` being checked (used in the
            error message).

    Raises:
        FKViolationError: if any child value is absent from *parent_values*.
    """
    orphans = [v for v in child_values if v not in parent_values]
    if orphans:
        sample = orphans[:5]
        raise FKViolationError(
            f"FK violation: {constraint.child_table}.{constraint.child_column} "
            f"→ {constraint.parent_table}.{constraint.parent_column}: "
            f"{len(orphans)} orphan(s), e.g. {sample}"
        )

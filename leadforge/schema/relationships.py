"""Shared FK-constraint primitives.

:class:`FKConstraint`, :class:`FKViolationError`, and :func:`validate_fk`
are scheme-agnostic utilities; both the lead-scoring and lifecycle schemes
build their FK catalogs (``ALL_CONSTRAINTS``, ``LIFECYCLE_CONSTRAINTS``) with
them.

Lead-scoring ``ALL_CONSTRAINTS`` lives in
:mod:`leadforge.schemes.lead_scoring.relationships`.

Lifecycle ``LIFECYCLE_CONSTRAINTS`` lives in
:mod:`leadforge.schemes.lifecycle.relationships`.
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
    _max_sample = 5
    orphan_count = 0
    orphan_sample: list[str] = []
    for v in child_values:
        if v not in parent_values:
            orphan_count += 1
            if len(orphan_sample) < _max_sample:
                orphan_sample.append(v)
    if orphan_count:
        raise FKViolationError(
            f"FK violation: {constraint.child_table}.{constraint.child_column} "
            f"→ {constraint.parent_table}.{constraint.parent_column}: "
            f"{orphan_count} orphan(s), e.g. {orphan_sample}"
        )

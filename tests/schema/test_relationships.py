"""Tests for leadforge.schema.relationships — FK constraints and validation."""

import dataclasses

import pytest

from leadforge.schema.relationships import (
    ALL_CONSTRAINTS,
    FKConstraint,
    FKViolationError,
    validate_fk,
)


def test_all_constraints_count() -> None:
    assert len(ALL_CONSTRAINTS) == 10


def test_all_constraints_are_fk_constraint_instances() -> None:
    for c in ALL_CONSTRAINTS:
        assert isinstance(c, FKConstraint)


def test_contacts_has_account_fk() -> None:
    c = next(c for c in ALL_CONSTRAINTS if c.child_table == "contacts")
    assert c.parent_table == "accounts"
    assert c.parent_column == "account_id"


def test_validate_fk_passes_when_all_present() -> None:
    constraint = FKConstraint("contacts", "account_id", "accounts", "account_id")
    parent_ids = {"acct_000001", "acct_000002"}
    child_values = ["acct_000001", "acct_000002", "acct_000001"]
    validate_fk(child_values, parent_ids, constraint)  # should not raise


def test_validate_fk_raises_on_orphan() -> None:
    constraint = FKConstraint("leads", "account_id", "accounts", "account_id")
    parent_ids = {"acct_000001"}
    child_values = ["acct_000001", "acct_MISSING"]
    with pytest.raises(FKViolationError, match="orphan"):
        validate_fk(child_values, parent_ids, constraint)


def test_validate_fk_error_message_contains_table_names() -> None:
    constraint = FKConstraint("touches", "lead_id", "leads", "lead_id")
    with pytest.raises(FKViolationError, match="touches"):
        validate_fk(["lead_MISSING"], set(), constraint)


def test_validate_fk_empty_child_passes() -> None:
    constraint = FKConstraint("sessions", "lead_id", "leads", "lead_id")
    validate_fk([], {"lead_000001"}, constraint)


def test_fk_constraint_is_frozen() -> None:
    c = FKConstraint("a", "b", "c", "d")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.child_table = "x"  # type: ignore[misc]

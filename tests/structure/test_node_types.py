"""Tests for leadforge.structure.node_types."""

from leadforge.structure.node_types import (
    LEAF_ONLY,
    REQUIRES_PARENT,
    ROOT_ELIGIBLE,
    NodeType,
)


def test_node_type_values_are_strings() -> None:
    for nt in NodeType:
        assert isinstance(nt.value, str)


def test_all_nine_node_types_defined() -> None:
    assert len(NodeType) == 9


def test_root_eligible_and_requires_parent_are_disjoint() -> None:
    assert ROOT_ELIGIBLE.isdisjoint(REQUIRES_PARENT)


def test_leaf_only_is_subset_of_requires_parent() -> None:
    assert LEAF_ONLY <= REQUIRES_PARENT


def test_outcome_is_leaf_only() -> None:
    assert NodeType.OUTCOME in LEAF_ONLY


def test_global_context_is_root_eligible() -> None:
    assert NodeType.GLOBAL_CONTEXT in ROOT_ELIGIBLE


def test_node_type_round_trips_via_value() -> None:
    for nt in NodeType:
        assert NodeType(nt.value) is nt

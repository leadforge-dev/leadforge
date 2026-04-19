"""Tests for leadforge.core.rng."""

import pytest

from leadforge.core.rng import RNGRoot


def test_rng_root_same_seed_same_sequence() -> None:
    """The same seed must always produce the same sequence."""
    r1 = RNGRoot(42).child("accounts")
    r2 = RNGRoot(42).child("accounts")
    assert [r1.random() for _ in range(20)] == [r2.random() for _ in range(20)]


def test_rng_root_different_seeds_different_sequences() -> None:
    r1 = RNGRoot(42).child("accounts")
    r2 = RNGRoot(99).child("accounts")
    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]
    assert seq1 != seq2


def test_named_children_are_independent() -> None:
    """Different child names must yield different sequences from the same root."""
    root = RNGRoot(42)
    r1 = root.child("accounts")
    r2 = root.child("contacts")
    seq1 = [r1.random() for _ in range(10)]
    seq2 = [r2.random() for _ in range(10)]
    assert seq1 != seq2


def test_child_reproducible_across_root_instances() -> None:
    """child() must be deterministic — same root seed → same child sequence."""
    name = "leads"
    seq1 = [RNGRoot(7).child(name).random() for _ in range(15)]
    seq2 = [RNGRoot(7).child(name).random() for _ in range(15)]
    assert seq1 == seq2


def test_rng_root_seed_property() -> None:
    root = RNGRoot(123)
    assert root.seed == 123


def test_rng_root_rejects_non_int_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        RNGRoot(3.14)  # type: ignore[arg-type]


def test_rng_root_rejects_bool_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        RNGRoot(True)  # type: ignore[arg-type]


def test_rng_root_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RNGRoot(-1)


def test_rng_root_repr() -> None:
    assert repr(RNGRoot(42)) == "RNGRoot(seed=42)"

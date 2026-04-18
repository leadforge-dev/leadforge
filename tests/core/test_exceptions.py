"""Tests for the exception hierarchy."""

import pytest

from leadforge.core.exceptions import (
    GraphConstructionError,
    InvalidConfigError,
    InvalidRecipeError,
    LeadforgeError,
    RenderError,
    SimulationError,
    ValidationError,
)


def test_all_exceptions_are_leadforge_errors() -> None:
    for exc_class in (
        InvalidRecipeError,
        InvalidConfigError,
        GraphConstructionError,
        SimulationError,
        RenderError,
        ValidationError,
    ):
        assert issubclass(exc_class, LeadforgeError)


def test_exceptions_are_catchable_as_base() -> None:
    with pytest.raises(LeadforgeError):
        raise InvalidRecipeError("unknown-recipe")


def test_exception_message_preserved() -> None:
    msg = "recipe 'foo' not found"
    exc = InvalidRecipeError(msg)
    assert str(exc) == msg

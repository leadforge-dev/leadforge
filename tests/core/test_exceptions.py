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


def test_invalid_config_error_on_string_count() -> None:
    """__post_init__ must raise InvalidConfigError for non-int count fields."""
    from leadforge.core.exceptions import InvalidConfigError
    from leadforge.core.models import GenerationConfig

    with pytest.raises(InvalidConfigError, match="n_leads"):
        GenerationConfig(n_leads="five_thousand")  # type: ignore[arg-type]


def test_invalid_config_error_on_bool_count() -> None:
    """bool is an int subclass and must be explicitly rejected."""
    from leadforge.core.exceptions import InvalidConfigError
    from leadforge.core.models import GenerationConfig

    with pytest.raises(InvalidConfigError, match="n_accounts"):
        GenerationConfig(n_accounts=True)  # type: ignore[arg-type]


def test_invalid_config_error_on_bad_exposure_mode() -> None:
    """Invalid exposure_mode string must raise InvalidConfigError, not ValueError."""
    from leadforge.core.exceptions import InvalidConfigError
    from leadforge.core.models import GenerationConfig

    with pytest.raises(InvalidConfigError, match="exposure_mode"):
        GenerationConfig(exposure_mode="not_a_mode")  # type: ignore[arg-type]


def test_invalid_config_error_on_bad_difficulty() -> None:
    """Invalid difficulty string must raise InvalidConfigError, not ValueError."""
    from leadforge.core.exceptions import InvalidConfigError
    from leadforge.core.models import GenerationConfig

    with pytest.raises(InvalidConfigError, match="difficulty"):
        GenerationConfig(difficulty="super_hard")  # type: ignore[arg-type]


def test_missing_sentinel_repr() -> None:
    """_MISSING sentinel must have a readable repr for help() / docs."""
    from leadforge.core.sentinels import _MISSING

    assert repr(_MISSING) == "<default>"


def test_missing_sentinel_is_singleton() -> None:
    from leadforge.core.sentinels import _MISSING, _MissingType

    assert _MissingType() is _MISSING

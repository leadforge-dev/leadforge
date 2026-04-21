"""Tests for leadforge.core.ids."""

import pytest

from leadforge.core.ids import ID_PREFIXES, make_id


def test_make_id_format() -> None:
    assert make_id("acct", 1) == "acct_000001"
    assert make_id("lead", 999) == "lead_000999"
    assert make_id("acct", 1_000_000) == "acct_1000000"


def test_make_id_zero_padded_to_six_digits() -> None:
    result = make_id("cnt", 42)
    assert result == "cnt_000042"
    assert len(result.split("_")[1]) >= 6


def test_make_id_deterministic() -> None:
    assert make_id("opp", 7) == make_id("opp", 7)


def test_make_id_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive int"):
        make_id("acct", 0)


def test_make_id_rejects_negative() -> None:
    with pytest.raises(ValueError, match="positive int"):
        make_id("acct", -1)


def test_make_id_rejects_bool() -> None:
    with pytest.raises(ValueError, match="positive int"):
        make_id("acct", True)  # type: ignore[arg-type]


def test_make_id_rejects_float() -> None:
    with pytest.raises((ValueError, TypeError)):
        make_id("acct", 1.0)  # type: ignore[arg-type]


def test_id_prefixes_covers_all_entities() -> None:
    expected = {
        "account",
        "contact",
        "lead",
        "touch",
        "session",
        "sales_activity",
        "opportunity",
        "customer",
        "subscription",
        "rep",
    }
    assert set(ID_PREFIXES.keys()) == expected


def test_id_prefixes_values_are_strings() -> None:
    for key, prefix in ID_PREFIXES.items():
        assert isinstance(prefix, str), f"{key} prefix is not a string"
        assert len(prefix) > 0

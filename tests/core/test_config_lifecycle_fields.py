"""Tests for the lifecycle-scheme fields on GenerationConfig (LTV-Pn.3)."""

from __future__ import annotations

import pytest

from leadforge.core.exceptions import InvalidConfigError
from leadforge.core.models import GenerationConfig


def test_defaults() -> None:
    c = GenerationConfig()
    assert c.n_customers == 1500
    assert c.forward_windows_days == (90, 365, 730)
    assert c.early_tenure_weeks == 4
    assert c.observation_date is None


def test_accepts_valid_overrides() -> None:
    c = GenerationConfig(
        n_customers=500,
        forward_windows_days=(30, 90),
        early_tenure_weeks=8,
        observation_date="2026-06-01",
    )
    assert c.n_customers == 500
    assert c.forward_windows_days == (30, 90)
    assert c.early_tenure_weeks == 8
    assert c.observation_date == "2026-06-01"


@pytest.mark.parametrize("bad", [0, -1, True])
def test_rejects_bad_n_customers(bad) -> None:
    with pytest.raises(InvalidConfigError, match="n_customers"):
        GenerationConfig(n_customers=bad)


@pytest.mark.parametrize("bad", [0, -4, True])
def test_rejects_bad_early_tenure(bad) -> None:
    with pytest.raises(InvalidConfigError, match="early_tenure_weeks"):
        GenerationConfig(early_tenure_weeks=bad)


def test_rejects_empty_windows() -> None:
    with pytest.raises(InvalidConfigError, match="non-empty tuple"):
        GenerationConfig(forward_windows_days=())


def test_rejects_nonpositive_window_entry() -> None:
    with pytest.raises(InvalidConfigError, match="forward_windows_days entry"):
        GenerationConfig(forward_windows_days=(90, 0, 365))


def test_rejects_unsorted_or_duplicate_windows() -> None:
    with pytest.raises(InvalidConfigError, match="strictly increasing"):
        GenerationConfig(forward_windows_days=(365, 90))
    with pytest.raises(InvalidConfigError, match="strictly increasing"):
        GenerationConfig(forward_windows_days=(90, 90, 365))


def test_rejects_bad_observation_date() -> None:
    with pytest.raises(InvalidConfigError, match="ISO date"):
        GenerationConfig(observation_date="06/01/2026")
    with pytest.raises(InvalidConfigError, match="observation_date"):
        GenerationConfig(observation_date=20260601)  # type: ignore[arg-type]


def test_lead_scoring_path_unaffected_by_defaults() -> None:
    # The lead-scoring scheme ignores the lifecycle fields; a default config
    # still constructs and the lifecycle fields carry their documented defaults.
    c = GenerationConfig(n_leads=100, snapshot_day=30)
    assert c.n_leads == 100
    assert c.forward_windows_days == (90, 365, 730)

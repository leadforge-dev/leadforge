"""Tests for leadforge.schema.tasks — TaskManifest and SplitSpec."""

import dataclasses

import pytest

from leadforge.schema.tasks import CONVERTED_WITHIN_90_DAYS, SplitSpec

# ---------------------------------------------------------------------------
# SplitSpec
# ---------------------------------------------------------------------------


def test_split_spec_valid() -> None:
    s = SplitSpec(train=0.7, valid=0.15, test=0.15)
    assert s.train == pytest.approx(0.7)


def test_split_spec_rejects_bad_sum() -> None:
    with pytest.raises(ValueError, match="sum"):
        SplitSpec(train=0.6, valid=0.2, test=0.1)


def test_split_spec_frozen() -> None:
    s = SplitSpec(0.7, 0.15, 0.15)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.train = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskManifest
# ---------------------------------------------------------------------------


def test_task_manifest_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONVERTED_WITHIN_90_DAYS.task_id = "other"  # type: ignore[misc]


def test_converted_within_90_days_id() -> None:
    assert CONVERTED_WITHIN_90_DAYS.task_id == "converted_within_90_days"


def test_converted_within_90_days_label_column() -> None:
    assert CONVERTED_WITHIN_90_DAYS.label_column == "converted_within_90_days"


def test_converted_within_90_days_window() -> None:
    assert CONVERTED_WITHIN_90_DAYS.label_window_days == 90


def test_converted_within_90_days_task_type() -> None:
    assert CONVERTED_WITHIN_90_DAYS.task_type == "binary_classification"


def test_converted_within_90_days_primary_table() -> None:
    assert CONVERTED_WITHIN_90_DAYS.primary_table == "leads"


def test_converted_within_90_days_split_sums_to_one() -> None:
    s = CONVERTED_WITHIN_90_DAYS.split
    assert s.train + s.valid + s.test == pytest.approx(1.0)


def test_task_manifest_to_dict_keys() -> None:
    d = CONVERTED_WITHIN_90_DAYS.to_dict()
    expected = {
        "task_id",
        "task_type",
        "label_column",
        "label_window_days",
        "primary_table",
        "split",
        "description",
    }
    assert set(d.keys()) == expected


def test_task_manifest_to_dict_split_is_dict() -> None:
    d = CONVERTED_WITHIN_90_DAYS.to_dict()
    assert isinstance(d["split"], dict)
    assert set(d["split"].keys()) == {"train", "valid", "test"}


def test_task_manifest_to_dict_is_json_serializable() -> None:
    import json

    d = CONVERTED_WITHIN_90_DAYS.to_dict()
    json.dumps(d)  # should not raise

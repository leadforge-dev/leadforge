"""Tests for leadforge.schema.features and leadforge.schema.dictionaries."""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from leadforge.schema.dictionaries import feature_dictionary_df, write_feature_dictionary
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, FeatureSpec

# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------


def test_feature_spec_is_frozen() -> None:
    f = FeatureSpec("x", "string", "desc", "account")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.name = "y"  # type: ignore[misc]


def test_lead_snapshot_features_non_empty() -> None:
    assert len(LEAD_SNAPSHOT_FEATURES) > 0


def test_exactly_one_target_feature() -> None:
    targets = [f for f in LEAD_SNAPSHOT_FEATURES if f.is_target]
    assert len(targets) == 1
    assert targets[0].name == "converted_within_90_days"


def test_target_is_last_feature() -> None:
    assert LEAD_SNAPSHOT_FEATURES[-1].is_target


def test_all_feature_names_unique() -> None:
    names = [f.name for f in LEAD_SNAPSHOT_FEATURES]
    assert len(names) == len(set(names))


def test_all_dtypes_are_valid_strings() -> None:
    valid = {"string", "Int64", "Float64", "boolean"}
    for f in LEAD_SNAPSHOT_FEATURES:
        assert f.dtype in valid, f"{f.name} has unknown dtype {f.dtype!r}"


def test_all_categories_are_known() -> None:
    valid = {"account", "contact", "lead_meta", "engagement", "sales", "target"}
    for f in LEAD_SNAPSHOT_FEATURES:
        assert f.category in valid, f"{f.name} has unknown category {f.category!r}"


def test_target_feature_category_is_target() -> None:
    for f in LEAD_SNAPSHOT_FEATURES:
        if f.is_target:
            assert f.category == "target"


def test_no_leakage_risk_on_target() -> None:
    for f in LEAD_SNAPSHOT_FEATURES:
        if f.is_target:
            assert not f.leakage_risk


def test_leakage_trap_implies_leakage_risk() -> None:
    """Pedagogical traps must also be leakage_risk — otherwise the redaction
    rule (drop ``leakage_risk and not is_leakage_trap``) can't tell them apart."""
    for f in LEAD_SNAPSHOT_FEATURES:
        if f.is_leakage_trap:
            assert f.leakage_risk, f"{f.name} is a leakage trap but not flagged leakage_risk"


def test_total_touches_all_is_leakage_trap() -> None:
    by_name = {f.name: f for f in LEAD_SNAPSHOT_FEATURES}
    f = by_name["total_touches_all"]
    assert f.leakage_risk
    assert f.is_leakage_trap


def test_current_stage_is_redacted_not_trap() -> None:
    """``current_stage`` must be redacted from student_public — not a trap."""
    by_name = {f.name: f for f in LEAD_SNAPSHOT_FEATURES}
    f = by_name["current_stage"]
    assert f.leakage_risk
    assert not f.is_leakage_trap


def test_student_public_redacted_set_excludes_traps() -> None:
    from leadforge.schema.features import STUDENT_PUBLIC_REDACTED_COLUMNS

    assert "current_stage" in STUDENT_PUBLIC_REDACTED_COLUMNS
    assert "total_touches_all" not in STUDENT_PUBLIC_REDACTED_COLUMNS


# ---------------------------------------------------------------------------
# feature_dictionary_df
# ---------------------------------------------------------------------------


def test_feature_dictionary_df_returns_dataframe() -> None:
    df = feature_dictionary_df()
    assert isinstance(df, pd.DataFrame)


def test_feature_dictionary_df_row_count_matches_features() -> None:
    df = feature_dictionary_df()
    assert len(df) == len(LEAD_SNAPSHOT_FEATURES)


def test_feature_dictionary_df_columns() -> None:
    df = feature_dictionary_df()
    expected = {
        "name",
        "dtype",
        "description",
        "category",
        "is_target",
        "leakage_risk",
        "is_leakage_trap",
    }
    assert set(df.columns) == expected


def test_feature_dictionary_df_target_row() -> None:
    df = feature_dictionary_df()
    target_rows = df[df["is_target"]]
    assert len(target_rows) == 1
    assert target_rows.iloc[0]["name"] == "converted_within_90_days"


# ---------------------------------------------------------------------------
# write_feature_dictionary
# ---------------------------------------------------------------------------


def test_write_feature_dictionary_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "feature_dictionary.csv"
    write_feature_dictionary(out)
    assert out.exists()


def test_write_feature_dictionary_csv_readable(tmp_path: Path) -> None:
    out = tmp_path / "feature_dictionary.csv"
    write_feature_dictionary(out)
    df = pd.read_csv(out)
    assert len(df) == len(LEAD_SNAPSHOT_FEATURES)
    assert "name" in df.columns


def test_write_feature_dictionary_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "feature_dictionary.csv"
    write_feature_dictionary(out)
    assert out.exists()

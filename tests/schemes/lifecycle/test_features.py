"""Tests for the lifecycle customer-snapshot feature catalog (LTV-Pl)."""

from leadforge.schemes.lifecycle.features import CUSTOMER_SNAPSHOT_FEATURES

_VALID_DTYPES = {"string", "Int64", "Float64", "boolean"}
_VALID_CATEGORIES = {"customer_meta", "account", "subscription", "health", "financial", "target"}


def test_feature_names_unique() -> None:
    names = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES]
    assert len(names) == len(set(names))


def test_dtypes_are_pandas_nullable() -> None:
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        assert f.dtype in _VALID_DTYPES, f"{f.name}: {f.dtype}"


def test_categories_valid() -> None:
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        assert f.category in _VALID_CATEGORIES, f"{f.name}: {f.category}"


def test_exactly_four_targets() -> None:
    targets = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES if f.is_target]
    assert targets == [
        "ltv_revenue_90d",
        "ltv_revenue_365d",
        "ltv_revenue_730d",
        "churned_within_180d",
    ]


def test_targets_are_in_target_category() -> None:
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        assert f.is_target == (f.category == "target"), f.name


def test_ltv_targets_are_continuous() -> None:
    for f in CUSTOMER_SNAPSHOT_FEATURES:
        if f.name.startswith("ltv_revenue_"):
            assert f.dtype == "Float64"
            assert f.non_negative


def test_trap_is_descriptive_not_redacted() -> None:
    """The mrr_change_full_period trap is flagged as leakage but retained in
    every exposure mode (deliberate pedagogical trap, design.md §7)."""
    (trap,) = [f for f in CUSTOMER_SNAPSHOT_FEATURES if f.name == "mrr_change_full_period"]
    assert trap.leakage_risk
    assert trap.redact_in_modes == frozenset()
    assert not trap.is_target


def test_trap_is_only_leakage_risk_column() -> None:
    risky = [f.name for f in CUSTOMER_SNAPSHOT_FEATURES if f.leakage_risk]
    assert risky == ["mrr_change_full_period"]


def test_identifiers_are_strings() -> None:
    for name in ("customer_id", "account_id"):
        (spec,) = [f for f in CUSTOMER_SNAPSHOT_FEATURES if f.name == name]
        assert spec.dtype == "string"
        assert spec.category == "customer_meta"


def test_no_mechanism_less_columns() -> None:
    """current_plan and downgrade_count are deliberately absent: the engine
    has no plan-change or downgrade mechanism, so they would be a duplicate
    column and a zero-variance column respectively (see features.py
    docstring).  Re-add them only together with the mechanism."""
    names = {f.name for f in CUSTOMER_SNAPSHOT_FEATURES}
    assert "current_plan" not in names
    assert "downgrade_count" not in names


def test_valid_mrr_change_counterpart_present() -> None:
    names = {f.name for f in CUSTOMER_SNAPSHOT_FEATURES}
    assert "mrr_change_at_snapshot" in names

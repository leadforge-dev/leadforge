"""Distributional sanity checks for generated bundles.

These checks verify that the generated data looks "reasonable" — conversion
rates are within expected bounds, feature values are in valid ranges, and
tables are non-degenerate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from leadforge.core.serialization import load_json


def check_realism(bundle_root: Path) -> list[str]:
    """Run distributional sanity checks on a written bundle.

    Returns a list of warning/error strings (empty = all checks pass).
    """
    manifest = load_json(bundle_root / "manifest.json")
    errors: list[str] = []
    errors.extend(_check_conversion_rate(bundle_root, manifest))
    errors.extend(_check_table_nonempty(manifest))
    errors.extend(_check_feature_ranges(bundle_root))
    errors.extend(_check_stage_distribution(bundle_root))
    return errors


def _check_conversion_rate(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Check that conversion rate is within plausible bounds."""
    errors: list[str] = []
    train_path = root / "tasks/converted_within_90_days/train.parquet"
    if not train_path.exists():
        return errors

    df = pd.read_parquet(train_path, columns=["converted_within_90_days"])
    if len(df) == 0:
        errors.append("Train split is empty")
        return errors

    rate = df["converted_within_90_days"].mean()

    # Absolute bounds — any reasonable simulation should land here.
    # The v1 engine typically produces rates in the 30–90% range depending
    # on population size and seed; these are wide guardrails for degeneracy.
    if rate < 0.01:
        errors.append(f"Conversion rate suspiciously low: {rate:.4f} (< 1%)")
    elif rate > 0.95:
        errors.append(f"Conversion rate suspiciously high: {rate:.4f} (> 95%)")

    return errors


def _check_table_nonempty(manifest: dict[str, Any]) -> list[str]:
    """Core tables should have at least 1 row."""
    errors: list[str] = []
    required_nonempty = {"accounts", "contacts", "leads"}
    raw_tables = manifest.get("tables", {})
    if not isinstance(raw_tables, dict):
        return errors

    for table_name in required_nonempty:
        info = raw_tables.get(table_name)
        if info is None:
            errors.append(f"Table '{table_name}' missing from manifest")
        elif isinstance(info, dict) and info.get("row_count", 0) == 0:
            errors.append(f"Table '{table_name}' has 0 rows")

    return errors


def _check_feature_ranges(root: Path) -> list[str]:
    """Spot-check that key features have valid values."""
    errors: list[str] = []
    train_path = root / "tasks/converted_within_90_days/train.parquet"
    if not train_path.exists():
        return errors

    df = pd.read_parquet(train_path)

    # Non-negative count features
    count_cols = [
        "touch_count",
        "session_count",
        "activity_count",
        "inbound_touch_count",
        "outbound_touch_count",
    ]
    for col in count_cols:
        if col in df.columns:
            min_val = df[col].min()
            if pd.notna(min_val) and min_val < 0:
                errors.append(f"Feature '{col}' has negative values (min={min_val})")

    # Boolean features should be True/False only
    bool_cols = ["is_mql", "is_sql", "has_open_opportunity", "converted_within_90_days"]
    for col in bool_cols:
        if col in df.columns:
            unique = set(df[col].dropna().unique())
            if unique - {True, False}:
                errors.append(f"Feature '{col}' has non-boolean values: {unique}")

    return errors


def _check_stage_distribution(root: Path) -> list[str]:
    """Check that leads span multiple funnel stages (not all stuck in one)."""
    errors: list[str] = []
    leads_path = root / "tables/leads.parquet"
    if not leads_path.exists():
        return errors

    df = pd.read_parquet(leads_path, columns=["current_stage"])
    if len(df) == 0:
        return errors

    n_stages = df["current_stage"].nunique()
    if n_stages < 2:
        errors.append(
            f"All {len(df)} leads are in a single funnel stage "
            f"('{df['current_stage'].iloc[0]}') — simulation may be degenerate"
        )

    return errors

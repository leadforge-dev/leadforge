"""Distributional sanity checks for generated bundles.

These checks verify that the generated data looks "reasonable" — conversion
rates are within expected bounds, feature values are in valid ranges, and
tables are non-degenerate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES

# Derive check lists from the canonical schema to avoid silent drift.
_COUNT_FEATURES = [f.name for f in LEAD_SNAPSHOT_FEATURES if f.dtype == "Int64"]
_BOOL_FEATURES = [f.name for f in LEAD_SNAPSHOT_FEATURES if f.dtype == "boolean"]


def check_realism(bundle_root: Path, manifest: dict[str, Any]) -> list[str]:
    """Run distributional sanity checks on a written bundle.

    Args:
        bundle_root: Path to the bundle directory.
        manifest: Parsed manifest dict (avoids re-reading manifest.json).

    Returns a list of warning/error strings (empty = all checks pass).
    """
    errors: list[str] = []
    errors.extend(_check_conversion_rate(bundle_root))
    errors.extend(_check_table_nonempty(bundle_root, manifest))
    errors.extend(_check_feature_ranges(bundle_root))
    errors.extend(_check_stage_distribution(bundle_root))
    return errors


def _check_conversion_rate(root: Path) -> list[str]:
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


def _check_table_nonempty(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Core tables should have at least 1 row (verified from actual files)."""
    errors: list[str] = []
    required_nonempty = {"accounts", "contacts", "leads"}

    for table_name in required_nonempty:
        parquet_path = root / f"tables/{table_name}.parquet"
        if not parquet_path.exists():
            errors.append(f"Table '{table_name}' file missing")
        else:
            meta = pq.read_metadata(parquet_path)
            if meta.num_rows == 0:
                errors.append(f"Table '{table_name}' has 0 rows")

    return errors


def _check_feature_ranges(root: Path) -> list[str]:
    """Spot-check that key features have valid values."""
    errors: list[str] = []
    train_path = root / "tasks/converted_within_90_days/train.parquet"
    if not train_path.exists():
        return errors

    # Only read the columns we actually check.
    needed_cols = _COUNT_FEATURES + _BOOL_FEATURES
    # Filter to columns that actually exist in the file.
    schema = pq.read_schema(train_path)
    all_cols = set(schema.names)
    read_cols = [c for c in needed_cols if c in all_cols]
    if not read_cols:
        return errors

    df = pd.read_parquet(train_path, columns=read_cols)

    # Non-negative count features
    for col in _COUNT_FEATURES:
        if col in df.columns:
            min_val = df[col].min()
            if pd.notna(min_val) and min_val < 0:
                errors.append(f"Feature '{col}' has negative values (min={min_val})")

    # Boolean features should have boolean dtype
    for col in _BOOL_FEATURES:
        if col in df.columns:
            if not pd.api.types.is_bool_dtype(df[col]):
                errors.append(f"Feature '{col}' has non-boolean dtype: {df[col].dtype}")

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

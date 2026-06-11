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

from leadforge.schemes.lead_scoring.features import LEAD_SNAPSHOT_FEATURES

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
    errors.extend(_check_conversion_rate(bundle_root, manifest))
    errors.extend(_check_table_nonempty(bundle_root, manifest))
    errors.extend(_check_feature_ranges(bundle_root, manifest))
    errors.extend(_check_stage_distribution(bundle_root))
    return errors


def _first_task_train_path(root: Path, manifest: dict[str, Any]) -> Path | None:
    """Return the train.parquet path of the primary task in the manifest."""
    tasks = manifest.get("tasks", {})
    if not isinstance(tasks, dict) or not tasks:
        return None
    primary = manifest.get("primary_task")
    task_id = primary if isinstance(primary, str) and primary in tasks else next(iter(tasks))
    path = root / f"tasks/{task_id}/train.parquet"
    return path if path.exists() else None


# The label column in the snapshot is always ``converted_within_90_days``
# (mirroring :class:`~leadforge.schemes.lead_scoring.entities.LeadRow`).  The task *directory*
# may vary via ``config.primary_task``, but the column inside does not.
_LABEL_COLUMN = "converted_within_90_days"


def _check_conversion_rate(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Check that conversion rate is within plausible bounds."""
    errors: list[str] = []
    train_path = _first_task_train_path(root, manifest)
    if train_path is None:
        return errors

    df = pd.read_parquet(train_path, columns=[_LABEL_COLUMN])
    if len(df) == 0:
        errors.append("Train split is empty")
        return errors

    rate = df[_LABEL_COLUMN].mean()

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


def _check_feature_ranges(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Spot-check that key features have valid values."""
    errors: list[str] = []
    train_path = _first_task_train_path(root, manifest)
    if train_path is None:
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
    """Check that leads span multiple funnel stages (not all stuck in one).

    ``current_stage`` is redacted from the relational ``leads.parquet`` in
    ``student_public`` mode (bundle schema v3 onward), so this check is a
    no-op there — the underlying simulation is identical to the
    ``research_instructor`` bundle, which still carries the column and will
    surface a degenerate simulation through the same check.
    """
    errors: list[str] = []
    leads_path = root / "tables/leads.parquet"
    if not leads_path.exists():
        return errors

    schema_names = set(pq.read_schema(leads_path).names)
    if "current_stage" not in schema_names:
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

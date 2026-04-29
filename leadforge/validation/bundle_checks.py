"""Bundle validation logic.

:func:`validate_bundle` performs all structural, integrity, FK, and leakage
checks on a written bundle directory.  It returns a list of human-readable
error strings (empty = pass).  The CLI ``validate`` command is a thin shell
around this function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from leadforge.core.hashing import file_sha256
from leadforge.core.serialization import load_json
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES
from leadforge.schema.relationships import ALL_CONSTRAINTS


def validate_bundle(bundle_root: Path) -> list[str]:
    """Run all validation checks on the bundle at *bundle_root*.

    Returns:
        A list of error strings.  An empty list means the bundle is valid.

    Raises:
        FileNotFoundError: if ``manifest.json`` does not exist.
        ``LeadforgeError``: if ``manifest.json`` is corrupt / unparseable.
    """
    manifest = load_json(bundle_root / "manifest.json")
    errors: list[str] = []
    errors.extend(_check_required_files(bundle_root))
    tables, table_errors = _check_tables(bundle_root, manifest)
    errors.extend(table_errors)
    errors.extend(_check_task_splits(bundle_root, manifest))
    errors.extend(_check_fk_integrity(tables))
    errors.extend(_check_leakage(bundle_root, manifest))
    return errors


# ------------------------------------------------------------------
# Internal check functions
# ------------------------------------------------------------------


def _check_required_files(root: Path) -> list[str]:
    errors: list[str] = []
    for fname in ("dataset_card.md", "feature_dictionary.csv"):
        if not (root / fname).exists():
            errors.append(f"Missing required file: {fname}")
    return errors


def _check_tables(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Validate table files.  Returns loaded DataFrames and errors."""
    errors: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    for table_name, info in manifest.get("tables", {}).items():
        rel_path = info.get("file", f"tables/{table_name}.parquet")
        abs_path = root / rel_path
        if not abs_path.exists():
            errors.append(f"Missing table file: {rel_path}")
            continue

        df = pd.read_parquet(abs_path)
        tables[table_name] = df

        expected_rows = info.get("row_count")
        if expected_rows is not None and len(df) != expected_rows:
            errors.append(f"Table {table_name}: expected {expected_rows} rows, got {len(df)}")

        expected_sha = info.get("sha256")
        if expected_sha is not None:
            actual_sha = file_sha256(abs_path)
            if actual_sha != expected_sha:
                errors.append(f"Table {table_name}: SHA-256 mismatch")

    return tables, errors


def _check_task_splits(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for task_id, task_info in manifest.get("tasks", {}).items():
        for split in ("train", "valid", "test"):
            rel_path = f"tasks/{task_id}/{split}.parquet"
            abs_path = root / rel_path
            if not abs_path.exists():
                errors.append(f"Missing task file: {rel_path}")
                continue

            df = pd.read_parquet(abs_path)
            expected_rows = task_info.get(f"{split}_rows")
            if expected_rows is not None and len(df) != expected_rows:
                errors.append(
                    f"Task {task_id}/{split}: expected {expected_rows} rows, got {len(df)}"
                )

            expected_sha = task_info.get(f"{split}_sha256")
            if expected_sha is not None:
                actual_sha = file_sha256(abs_path)
                if actual_sha != expected_sha:
                    errors.append(f"Task {task_id}/{split}: SHA-256 mismatch")

    return errors


def _check_fk_integrity(tables: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    for fk in ALL_CONSTRAINTS:
        child_df = tables.get(fk.child_table)
        parent_df = tables.get(fk.parent_table)
        if child_df is None or parent_df is None:
            missing = fk.child_table if child_df is None else fk.parent_table
            errors.append(
                f"FK check skipped: {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} "
                f"(table '{missing}' not loaded)"
            )
            continue
        if fk.child_column not in child_df.columns:
            continue
        if fk.parent_column not in parent_df.columns:
            continue

        child_vals = set(child_df[fk.child_column].dropna())
        parent_vals = set(parent_df[fk.parent_column].dropna())
        orphans = child_vals - parent_vals
        if orphans:
            sample = list(orphans)[:3]
            errors.append(
                f"FK violation: {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column}: "
                f"{len(orphans)} orphan(s), e.g. {sample}"
            )

    return errors


def _check_leakage(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Check all task splits for unexpected columns."""
    errors: list[str] = []
    expected_columns = {f.name for f in LEAD_SNAPSHOT_FEATURES}
    for task_id in manifest.get("tasks", {}):
        for split in ("train", "valid", "test"):
            split_path = root / f"tasks/{task_id}/{split}.parquet"
            if split_path.exists():
                actual_columns = set(pd.read_parquet(split_path, columns=[]).columns)
                extra = actual_columns - expected_columns
                if extra:
                    errors.append(
                        f"Task {task_id}/{split}: unexpected columns (possible leakage): {extra}"
                    )
    return errors

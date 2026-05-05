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
import pyarrow.parquet as pq

from leadforge.core.enums import ExposureMode
from leadforge.core.exceptions import LeadforgeError
from leadforge.core.hashing import file_sha256
from leadforge.core.serialization import load_json
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, redacted_columns_for
from leadforge.schema.relationships import ALL_CONSTRAINTS
from leadforge.validation.difficulty import check_difficulty
from leadforge.validation.realism import check_realism
from leadforge.validation.relational_leakage import (
    BANNED_TABLES,
    LeakageReport,
    run_all_probes,
)


def validate_bundle(bundle_root: Path, *, include_realism: bool = True) -> list[str]:
    """Run all validation checks on the bundle at *bundle_root*.

    Args:
        bundle_root: Path to the bundle directory.
        include_realism: If True (default), also run distributional sanity
            and difficulty-adherence checks.

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
    errors.extend(_check_fk_integrity(tables, manifest))
    errors.extend(_check_leakage(bundle_root, manifest))
    errors.extend(_check_exposure_redaction(bundle_root, manifest))
    errors.extend(_check_relational_leakage(bundle_root, manifest))

    if include_realism:
        errors.extend(check_realism(bundle_root, manifest))
        errors.extend(check_difficulty(manifest))

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
    raw_tables = manifest.get("tables", {})
    if not isinstance(raw_tables, dict):
        errors.append("Malformed manifest: 'tables' is not a JSON object")
        return tables, errors
    for table_name, info in raw_tables.items():
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
    raw_tasks = manifest.get("tasks", {})
    if not isinstance(raw_tasks, dict):
        errors.append("Malformed manifest: 'tasks' is not a JSON object")
        return errors
    for task_id, task_info in raw_tasks.items():
        for split in ("train", "valid", "test"):
            rel_path = f"tasks/{task_id}/{split}.parquet"
            abs_path = root / rel_path
            if not abs_path.exists():
                errors.append(f"Missing task file: {rel_path}")
                continue

            expected_rows = task_info.get(f"{split}_rows")
            if expected_rows is not None:
                meta = pq.read_metadata(abs_path)
                if meta.num_rows != expected_rows:
                    errors.append(
                        f"Task {task_id}/{split}: expected"
                        f" {expected_rows} rows, got {meta.num_rows}"
                    )

            expected_sha = task_info.get(f"{split}_sha256")
            if expected_sha is not None:
                actual_sha = file_sha256(abs_path)
                if actual_sha != expected_sha:
                    errors.append(f"Task {task_id}/{split}: SHA-256 mismatch")

    return errors


def _check_fk_integrity(tables: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> list[str]:
    # In snapshot-safe (public) bundles ``customers`` / ``subscriptions``
    # are intentionally absent — emitting "FK check skipped" for them
    # would be a false positive.  The expected-absent set is the same
    # ``BANNED_TABLES`` constant the writer omits.
    snapshot_safe = bool(manifest.get("relational_snapshot_safe", False))
    expected_absent = set(BANNED_TABLES) if snapshot_safe else set()

    errors: list[str] = []
    for fk in ALL_CONSTRAINTS:
        child_df = tables.get(fk.child_table)
        parent_df = tables.get(fk.parent_table)
        if child_df is None or parent_df is None:
            missing = fk.child_table if child_df is None else fk.parent_table
            if missing in expected_absent:
                continue
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
    raw_tasks = manifest.get("tasks", {})
    if not isinstance(raw_tasks, dict):
        return errors
    expected_columns = {f.name for f in LEAD_SNAPSHOT_FEATURES}
    for task_id in raw_tasks:
        for split in ("train", "valid", "test"):
            split_path = root / f"tasks/{task_id}/{split}.parquet"
            if split_path.exists():
                actual_columns = set(pq.read_schema(split_path).names)
                extra = actual_columns - expected_columns
                if extra:
                    errors.append(
                        f"Task {task_id}/{split}: unexpected columns (possible leakage): {extra}"
                    )
    return errors


def _check_exposure_redaction(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Enforce the exposure-mode redaction contract.

    The expected redaction set is derived **directly from
    LEAD_SNAPSHOT_FEATURES** via :func:`redacted_columns_for`, *not* from
    the bundle filter the writer used.  That keeps this check independent
    of the writer's machinery: a future bug in the filter that silently
    skips a redaction will be caught here, because the validator's
    expected set still comes from the feature spec.

    Two things are checked:

    1. No expected-redacted column appears in any task split or in the
       feature dictionary (the actual leakage invariant).
    2. ``manifest.redacted_columns`` matches the expected set exactly
       (the bundle is self-describing and accurate).
    """
    errors: list[str] = []
    mode_str = manifest.get("exposure_mode")
    if not mode_str:
        return errors
    try:
        mode = ExposureMode(mode_str)
    except ValueError:
        errors.append(f"Manifest exposure_mode is unknown: {mode_str!r}")
        return errors

    expected = redacted_columns_for(mode)

    # Cross-check the manifest's self-reported redaction set.
    declared_raw = manifest.get("redacted_columns")
    if declared_raw is None:
        if expected:
            errors.append(
                "manifest.redacted_columns is missing; expected "
                f"{sorted(expected)} for {mode.value}"
            )
    elif isinstance(declared_raw, list):
        declared = set(declared_raw)
        if declared != set(expected):
            errors.append(
                "manifest.redacted_columns disagrees with feature spec for "
                f"{mode.value}: declared={sorted(declared)} expected={sorted(expected)}"
            )

    if not expected:
        return errors

    raw_tasks = manifest.get("tasks", {})
    if isinstance(raw_tasks, dict):
        for task_id in raw_tasks:
            for split in ("train", "valid", "test"):
                split_path = root / f"tasks/{task_id}/{split}.parquet"
                if split_path.exists():
                    actual = set(pq.read_schema(split_path).names)
                    leaked = sorted(actual & expected)
                    if leaked:
                        errors.append(
                            f"Task {task_id}/{split}: redacted columns present in "
                            f"{mode.value} bundle: {leaked}"
                        )

    fd_path = root / "feature_dictionary.csv"
    if fd_path.exists():
        fd = pd.read_csv(fd_path)
        if "name" in fd.columns:
            present = set(fd["name"].astype(str).tolist())
            leaked = sorted(present & expected)
            if leaked:
                errors.append(
                    f"feature_dictionary.csv: redacted columns present in "
                    f"{mode.value} bundle: {leaked}"
                )

    return errors


def _check_relational_leakage(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Run :func:`run_all_probes` on snapshot-safe (public) bundles.

    Skips ``research_instructor`` bundles entirely — they retain the
    full hidden truth (label column, customers, subscriptions,
    ``close_outcome``) by design, so the leakage probes would be a
    false positive there.  The bonus-model probe stays off (PR 3.3
    will calibrate per-tier bands).

    Each :class:`~leadforge.validation.relational_leakage.LeakageFinding`
    is rendered as one error string keeping the existing
    ``validate_bundle`` contract (return list of strings, empty = pass).
    """
    mode_str = manifest.get("exposure_mode")
    if mode_str != ExposureMode.student_public.value:
        return []

    snapshot_day = manifest.get("snapshot_day")
    if snapshot_day is None:
        return [
            "Cannot run relational-leakage probes: manifest.snapshot_day is null "
            "but exposure_mode is student_public (snapshot-safe contract requires "
            "a windowed snapshot)."
        ]
    if not isinstance(snapshot_day, int) or isinstance(snapshot_day, bool):
        return [f"Manifest snapshot_day must be an int, got {type(snapshot_day).__name__}"]

    try:
        report: LeakageReport = run_all_probes(root, snapshot_day=snapshot_day)
    except (FileNotFoundError, ValueError, LeadforgeError) as exc:
        return [f"Relational-leakage probe failed: {type(exc).__name__}: {exc}"]

    return [f"Relational leakage [{f.channel}] {f.detail}: {f.message}" for f in report.findings]

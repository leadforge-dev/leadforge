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
from leadforge.validation.leakage_probes import (
    BANNED_TABLES,
    LeakageReport,
    probe_banned_columns,
    probe_banned_tables,
    probe_deterministic_reconstruction,
    run_all_probes,
)
from leadforge.validation.realism import check_realism


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
    #
    # Strict bool check: ``bool(...)`` would coerce malformed manifest
    # values (the JSON string ``"false"`` is truthy; the int ``1`` would
    # masquerade as snapshot-safe) and silently suppress real FK errors.
    # Surface the bad value instead.
    errors: list[str] = []
    raw_flag = manifest.get("relational_snapshot_safe", False)
    if not isinstance(raw_flag, bool):
        errors.append(
            f"manifest.relational_snapshot_safe must be a JSON boolean, got "
            f"{type(raw_flag).__name__}={raw_flag!r}"
        )
        snapshot_safe = False
    else:
        snapshot_safe = raw_flag
    expected_absent = set(BANNED_TABLES) if snapshot_safe else set()

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
    """Run the relational-leakage probes on snapshot-safe (public) bundles.

    Skips ``research_instructor`` bundles entirely — they retain the
    full hidden truth (label column, customers, subscriptions,
    ``close_outcome``) by design, so the probes would be a false
    positive there.  The bonus-model probe stays off (PR 3.3 will
    calibrate per-tier bands).

    The structural probes (banned columns, banned tables, deterministic
    join reconstruction) do not depend on ``snapshot_day``.  They run
    even if the manifest is missing or has a malformed ``snapshot_day``
    — those are exactly the cases where users *most* need leakage
    detection, not least.  Only the snapshot-window probe is skipped
    when ``snapshot_day`` is unavailable, with an explicit error
    surfaced so the gap is visible.

    Each :class:`~leadforge.validation.leakage_probes.LeakageFinding`
    is rendered as one error string, keeping the existing
    ``validate_bundle`` contract (return list of strings, empty = pass).
    """
    mode_str = manifest.get("exposure_mode")
    if mode_str != ExposureMode.student_public.value:
        return []

    snapshot_day = manifest.get("snapshot_day")
    snapshot_day_usable = (
        isinstance(snapshot_day, int) and not isinstance(snapshot_day, bool) and snapshot_day >= 0
    )

    errors: list[str] = []
    try:
        if snapshot_day_usable and isinstance(snapshot_day, int):
            report: LeakageReport = run_all_probes(root, snapshot_day=snapshot_day)
        else:
            # Run the structural subset that does not need ``snapshot_day``.
            tables = _read_relational_tables(root)
            findings = list(probe_banned_columns(tables))
            findings += list(probe_banned_tables(tables.keys()))
            findings += list(probe_deterministic_reconstruction(tables))
            report = LeakageReport(findings=tuple(findings))
            errors.append(
                "Relational leakage [snapshot_window] manifest.snapshot_day is "
                f"missing or malformed ({snapshot_day!r}); skipping the snapshot-"
                "window probe.  Structural probes (banned columns / tables / "
                "join reconstruction) ran normally."
            )
    except (FileNotFoundError, ValueError, LeadforgeError) as exc:
        return errors + [f"Relational-leakage probe failed: {type(exc).__name__}: {exc}"]

    errors.extend(
        f"Relational leakage [{f.channel}] {f.detail}: {f.message}" for f in report.findings
    )
    return errors


def _read_relational_tables(root: Path) -> dict[str, pd.DataFrame]:
    """Read every public + banned-table parquet under ``<root>/tables/``.

    Mirrors the read logic in
    :func:`leadforge.validation.leakage_probes.run_all_probes` but
    is reusable for the snapshot_day-missing path above.
    """
    from leadforge.validation.leakage_probes import (
        BANNED_TABLES as _BANNED,
    )

    tables_dir = root / "tables"
    if not tables_dir.is_dir() or not (tables_dir / "leads.parquet").exists():
        raise FileNotFoundError(f"missing tables/leads.parquet under {root}")

    public_tables = (
        "accounts",
        "contacts",
        "leads",
        "touches",
        "sessions",
        "sales_activities",
        "opportunities",
    )
    out: dict[str, pd.DataFrame] = {}
    for name in (*public_tables, *_BANNED):
        path = tables_dir / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out

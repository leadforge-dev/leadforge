"""leadforge validate command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer


def validate(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
) -> None:
    """Run schema and artifact validation on a generated bundle."""
    root = Path(bundle_path)
    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Manifest presence and parse
    # ------------------------------------------------------------------
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"FAIL: no manifest.json in {root}", err=True)
        raise typer.Exit(1)

    manifest = json.loads(manifest_path.read_text())

    # ------------------------------------------------------------------
    # 2. Required top-level files
    # ------------------------------------------------------------------
    for fname in ("dataset_card.md", "feature_dictionary.csv"):
        if not (root / fname).exists():
            errors.append(f"Missing required file: {fname}")

    # ------------------------------------------------------------------
    # 3. Table files exist + row counts + SHA-256 hashes
    # ------------------------------------------------------------------
    import pandas as pd

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
            actual_sha = _sha256(abs_path)
            if actual_sha != expected_sha:
                errors.append(f"Table {table_name}: SHA-256 mismatch")

    # ------------------------------------------------------------------
    # 4. Task split files exist + row counts + hashes
    # ------------------------------------------------------------------
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
                actual_sha = _sha256(abs_path)
                if actual_sha != expected_sha:
                    errors.append(f"Task {task_id}/{split}: SHA-256 mismatch")

    # ------------------------------------------------------------------
    # 5. FK integrity
    # ------------------------------------------------------------------
    from leadforge.schema.relationships import ALL_CONSTRAINTS

    for fk in ALL_CONSTRAINTS:
        child_df = tables.get(fk.child_table)
        parent_df = tables.get(fk.parent_table)
        if child_df is None or parent_df is None:
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

    # ------------------------------------------------------------------
    # 6. Leakage check — no post-anchor features in task splits
    # ------------------------------------------------------------------
    from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES

    expected_columns = {f.name for f in LEAD_SNAPSHOT_FEATURES}
    for task_id in manifest.get("tasks", {}):
        train_path = root / f"tasks/{task_id}/train.parquet"
        if train_path.exists():
            actual_columns = set(pd.read_parquet(train_path).columns)
            extra = actual_columns - expected_columns
            if extra:
                errors.append(f"Task {task_id}: unexpected columns (possible leakage): {extra}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if errors:
        typer.echo(f"FAIL: {len(errors)} validation error(s):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"OK: bundle at {root} passed all checks.")


def _sha256(path: Path) -> str:
    """Return hex-encoded SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

"""leadforge inspect command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from leadforge.core.exceptions import LeadforgeError
from leadforge.core.serialization import load_json


def inspect(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
    json_output: bool = typer.Option(  # noqa: FBT001
        False,
        "--json",
        "-j",
        help="Emit the parsed manifest as JSON to stdout (pipe-friendly).",
    ),
) -> None:
    """Inspect a generated dataset bundle and print a summary."""
    root = Path(bundle_path)

    if not root.exists():
        typer.echo(f"Error: path does not exist: {root}", err=True)
        raise typer.Exit(1)
    if not root.is_dir():
        typer.echo(f"Error: not a directory (expected a bundle dir): {root}", err=True)
        raise typer.Exit(1)

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"Error: no manifest.json found in {root}", err=True)
        raise typer.Exit(1)

    try:
        manifest = load_json(manifest_path)
    except LeadforgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    if not isinstance(manifest, dict):
        typer.echo("Error: manifest.json is not a JSON object", err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(manifest, indent=2))
        return

    typer.echo(f"Bundle: {root}")
    typer.echo(f"  Recipe:        {manifest.get('recipe_id', '?')}")
    # v6+ field: which peer generation scheme produced the bundle.  Conditional
    # so pre-v6 bundles render without a "?" placeholder row.
    if "generation_scheme" in manifest:
        typer.echo(f"  Scheme:        {manifest['generation_scheme']}")
    typer.echo(f"  Seed:          {manifest.get('seed', '?')}")
    typer.echo(f"  Mode:          {manifest.get('exposure_mode', '?')}")
    typer.echo(f"  Difficulty:    {manifest.get('difficulty', '?')}")
    typer.echo(f"  Horizon days:  {manifest.get('horizon_days', '?')}")
    typer.echo(f"  Generated at:  {manifest.get('generation_timestamp', '?')}")
    typer.echo(f"  Package:       leadforge {manifest.get('package_version', '?')}")
    typer.echo(f"  Schema ver:    {manifest.get('bundle_schema_version', '?')}")

    # v3+ fields — only print rows for keys actually present in the manifest,
    # so older (v2) bundles render cleanly without "?" placeholders.
    if "primary_task" in manifest:
        typer.echo(f"  Primary task:  {manifest['primary_task']}")
    if "label_window_days" in manifest:
        typer.echo(f"  Label window:  {manifest['label_window_days']} days")
    if "snapshot_day" in manifest:
        snapshot_day = manifest["snapshot_day"]
        if snapshot_day is None:
            typer.echo("  Snapshot day:  (full horizon, no windowing)")
        else:
            typer.echo(f"  Snapshot day:  {snapshot_day} days")
    if "redacted_columns" in manifest:
        cols = manifest["redacted_columns"] or []
        if cols:
            noun = "column" if len(cols) == 1 else "columns"
            if len(cols) <= 4:
                names = ", ".join(cols)
            else:
                names = ", ".join(cols[:3]) + ", ..."
            typer.echo(f"  Redactions:    {len(cols)} {noun} [{names}]")

    typer.echo(f"  Motif family:  {manifest.get('motif_family') or '?'}")

    typer.echo("")
    typer.echo("Tables:")
    tables = manifest.get("tables", {})
    if isinstance(tables, dict):
        for name, info in tables.items():
            row_count = _safe_get(info, "row_count", "?")
            typer.echo(f"  {name:25s}  {row_count:>8} rows")

    tasks = manifest.get("tasks", {})
    if isinstance(tasks, dict) and tasks:
        typer.echo("")
        typer.echo("Tasks:")
        for task_id, info in tasks.items():
            train = _safe_get(info, "train_rows", "?")
            valid = _safe_get(info, "valid_rows", "?")
            test = _safe_get(info, "test_rows", "?")
            typer.echo(f"  {task_id}")
            typer.echo(f"    train={train}  valid={valid}  test={test}")

    has_metadata = (root / "metadata").is_dir()
    typer.echo("")
    typer.echo(f"Metadata dir:    {'present' if has_metadata else 'absent'}")


def _safe_get(obj: Any, key: str, default: str = "?") -> Any:
    """Get a key from *obj* if it's a dict, else return *default*."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

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
    typer.echo(f"  Seed:          {manifest.get('seed', '?')}")
    typer.echo(f"  Mode:          {manifest.get('exposure_mode', '?')}")
    typer.echo(f"  Difficulty:    {manifest.get('difficulty', '?')}")
    typer.echo(f"  Horizon days:  {manifest.get('horizon_days', '?')}")
    typer.echo(f"  Generated at:  {manifest.get('generation_timestamp', '?')}")
    typer.echo(f"  Package:       leadforge {manifest.get('package_version', '?')}")
    typer.echo(f"  Schema ver:    {manifest.get('bundle_schema_version', '?')}")
    typer.echo(f"  Primary task:  {manifest.get('primary_task', '?')}")
    typer.echo(f"  Label window:  {manifest.get('label_window_days', '?')} days")
    typer.echo(f"  Snapshot day:  {_format_snapshot_day(manifest)}")
    typer.echo(f"  Redactions:    {_format_redactions(manifest)}")
    typer.echo(f"  Motif family:  {manifest.get('motif_family', '?')}")

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


def _format_snapshot_day(manifest: dict[str, Any]) -> str:
    """Format the ``snapshot_day`` field, annotating the full-horizon case."""
    if "snapshot_day" not in manifest:
        return "?"
    snapshot_day = manifest.get("snapshot_day")
    horizon_days = manifest.get("horizon_days")
    if snapshot_day is None or (
        isinstance(snapshot_day, int)
        and isinstance(horizon_days, int)
        and snapshot_day == horizon_days
    ):
        return "(full horizon, no windowing)"
    return f"{snapshot_day} days"


def _format_redactions(manifest: dict[str, Any]) -> str:
    """Format the ``redacted_columns`` field as count + list (full or truncated)."""
    if "redacted_columns" not in manifest:
        return "?"
    cols = manifest.get("redacted_columns") or []
    if not isinstance(cols, list):
        return "?"
    if not cols:
        return "0 column(s) []"
    if len(cols) <= 4:
        return f"{len(cols)} column(s) [{', '.join(cols)}]"
    head = ", ".join(cols[:3])
    return f"{len(cols)} column(s) [{head}, ...] ({len(cols)} total)"


def _safe_get(obj: Any, key: str, default: str = "?") -> Any:
    """Get a key from *obj* if it's a dict, else return *default*."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

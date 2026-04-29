"""leadforge inspect command."""

from __future__ import annotations

from pathlib import Path

import typer

from leadforge.core.exceptions import LeadforgeError
from leadforge.core.serialization import load_json


def inspect(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
) -> None:
    """Inspect a generated dataset bundle and print a summary."""
    root = Path(bundle_path)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        typer.echo(f"Error: no manifest.json found in {root}", err=True)
        raise typer.Exit(1)

    try:
        manifest = load_json(manifest_path)
    except LeadforgeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Bundle: {root}")
    typer.echo(f"  Recipe:        {manifest.get('recipe_id', '?')}")
    typer.echo(f"  Seed:          {manifest.get('seed', '?')}")
    typer.echo(f"  Mode:          {manifest.get('exposure_mode', '?')}")
    typer.echo(f"  Difficulty:    {manifest.get('difficulty', '?')}")
    typer.echo(f"  Horizon days:  {manifest.get('horizon_days', '?')}")
    typer.echo(f"  Generated at:  {manifest.get('generation_timestamp', '?')}")
    typer.echo(f"  Package:       leadforge {manifest.get('package_version', '?')}")
    typer.echo(f"  Schema ver:    {manifest.get('bundle_schema_version', '?')}")
    typer.echo(f"  Motif family:  {manifest.get('motif_family', '?')}")

    typer.echo("")
    typer.echo("Tables:")
    tables = manifest.get("tables", {})
    for name, info in tables.items():
        typer.echo(f"  {name:25s}  {info.get('row_count', '?'):>8} rows")

    tasks = manifest.get("tasks", {})
    if tasks:
        typer.echo("")
        typer.echo("Tasks:")
        for task_id, info in tasks.items():
            train = info.get("train_rows", "?")
            valid = info.get("valid_rows", "?")
            test = info.get("test_rows", "?")
            typer.echo(f"  {task_id}")
            typer.echo(f"    train={train}  valid={valid}  test={test}")

    has_metadata = (root / "metadata").is_dir()
    typer.echo("")
    typer.echo(f"Metadata dir:    {'present' if has_metadata else 'absent'}")

"""leadforge validate command."""

from __future__ import annotations

from pathlib import Path

import typer

from leadforge.core.exceptions import LeadforgeError


def validate(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
) -> None:
    """Run schema and artifact validation on a generated bundle."""
    from leadforge.validation.bundle_checks import validate_bundle

    root = Path(bundle_path)

    if not root.exists():
        typer.echo(f"FAIL: path does not exist: {root}", err=True)
        raise typer.Exit(1)
    if not root.is_dir():
        typer.echo(f"FAIL: not a directory: {root}", err=True)
        raise typer.Exit(1)
    if not (root / "manifest.json").exists():
        typer.echo(f"FAIL: no manifest.json in {root}", err=True)
        raise typer.Exit(1)

    try:
        errors = validate_bundle(root)
    except LeadforgeError as exc:
        typer.echo(f"FAIL: {exc}", err=True)
        raise typer.Exit(1) from None

    if errors:
        typer.echo(f"FAIL: {len(errors)} validation error(s):", err=True)
        for e in errors:
            typer.echo(f"  - {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"OK: bundle at {root} passed all checks.")

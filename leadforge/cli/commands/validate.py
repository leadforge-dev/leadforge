"""leadforge validate command."""

import typer


def validate(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
) -> None:
    """Run schema and artifact validation on a generated bundle."""
    typer.echo(
        "The 'validate' command is not yet implemented. Coming in v0.5.0.",
        err=True,
    )
    raise typer.Exit(1)

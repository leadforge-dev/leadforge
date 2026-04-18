"""leadforge inspect command."""

import typer


def inspect(
    bundle_path: str = typer.Argument(..., help="Path to a generated bundle directory."),
) -> None:
    """Inspect a generated dataset bundle and print a summary."""
    typer.echo(
        "The 'inspect' command is not yet implemented. Coming in v0.4.0.",
        err=True,
    )
    raise typer.Exit(1)

"""leadforge list-recipes command."""

import typer
from rich.console import Console
from rich.table import Table

from leadforge.recipes.registry import list_recipes


def list_recipes_cmd() -> None:
    """List all available generation recipes."""
    recipes = list_recipes()
    if not recipes:
        typer.echo("No recipes found.")
        raise typer.Exit()

    console = Console()
    table = Table(title="Available Recipes", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Primary Task", style="green")
    table.add_column("Modes")
    table.add_column("Difficulty")

    for r in recipes:
        table.add_row(
            r.get("id", ""),
            r.get("title", ""),
            r.get("primary_task", ""),
            ", ".join(r.get("supported_modes", [])),
            ", ".join(r.get("supported_difficulty", [])),
        )

    console.print(table)

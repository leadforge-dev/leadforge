"""leadforge CLI entrypoint."""

import typer

from leadforge.cli.commands.generate import generate
from leadforge.cli.commands.inspect import inspect
from leadforge.cli.commands.list_recipes import list_recipes_cmd
from leadforge.cli.commands.validate import validate
from leadforge.version import __version__

app = typer.Typer(
    name="leadforge",
    help="Generate synthetic CRM and GTM datasets from simulated commercial worlds.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"leadforge {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: FBT001
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


app.command("list-recipes")(list_recipes_cmd)
app.command("generate")(generate)
app.command("inspect")(inspect)
app.command("validate")(validate)

"""leadforge generate command."""

import typer


def generate(
    recipe: str = typer.Option(..., "--recipe", "-r", help="Recipe ID to use."),
    seed: int = typer.Option(..., "--seed", help="Random seed for deterministic generation."),
    mode: str = typer.Option(
        ...,
        "--mode",
        help="Exposure mode: student_public or research_instructor.",
    ),
    out: str = typer.Option(..., "--out", help="Output directory for the generated bundle."),
    difficulty: str = typer.Option(
        "intermediate",
        "--difficulty",
        help="Difficulty profile: intro, intermediate, or advanced.",
    ),
    n_accounts: int | None = typer.Option(None, "--n-accounts", help="Number of accounts."),
    n_contacts: int | None = typer.Option(None, "--n-contacts", help="Number of contacts."),
    n_leads: int | None = typer.Option(None, "--n-leads", help="Number of leads."),
    horizon_days: int | None = typer.Option(
        None, "--horizon-days", help="Simulation horizon in days."
    ),
    override: str | None = typer.Option(
        None, "--override", help="Path to a YAML config override file."
    ),
) -> None:
    """Generate a synthetic CRM dataset bundle from a recipe."""
    typer.echo(
        "The 'generate' command is not yet implemented. Coming in v0.2.0.",
        err=True,
    )
    raise typer.Exit(1)

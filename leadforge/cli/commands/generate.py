"""leadforge generate command."""

from __future__ import annotations

from pathlib import Path

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
    from leadforge.api.generator import Generator
    from leadforge.core.serialization import load_yaml

    override_dict: dict | None = None
    if override is not None:
        override_dict = load_yaml(Path(override))

    gen = Generator.from_recipe(
        recipe,
        seed=seed,
        exposure_mode=mode,
        difficulty=difficulty,
        n_accounts=n_accounts,
        n_contacts=n_contacts,
        n_leads=n_leads,
        horizon_days=horizon_days,
        override=override_dict,
    )

    generate_kwargs: dict[str, int] = {}
    if n_accounts is not None:
        generate_kwargs["n_accounts"] = n_accounts
    if n_contacts is not None:
        generate_kwargs["n_contacts"] = n_contacts
    if n_leads is not None:
        generate_kwargs["n_leads"] = n_leads

    typer.echo(f"Generating bundle with recipe '{recipe}', seed={seed}, mode={mode} ...")
    bundle = gen.generate(**generate_kwargs)

    typer.echo(f"Writing bundle to {out} ...")
    bundle.save(out)

    typer.echo(f"Done. Bundle written to {out}")

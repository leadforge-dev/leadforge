"""leadforge generate command."""

from __future__ import annotations

from pathlib import Path

import typer

from leadforge.core.exceptions import LeadforgeError


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
    n_accounts: int | None = typer.Option(
        None, "--n-accounts", help="Override recipe default account count."
    ),
    n_contacts: int | None = typer.Option(
        None, "--n-contacts", help="Override recipe default contact count."
    ),
    n_leads: int | None = typer.Option(
        None, "--n-leads", help="Override recipe default lead count."
    ),
    horizon_days: int | None = typer.Option(
        None, "--horizon-days", help="Override recipe default simulation horizon in days."
    ),
    primary_task: str | None = typer.Option(
        None,
        "--primary-task",
        help="Override recipe default task identifier (e.g. converted_within_60_days).",
    ),
    label_window_days: int | None = typer.Option(
        None,
        "--label-window-days",
        help="Override recipe default label observation window in days.",
    ),
    snapshot_day: int | None = typer.Option(
        None,
        "--snapshot-day",
        help="Override recipe default snapshot day for windowed feature aggregation.",
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
        override_path = Path(override)
        if not override_path.exists():
            typer.echo(f"Error: override file not found: {override_path}", err=True)
            raise typer.Exit(1)
        try:
            loaded = load_yaml(override_path)
        except LeadforgeError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from None
        if loaded is not None and not isinstance(loaded, dict):
            typer.echo(
                "Error: override file must contain a YAML mapping at the top level.",
                err=True,
            )
            raise typer.Exit(1)
        override_dict = loaded

    try:
        gen = Generator.from_recipe(
            recipe,
            seed=seed,
            exposure_mode=mode,
            difficulty=difficulty,
            n_accounts=n_accounts,
            n_contacts=n_contacts,
            n_leads=n_leads,
            horizon_days=horizon_days,
            primary_task=primary_task,
            label_window_days=label_window_days,
            snapshot_day=snapshot_day,
            override=override_dict,
        )
    except (LeadforgeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Generating bundle with recipe '{recipe}', seed={seed}, mode={mode} ...")

    try:
        bundle = gen.generate()
    except (LeadforgeError, RuntimeError) as exc:
        typer.echo(f"Error during generation: {exc}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"Writing bundle to {out} ...")
    bundle.save(out)

    typer.echo(f"Done. Bundle written to {out}")

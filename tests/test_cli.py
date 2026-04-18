"""CLI smoke tests."""

from typer.testing import CliRunner

from leadforge.cli.main import app

runner = CliRunner()


def test_help_exits_clean() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "leadforge" in result.output


def test_list_recipes_exits_clean() -> None:
    result = runner.invoke(app, ["list-recipes"])
    assert result.exit_code == 0


def test_list_recipes_shows_v1_recipe() -> None:
    result = runner.invoke(app, ["list-recipes"])
    assert "b2b_saas_procurement_v1" in result.output


def test_generate_stub_exits_nonzero() -> None:
    result = runner.invoke(
        app, ["generate", "--recipe", "x", "--seed", "1", "--mode", "y", "--out", "/tmp"]
    )
    assert result.exit_code != 0


def test_inspect_stub_exits_nonzero() -> None:
    result = runner.invoke(app, ["inspect", "/nonexistent"])
    assert result.exit_code != 0


def test_validate_stub_exits_nonzero() -> None:
    result = runner.invoke(app, ["validate", "/nonexistent"])
    assert result.exit_code != 0

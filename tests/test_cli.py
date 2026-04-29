"""CLI tests — smoke tests + generate/inspect/validate integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from leadforge.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper — generate a small bundle to a temp dir
# ---------------------------------------------------------------------------

_GENERATE_ARGS = [
    "generate",
    "--recipe",
    "b2b_saas_procurement_v1",
    "--seed",
    "42",
    "--mode",
    "student_public",
    "--n-leads",
    "30",
    "--n-accounts",
    "15",
    "--n-contacts",
    "45",
]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a small bundle once and reuse across tests in this module."""
    out = tmp_path_factory.mktemp("bundle")
    result = runner.invoke(app, [*_GENERATE_ARGS, "--out", str(out)])
    assert result.exit_code == 0, f"generate failed:\n{result.output}"
    return out


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    def test_exits_zero(self, bundle_dir: Path) -> None:
        # bundle_dir fixture already asserts exit_code == 0
        assert (bundle_dir / "manifest.json").exists()

    def test_writes_core_files(self, bundle_dir: Path) -> None:
        assert (bundle_dir / "dataset_card.md").exists()
        assert (bundle_dir / "feature_dictionary.csv").exists()
        assert (bundle_dir / "tables").is_dir()
        assert (bundle_dir / "tasks").is_dir()

    def test_manifest_has_expected_keys(self, bundle_dir: Path) -> None:
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        assert manifest["recipe_id"] == "b2b_saas_procurement_v1"
        assert manifest["seed"] == 42
        assert manifest["exposure_mode"] == "student_public"
        assert "tables" in manifest
        assert "tasks" in manifest

    def test_no_metadata_in_student_public(self, bundle_dir: Path) -> None:
        assert not (bundle_dir / "metadata").exists()

    def test_invalid_recipe_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "nonexistent_recipe",
                "--seed",
                "1",
                "--mode",
                "student_public",
                "--out",
                str(tmp_path / "bad"),
            ],
        )
        assert result.exit_code != 0

    def test_invalid_mode_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "b2b_saas_procurement_v1",
                "--seed",
                "1",
                "--mode",
                "invalid_mode",
                "--out",
                str(tmp_path / "bad"),
            ],
        )
        assert result.exit_code != 0

    def test_research_instructor_mode_has_metadata(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "b2b_saas_procurement_v1",
                "--seed",
                "7",
                "--mode",
                "research_instructor",
                "--n-leads",
                "20",
                "--n-accounts",
                "10",
                "--n-contacts",
                "30",
                "--out",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, f"generate failed:\n{result.output}"
        assert (tmp_path / "metadata").is_dir()

    def test_output_message(self, tmp_path: Path) -> None:
        out = tmp_path / "msg_test"
        result = runner.invoke(app, [*_GENERATE_ARGS, "--out", str(out)])
        assert result.exit_code == 0
        assert "Generating bundle" in result.output
        assert "Done" in result.output


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    def test_exits_zero(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert result.exit_code == 0

    def test_shows_recipe(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert "b2b_saas_procurement_v1" in result.output

    def test_shows_seed(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert "42" in result.output

    def test_shows_tables(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert "accounts" in result.output
        assert "leads" in result.output

    def test_shows_tasks(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert "converted_within_90_days" in result.output

    def test_missing_bundle_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["inspect", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_valid_bundle_passes(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["validate", str(bundle_dir)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_missing_bundle_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_corrupt_manifest_fails(self, tmp_path: Path, bundle_dir: Path) -> None:
        """A bundle with a tampered row count should fail validation."""
        import shutil

        corrupt = tmp_path / "corrupt_bundle"
        shutil.copytree(bundle_dir, corrupt)

        manifest = json.loads((corrupt / "manifest.json").read_text())
        # Tamper with a table row count
        first_table = next(iter(manifest["tables"]))
        manifest["tables"][first_table]["row_count"] = 999999
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        result = runner.invoke(app, ["validate", str(corrupt)])
        assert result.exit_code != 0
        assert "FAIL" in result.output

    def test_missing_table_file_fails(self, tmp_path: Path, bundle_dir: Path) -> None:
        """Removing a table Parquet file should cause validation failure."""
        import shutil

        corrupt = tmp_path / "missing_table_bundle"
        shutil.copytree(bundle_dir, corrupt)

        # Remove one table file
        manifest = json.loads((corrupt / "manifest.json").read_text())
        first_table = next(iter(manifest["tables"]))
        (corrupt / f"tables/{first_table}.parquet").unlink()

        result = runner.invoke(app, ["validate", str(corrupt)])
        assert result.exit_code != 0
        assert "FAIL" in result.output

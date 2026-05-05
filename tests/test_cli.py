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
    """Generate a small bundle once and reuse across tests in this module.

    WARNING: this fixture is module-scoped for performance (avoids re-running
    the full generate pipeline per test).  Tests MUST NOT mutate this directory.
    Tests that need to tamper with bundle contents should ``shutil.copytree``
    into their own ``tmp_path`` first (see ``TestValidateCommand``).
    """
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
        assert "Error" in result.output

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
        assert "Error" in result.output

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

    def test_override_flag(self, tmp_path: Path) -> None:
        """--override with a valid YAML file should work."""
        override_file = tmp_path / "override.yaml"
        override_file.write_text("n_leads: 25\n")
        out = tmp_path / "override_out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "b2b_saas_procurement_v1",
                "--seed",
                "1",
                "--mode",
                "student_public",
                "--override",
                str(override_file),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"generate failed:\n{result.output}"
        assert (out / "manifest.json").exists()

    def test_override_missing_file_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "b2b_saas_procurement_v1",
                "--seed",
                "1",
                "--mode",
                "student_public",
                "--override",
                str(tmp_path / "nope.yaml"),
                "--out",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_difficulty_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "diff_out"
        result = runner.invoke(
            app,
            [
                "generate",
                "--recipe",
                "b2b_saas_procurement_v1",
                "--seed",
                "1",
                "--mode",
                "student_public",
                "--difficulty",
                "intro",
                "--n-leads",
                "20",
                "--n-accounts",
                "10",
                "--n-contacts",
                "30",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"generate failed:\n{result.output}"
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["difficulty"] == "intro"

    def test_snapshot_day_flag(self, tmp_path: Path) -> None:
        out = tmp_path / "snap_out"
        result = runner.invoke(
            app,
            [
                *_GENERATE_ARGS,
                "--snapshot-day",
                "45",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"generate failed:\n{result.output}"
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["snapshot_day"] == 45

    def test_primary_task_and_label_window_flags(self, tmp_path: Path) -> None:
        out = tmp_path / "task_out"
        result = runner.invoke(
            app,
            [
                *_GENERATE_ARGS,
                "--primary-task",
                "converted_within_60_days",
                "--label-window-days",
                "60",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, f"generate failed:\n{result.output}"
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["primary_task"] == "converted_within_60_days"
        assert manifest["label_window_days"] == 60

    def test_no_window_flags_uses_recipe_defaults(self, bundle_dir: Path) -> None:
        """Regression guard: omitting new flags should still use recipe defaults."""
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        # Recipe defaults from b2b_saas_procurement_v1
        assert manifest["primary_task"] == "converted_within_90_days"
        assert manifest["label_window_days"] == 90
        assert manifest["snapshot_day"] == 30


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    def test_inspect_output(self, bundle_dir: Path) -> None:
        """Single invocation, multiple assertions."""
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert result.exit_code == 0
        output = result.output
        assert "b2b_saas_procurement_v1" in output
        assert "42" in output
        assert "accounts" in output
        assert "leads" in output
        assert "converted_within_90_days" in output
        assert "Metadata dir:" in output

    def test_missing_bundle_fails(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["inspect", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_file_instead_of_dir_fails(self, bundle_dir: Path) -> None:
        """Passing a file path instead of a directory should error clearly."""
        result = runner.invoke(app, ["inspect", str(bundle_dir / "manifest.json")])
        assert result.exit_code != 0
        assert "not a directory" in result.output

    def test_inspect_surfaces_v4_fields(self, bundle_dir: Path) -> None:
        """inspect prints primary_task, label_window_days, snapshot_day, redactions."""
        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        result = runner.invoke(app, ["inspect", str(bundle_dir)])
        assert result.exit_code == 0
        output = result.output
        assert "Primary task:" in output
        assert manifest["primary_task"] in output
        assert "Label window:" in output
        assert str(manifest["label_window_days"]) in output
        assert "Snapshot day:" in output
        assert str(manifest["snapshot_day"]) in output
        assert "Redactions:" in output

    def test_inspect_snapshot_day_none_annotation(self, tmp_path: Path) -> None:
        """A manifest with snapshot_day=None prints the full-horizon annotation."""
        bundle = tmp_path / "manual"
        bundle.mkdir()
        manifest = {
            "bundle_schema_version": "4",
            "package_version": "1.0.0",
            "recipe_id": "x",
            "seed": 1,
            "exposure_mode": "student_public",
            "difficulty": "intro",
            "horizon_days": 90,
            "primary_task": "converted_within_90_days",
            "label_window_days": 90,
            "snapshot_day": None,
            "motif_family": "fit_dominant",
            "redacted_columns": [],
            "tables": {},
            "tasks": {},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        assert "(full horizon, no windowing)" in result.output

    def test_inspect_redactions_empty(self, tmp_path: Path) -> None:
        bundle = tmp_path / "redact_empty"
        bundle.mkdir()
        manifest = {"redacted_columns": [], "tables": {}, "tasks": {}}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        assert "Redactions:" in result.output
        assert "0 column(s)" in result.output

    def test_inspect_redactions_short_list(self, tmp_path: Path) -> None:
        bundle = tmp_path / "redact_short"
        bundle.mkdir()
        manifest = {
            "redacted_columns": ["col_a", "col_b"],
            "tables": {},
            "tasks": {},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        assert "2 column(s)" in result.output
        assert "col_a" in result.output
        assert "col_b" in result.output
        assert "..." not in result.output.split("Redactions:")[1].splitlines()[0]

    def test_inspect_redactions_long_list_truncates(self, tmp_path: Path) -> None:
        bundle = tmp_path / "redact_long"
        bundle.mkdir()
        cols = ["c1", "c2", "c3", "c4", "c5", "c6"]
        manifest = {
            "redacted_columns": cols,
            "tables": {},
            "tasks": {},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        result = runner.invoke(app, ["inspect", str(bundle)])
        assert result.exit_code == 0
        line = [line for line in result.output.splitlines() if "Redactions:" in line][0]
        assert "6 column(s)" in line
        assert "c1" in line
        assert "..." in line
        assert "(6 total)" in line
        # Tail elements should not all appear
        assert "c5" not in line
        assert "c6" not in line

    def test_inspect_json_emits_valid_json(self, bundle_dir: Path) -> None:
        """--json emits valid JSON containing all the human-readable keys."""
        result = runner.invoke(app, ["inspect", str(bundle_dir), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
        for key in (
            "recipe_id",
            "seed",
            "exposure_mode",
            "difficulty",
            "horizon_days",
            "package_version",
            "bundle_schema_version",
            "primary_task",
            "label_window_days",
            "snapshot_day",
            "redacted_columns",
            "motif_family",
            "tables",
            "tasks",
        ):
            assert key in parsed, f"missing key: {key}"

    def test_inspect_json_short_flag(self, bundle_dir: Path) -> None:
        """-j short flag works the same as --json."""
        result = runner.invoke(app, ["inspect", str(bundle_dir), "-j"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["recipe_id"] == "b2b_saas_procurement_v1"

    def test_inspect_json_and_plain_no_cross_contamination(self, bundle_dir: Path) -> None:
        """JSON mode and plain mode return distinct output formats."""
        plain = runner.invoke(app, ["inspect", str(bundle_dir)])
        json_out = runner.invoke(app, ["inspect", str(bundle_dir), "--json"])
        assert plain.exit_code == 0
        assert json_out.exit_code == 0
        # Plain mode has the "Bundle:" header
        assert "Bundle:" in plain.output
        # JSON mode is parseable
        parsed = json.loads(json_out.output)
        assert isinstance(parsed, dict)
        # Plain output should NOT be parseable as JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(plain.output)


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

    def test_file_instead_of_dir_fails(self, bundle_dir: Path) -> None:
        result = runner.invoke(app, ["validate", str(bundle_dir / "manifest.json")])
        assert result.exit_code != 0
        assert "not a directory" in result.output

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
        # Should also report skipped FK checks for the missing table
        assert "FK check skipped" in result.output

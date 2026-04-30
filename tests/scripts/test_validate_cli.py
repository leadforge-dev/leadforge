"""Tests for scripts/validate_lead_scoring_dataset.py CLI entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_v5_dataset, save_csv

# ---------------------------------------------------------------------------
# Import the script module
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_lead_scoring_dataset.py"


# ---------------------------------------------------------------------------
# Session-scoped fixtures (avoid regenerating data per test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def valid_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a CSV that passes all validation checks.

    Uses ``inject_signal=True`` so the LR baseline achieves AUC >= 0.62
    (the shifted sigmoid with bias -0.85 targets ~30% conversion rate).
    """
    tmp = tmp_path_factory.mktemp("validate_cli")
    df = make_v5_dataset(n=400, inject_signal=True, seed=42)
    # Precondition: conversion rate must be in [15%, 40%] for checks to pass
    rate = df["converted"].mean()
    assert 0.15 <= rate <= 0.40, (
        f"Fixture conversion rate {rate:.1%} outside [15%, 40%]; "
        f"adjust sigmoid bias in make_v5_dataset(inject_signal=True)"
    )
    return save_csv(df, tmp, "valid.csv")


@pytest.fixture(scope="session")
def invalid_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a CSV missing the target column (will fail validation)."""
    tmp = tmp_path_factory.mktemp("validate_cli_invalid")
    df = pd.DataFrame({"industry": ["a", "b"], "region": ["US", "UK"]})
    return save_csv(df, tmp, "invalid.csv")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateCLI:
    def test_valid_csv_exit_code_zero(self, valid_csv):
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT_PATH), "--csv", str(valid_csv)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_invalid_csv_exit_code_one(self, invalid_csv):
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT_PATH), "--csv", str(invalid_csv)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1

    def test_out_json_flag(self, valid_csv, tmp_path):
        json_path = tmp_path / "report.json"
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(valid_csv),
                "--out-json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert json_path.exists()
        report = json.loads(json_path.read_text())
        assert "passed" in report
        assert "checks" in report

    def test_emit_release_snippet_flag(self, valid_csv):
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(valid_csv),
                "--emit-release-snippet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "RELEASE SNIPPET" in result.stdout

    def test_enforce_1000_flag_fails_on_small(self, valid_csv):
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(valid_csv),
                "--enforce-1000",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1

    def test_missing_csv_arg_fails(self):
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

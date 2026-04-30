"""Tests for scripts/validate_lead_scoring_dataset.py CLI entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import the script module
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_lead_scoring_dataset.py"


def _make_valid_csv(path: Path, n: int = 400, seed: int = 42) -> Path:
    """Write a small CSV that passes validation (including baseline AUC ≥ 0.62).

    Injects real feature-target correlation so the LR baseline achieves a
    reasonable AUC despite the small sample size.
    """
    rng = np.random.RandomState(seed)

    # Generate a latent score and derive conversion from it, ensuring signal.
    # Shift bias so base rate ≈ 30%.
    latent = rng.normal(0, 1, size=n)
    prob = 1 / (1 + np.exp(-(1.5 * latent - 0.85)))  # shifted sigmoid
    converted = (rng.random(n) < prob).astype(int)

    # Correlated numeric features (positive correlation with latent).
    inbound = np.clip(rng.poisson(3, size=n) + (latent * 1.5).astype(int), 0, None)
    web_sessions = np.clip(rng.poisson(4, size=n) + (latent * 1.0).astype(int), 0, None).astype(
        float
    )
    demo_completed = (latent + rng.normal(0, 0.8, size=n) > 0.3).astype(int)
    opp_created = (latent + rng.normal(0, 0.8, size=n) > 0.0).astype(int)

    df = pd.DataFrame(
        {
            "industry": rng.choice(
                ["manufacturing", "logistics", "services", "healthcare"], size=n
            ),
            "region": rng.choice(["US", "UK"], size=n),
            "company_size": rng.choice(["200-499", "500-999", "1000-1999", "2000+"], size=n),
            "company_revenue": rng.choice(
                ["$1M-$10M", "$10M-$50M", "$50M-$200M", "$200M+"], size=n
            ),
            "contact_role": rng.choice(
                ["finance", "ap_manager", "it_director", "procurement"], size=n
            ),
            "seniority": rng.choice(
                ["individual_contributor", "manager", "director", "vp", "c_suite"], size=n
            ),
            "lead_source": rng.choice(
                ["inbound_marketing", "sdr_outbound", "partner_referral"], size=n
            ),
            "opportunity_created": opp_created,
            "demo_completed": demo_completed,
            "expected_acv": rng.uniform(18_000, 120_000, size=n).round(0),
            "inbound_touches": inbound,
            "outbound_touches": rng.poisson(2, size=n),
            "touches_week_1": rng.poisson(2, size=n),
            "days_since_first_touch": rng.uniform(0, 14, size=n).round(1),
            "web_sessions": web_sessions,
            "sales_activities": rng.poisson(3, size=n),
            "days_since_last_touch": rng.uniform(0, 14, size=n).round(1),
            "__leakage__total_touches_90d": converted * rng.poisson(8, size=n)
            + rng.poisson(3, size=n),
            "converted": converted,
        }
    )
    # Inject small missingness to be realistic
    miss_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    df.loc[miss_idx, "web_sessions"] = np.nan

    csv_path = path / "valid.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _make_invalid_csv(path: Path) -> Path:
    """Write a CSV missing the target column (will fail validation)."""
    df = pd.DataFrame({"industry": ["a", "b"], "region": ["US", "UK"]})
    csv_path = path / "invalid.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateCLI:
    def test_valid_csv_exit_code_zero(self, tmp_path):
        csv_path = _make_valid_csv(tmp_path)
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT_PATH), "--csv", str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_invalid_csv_exit_code_one(self, tmp_path):
        csv_path = _make_invalid_csv(tmp_path)
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(_SCRIPT_PATH), "--csv", str(csv_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 1

    def test_out_json_flag(self, tmp_path):
        csv_path = _make_valid_csv(tmp_path)
        json_path = tmp_path / "report.json"
        subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(csv_path),
                "--out-json",
                str(json_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # JSON report should be written regardless of pass/fail
        assert json_path.exists()
        report = json.loads(json_path.read_text())
        assert "passed" in report
        assert "checks" in report

    def test_emit_release_snippet_flag(self, tmp_path):
        csv_path = _make_valid_csv(tmp_path)
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(csv_path),
                "--emit-release-snippet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Snippet should be emitted regardless of pass/fail
        assert "RELEASE SNIPPET" in result.stdout

    def test_enforce_1000_flag_fails_on_small(self, tmp_path):
        csv_path = _make_valid_csv(tmp_path, n=200)
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(_SCRIPT_PATH),
                "--csv",
                str(csv_path),
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

"""Tests for ``scripts/probe_relational_leakage.py``.

Locks the deterministic reconstruction function (which PR 3.1 will lift
into ``leadforge/validation/leakage_probes.py``), the binary-metrics
helper, and the CLI entrypoint.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_relational_leakage.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_relational_leakage", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
probe_module = importlib.util.module_from_spec(_spec)
sys.modules["probe_relational_leakage"] = probe_module
_spec.loader.exec_module(probe_module)


def _toy_bundle() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A 3-lead toy bundle covering each path through the join graph.

    - lead_001: converted; opp closed_won; customer + subscription rows.
                (Hits A, B, C, D, E.)
    - lead_002: converted; opp exists but close_outcome NaN; customer row exists,
                no subscription. (Hits A and C — and therefore E — but NOT B or D.)
    - lead_003: not converted; opp exists but is open; no customer/sub.
                (Hits none.)
    """
    leads = pd.DataFrame(
        [
            {"lead_id": "lead_001", "converted_within_90_days": True},
            {"lead_id": "lead_002", "converted_within_90_days": True},
            {"lead_id": "lead_003", "converted_within_90_days": False},
        ]
    )
    opportunities = pd.DataFrame(
        [
            {
                "opportunity_id": "opp_001",
                "lead_id": "lead_001",
                "close_outcome": "closed_won",
                "estimated_acv": 50_000,
            },
            {
                "opportunity_id": "opp_002",
                "lead_id": "lead_002",
                "close_outcome": None,
                "estimated_acv": 30_000,
            },
            {
                "opportunity_id": "opp_003",
                "lead_id": "lead_003",
                "close_outcome": None,
                "estimated_acv": 20_000,
            },
        ]
    )
    customers = pd.DataFrame(
        [
            {"customer_id": "cust_001", "opportunity_id": "opp_001", "account_id": "acct_a"},
            {"customer_id": "cust_002", "opportunity_id": "opp_002", "account_id": "acct_b"},
        ]
    )
    subscriptions = pd.DataFrame(
        [
            {"subscription_id": "sub_001", "customer_id": "cust_001", "plan_name": "starter"},
        ]
    )
    return leads, opportunities, customers, subscriptions


def test_deterministic_reconstruction_paths() -> None:
    leads, opportunities, customers, subscriptions = _toy_bundle()

    paths = probe_module.deterministic_relational_reconstruction(
        leads, opportunities, customers, subscriptions
    )

    expected = {
        "path_a_direct_label": [True, True, False],
        "path_b_opportunity_won": [True, False, False],
        "path_c_customer_exists": [True, True, False],
        "path_d_subscription_exists": [True, False, False],
        "path_e_or_b_c_d": [True, True, False],
    }
    for col, vals in expected.items():
        assert list(paths[col].astype(bool)) == vals, f"{col}: {list(paths[col].astype(bool))}"


def test_deterministic_reconstruction_phase2_success_state() -> None:
    """Phase 2 success state — label dropped, customers/subs absent —
    collapses paths A/C/D/E to all-False without crashing."""
    leads, opportunities, _customers, _subscriptions = _toy_bundle()
    leads_safe = leads.drop(columns=["converted_within_90_days"])
    opps_safe = opportunities.drop(columns=["close_outcome"])
    empty_customers = pd.DataFrame(
        {"customer_id": pd.Series(dtype=str), "opportunity_id": pd.Series(dtype=str)}
    )
    empty_subscriptions = pd.DataFrame(
        {"subscription_id": pd.Series(dtype=str), "customer_id": pd.Series(dtype=str)}
    )

    paths = probe_module.deterministic_relational_reconstruction(
        leads_safe, opps_safe, empty_customers, empty_subscriptions
    )
    assert not paths["path_a_direct_label"].any()
    assert not paths["path_b_opportunity_won"].any()
    assert not paths["path_c_customer_exists"].any()
    assert not paths["path_d_subscription_exists"].any()
    assert not paths["path_e_or_b_c_d"].any()


def test_deterministic_reconstruction_rejects_duplicate_lead_id() -> None:
    """A validator cannot operate on non-unique keys — must raise, not silently
    misalign."""
    leads, opportunities, customers, subscriptions = _toy_bundle()
    dup = pd.concat([leads, leads.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="lead_id must be unique"):
        probe_module.deterministic_relational_reconstruction(
            dup, opportunities, customers, subscriptions
        )


def test_binary_metrics_perfect_prediction() -> None:
    y_true = pd.Series([True, True, False, False])
    y_pred = pd.Series([True, True, False, False])
    m = probe_module._binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)


def test_binary_metrics_all_negative_prediction() -> None:
    """Recall undefined-but-not-NaN protection: when no positives predicted but
    positives exist, precision is NaN (no positive predictions) and recall = 0."""
    y_true = pd.Series([True, False, False, False])
    y_pred = pd.Series([False, False, False, False])
    m = probe_module._binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.75
    assert m["recall"] == 0.0
    assert m["precision"] != m["precision"]  # NaN: no positive predictions
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 0, 1, 3)


def test_binary_metrics_mixed() -> None:
    y_true = pd.Series([True, True, False, False])
    y_pred = pd.Series([True, False, True, False])
    m = probe_module._binary_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


# ---------------------------------------------------------------------------
# CLI smoke test — runs against the real alpha bundle if present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "release" / "intermediate" / "tables" / "leads.parquet").exists(),
    reason="alpha bundle not present (clean checkout)",
)
def test_cli_smoke_reports_numeric_accuracy() -> None:
    """Acceptance criterion: ``probe_relational_leakage.py release/intermediate``
    exits 0 and reports a numeric reconstruction accuracy."""
    result = subprocess.run(  # noqa: S603 — controlled args, sys.executable
        [sys.executable, str(_SCRIPT_PATH), str(_REPO_ROOT / "release" / "intermediate")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "path_e_or_b_c_d" in result.stdout
    # The accuracy column should contain at least one float-like token.
    assert any(tok.replace(".", "").isdigit() for tok in result.stdout.split())


@pytest.mark.skipif(
    not (_REPO_ROOT / "release" / "intermediate" / "tables" / "leads.parquet").exists(),
    reason="alpha bundle not present (clean checkout)",
)
def test_cli_max_accuracy_gate_fails_on_alpha() -> None:
    """Phase-2 CI gate flag: alpha bundles trivially exceed any sane threshold,
    so ``--max-accuracy 0.65`` must exit 2."""
    result = subprocess.run(  # noqa: S603 — controlled args, sys.executable
        [
            sys.executable,
            str(_SCRIPT_PATH),
            str(_REPO_ROOT / "release" / "intermediate"),
            "--max-accuracy",
            "0.65",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert "GATE FAILURE" in result.stderr

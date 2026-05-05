"""Tests for ``scripts/probe_relational_leakage.py``.

Exercises the deterministic reconstruction function on a hand-built four-lead
frame that covers each leakage path independently — this is the function PR
3.1 will lift into ``leadforge/validation/leakage_probes.py``, so locking
its behaviour now matters.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_relational_leakage.py"
_spec = importlib.util.spec_from_file_location("probe_relational_leakage", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
probe_module = importlib.util.module_from_spec(_spec)
sys.modules["probe_relational_leakage"] = probe_module
_spec.loader.exec_module(probe_module)


def _toy_bundle() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct a 4-lead toy bundle covering every leakage path independently.

    - lead_001: converted; opp closed_won; customer + subscription rows.
    - lead_002: converted via stale flag in leads.parquet; no joined opp.
                (Tests path A in isolation.)
    - lead_003: converted by ground truth; opp exists but close_outcome NaN.
                customer row exists. (Tests path C reaching when path B can't.)
    - lead_004: not converted; opp exists but is open; no customer/sub.
    """
    leads = pd.DataFrame(
        [
            {"lead_id": "lead_001", "converted_within_90_days": True},
            {"lead_id": "lead_002", "converted_within_90_days": True},
            {"lead_id": "lead_003", "converted_within_90_days": True},
            {"lead_id": "lead_004", "converted_within_90_days": False},
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
                "opportunity_id": "opp_003",
                "lead_id": "lead_003",
                "close_outcome": None,
                "estimated_acv": 30_000,
            },
            {
                "opportunity_id": "opp_004",
                "lead_id": "lead_004",
                "close_outcome": None,
                "estimated_acv": 20_000,
            },
        ]
    )
    customers = pd.DataFrame(
        [
            {"customer_id": "cust_001", "opportunity_id": "opp_001", "account_id": "acct_a"},
            {"customer_id": "cust_003", "opportunity_id": "opp_003", "account_id": "acct_c"},
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

    # Path A — direct read of the label.
    assert bool(paths.loc["lead_001", "path_a_direct_label"]) is True
    assert bool(paths.loc["lead_002", "path_a_direct_label"]) is True
    assert bool(paths.loc["lead_003", "path_a_direct_label"]) is True
    assert bool(paths.loc["lead_004", "path_a_direct_label"]) is False

    # Path B — opportunity closed_won. Only lead_001.
    assert bool(paths.loc["lead_001", "path_b_opportunity_won"]) is True
    assert bool(paths.loc["lead_002", "path_b_opportunity_won"]) is False
    assert bool(paths.loc["lead_003", "path_b_opportunity_won"]) is False
    assert bool(paths.loc["lead_004", "path_b_opportunity_won"]) is False

    # Path C — customer existence. lead_001 + lead_003.
    assert bool(paths.loc["lead_001", "path_c_customer_exists"]) is True
    assert bool(paths.loc["lead_002", "path_c_customer_exists"]) is False
    assert bool(paths.loc["lead_003", "path_c_customer_exists"]) is True
    assert bool(paths.loc["lead_004", "path_c_customer_exists"]) is False

    # Path D — subscription existence. Only lead_001.
    assert bool(paths.loc["lead_001", "path_d_subscription_exists"]) is True
    assert bool(paths.loc["lead_002", "path_d_subscription_exists"]) is False
    assert bool(paths.loc["lead_003", "path_d_subscription_exists"]) is False
    assert bool(paths.loc["lead_004", "path_d_subscription_exists"]) is False

    # Path E — OR of B/C/D. lead_001 + lead_003.
    assert bool(paths.loc["lead_001", "path_e_or_b_c_d"]) is True
    assert bool(paths.loc["lead_002", "path_e_or_b_c_d"]) is False
    assert bool(paths.loc["lead_003", "path_e_or_b_c_d"]) is True
    assert bool(paths.loc["lead_004", "path_e_or_b_c_d"]) is False


def test_deterministic_reconstruction_handles_missing_label_column() -> None:
    """When public ``leads`` has had the label dropped (Phase 2 fix), path A
    must collapse to all-False rather than crash."""
    leads, opportunities, customers, subscriptions = _toy_bundle()
    leads_safe = leads.drop(columns=["converted_within_90_days"])

    paths = probe_module.deterministic_relational_reconstruction(
        leads_safe, opportunities, customers, subscriptions
    )
    assert not paths["path_a_direct_label"].any()
    # Other paths still fire.
    assert bool(paths.loc["lead_001", "path_e_or_b_c_d"]) is True

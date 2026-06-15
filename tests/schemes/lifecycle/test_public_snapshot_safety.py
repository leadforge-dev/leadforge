"""student_public snapshot-safety for the lifecycle scheme (LTV-Pn.4c)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from leadforge.core.models import GenerationConfig
from leadforge.schemes import get_scheme
from leadforge.schemes.lifecycle.render.relational import to_dataframes
from leadforge.schemes.lifecycle.render.relational_snapshot_safe import (
    LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS,
    to_dataframes_snapshot_safe,
)

_TS = "2026-01-01T00:00:00+00:00"
_EVENT_TS = {
    "subscription_events": "event_timestamp",
    "health_signals": "period_start",
    "invoices": "invoice_date",
}
_AT_SIGNING_SUB_COLS = {
    "subscription_id",
    "customer_id",
    "plan_name",
    "subscription_start_at",
    "contract_term_months",
}


def _public_bundle(tmp_path: Path, *, n_customers: int = 150) -> tuple[Path, str]:
    scheme = get_scheme("lifecycle")
    cfg = GenerationConfig(
        seed=42,
        n_customers=n_customers,
        recipe_id="b2b_saas_ltv_v1",
        exposure_mode="student_public",
    )
    bundle = scheme.build_world(cfg, narrative=None)
    out = tmp_path / "bundle"
    scheme.write_bundle(bundle, str(out), generation_timestamp=_TS)
    obs = json.loads((out / "manifest.json").read_text())["observation_date"]
    return out, obs


# ---------------------------------------------------------------------------
# Relational snapshot-safety (the published tables)
# ---------------------------------------------------------------------------


def test_event_tables_filtered_to_observation_date(tmp_path) -> None:
    out, obs = _public_bundle(tmp_path)
    for table, ts_col in _EVENT_TS.items():
        df = pd.read_parquet(out / "tables" / f"{table}.parquet")
        assert (df[ts_col] <= obs).all(), f"{table} has rows after {obs}"


def test_subscriptions_drops_stateful_columns(tmp_path) -> None:
    out, _ = _public_bundle(tmp_path)
    cols = set(pd.read_parquet(out / "tables" / "subscriptions.parquet").columns)
    assert cols == _AT_SIGNING_SUB_COLS
    assert not (cols & set(LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS))


def test_no_target_columns_in_any_public_relational_table(tmp_path) -> None:
    out, _ = _public_bundle(tmp_path)
    banned_targets = {
        "ltv_revenue_90d",
        "ltv_revenue_365d",
        "ltv_revenue_730d",
        "churned_within_180d",
    }
    for parquet in (out / "tables").glob("*.parquet"):
        cols = set(pd.read_parquet(parquet).columns)
        assert not (cols & banned_targets), f"{parquet.name} leaks a target column"


def test_no_metadata_dir_in_public(tmp_path) -> None:
    out, _ = _public_bundle(tmp_path)
    assert not (out / "metadata").exists()


def test_manifest_records_snapshot_safe_and_redactions(tmp_path) -> None:
    out, _ = _public_bundle(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    assert m["relational_snapshot_safe"] is True
    assert m["structural_redactions"] == {
        "columns": {"subscriptions": sorted(LIFECYCLE_BANNED_SUBSCRIPTION_COLUMNS)},
        "omitted_tables": [],
    }


def test_public_tasks_still_single_target_and_keep_trap(tmp_path) -> None:
    out, _ = _public_bundle(tmp_path)
    targets = {"ltv_revenue_90d", "ltv_revenue_365d", "ltv_revenue_730d", "churned_within_180d"}
    for td in (out / "tasks").iterdir():
        manifest = json.loads((td / "task_manifest.json").read_text())
        df = pd.read_parquet(td / "train.parquet")
        assert targets & set(df.columns) == {manifest["label_column"]}
        assert "mrr_change_full_period" in df.columns  # deliberate trap, all modes


def test_public_bundle_deterministic(tmp_path) -> None:
    import hashlib

    def hashes(root: Path) -> dict[str, str]:
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    a, _ = _public_bundle(tmp_path / "a")
    b, _ = _public_bundle(tmp_path / "b")
    assert hashes(a) == hashes(b)


# ---------------------------------------------------------------------------
# Early-pLTV family is omitted from public (its target is relational-reconstructible)
# ---------------------------------------------------------------------------


def test_public_omits_early_pltv_family(tmp_path) -> None:
    """The tenure-anchored early-pLTV tasks are NOT published in student_public:
    their forward window precedes observation_date, so the public event tables
    (<= observation_date) would let a join reconstruct the target."""
    out, _ = _public_bundle(tmp_path)
    task_dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert not any(t.startswith("early_") for t in task_dirs), task_dirs
    assert task_dirs == {
        "pltv_revenue_90d",
        "pltv_revenue_365d",
        "pltv_revenue_730d",
        "churned_within_180d",
    }


def test_published_calendar_target_not_reconstructible_from_public_relational(tmp_path) -> None:
    """Leakage probe: the published pltv_revenue_90d target (revenue AFTER
    observation_date) cannot be recovered from the public invoices table (which
    is filtered to <= observation_date).  Customers with a nonzero target must
    reconstruct to a strictly smaller value (typically 0)."""
    out, obs = _public_bundle(tmp_path, n_customers=200)
    invoices = pd.read_parquet(out / "tables" / "invoices.parquet")
    paid = invoices[invoices.payment_status.isin(["paid", "recovered"])]
    bound = (date.fromisoformat(obs) + timedelta(days=90)).isoformat()
    task = pd.read_parquet(out / "tasks" / "pltv_revenue_90d" / "train.parquet")

    nonzero = task[task["ltv_revenue_90d"] > 0]
    assert len(nonzero) > 0, "fixture should have customers with forward revenue"
    leaked = 0
    for _, row in nonzero.head(60).iterrows():
        inv = paid[paid.customer_id == row["customer_id"]]
        # Everything reconstructible from public relational is <= obs < window.
        recon = float(inv[(inv.invoice_date > obs) & (inv.invoice_date <= bound)].amount_usd.sum())
        if abs(recon - float(row["ltv_revenue_90d"])) < 1e-6:
            leaked += 1
    assert leaked == 0, f"{leaked} calendar targets reconstructible from public relational"


# ---------------------------------------------------------------------------
# Instructor mode is unaffected
# ---------------------------------------------------------------------------


def test_instructor_keeps_full_subscriptions_and_metadata(tmp_path) -> None:
    scheme = get_scheme("lifecycle")
    cfg = GenerationConfig(
        seed=42, n_customers=80, recipe_id="b2b_saas_ltv_v1", exposure_mode="research_instructor"
    )
    out = tmp_path / "inst"
    scheme.write_bundle(scheme.build_world(cfg, narrative=None), str(out), generation_timestamp=_TS)
    subs = set(pd.read_parquet(out / "tables" / "subscriptions.parquet").columns)
    assert {"churn_at", "current_mrr", "subscription_status"} <= subs
    assert (out / "metadata").is_dir()
    m = json.loads((out / "manifest.json").read_text())
    assert m["relational_snapshot_safe"] is False
    # Instructor keeps BOTH regimes — the early family is full-truth here.
    task_dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert sum(t.startswith("early_") for t in task_dirs) == 4
    assert len(task_dirs) == 8


# ---------------------------------------------------------------------------
# to_dataframes_snapshot_safe unit behaviour
# ---------------------------------------------------------------------------


def test_snapshot_safe_passes_through_accounts_and_customers(tmp_path) -> None:
    scheme = get_scheme("lifecycle")
    cfg = GenerationConfig(seed=1, n_customers=60, recipe_id="b2b_saas_ltv_v1")
    arts = scheme.build_world(cfg, narrative=None).artifacts
    full = to_dataframes(arts.simulation_result, arts.population)
    safe = to_dataframes_snapshot_safe(full, cutoff=arts.population.observation_date)
    for name in ("accounts", "customers"):
        assert full[name].equals(safe[name])


def test_snapshot_safe_rejects_empty_cutoff(tmp_path) -> None:
    scheme = get_scheme("lifecycle")
    cfg = GenerationConfig(seed=1, n_customers=30, recipe_id="b2b_saas_ltv_v1")
    arts = scheme.build_world(cfg, narrative=None).artifacts
    full = to_dataframes(arts.simulation_result, arts.population)
    with pytest.raises(ValueError, match="cutoff"):
        to_dataframes_snapshot_safe(full, cutoff="")


def test_snapshot_safe_does_not_mutate_input(tmp_path) -> None:
    scheme = get_scheme("lifecycle")
    cfg = GenerationConfig(seed=1, n_customers=60, recipe_id="b2b_saas_ltv_v1")
    arts = scheme.build_world(cfg, narrative=None).artifacts
    full = to_dataframes(arts.simulation_result, arts.population)
    before = {k: len(v) for k, v in full.items()}
    to_dataframes_snapshot_safe(full, cutoff=arts.population.observation_date)
    assert {k: len(v) for k, v in full.items()} == before
    assert "churn_at" in full["subscriptions"].columns  # original untouched

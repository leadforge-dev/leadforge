"""Tests for ``leadforge/validation/leakage_probes.py``.

Covers every probe family in the unified leakage taxonomy:

* relational / time-window / direct (lifted from PR 2.1's
  ``test_relational_leakage.py``);
* split — ID-overlap, near-duplicate row collisions, label drift;
* model-realism — opt-in calibrated baselines (``probe_id_only_baseline``,
  ``probe_feature_subset_baseline``, ``probe_bonus_model_auc``);
* meta — every ``probe_*`` function in the module is registered in
  :data:`leadforge.validation.leakage_probes.PROBE_REGISTRY` so a
  future "I added a probe but forgot to wire it" regression fails
  loudly here.

For the structural probes each is exercised against two configurations:

* a *clean* bundle, produced by running the same source frames through
  :func:`leadforge.render.relational_snapshot_safe.to_dataframes_snapshot_safe`,
  on which every probe must produce zero findings;
* a *tampered* bundle, in which one leakage channel at a time is
  re-introduced, on which the matching probe must fire with a finding
  that pins the channel and the offending detail.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from leadforge.render.relational_snapshot_safe import to_dataframes_snapshot_safe
from leadforge.validation import leakage_probes
from leadforge.validation.leakage_probes import (
    CHANNEL_BANNED_COLUMN,
    CHANNEL_BANNED_TABLE,
    CHANNEL_BONUS_MODEL,
    CHANNEL_FEATURE_SUBSET_BASELINE,
    CHANNEL_ID_ONLY_BASELINE,
    CHANNEL_JOIN_RECONSTRUCTION,
    CHANNEL_SPLIT_ID_OVERLAP,
    CHANNEL_SPLIT_LABEL_DRIFT,
    CHANNEL_SPLIT_NEAR_DUPLICATE,
    PROBE_REGISTRY,
    LeakageFinding,
    LeakageReport,
    RelationalLeakageError,
    deterministic_relational_reconstruction,
    probe_banned_columns,
    probe_banned_tables,
    probe_bonus_model_auc,
    probe_deterministic_reconstruction,
    probe_feature_subset_baseline,
    probe_id_only_baseline,
    probe_snapshot_window,
    probe_split_id_overlap,
    probe_split_label_drift,
    probe_split_near_duplicates,
    run_all_probes,
    run_all_probes_on_dataframes,
    run_split_probes,
)

ANCHOR = pd.Timestamp("2026-01-01")
SNAPSHOT_DAY = 10


def _ts(offset_days: int) -> str:
    return (ANCHOR + pd.Timedelta(days=offset_days)).isoformat()


def _full_horizon_bundle(*, n_each: int = 25) -> dict[str, pd.DataFrame]:
    """A balanced bundle with 2*n_each leads — half converted, half not.

    Both converted (``lead_C_*``) and unconverted (``lead_U_*``) leads
    own a single opportunity; only converted leads own a customer and a
    subscription.  This mirrors the realistic shape — opportunities for
    everyone, conversion-conditional only customers/subscriptions — so
    that the structural probes have something to bite on without
    accidentally turning ``n_opps`` into a perfect predictor for the
    opt-in bonus-model probe.
    """
    leads_rows: list[dict] = []
    opps_rows: list[dict] = []
    cust_rows: list[dict] = []
    sub_rows: list[dict] = []
    touches_rows: list[dict] = []
    sessions_rows: list[dict] = []
    activities_rows: list[dict] = []

    for i in range(n_each):
        cid = f"lead_C_{i:03d}"
        ucid = f"lead_U_{i:03d}"
        leads_rows.append(
            {
                "lead_id": cid,
                "account_id": f"acct_C_{i:03d}",
                "contact_id": f"con_C_{i:03d}",
                "lead_created_at": ANCHOR.isoformat(),
                "current_stage": "closed_won",
                "is_sql": True,
                "converted_within_90_days": True,
                "conversion_timestamp": _ts(40),
            }
        )
        leads_rows.append(
            {
                "lead_id": ucid,
                "account_id": f"acct_U_{i:03d}",
                "contact_id": f"con_U_{i:03d}",
                "lead_created_at": ANCHOR.isoformat(),
                "current_stage": "discovery",
                "is_sql": False,
                "converted_within_90_days": False,
                "conversion_timestamp": None,
            }
        )
        # ACV drawn from the same distribution for both classes — keeps
        # ``max_acv`` / ``mean_acv`` uninformative for the bonus probe
        # so its enabled-but-clean test path is stable.
        acv = 30_000 + (i % 10) * 2_000
        opps_rows.append(
            {
                "opportunity_id": f"opp_C_{i:03d}",
                "lead_id": cid,
                "created_at": _ts(5),
                "stage": "closed_won",
                "estimated_acv": acv,
                "close_outcome": "closed_won",
                "closed_at": _ts(40),
            }
        )
        opps_rows.append(
            {
                "opportunity_id": f"opp_U_{i:03d}",
                "lead_id": ucid,
                "created_at": _ts(6),
                "stage": "negotiation",
                "estimated_acv": acv,
                "close_outcome": None,
                "closed_at": None,
            }
        )
        cust_rows.append(
            {
                "customer_id": f"cu_{i:03d}",
                "opportunity_id": f"opp_C_{i:03d}",
                "account_id": f"acct_C_{i:03d}",
                "customer_start_at": _ts(40),
            }
        )
        sub_rows.append(
            {
                "subscription_id": f"sub_{i:03d}",
                "customer_id": f"cu_{i:03d}",
                "plan_name": "starter",
                "subscription_start_at": _ts(40),
                "subscription_status": "active",
            }
        )
        touches_rows.append({"touch_id": f"t_C_{i:03d}", "lead_id": cid, "touch_timestamp": _ts(2)})
        sessions_rows.append(
            {"session_id": f"s_C_{i:03d}", "lead_id": cid, "session_timestamp": _ts(3)}
        )
        activities_rows.append(
            {"activity_id": f"a_C_{i:03d}", "lead_id": cid, "activity_timestamp": _ts(7)}
        )
        touches_rows.append(
            {"touch_id": f"t_U_{i:03d}", "lead_id": ucid, "touch_timestamp": _ts(2)}
        )

    return {
        "accounts": pd.DataFrame(
            [{"account_id": r["account_id"]} for r in leads_rows]
        ).drop_duplicates(),
        "contacts": pd.DataFrame(
            [{"contact_id": r["contact_id"], "account_id": r["account_id"]} for r in leads_rows]
        ).drop_duplicates(),
        "leads": pd.DataFrame(leads_rows),
        "touches": pd.DataFrame(touches_rows),
        "sessions": pd.DataFrame(sessions_rows),
        "sales_activities": pd.DataFrame(activities_rows),
        "opportunities": pd.DataFrame(opps_rows),
        "customers": pd.DataFrame(cust_rows),
        "subscriptions": pd.DataFrame(sub_rows),
    }


def _clean_bundle() -> dict[str, pd.DataFrame]:
    """Snapshot-safe bundle — every probe should report zero findings."""
    return to_dataframes_snapshot_safe(_full_horizon_bundle(), snapshot_day=SNAPSHOT_DAY)


def _label_for(bundle: dict[str, pd.DataFrame]) -> pd.Series:
    """Held-back ground truth for the bonus-model probe on a clean bundle."""
    src = _full_horizon_bundle()
    return src["leads"].set_index("lead_id")["converted_within_90_days"]


# ---------------------------------------------------------------------------
# Clean bundle — every probe must be silent.
# ---------------------------------------------------------------------------


def test_clean_bundle_passes_all_probes() -> None:
    """Default orchestrator (structural probes only) must report zero findings."""
    tables = _clean_bundle()
    report = run_all_probes_on_dataframes(tables, snapshot_day=SNAPSHOT_DAY)
    assert report.ok, [f"[{f.channel}] {f.detail}: {f.message}" for f in report.findings]


def test_clean_bundle_passes_with_bonus_probe_enabled() -> None:
    """Bonus probe is opt-in; when enabled with a generous threshold, the
    clean bundle still reports zero findings — exercising the full code
    path without flaking on synthetic-data noise."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    report = run_all_probes_on_dataframes(
        tables,
        snapshot_day=SNAPSHOT_DAY,
        bonus_model_max_auc=0.95,
        label=_label_for(tables),
    )
    assert report.ok, [f"[{f.channel}] {f.detail}: {f.message}" for f in report.findings]


def test_clean_bundle_individual_probes_silent() -> None:
    tables = _clean_bundle()
    assert probe_banned_columns(tables) == []
    assert probe_banned_tables(tables.keys()) == []
    assert probe_deterministic_reconstruction(tables) == []
    assert probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY) == []


# ---------------------------------------------------------------------------
# Banned columns (Path A).
# ---------------------------------------------------------------------------


def test_banned_column_in_leads_fires() -> None:
    tables = _clean_bundle()
    tables["leads"] = tables["leads"].assign(converted_within_90_days=[True] * len(tables["leads"]))
    findings = probe_banned_columns(tables)
    assert any(
        f.channel == CHANNEL_BANNED_COLUMN and f.detail == "leads.converted_within_90_days"
        for f in findings
    )


def test_banned_column_in_opportunities_fires() -> None:
    tables = _clean_bundle()
    tables["opportunities"] = tables["opportunities"].assign(
        close_outcome=["closed_won"] * len(tables["opportunities"])
    )
    findings = probe_banned_columns(tables)
    assert any(
        f.channel == CHANNEL_BANNED_COLUMN and f.detail == "opportunities.close_outcome"
        for f in findings
    )


# ---------------------------------------------------------------------------
# Banned tables.
# ---------------------------------------------------------------------------


def test_banned_table_customers_present_fires() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    findings = probe_banned_tables(tables.keys())
    assert any(f.channel == CHANNEL_BANNED_TABLE and f.detail == "customers" for f in findings)


def test_banned_table_subscriptions_present_fires() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["subscriptions"] = src["subscriptions"]
    findings = probe_banned_tables(tables.keys())
    assert any(f.channel == CHANNEL_BANNED_TABLE and f.detail == "subscriptions" for f in findings)


# ---------------------------------------------------------------------------
# Deterministic paths B / C / D.
# ---------------------------------------------------------------------------


def test_deterministic_path_b_fires_when_close_outcome_present() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["opportunities"] = src["opportunities"]  # restores close_outcome
    findings = probe_deterministic_reconstruction(tables)
    assert any(f.detail == "path_b_opportunity_won" for f in findings)


def test_deterministic_path_c_fires_when_customers_present() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    findings = probe_deterministic_reconstruction(tables)
    assert any(f.detail == "path_c_customer_exists" for f in findings)


def test_deterministic_path_d_fires_when_subscriptions_present() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    tables["subscriptions"] = src["subscriptions"]
    findings = probe_deterministic_reconstruction(tables)
    assert any(f.detail == "path_d_subscription_exists" for f in findings)


def test_deterministic_probe_does_not_flag_path_a() -> None:
    """Path A is covered by probe_banned_columns; the deterministic probe
    must remain silent on it to avoid double-counting the same finding."""
    tables = _clean_bundle()
    tables["leads"] = tables["leads"].assign(converted_within_90_days=[True] * len(tables["leads"]))
    findings = probe_deterministic_reconstruction(tables)
    assert all(f.detail != "path_a_direct_label" for f in findings)


# ---------------------------------------------------------------------------
# Snapshot window.
# ---------------------------------------------------------------------------


def test_snapshot_window_fires_on_late_event() -> None:
    tables = _clean_bundle()
    leaked = pd.DataFrame(
        [
            {
                "touch_id": "t_late",
                "lead_id": tables["leads"]["lead_id"].iloc[0],
                "touch_timestamp": _ts(SNAPSHOT_DAY + 5),
            }
        ]
    )
    tables["touches"] = pd.concat([tables["touches"], leaked], ignore_index=True)
    findings = probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY)
    assert any(f.detail == "touches.touch_timestamp" for f in findings)


def test_snapshot_window_silent_on_clean_bundle() -> None:
    tables = _clean_bundle()
    assert probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY) == []


def test_snapshot_window_negative_day_raises() -> None:
    tables = _clean_bundle()
    with pytest.raises(ValueError, match="non-negative"):
        probe_snapshot_window(tables, snapshot_day=-1)


def test_snapshot_window_duplicate_lead_id_raises() -> None:
    """Duplicate lead_ids would broadcast in the merge; matches the
    same invariant asserted by deterministic_relational_reconstruction."""
    tables = _clean_bundle()
    tables["leads"] = pd.concat([tables["leads"], tables["leads"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="lead_id must be unique"):
        probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY)


def test_snapshot_window_nat_lead_created_at_raises() -> None:
    """NaT in the anchor propagates to NaT cutoffs, masking violations
    via the fillna(False) in the count expression.  The probe must
    refuse to operate on a malformed anchor."""
    tables = _clean_bundle()
    tables["leads"] = tables["leads"].copy()
    tables["leads"].loc[0, "lead_created_at"] = None
    with pytest.raises(ValueError, match="unparseable / null"):
        probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY)


def test_snapshot_window_orphan_event_raises() -> None:
    """An event row whose lead_id is absent from leads gets a NaT cutoff
    after the left-merge; the count would silently miss it.  Treat
    orphan events as a structural violation and raise."""
    tables = _clean_bundle()
    orphan = pd.DataFrame(
        [{"touch_id": "t_orphan", "lead_id": "lead_does_not_exist", "touch_timestamp": _ts(2)}]
    )
    tables["touches"] = pd.concat([tables["touches"], orphan], ignore_index=True)
    with pytest.raises(ValueError, match="touches.parquet has 1 row.*absent from leads"):
        probe_snapshot_window(tables, snapshot_day=SNAPSHOT_DAY)


# ---------------------------------------------------------------------------
# Bonus-model probe.
# ---------------------------------------------------------------------------


def test_bonus_model_probe_fires_when_customers_reintroduced() -> None:
    """Re-adding customers to a clean bundle gives the model n_customers as
    a perfect predictor — AUC should saturate near 1.0 and exceed any sane
    band (we use a deliberately tight 0.95 to guard against flakiness on
    small synthetic data)."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    tables["subscriptions"] = src["subscriptions"]

    label = _label_for(tables)
    findings = probe_bonus_model_auc(tables, max_auc=0.95, label=label)
    assert findings, "expected the bonus model to detect the customers leak"
    assert all(f.channel == CHANNEL_BONUS_MODEL for f in findings)


def test_bonus_model_probe_skipped_without_label() -> None:
    """If `leads.converted_within_90_days` is redacted (clean bundle) and
    no `label` is supplied, the probe has nothing to score against and
    must skip cleanly with zero findings, not error."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    assert probe_bonus_model_auc(tables, max_auc=0.65, label=None) == []


def test_bonus_model_probe_rejects_misaligned_label() -> None:
    """A label not indexed by lead_id would silently misalign and skip the
    probe via the binary-cardinality gate — defeating the validator.
    The probe must reject it loudly instead."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    bad_label = src["leads"]["converted_within_90_days"].reset_index(drop=True)
    assert bad_label.index.name != "lead_id"
    with pytest.raises(ValueError, match="indexed by lead_id"):
        probe_bonus_model_auc(tables, max_auc=0.65, label=bad_label)


def test_bonus_model_probe_rejects_partial_label() -> None:
    """A label that covers only a subset of bundle lead_ids would
    introduce NaN after reindex; the resulting astype(int) cast used to
    crash with an opaque error.  The probe must raise a clear
    ValueError naming the gap instead."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    full_label = _label_for(tables)
    partial = full_label.iloc[:-5]
    with pytest.raises(ValueError, match="missing values for"):
        probe_bonus_model_auc(tables, max_auc=0.65, label=partial)


def test_bonus_model_probe_skips_when_class_too_small_for_cv() -> None:
    """When the smaller class has fewer than 2 members,
    StratifiedKFold cannot run; the probe must skip silently rather
    than raise a sklearn-internal error."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    leads = tables["leads"].head(6).copy()
    tables["leads"] = leads
    label = pd.Series(
        [True, True, True, True, True, False],
        index=leads["lead_id"].to_numpy(),
        name="converted_within_90_days",
    )
    label.index.name = "lead_id"
    assert probe_bonus_model_auc(tables, max_auc=0.65, label=label) == []


def test_bonus_model_probe_uses_smaller_n_splits_when_class_count_lt_5() -> None:
    """When 2 <= min_class_count < 5, the probe must size n_splits
    accordingly — exercised by labelling only 3 positives in a 12-lead
    bundle (n_splits should drop to 3)."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    leads = tables["leads"].head(12).copy()
    tables["leads"] = leads
    tables["customers"] = src["customers"]  # leak signal so AUC saturates
    tables["subscriptions"] = src["subscriptions"]

    truth = [True, True, True] + [False] * 9
    label = pd.Series(truth, index=leads["lead_id"].to_numpy(), name="converted_within_90_days")
    label.index.name = "lead_id"
    findings = probe_bonus_model_auc(tables, max_auc=0.5, label=label)
    assert findings, "expected the bonus model to fire — n_splits should adapt to min_class=3"
    assert all(f.message.startswith("3-fold") for f in findings), (
        "n_splits must downshift to 3 when the minority class has only 3 members"
    )


def test_orchestrator_skips_bonus_probe_by_default() -> None:
    """run_all_probes_on_dataframes(... bonus_model_max_auc=None) must skip
    the bonus probe even when customers/subscriptions would trigger it.
    This is the post-PR-2.1 default; PR 3.3 calibrates and turns it on."""
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    tables["subscriptions"] = src["subscriptions"]

    # The bonus probe would fire here (customers+subs reintroduced), but
    # the structural probes will fire first regardless.  Filter to bonus-
    # channel findings to assert the bonus probe is the one that didn't run.
    report = run_all_probes_on_dataframes(tables, snapshot_day=SNAPSHOT_DAY)
    assert all(f.channel != CHANNEL_BONUS_MODEL for f in report.findings)


def test_orchestrator_runs_bonus_probe_when_enabled() -> None:
    pytest.importorskip("sklearn")
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    tables["customers"] = src["customers"]
    tables["subscriptions"] = src["subscriptions"]

    report = run_all_probes_on_dataframes(
        tables,
        snapshot_day=SNAPSHOT_DAY,
        bonus_model_max_auc=0.95,
        label=_label_for(tables),
    )
    assert any(f.channel == CHANNEL_BONUS_MODEL for f in report.findings)


# ---------------------------------------------------------------------------
# Orchestrator + report ergonomics.
# ---------------------------------------------------------------------------


def test_orchestrator_aggregates_findings_across_channels() -> None:
    tables = _clean_bundle()
    src = _full_horizon_bundle()
    # Re-introduce two channels at once: the label column AND customers.
    tables["leads"] = tables["leads"].assign(converted_within_90_days=[True] * len(tables["leads"]))
    tables["customers"] = src["customers"]

    report = run_all_probes_on_dataframes(tables, snapshot_day=SNAPSHOT_DAY)
    assert not report.ok
    channels = {f.channel for f in report.findings}
    assert CHANNEL_BANNED_COLUMN in channels
    assert CHANNEL_BANNED_TABLE in channels
    assert CHANNEL_JOIN_RECONSTRUCTION in channels


def test_report_raise_if_failing_carries_report() -> None:
    tables = _clean_bundle()
    tables["leads"] = tables["leads"].assign(converted_within_90_days=[True] * len(tables["leads"]))
    report = run_all_probes_on_dataframes(tables, snapshot_day=SNAPSHOT_DAY)
    with pytest.raises(RelationalLeakageError) as excinfo:
        report.raise_if_failing()
    assert excinfo.value.report is report
    assert "banned_column" in str(excinfo.value)


def test_report_ok_property() -> None:
    assert LeakageReport(findings=()).ok is True
    f = LeakageFinding(channel=CHANNEL_BANNED_COLUMN, detail="leads.x", message="...")
    assert LeakageReport(findings=(f,)).ok is False


# ---------------------------------------------------------------------------
# File-based orchestrator.
# ---------------------------------------------------------------------------


def _write_bundle_to_disk(tables: dict[str, pd.DataFrame], root: Path) -> Path:
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(tables_dir / f"{name}.parquet", index=False)
    return root


def test_run_all_probes_reads_bundle_from_disk(tmp_path: Path) -> None:
    bundle_dir = _write_bundle_to_disk(_clean_bundle(), tmp_path / "clean")
    report = run_all_probes(bundle_dir, snapshot_day=SNAPSHOT_DAY)
    assert report.ok, [f"[{f.channel}] {f.detail}: {f.message}" for f in report.findings]


def test_run_all_probes_detects_disk_bundle_leakage(tmp_path: Path) -> None:
    src = _full_horizon_bundle()
    bundle_dir = _write_bundle_to_disk(src, tmp_path / "leaky")
    report = run_all_probes(bundle_dir, snapshot_day=SNAPSHOT_DAY)
    assert not report.ok
    channels = {f.channel for f in report.findings}
    assert CHANNEL_BANNED_COLUMN in channels
    assert CHANNEL_BANNED_TABLE in channels
    assert CHANNEL_JOIN_RECONSTRUCTION in channels


def test_run_all_probes_missing_tables_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing tables/"):
        run_all_probes(tmp_path, snapshot_day=SNAPSHOT_DAY)


def test_run_all_probes_missing_leads_raises(tmp_path: Path) -> None:
    (tmp_path / "tables").mkdir()
    with pytest.raises(FileNotFoundError, match="leads.parquet"):
        run_all_probes(tmp_path, snapshot_day=SNAPSHOT_DAY)


# ---------------------------------------------------------------------------
# Lifted function — sanity check that the package and the script agree.
# ---------------------------------------------------------------------------


def test_deterministic_function_matches_script_export() -> None:
    """The script re-exports the lifted function; calling each must yield
    identical output on the same inputs."""
    import importlib.util
    import sys

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "probe_relational_leakage.py"
    spec = importlib.util.spec_from_file_location("probe_relational_leakage_check", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_relational_leakage_check"] = module
    spec.loader.exec_module(module)

    src = _full_horizon_bundle()
    a = deterministic_relational_reconstruction(
        src["leads"], src["opportunities"], src["customers"], src["subscriptions"]
    )
    b = module.deterministic_relational_reconstruction(
        src["leads"], src["opportunities"], src["customers"], src["subscriptions"]
    )
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# §8.1 Direct — caller-supplied banned sets.
# ---------------------------------------------------------------------------


def test_probe_banned_columns_accepts_custom_banned_set() -> None:
    """A non-relational caller (e.g. flat-CSV exporter) can ban its own
    column list without depending on the snapshot-safe defaults."""
    tables = _clean_bundle()
    tables["leads"] = tables["leads"].assign(secret_score=0.5)
    custom = {"leads": ("secret_score",)}
    findings = probe_banned_columns(tables, banned=custom)
    assert any(f.detail == "leads.secret_score" for f in findings)
    # Default ban list is unaffected.
    assert probe_banned_columns(tables) == []


def test_probe_banned_tables_accepts_custom_banned_set() -> None:
    """Same generalisation for banned tables — caller can declare an
    arbitrary blocklist (e.g. ``raw_telemetry``)."""
    tables = _clean_bundle()
    tables["raw_telemetry"] = pd.DataFrame({"x": [1]})
    findings = probe_banned_tables(tables.keys(), banned=("raw_telemetry",))
    assert any(f.detail == "raw_telemetry" for f in findings)
    # Default ban list is unaffected.
    assert probe_banned_tables(tables.keys()) == []


# ---------------------------------------------------------------------------
# §8.4 Split — ID overlap / near-duplicates / label drift.
# ---------------------------------------------------------------------------


def _split_frames() -> dict[str, pd.DataFrame]:
    """Three disjoint splits with shared schema, no contamination."""
    train = pd.DataFrame(
        {
            "lead_id": [f"l_{i:03d}" for i in range(20)],
            "account_id": [f"a_{i:03d}" for i in range(20)],
            "f1": [float(i) for i in range(20)],
            "f2": [i * 0.5 for i in range(20)],
            "label": [True] * 5 + [False] * 15,
        }
    )
    valid = pd.DataFrame(
        {
            "lead_id": [f"l_{i:03d}" for i in range(20, 25)],
            "account_id": [f"a_{i:03d}" for i in range(20, 25)],
            "f1": [float(i) for i in range(20, 25)],
            "f2": [i * 0.5 for i in range(20, 25)],
            "label": [True, False, True, False, False],
        }
    )
    test = pd.DataFrame(
        {
            "lead_id": [f"l_{i:03d}" for i in range(25, 30)],
            "account_id": [f"a_{i:03d}" for i in range(25, 30)],
            "f1": [float(i) for i in range(25, 30)],
            "f2": [i * 0.5 for i in range(25, 30)],
            "label": [True, False, False, False, True],
        }
    )
    return {"train": train, "valid": valid, "test": test}


def test_split_id_overlap_silent_on_clean_splits() -> None:
    assert probe_split_id_overlap(_split_frames()) == []


def test_split_id_overlap_fires_when_lead_id_shared() -> None:
    splits = _split_frames()
    # Force a leak: copy first train row into test.
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)], ignore_index=True)
    findings = probe_split_id_overlap(splits)
    assert any(f.channel == CHANNEL_SPLIT_ID_OVERLAP for f in findings)
    assert any("lead_id" in f.detail for f in findings)


def test_split_id_overlap_audits_account_id_when_requested() -> None:
    splits = _split_frames()
    # Same account_id appears in train and valid by design.
    shared = splits["train"]["account_id"].head(5).values
    splits["valid"] = splits["valid"].assign(account_id=shared)
    findings = probe_split_id_overlap(splits, id_columns=("account_id",))
    assert any(f.detail.startswith("account_id:") for f in findings)


def test_split_id_overlap_skips_missing_columns() -> None:
    """Splits without an audited id column are silently skipped per-split,
    not a probe-level error.  Allows the probe to be wired generically
    over heterogeneous task schemas."""
    splits = _split_frames()
    splits["valid"] = splits["valid"].drop(columns=["account_id"])
    # account_id audit: train+test still overlap-free, valid skipped — no findings.
    assert probe_split_id_overlap(splits, id_columns=("account_id",)) == []


def test_split_near_duplicates_silent_on_clean_splits() -> None:
    splits = _split_frames()
    findings = probe_split_near_duplicates(splits, feature_columns=("f1", "f2"), decimals=4)
    assert findings == []


def test_split_near_duplicates_fires_on_rounded_match() -> None:
    splits = _split_frames()
    # Inject a row in test whose features round to a train row's features.
    leak_row = splits["train"].iloc[[0]].copy()
    leak_row["lead_id"] = "leak_001"  # different ID, near-duplicate features
    splits["test"] = pd.concat([splits["test"], leak_row], ignore_index=True)
    findings = probe_split_near_duplicates(splits, feature_columns=("f1", "f2"))
    assert any(f.channel == CHANNEL_SPLIT_NEAR_DUPLICATE for f in findings)


def test_split_near_duplicates_skipped_when_no_columns() -> None:
    assert probe_split_near_duplicates(_split_frames(), feature_columns=()) == []


def test_split_label_drift_skipped_below_threshold() -> None:
    splits = _split_frames()
    # Rates are 0.25, 0.4, 0.4 — drift = 0.15.  Pass a threshold above that.
    assert probe_split_label_drift(splits, label_col="label", max_drift=0.5) == []


def test_split_label_drift_fires_when_drift_exceeds_threshold() -> None:
    splits = _split_frames()
    # Force a drift: relabel everything in test to True.
    splits["test"] = splits["test"].assign(label=[True] * len(splits["test"]))
    findings = probe_split_label_drift(splits, label_col="label", max_drift=0.1)
    assert any(f.channel == CHANNEL_SPLIT_LABEL_DRIFT for f in findings)


def test_split_label_drift_negative_threshold_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        probe_split_label_drift(_split_frames(), label_col="label", max_drift=-0.1)


# ---------------------------------------------------------------------------
# §8.5 Model realism — opt-in calibrated baselines.
# ---------------------------------------------------------------------------


def test_probe_id_only_baseline_silent_on_random_splits() -> None:
    """Hashed IDs alone must not predict the label on a clean random split."""
    pytest.importorskip("sklearn")
    splits = _split_frames()
    # max_auc=0.95 is generous: on this 20+5+5 random split a HistGBM
    # trained on hashed IDs should hover near 0.5.  We're not testing
    # the literal AUC, only that the probe doesn't fire spuriously.
    assert probe_id_only_baseline(splits, label_col="label", max_auc=0.95) == []


def test_probe_id_only_baseline_runs_cleanly_on_partitioned_ids() -> None:
    """Wiring smoke test: even when train lead_ids cleanly partition by
    label, the probe completes without error.

    A "fires" demonstration is structurally impossible from hashed
    lead_ids alone: each hash is independent, so HistGBM cannot
    generalise from train hashes to disjoint test hashes — AUC stays
    near 0.5 by construction.  The positive-fire path for the
    model-realism family is covered by
    :func:`test_probe_feature_subset_baseline_fires_when_subset_predicts_label`;
    here we just assert the probe runs end-to-end with a non-trivial
    label.
    """
    pytest.importorskip("sklearn")
    train = pd.DataFrame(
        {
            "lead_id": [f"POS_{i:03d}" for i in range(40)] + [f"NEG_{i:03d}" for i in range(40)],
            "label": [True] * 40 + [False] * 40,
        }
    )
    test = pd.DataFrame(
        {
            "lead_id": [f"POS_{i:03d}" for i in range(40, 50)]
            + [f"NEG_{i:03d}" for i in range(40, 50)],
            "label": [True] * 10 + [False] * 10,
        }
    )
    findings = probe_id_only_baseline(
        {"train": train, "test": test},
        label_col="label",
        max_auc=0.6,
        id_columns=("lead_id",),
    )
    assert isinstance(findings, list)


def test_probe_id_only_baseline_skips_without_train() -> None:
    pytest.importorskip("sklearn")
    splits = _split_frames()
    del splits["train"]
    assert probe_id_only_baseline(splits, label_col="label", max_auc=0.6) == []


def test_probe_feature_subset_baseline_silent_when_features_uninformative() -> None:
    """Numeric features that carry no signal must not exceed even a
    tight band — confirms the wiring runs without spuriously firing."""
    pytest.importorskip("sklearn")
    splits = _split_frames()
    findings = probe_feature_subset_baseline(
        splits,
        feature_columns=("f1", "f2"),
        label_col="label",
        max_auc=0.99,
        name="numeric_only",
    )
    assert findings == []


def test_probe_feature_subset_baseline_fires_when_subset_predicts_label() -> None:
    """A leakage-equivalent feature in the subset must trip the baseline.

    Build a train+test where ``leak`` is monotonic in the label;
    HistGBM should saturate AUC and exceed any sane band.
    """
    pytest.importorskip("sklearn")
    rng_pos = pd.Series(range(40), name="leak").astype(float) + 100.0
    rng_neg = pd.Series(range(40), name="leak").astype(float)
    train = pd.DataFrame(
        {
            "lead_id": [f"l_{i}" for i in range(80)],
            "leak": pd.concat([rng_pos, rng_neg], ignore_index=True),
            "label": [True] * 40 + [False] * 40,
        }
    )
    test = pd.DataFrame(
        {
            "lead_id": [f"t_{i}" for i in range(20)],
            "leak": [200.0] * 10 + [0.0] * 10,
            "label": [True] * 10 + [False] * 10,
        }
    )
    findings = probe_feature_subset_baseline(
        {"train": train, "test": test},
        feature_columns=("leak",),
        label_col="label",
        max_auc=0.6,
        name="leak_subset",
    )
    assert findings, "expected the leakage-equivalent feature subset to exceed max_auc"
    assert all(f.channel == CHANNEL_FEATURE_SUBSET_BASELINE for f in findings)


def test_probe_feature_subset_baseline_skips_when_no_columns_present() -> None:
    pytest.importorskip("sklearn")
    splits = _split_frames()
    assert (
        probe_feature_subset_baseline(
            splits,
            feature_columns=("nonexistent",),
            label_col="label",
            max_auc=0.6,
            name="ghost",
        )
        == []
    )


# ---------------------------------------------------------------------------
# Split-orchestrator wiring.
# ---------------------------------------------------------------------------


def test_run_split_probes_runs_id_overlap_by_default() -> None:
    splits = _split_frames()
    splits["test"] = pd.concat([splits["test"], splits["train"].head(1)], ignore_index=True)
    report = run_split_probes(splits)
    assert any(f.channel == CHANNEL_SPLIT_ID_OVERLAP for f in report.findings)


def test_run_split_probes_skips_opt_in_probes_by_default() -> None:
    """Without explicit thresholds, the orchestrator must not run the
    label-drift, ID-only, or feature-subset probes — calibrated bands
    are PR 3.3's job."""
    pytest.importorskip("sklearn")
    splits = _split_frames()
    report = run_split_probes(splits)
    channels = {f.channel for f in report.findings}
    assert CHANNEL_SPLIT_LABEL_DRIFT not in channels
    assert CHANNEL_ID_ONLY_BASELINE not in channels
    assert CHANNEL_FEATURE_SUBSET_BASELINE not in channels


def test_run_split_probes_runs_all_when_enabled() -> None:
    pytest.importorskip("sklearn")
    splits = _split_frames()
    # Force drift so the label-drift probe has something to surface.
    splits["test"] = splits["test"].assign(label=[True] * len(splits["test"]))
    report = run_split_probes(
        splits,
        label_col="label",
        near_duplicate_columns=("f1", "f2"),
        label_drift_max=0.05,
        id_only_max_auc=0.95,
        feature_subsets={"numeric_only": (0.95, ("f1", "f2"))},
    )
    # At minimum, label-drift must fire.  ID-only / feature-subset may
    # or may not fire on this synthetic data; assertion focuses on the
    # opt-in probes' wiring, not their statistical behaviour.
    channels = {f.channel for f in report.findings}
    assert CHANNEL_SPLIT_LABEL_DRIFT in channels


# ---------------------------------------------------------------------------
# Meta — every probe is registered.
# ---------------------------------------------------------------------------


def test_probe_registry_covers_every_module_level_probe() -> None:
    """Any function named ``probe_*`` in :mod:`leakage_probes` must be
    listed in :data:`PROBE_REGISTRY`.  Guards against the
    "I-added-a-probe-but-forgot-to-register-it" regression so the
    orchestrators stay authoritative."""
    module_probes = {
        name
        for name, obj in inspect.getmembers(leakage_probes, inspect.isfunction)
        if name.startswith("probe_") and obj.__module__ == leakage_probes.__name__
    }
    registered = {f"probe_{spec.name}" for spec in PROBE_REGISTRY.values()}
    missing = module_probes - registered
    extra = registered - module_probes
    assert not missing, f"probes defined but not registered: {sorted(missing)}"
    assert not extra, f"registered probes that don't exist: {sorted(extra)}"


def test_probe_registry_taxonomies_are_known() -> None:
    """Every spec carries one of the five documented taxonomies."""
    valid = {"direct", "time_window", "relational", "split", "model_realism"}
    for spec in PROBE_REGISTRY.values():
        assert spec.taxonomy in valid, f"{spec.name} has unknown taxonomy {spec.taxonomy!r}"


def test_probe_registry_callables_are_callable() -> None:
    for spec in PROBE_REGISTRY.values():
        assert callable(spec.callable), f"{spec.name}.callable is not callable"

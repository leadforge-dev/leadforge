"""Tests for ``scripts/audit_channel_signal.py``.

Exercises the per-channel rollup, univariate-AUC scorer, and the JSON +
markdown rendering paths.  A determinism guard ensures the script's
output is byte-identical across runs against the committed
``release/`` bundles.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_channel_signal.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("audit_channel_signal", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
audit_module = importlib.util.module_from_spec(_spec)
sys.modules["audit_channel_signal"] = audit_module
_spec.loader.exec_module(audit_module)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def _toy_train(n_per_channel: int = 20) -> pd.DataFrame:
    """Three channels with deliberately different conversion rates.

    Channel rates: ``A`` 100%, ``B`` 50%, ``C`` 0%.  Univariate AUC for
    a perfectly separating saturated classifier on this is 1.0 only if
    ``B`` is treated as a tied middle class — otherwise it's the
    standard 1-D Bayes AUC against a 3-bucket score.
    """

    rows = []
    for ch, rate in [("A", 1.0), ("B", 0.5), ("C", 0.0)]:
        for i in range(n_per_channel):
            rows.append(
                {
                    "lead_source": ch,
                    "first_touch_channel": ch,
                    "converted_within_90_days": bool(i < int(rate * n_per_channel)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-channel rollup
# ---------------------------------------------------------------------------


def test_audit_channel_returns_per_channel_stats() -> None:
    df = _toy_train()
    audit = audit_module.audit_channel(df, "lead_source")
    assert audit.column == "lead_source"
    assert audit.n_total == 60
    assert audit.overall_conversion_rate == pytest.approx(0.5)
    names = [c.name for c in audit.channels]
    assert names == ["A", "B", "C"]  # sorted by name
    by_name = {c.name: c for c in audit.channels}
    assert by_name["A"].conversion_rate == pytest.approx(1.0)
    assert by_name["B"].conversion_rate == pytest.approx(0.5)
    assert by_name["C"].conversion_rate == pytest.approx(0.0)
    assert audit.rate_spread == pytest.approx(1.0)


def test_audit_channel_univariate_auc_perfectly_separable() -> None:
    df = _toy_train()
    audit = audit_module.audit_channel(df, "lead_source")
    # 20 pos from A (rate 1.0), 10 pos / 10 neg from B (rate 0.5, tied),
    # 20 neg from C (rate 0.0).  Pair-counting AUC:
    #   A_pos vs B_neg : 200 wins
    #   A_pos vs C_neg : 400 wins
    #   B_pos vs B_neg : 100 ties → +50
    #   B_pos vs C_neg : 200 wins
    # → 850 / 900 = 17/18.
    assert audit.univariate_auc == pytest.approx(17 / 18)


def test_audit_channel_handles_single_class_label() -> None:
    df = _toy_train()
    df["converted_within_90_days"] = False
    audit = audit_module.audit_channel(df, "lead_source")
    assert audit.univariate_auc == 0.5  # AUC undefined → reported as chance


def test_audit_channel_raises_on_missing_column() -> None:
    df = _toy_train()
    with pytest.raises(KeyError):
        audit_module.audit_channel(df, "no_such_column")


def test_audit_tier_runs_every_channel_column() -> None:
    df = _toy_train()
    tier = audit_module.audit_tier(df, "intro")
    cols = {c.column for c in tier.columns}
    assert cols == {"lead_source", "first_touch_channel"}
    assert tier.tier == "intro"
    assert tier.n_leads == 60


# ---------------------------------------------------------------------------
# Build / render
# ---------------------------------------------------------------------------


def test_build_report_round_trips_through_render_json() -> None:
    df = _toy_train()
    tier = audit_module.audit_tier(df, "intro")
    report = audit_module.AuditReport(
        release_dir="release",
        task="converted_within_90_days",
        label_column="converted_within_90_days",
        channel_columns=audit_module.CHANNEL_COLUMNS,
        tiers=(tier,),
        industry_mql_to_sql_benchmarks=audit_module.INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )
    js = audit_module.render_json(report)
    parsed = json.loads(js)
    assert parsed["tiers"][0]["tier"] == "intro"
    assert parsed["industry_mql_to_sql_benchmarks"]["SEO"] == pytest.approx(0.51)


def test_render_markdown_includes_verdict_section() -> None:
    df = _toy_train()
    tier = audit_module.audit_tier(df, "intro")
    report = audit_module.AuditReport(
        release_dir="release",
        task="converted_within_90_days",
        label_column="converted_within_90_days",
        channel_columns=audit_module.CHANNEL_COLUMNS,
        tiers=(tier,),
        industry_mql_to_sql_benchmarks=audit_module.INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )
    md = audit_module.render_markdown(report)
    assert "## Verdict" in md
    assert "## Industry benchmark band" in md
    assert "Tier: `intro`" in md


# ---------------------------------------------------------------------------
# CLI determinism (guards against accidental nondeterminism in either
# the audit functions or the rendering layer)
# ---------------------------------------------------------------------------


_INTRO_TRAIN = (
    _REPO_ROOT / "release" / "intro" / "tasks" / "converted_within_90_days" / "train.parquet"
)


@pytest.mark.skipif(
    not _INTRO_TRAIN.exists(),
    reason="release/intro bundle not present; skipping determinism guard",
)
def test_release_audit_is_deterministic(tmp_path: Path) -> None:
    """Two back-to-back runs against the committed release bundle must
    produce byte-identical JSON and markdown output."""

    out_md_a = tmp_path / "a.md"
    out_json_a = tmp_path / "a.json"
    out_md_b = tmp_path / "b.md"
    out_json_b = tmp_path / "b.json"

    rc_a = audit_module.main(
        [
            "--release-dir",
            str(_REPO_ROOT / "release"),
            "--out-md",
            str(out_md_a),
            "--out-json",
            str(out_json_a),
        ]
    )
    rc_b = audit_module.main(
        [
            "--release-dir",
            str(_REPO_ROOT / "release"),
            "--out-md",
            str(out_md_b),
            "--out-json",
            str(out_json_b),
        ]
    )
    assert rc_a == 0
    assert rc_b == 0
    assert out_md_a.read_bytes() == out_md_b.read_bytes()
    assert out_json_a.read_bytes() == out_json_b.read_bytes()


def test_main_reports_missing_release_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = audit_module.main(
        [
            "--release-dir",
            str(tmp_path / "nope"),
            "--out-md",
            str(tmp_path / "audit.md"),
            "--out-json",
            str(tmp_path / "audit.json"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "release directory not found" in captured.err


def test_main_reports_missing_train_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Empty release dir — tier subdirectory missing.
    (tmp_path / "release").mkdir()
    rc = audit_module.main(
        [
            "--release-dir",
            str(tmp_path / "release"),
            "--tier",
            "intro",
            "--out-md",
            str(tmp_path / "audit.md"),
            "--out-json",
            str(tmp_path / "audit.json"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "missing train split" in captured.err

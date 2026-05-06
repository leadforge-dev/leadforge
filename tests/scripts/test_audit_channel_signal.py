"""Tests for ``scripts/audit_channel_signal.py``.

Exercises the per-channel rollup, in-sample / out-of-sample univariate
AUC scorers, the JSON + markdown rendering paths, and two integrity
properties against the committed ``release/`` bundles:

1. ``lead_source`` and ``first_touch_channel`` carry identical values in
   every tier (the feature dictionary's claim).
2. The committed ``docs/release/channel_signal_audit.{md,json}`` are
   byte-identical to a fresh run of the audit script.

Both properties fail loudly if the bundles are regenerated without
re-running the audit, or if the simulator ever diverges the two
channel columns.
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


_INTRO_TRAIN = (
    _REPO_ROOT / "release" / "intro" / "tasks" / "converted_within_90_days" / "train.parquet"
)
_RELEASE_BUNDLES_PRESENT = _INTRO_TRAIN.exists()

_TIERS = ("intro", "intermediate", "advanced")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _toy_split(n_per_channel: int = 20) -> pd.DataFrame:
    """Three channels with deliberately different conversion rates.

    Channel rates: ``A`` 100%, ``B`` 50%, ``C`` 0%.
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
    train = _toy_split()
    audit = audit_module.audit_channel(train, "lead_source", test=train)
    assert audit.column == "lead_source"
    assert audit.n_train == 60
    assert audit.train_conversion_rate == pytest.approx(0.5)
    names = [c.name for c in audit.channels]
    assert names == ["A", "B", "C"]  # sorted by name
    by_name = {c.name: c for c in audit.channels}
    assert by_name["A"].conversion_rate == pytest.approx(1.0)
    assert by_name["B"].conversion_rate == pytest.approx(0.5)
    assert by_name["C"].conversion_rate == pytest.approx(0.0)
    assert audit.rate_spread == pytest.approx(1.0)


def test_audit_channel_in_sample_auc_pair_counting() -> None:
    """Closed-form check of the in-sample univariate AUC.

    20 pos from A (rate 1.0), 10 pos / 10 neg from B (rate 0.5, tied),
    20 neg from C (rate 0.0).  Pair-counting AUC:
        A_pos vs B_neg : 200 wins
        A_pos vs C_neg : 400 wins
        B_pos vs B_neg : 100 ties → +50
        B_pos vs C_neg : 200 wins
    → 850 / 900 = 17/18.
    """

    train = _toy_split()
    audit = audit_module.audit_channel(train, "lead_source", test=train)
    assert audit.univariate_auc_in_sample == pytest.approx(17 / 18)


def test_audit_channel_oos_auc_matches_in_sample_when_test_is_train() -> None:
    """When the test split is the train split, OOS AUC == in-sample AUC."""

    train = _toy_split()
    audit = audit_module.audit_channel(train, "lead_source", test=train)
    assert audit.univariate_auc_out_of_sample == pytest.approx(audit.univariate_auc_in_sample)


def test_audit_channel_oos_auc_handles_unseen_test_categories() -> None:
    """Test categories not present in train get the train base rate fallback."""

    train = _toy_split()
    test = pd.DataFrame(
        {
            "lead_source": ["A", "B", "C", "Z", "Z"],  # Z is unseen
            "first_touch_channel": ["A", "B", "C", "Z", "Z"],
            "converted_within_90_days": [True, True, False, True, False],
        }
    )
    audit = audit_module.audit_channel(train, "lead_source", test=test)
    # AUC is well-defined (no NaN) — the unseen categories fall back to
    # the train base rate (0.5), which produces ties against any seen
    # category whose rate also equals 0.5.
    assert 0.0 <= audit.univariate_auc_out_of_sample <= 1.0


def test_audit_channel_handles_single_class_label() -> None:
    train = _toy_split()
    train["converted_within_90_days"] = False
    audit = audit_module.audit_channel(train, "lead_source", test=train)
    assert audit.univariate_auc_in_sample == 0.5
    assert audit.univariate_auc_out_of_sample == 0.5


def test_audit_channel_raises_on_missing_column() -> None:
    train = _toy_split()
    with pytest.raises(KeyError):
        audit_module.audit_channel(train, "no_such_column", test=train)


def test_audit_tier_runs_every_channel_column() -> None:
    train = _toy_split()
    tier = audit_module.audit_tier(train, "intro", test=train)
    cols = {c.column for c in tier.columns}
    assert cols == {"lead_source", "first_touch_channel"}
    assert tier.tier == "intro"
    assert tier.n_train == 60
    assert tier.n_test == 60


# ---------------------------------------------------------------------------
# Build / render
# ---------------------------------------------------------------------------


def test_render_json_round_trip() -> None:
    train = _toy_split()
    tier = audit_module.audit_tier(train, "intro", test=train)
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
    # Industry benchmarks render as a {name: rate} dict in the JSON
    # (renderer converts the immutable tuple-of-pairs back).
    assert parsed["industry_mql_to_sql_benchmarks"]["SEO"] == pytest.approx(0.51)


def test_render_markdown_collapses_identical_columns() -> None:
    """When two columns produce identical audits, the renderer groups them."""

    train = _toy_split()  # lead_source == first_touch_channel by construction
    tier = audit_module.audit_tier(train, "intro", test=train)
    report = audit_module.AuditReport(
        release_dir="release",
        task="converted_within_90_days",
        label_column="converted_within_90_days",
        channel_columns=audit_module.CHANNEL_COLUMNS,
        tiers=(tier,),
        industry_mql_to_sql_benchmarks=audit_module.INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )
    md = audit_module.render_markdown(report)
    assert "audit values identical" in md
    # Each tier should render the columns once, not twice.
    assert md.count("Per-channel rate spread") == 1


def test_render_markdown_renders_distinct_columns_separately() -> None:
    """When two columns differ, the renderer keeps them in separate sections."""

    train = _toy_split()
    train["first_touch_channel"] = "A"  # force divergence from lead_source
    tier = audit_module.audit_tier(train, "intro", test=train)
    report = audit_module.AuditReport(
        release_dir="release",
        task="converted_within_90_days",
        label_column="converted_within_90_days",
        channel_columns=audit_module.CHANNEL_COLUMNS,
        tiers=(tier,),
        industry_mql_to_sql_benchmarks=audit_module.INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )
    md = audit_module.render_markdown(report)
    assert "audit values identical" not in md
    assert md.count("Per-channel rate spread") == 2


def test_render_markdown_includes_discussion_section() -> None:
    train = _toy_split()
    tier = audit_module.audit_tier(train, "intro", test=train)
    report = audit_module.AuditReport(
        release_dir="release",
        task="converted_within_90_days",
        label_column="converted_within_90_days",
        channel_columns=audit_module.CHANNEL_COLUMNS,
        tiers=(tier,),
        industry_mql_to_sql_benchmarks=audit_module.INDUSTRY_MQL_TO_SQL_BENCHMARKS,
    )
    md = audit_module.render_markdown(report)
    assert "## Discussion" in md
    assert "## Industry benchmark (context, not target)" in md


# ---------------------------------------------------------------------------
# CLI determinism + error paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release/intro bundle not present")
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


# ---------------------------------------------------------------------------
# Integrity properties against the committed release/ bundles
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release/ bundles not present")
@pytest.mark.parametrize("tier", _TIERS)
def test_lead_source_equals_first_touch_channel_in_v1(tier: str) -> None:
    """Locks the feature-dict claim that the two channel columns are
    identical in v1.  If the simulator ever diverges them, this test
    fails and ``docs/release/feature_dictionary.md`` must be updated."""

    for split in ("train", "test", "valid"):
        df = audit_module.load_split(_REPO_ROOT / "release", tier, split)
        assert (df["lead_source"] == df["first_touch_channel"]).all(), (
            f"{tier}/{split}: lead_source diverges from first_touch_channel"
        )


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release/ bundles not present")
def test_committed_audit_artifacts_match_fresh_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh audit run against the committed bundles must match the
    committed ``docs/release/channel_signal_audit.{md,json}`` exactly.

    If this fails, the bundles drifted without re-running the audit.
    Regenerate via ``python scripts/audit_channel_signal.py`` from the
    repo root.
    """

    # The committed JSON records ``release_dir`` as the literal path
    # the developer passed on the command line.  Re-run the audit
    # exactly as the developer would: from the repo root, with the
    # default (relative) ``release`` argument.
    monkeypatch.chdir(_REPO_ROOT)

    out_md = tmp_path / "audit.md"
    out_json = tmp_path / "audit.json"
    rc = audit_module.main(
        [
            "--out-md",
            str(out_md),
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    committed_md = (_REPO_ROOT / "docs" / "release" / "channel_signal_audit.md").read_bytes()
    committed_json = (_REPO_ROOT / "docs" / "release" / "channel_signal_audit.json").read_bytes()
    assert out_md.read_bytes() == committed_md
    assert out_json.read_bytes() == committed_json

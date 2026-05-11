"""Tests for ``scripts/verify_claims_register.py``.

The verifier is the gate the PR was missing — without these tests it
would be easy to soften the numeric matcher into a no-op (e.g.  by
making ``_extract_numerics`` return ``[]`` for everything) and still
have CI pass.  Each test exercises one drift mode the verifier is
meant to catch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "verify_claims_register.py"


def _load_module() -> ModuleType:
    # Register in ``sys.modules`` BEFORE ``exec_module`` so dataclasses
    # in the module can resolve their own ``__module__``.
    spec = importlib.util.spec_from_file_location("verify_claims_register", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_claims_register"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Multi-path expansion (brace + comma compositionally)
# ---------------------------------------------------------------------------


def test_brace_expansion_splits_choices() -> None:
    mod = _load_module()
    out = mod._expand_multipath("$.x.{a, b, c}.y")
    assert out == ["$.x.a.y", "$.x.b.y", "$.x.c.y"]


def test_comma_split_on_dollar_rooted_paths() -> None:
    mod = _load_module()
    out = mod._expand_multipath("$.a, $.b.c")
    assert set(out) == {"$.a", "$.b.c"}


def test_brace_and_comma_chained() -> None:
    """Both forms can appear in a single backing_path."""

    mod = _load_module()
    out = mod._expand_multipath("$.tables.*.sha256, $.tasks.*.{train,valid,test}_sha256")
    assert set(out) == {
        "$.tables.*.sha256",
        "$.tasks.*.train_sha256",
        "$.tasks.*.valid_sha256",
        "$.tasks.*.test_sha256",
    }


# ---------------------------------------------------------------------------
# Wildcard resolution
# ---------------------------------------------------------------------------


def test_wildcard_fans_out_across_dict_values() -> None:
    mod = _load_module()
    data = {"tables": {"accounts": {"sha256": "a"}, "leads": {"sha256": "b"}}}
    ok, value = mod._resolve_dict_path(data, ["tables", "*", "sha256"])
    assert ok
    assert set(value) == {"a", "b"}


def test_wildcard_fails_on_non_dict() -> None:
    mod = _load_module()
    ok, _ = mod._resolve_dict_path({"x": 5}, ["x", "*", "y"])
    assert not ok


# ---------------------------------------------------------------------------
# Numeric extraction — drift modes
# ---------------------------------------------------------------------------


def test_strong_numerics_include_decimals_percents_commas() -> None:
    mod = _load_module()
    cands = mod._extract_numerics(
        "intro 0.879, intermediate 0.886, advanced 0.886. Conversion 42.67%. Leads 5,000."
    )
    assert 0.879 in cands.strong
    assert 0.886 in cands.strong
    assert 0.4267 in cands.strong
    assert 5000.0 in cands.strong


def test_strong_regex_handles_sentence_ending_period() -> None:
    """Trailing dot must not eat the final digit of the last token."""

    mod = _load_module()
    cands = mod._extract_numerics("advanced 0.351.")
    assert 0.351 in cands.strong


def test_weak_bare_integers_excluded_from_strong_bucket() -> None:
    mod = _load_module()
    cands = mod._extract_numerics("seed 42, version 5")
    assert 42.0 not in cands.strong
    assert 5.0 not in cands.strong
    assert 42 in cands.weak
    assert 5 in cands.weak


def test_year_inside_word_does_not_become_a_candidate() -> None:
    """``v1`` and ``2024-2026`` are version refs / years; not data."""

    mod = _load_module()
    cands = mod._extract_numerics("v1 release 2024-2026")
    assert cands.strong == ()
    # ``v1`` is preceded by a word char — bare integer regex excludes
    # it via ``(?<![\w.])``.  The year tokens land in weak (so JSON
    # values that are integer years would still match) but never in
    # strong.
    assert 1 not in cands.weak  # ``v1`` rejected by lookbehind
    assert 2024 in cands.weak
    assert 2026 in cands.weak


# ---------------------------------------------------------------------------
# End-to-end verify_claims drift detection
# ---------------------------------------------------------------------------


def _write_claims(release_dir: Path, claims: list[dict]) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    source = release_dir / "claims_register_source.yaml"
    source.write_text(yaml.safe_dump({"claims": claims}), encoding="utf-8")
    return source


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_passes_on_aligned_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    release_dir = tmp_path / "release"
    _write_artifact(release_dir / "metrics.json", {"tiers": {"intro": {"lr_auc": 0.879}}})
    source = _write_claims(
        release_dir,
        [
            {
                "id": "c01",
                "text": "LR AUC for intro is 0.879.",
                "category": "calibration",
                "backing_artifact": "release/metrics.json",
                "backing_path": "$.tiers.intro.lr_auc",
                "verifier": "test",
            }
        ],
    )
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert failures == []


def test_verify_flags_numeric_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claim text says 0.879 but the artifact says 0.823 — verifier must surface."""

    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    release_dir = tmp_path / "release"
    _write_artifact(release_dir / "metrics.json", {"tiers": {"intro": {"lr_auc": 0.823}}})
    source = _write_claims(
        release_dir,
        [
            {
                "id": "c01",
                "text": "LR AUC for intro is 0.879.",
                "category": "calibration",
                "backing_artifact": "release/metrics.json",
                "backing_path": "$.tiers.intro.lr_auc",
                "verifier": "test",
            }
        ],
    )
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert len(failures) == 1
    assert "0.823" in failures[0].message
    assert failures[0].claim_id == "c01"


def test_verify_flags_unresolvable_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claim points at a key that doesn't exist."""

    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    release_dir = tmp_path / "release"
    _write_artifact(release_dir / "metrics.json", {"tiers": {"intro": {}}})
    source = _write_claims(
        release_dir,
        [
            {
                "id": "c01",
                "text": "LR AUC for intro is 0.879.",
                "category": "calibration",
                "backing_artifact": "release/metrics.json",
                "backing_path": "$.tiers.intro.lr_auc",
                "verifier": "test",
            }
        ],
    )
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert any("did not resolve" in f.message for f in failures)


def test_verify_skips_prose_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``backing_path: n/a (prose)`` is a soft claim — no JSON to walk."""

    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    release_dir = tmp_path / "release"
    docs = release_dir / "docs"
    docs.mkdir(parents=True)
    (docs / "audit.md").write_text("# audit\n", encoding="utf-8")
    source = _write_claims(
        release_dir,
        [
            {
                "id": "c01",
                "text": "lead_source is weakly informative.",
                "category": "limitations",
                "backing_artifact": "release/docs/audit.md",
                "backing_path": "n/a (prose)",
                "verifier": "test",
            }
        ],
    )
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert failures == []


def test_verify_skip_missing_tier_unless_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing per-tier manifest.json is tolerated on fresh checkouts;
    ``--strict`` upgrades it to a failure."""

    mod = _load_module()
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    release_dir = tmp_path / "release"
    source = _write_claims(
        release_dir,
        [
            {
                "id": "c01",
                "text": "Per-tier composition.",
                "category": "composition",
                "backing_artifact": "release/<tier>/manifest.json",
                "backing_path": "$.n_leads",
                "verifier": "leadforge validate",
            }
        ],
    )
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert failures == []
    failures_strict = mod.verify_claims(source, release_dir, strict=True)
    assert len(failures_strict) == 3  # one per tier


# ---------------------------------------------------------------------------
# Audit-sync gate against the real release tree
# ---------------------------------------------------------------------------


def test_committed_claims_register_verifies_against_release_tree() -> None:
    """The shipped claims register resolves cleanly against the
    shipped artifacts.  This is the gate that catches the case where a
    numeric value in a claim drifts without anyone re-running the
    metrics builder."""

    mod = _load_module()
    source = _REPO_ROOT / "release" / "claims_register_source.yaml"
    release_dir = _REPO_ROOT / "release"
    if not source.is_file():
        pytest.skip("claims_register_source.yaml missing on this checkout")
    failures = mod.verify_claims(source, release_dir, strict=False)
    assert failures == [], (
        "claim drift detected — run scripts/verify_claims_register.py for details"
    )

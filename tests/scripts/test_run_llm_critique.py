"""Tests for ``scripts/run_llm_critique.py``.

No live API.  The canned-client fake from
``tests/validation/test_llm_critique.py`` is replicated here as a
local helper rather than re-imported across the test boundary, so a
breakage in the validation tests doesn't cascade into the driver
tests.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from leadforge.validation.llm_critique import (
    ANTHROPIC_API_KEY_ENV,
    SYSTEM_PROMPT_CLOSE,
    SYSTEM_PROMPT_OPEN,
    USER_CUE_CLOSE,
    USER_CUE_OPEN,
    LLMCritiqueClient,
)

# ---------------------------------------------------------------------------
# Module loader — scripts/ is not on sys.path, so load by file path
# ---------------------------------------------------------------------------


# The driver lives under ``scripts/`` which isn't a package; load it
# by file path the same way ``tests/scripts/test_validate_release_candidate.py``
# does.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_llm_critique.py"
_spec = importlib.util.spec_from_file_location("scripts_run_llm_critique", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
run_llm_critique = importlib.util.module_from_spec(_spec)
sys.modules["scripts_run_llm_critique"] = run_llm_critique
_spec.loader.exec_module(run_llm_critique)


# ---------------------------------------------------------------------------
# Fixture builder — minimal release dir + minimal rubric file
# ---------------------------------------------------------------------------


def _well_formed_payload() -> dict:
    return {
        "release_id": "leadforge-lead-scoring-v1",
        "overall_score": 7,
        "overall_assessment": "Bundle in good shape; one medium finding.",
        "findings": [
            {
                "id": "F001",
                "severity": "medium",
                "category": "documentation",
                "rubric_dimension": "D1",
                "claim": "Stale claim X.",
                "evidence": "release/README.md line 42.",
                "reproducer": "grep -n foo release/README.md",
                "suggested_fix": "Update to bar.",
            }
        ],
        "missing_sections": [],
        "questions_for_maintainer": [],
    }


def _high_severity_payload() -> dict:
    payload = _well_formed_payload()
    payload["findings"][0]["severity"] = "high"
    payload["findings"][0]["category"] = "critical-leakage"
    payload["findings"][0]["rubric_dimension"] = "D2"
    return payload


def _write_minimal_release(tmp_path: Path, *, tier: str = "intermediate") -> Path:
    repo_root = tmp_path
    release_dir = repo_root / "release"
    bundle_dir = release_dir / tier
    (bundle_dir / "tasks" / "converted_within_90_days").mkdir(parents=True, exist_ok=True)
    (release_dir / "validation").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "release").mkdir(parents=True, exist_ok=True)

    (release_dir / "README.md").write_text("# Card\n", encoding="utf-8")
    (bundle_dir / "dataset_card.md").write_text("# Tier card\n", encoding="utf-8")
    (repo_root / "docs" / "release" / "generation_method.md").write_text(
        "# Method\n", encoding="utf-8"
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"bundle_schema_version": "5", "exposure_mode": "student_public"}),
        encoding="utf-8",
    )
    (bundle_dir / "feature_dictionary.csv").write_text(
        "name,dtype,description,leakage_risk\nlead_id,string,id,False\n",
        encoding="utf-8",
    )
    (release_dir / "validation" / "validation_report.md").write_text("# Report\n", encoding="utf-8")
    (release_dir / "validation" / "validation_report.json").write_text(
        json.dumps({"tiers": {tier: {}}}),
        encoding="utf-8",
    )
    df = pd.DataFrame({"lead_id": ["L1", "L2"], "converted_within_90_days": [0, 1]})
    df.to_parquet(bundle_dir / "tasks" / "converted_within_90_days" / "test.parquet")
    (repo_root / "docs" / "release" / "break_me_guide.md").write_text(
        "# Break me\n", encoding="utf-8"
    )
    return release_dir


def _write_minimal_rubric(tmp_path: Path) -> Path:
    """Write a minimal rubric file with the two required section markers."""

    rubric_path = tmp_path / "docs" / "release" / "llm_critique_prompt.md"
    rubric_path.parent.mkdir(parents=True, exist_ok=True)
    rubric_path.write_text(
        f"prelude\n\n{SYSTEM_PROMPT_OPEN}\n\nMinimal system prompt.\n\n"
        f"{SYSTEM_PROMPT_CLOSE}\n\n{USER_CUE_OPEN}\n\nApply the rubric.\n\n"
        f"{USER_CUE_CLOSE}\n",
        encoding="utf-8",
    )
    return rubric_path


@dataclass(frozen=True)
class _CannedClient:
    canned: str

    def run(
        self,
        *,
        system_prompt: str,
        input_bundle_text: str,
        user_cue: str,
        model: str,
        max_tokens: int,
        effort: str,
    ) -> str:
        # Confirm the driver passed every prompt-shape field through.
        assert system_prompt
        assert input_bundle_text
        assert user_cue
        return self.canned


def _config(
    tmp_path: Path,
    rubric: Path,
    release: Path,
    *,
    dry_run: bool = False,
    no_execute: bool = False,
    out_tag: str | None = None,
) -> Any:
    return run_llm_critique.DriverConfig(
        release_dir=release,
        out_dir=tmp_path / "out",
        prompt=rubric,
        model="claude-opus-4-7",
        tier="intermediate",
        effort="high",
        max_tokens=16000,
        out_tag=out_tag,
        dry_run=dry_run,
        no_execute=no_execute,
    )


# ---------------------------------------------------------------------------
# Skip-cleanly path
# ---------------------------------------------------------------------------


class TestSkipCleanly:
    def test_skips_when_key_unset(self, tmp_path: Path) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release)
        result = run_llm_critique.run_critique(config, env={})
        assert result.skipped is True
        assert result.skip_reason is not None
        assert "ANTHROPIC_API_KEY" in result.skip_reason
        assert result.written_files == ()
        # No I/O: out-dir should not have been created.
        assert not (tmp_path / "out").exists()

    def test_skips_when_key_empty(self, tmp_path: Path) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release)
        result = run_llm_critique.run_critique(config, env={ANTHROPIC_API_KEY_ENV: "   "})
        assert result.skipped is True
        assert result.written_files == ()


# ---------------------------------------------------------------------------
# Live happy path (with canned client)
# ---------------------------------------------------------------------------


class TestLivePath:
    def test_writes_both_outputs(self, tmp_path: Path) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release)
        client: LLMCritiqueClient = _CannedClient(json.dumps(_well_formed_payload()))
        result = run_llm_critique.run_critique(
            config,
            client=client,
            env={ANTHROPIC_API_KEY_ENV: "sk-ant-fake"},
        )
        assert result.skipped is False
        assert result.result is not None
        assert result.result.overall_score == 7
        # Two files written: timestamped raw + canonical summary.
        assert len(result.written_files) == 2
        raw, summary = result.written_files
        assert raw.exists()
        assert summary.exists()
        assert summary.name == "llm_critique_summary.md"
        assert raw.name.startswith("llm_critique_raw_")
        assert raw.suffix == ".json"
        # Raw JSON is parseable and matches the result.
        on_disk = json.loads(raw.read_text(encoding="utf-8"))
        assert on_disk["overall_score"] == 7

    def test_high_severity_finding_does_not_short_circuit_writes(self, tmp_path: Path) -> None:
        # Even when there's a high-severity finding, the outputs are
        # written.  The exit code is 1, but the maintainer needs the
        # files on disk to adjudicate.
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release)
        client: LLMCritiqueClient = _CannedClient(json.dumps(_high_severity_payload()))
        result = run_llm_critique.run_critique(
            config,
            client=client,
            env={ANTHROPIC_API_KEY_ENV: "sk"},
        )
        assert result.result is not None
        assert run_llm_critique.has_unresolved_high_severity(result.result)
        assert len(result.written_files) == 2

    def test_out_tag_suffixes_filename(self, tmp_path: Path) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release, out_tag="adj1")
        client: LLMCritiqueClient = _CannedClient(json.dumps(_well_formed_payload()))
        result = run_llm_critique.run_critique(
            config, client=client, env={ANTHROPIC_API_KEY_ENV: "sk"}
        )
        raw = result.written_files[0]
        assert raw.name.endswith("_adj1.json")


# ---------------------------------------------------------------------------
# Dry-run path
# ---------------------------------------------------------------------------


class TestNoExecute:
    def test_no_execute_does_not_read_release_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # --no-execute must short-circuit BEFORE _preflight; pointing
        # --release-dir at a non-existent path proves no I/O occurred.
        rubric = _write_minimal_rubric(tmp_path)
        # build_anthropic_client is called to confirm SDK importability;
        # stub it so no SDK is required.
        canned = _CannedClient(canned="{}")
        monkeypatch.setattr(run_llm_critique, "build_anthropic_client", lambda: canned)
        config = run_llm_critique.DriverConfig(
            release_dir=tmp_path / "no-such-release",  # would FileNotFoundError if read
            out_dir=tmp_path / "out",
            prompt=rubric,
            model="claude-opus-4-7",
            tier="intermediate",
            effort="high",
            max_tokens=16000,
            out_tag=None,
            dry_run=False,
            no_execute=True,
        )
        result = run_llm_critique.run_critique(config, env={ANTHROPIC_API_KEY_ENV: "sk-ant-fake"})
        assert result.skipped is True
        assert "no-execute" in (result.skip_reason or "")
        # No out-dir created.
        assert not (tmp_path / "out").exists()

    def test_no_execute_without_key_fails_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # main() catches MissingCredentialsError → exit code 2 with a
        # pre-flight error on stderr.  --no-execute is the smoke gate;
        # it's supposed to fail loud when creds are missing.
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
        rc = run_llm_critique.main(
            [
                "--release-dir",
                str(release),
                "--out-dir",
                str(tmp_path / "out"),
                "--prompt",
                str(rubric),
                "--no-execute",
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.err


class TestDryRun:
    def test_writes_input_bundle_only(self, tmp_path: Path) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        config = _config(tmp_path, rubric, release, dry_run=True)
        result = run_llm_critique.run_critique(config, env={ANTHROPIC_API_KEY_ENV: ""})
        # Dry-run sidesteps the credentials gate.
        assert result.skipped is True
        assert "dry-run" in (result.skip_reason or "")
        assert len(result.written_files) == 1
        dry = result.written_files[0]
        assert dry.name.startswith("llm_critique_input_")
        # The raw JSON / summary are NOT written.
        assert not (tmp_path / "out" / "llm_critique_summary.md").exists()


# ---------------------------------------------------------------------------
# Schema-validation failure → exit code 2
# ---------------------------------------------------------------------------


class TestSchemaFailure:
    def test_main_returns_2_on_malformed_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        # Stub build_anthropic_client so main() (which calls it implicitly
        # via run_critique on the live path) returns a canned malformed
        # client without touching the SDK.
        bad_client = _CannedClient(canned="not json at all")

        def _fake_builder() -> _CannedClient:
            return bad_client

        monkeypatch.setattr(run_llm_critique, "build_anthropic_client", _fake_builder)
        monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-fake")

        argv = [
            "--release-dir",
            str(release),
            "--out-dir",
            str(tmp_path / "out"),
            "--prompt",
            str(rubric),
        ]
        rc = run_llm_critique.main(argv)
        assert rc == 2
        captured = capsys.readouterr()
        assert "schema-validation error" in captured.err


# ---------------------------------------------------------------------------
# main() exit-code policy on the happy + high-severity paths
# ---------------------------------------------------------------------------


class TestMainExitCodes:
    def test_pass_returns_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        canned = _CannedClient(json.dumps(_well_formed_payload()))
        monkeypatch.setattr(run_llm_critique, "build_anthropic_client", lambda: canned)
        monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-fake")
        rc = run_llm_critique.main(
            [
                "--release-dir",
                str(release),
                "--out-dir",
                str(tmp_path / "out"),
                "--prompt",
                str(rubric),
            ]
        )
        assert rc == 0

    def test_high_severity_returns_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        canned = _CannedClient(json.dumps(_high_severity_payload()))
        monkeypatch.setattr(run_llm_critique, "build_anthropic_client", lambda: canned)
        monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-fake")
        rc = run_llm_critique.main(
            [
                "--release-dir",
                str(release),
                "--out-dir",
                str(tmp_path / "out"),
                "--prompt",
                str(rubric),
            ]
        )
        assert rc == 1

    def test_skip_cleanly_returns_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rubric = _write_minimal_rubric(tmp_path)
        release = _write_minimal_release(tmp_path)
        monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
        rc = run_llm_critique.main(
            [
                "--release-dir",
                str(release),
                "--out-dir",
                str(tmp_path / "out"),
                "--prompt",
                str(rubric),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "SKIPPED" in captured.out

    def test_pre_flight_returns_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Missing release dir → pre-flight failure.
        monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, "sk-ant-fake")
        rc = run_llm_critique.main(
            [
                "--release-dir",
                str(tmp_path / "no-such-release"),
                "--out-dir",
                str(tmp_path / "out"),
                "--prompt",
                str(tmp_path / "no-such-prompt"),
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "pre-flight" in captured.err

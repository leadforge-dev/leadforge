"""Tests for :mod:`leadforge.validation.llm_critique`.

No live API calls.  The Anthropic implementation is exercised only
indirectly via the :class:`leadforge.validation.llm_critique.LLMCritiqueClient`
protocol; tests substitute a small in-process fake.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
)
from leadforge.validation.llm_critique import (
    ANTHROPIC_API_KEY_ENV,
    DEFAULT_THINKING_MODE,
    SYSTEM_PROMPT_CLOSE,
    SYSTEM_PROMPT_OPEN,
    USER_CUE_CLOSE,
    USER_CUE_OPEN,
    VALID_CATEGORIES,
    VALID_RUBRIC_DIMENSIONS,
    VALID_SEVERITIES,
    CritiqueResult,
    CritiqueValidationError,
    Finding,
    LLMCritiqueClient,
    MissingCredentialsError,
    api_key_or_skip,
    build_input_bundle,
    has_anthropic_credentials,
    has_unresolved_high_severity,
    parse_critique_response,
    parse_rubric_prompt,
    raw_output_path,
    render_markdown_summary,
    result_to_dict,
    result_to_json,
    summary_output_path,
)

# ---------------------------------------------------------------------------
# Fixture builders — minimal synthetic release dir
# ---------------------------------------------------------------------------


def _write_minimal_release(
    tmp_path: Path,
    *,
    tier: str = "intermediate",
    n_test_rows: int = 5,
) -> Path:
    """Build a minimal release directory exercising the bundle builder.

    Only the files :func:`build_input_bundle` reads need to exist;
    every other Phase 6 artefact is irrelevant here.
    """

    repo_root = tmp_path
    release_dir = repo_root / "release"
    bundle_dir = release_dir / tier

    (release_dir).mkdir(parents=True, exist_ok=True)
    (bundle_dir).mkdir(parents=True, exist_ok=True)
    (bundle_dir / "tasks" / "converted_within_90_days").mkdir(parents=True, exist_ok=True)
    (release_dir / "validation").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "release").mkdir(parents=True, exist_ok=True)

    # Top-level dataset card (release/README.md).
    (release_dir / "README.md").write_text(
        "# leadforge-lead-scoring-v1\n\nDataset card body.\n",
        encoding="utf-8",
    )

    # Per-tier dataset card.
    (bundle_dir / "dataset_card.md").write_text(
        f"# {tier} tier\n\nPer-tier card.\n", encoding="utf-8"
    )

    # generation_method.md.
    (repo_root / "docs" / "release" / "generation_method.md").write_text(
        "# Generation method\n\nDGP summary.\n", encoding="utf-8"
    )

    # manifest.json.
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": "5",
                "package_version": "1.0.0",
                "recipe_id": "b2b_saas_procurement_v1",
                "seed": 42,
                "exposure_mode": "student_public",
                "difficulty": tier,
                "relational_snapshot_safe": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # feature_dictionary.csv.
    (bundle_dir / "feature_dictionary.csv").write_text(
        "name,dtype,description,leakage_risk\n"
        "lead_id,string,Stable lead identifier,False\n"
        "industry,string,Industry segment,False\n"
        "converted_within_90_days,int,Target,False\n",
        encoding="utf-8",
    )

    # validation_report.{md,json}.
    (release_dir / "validation" / "validation_report.md").write_text(
        "# Validation report\n\nMetrics.\n", encoding="utf-8"
    )
    (release_dir / "validation" / "validation_report.json").write_text(
        json.dumps({"tiers": {tier: {"medians": {"average_precision": 0.42}}}}),
        encoding="utf-8",
    )

    # Test split — render via parquet so build_input_bundle can read it.
    df = pd.DataFrame(
        {
            "lead_id": [f"lead_{i:05d}" for i in range(n_test_rows)],
            "industry": ["logistics"] * n_test_rows,
            "converted_within_90_days": [i % 2 for i in range(n_test_rows)],
        }
    )
    df.to_parquet(bundle_dir / "tasks" / "converted_within_90_days" / "test.parquet")

    # break_me_guide.md.
    (repo_root / "docs" / "release" / "break_me_guide.md").write_text(
        "# Break me guide\n\nNine patterns.\n", encoding="utf-8"
    )

    return release_dir


def _well_formed_response_payload(*, severity: str = "medium") -> dict:
    """Build a payload that satisfies the schema validator."""
    return {
        "release_id": "leadforge-lead-scoring-v1",
        "overall_score": 7,
        "overall_assessment": ("Bundle is in good shape; one medium finding worth addressing."),
        "findings": [
            {
                "id": "F001",
                "severity": severity,
                "category": "documentation",
                "rubric_dimension": "D1",
                "claim": "Dataset card claim X is stale.",
                "evidence": "release/README.md line 42 references 'foo'.",
                "reproducer": "grep -n 'foo' release/README.md",
                "suggested_fix": "Update to 'bar'.",
            }
        ],
        "missing_sections": ["missing: maintenance plan — needed for HF README"],
        "questions_for_maintainer": [
            "Is the channel-signal audit a fixed snapshot or live recomputed?"
        ],
    }


# ---------------------------------------------------------------------------
# Skip-cleanly path — has_anthropic_credentials / api_key_or_skip
# ---------------------------------------------------------------------------


class TestCredentialsGate:
    def test_unset_means_absent(self) -> None:
        assert has_anthropic_credentials({}) is False

    def test_empty_string_means_absent(self) -> None:
        assert has_anthropic_credentials({ANTHROPIC_API_KEY_ENV: ""}) is False

    def test_whitespace_only_means_absent(self) -> None:
        assert has_anthropic_credentials({ANTHROPIC_API_KEY_ENV: "   \t\n"}) is False

    def test_real_value_means_present(self) -> None:
        assert has_anthropic_credentials({ANTHROPIC_API_KEY_ENV: "sk-ant-something"}) is True

    def test_api_key_or_skip_returns_stripped(self) -> None:
        assert api_key_or_skip({ANTHROPIC_API_KEY_ENV: "  sk-ant  "}) == "sk-ant"

    def test_api_key_or_skip_raises_on_absent(self) -> None:
        with pytest.raises(MissingCredentialsError):
            api_key_or_skip({})


# ---------------------------------------------------------------------------
# Rubric prompt parser
# ---------------------------------------------------------------------------


class TestParseRubricPrompt:
    def test_extracts_both_sections(self) -> None:
        rubric = (
            f"prelude\n\n{SYSTEM_PROMPT_OPEN}\n\nSYS\n\n{SYSTEM_PROMPT_CLOSE}\n\n"
            f"middle\n\n{USER_CUE_OPEN}\n\nCUE\n\n{USER_CUE_CLOSE}\n\nepilogue"
        )
        sys_prompt, cue = parse_rubric_prompt(rubric)
        assert sys_prompt == "SYS"
        assert cue == "CUE"

    def test_missing_system_prompt_raises(self) -> None:
        rubric = f"{USER_CUE_OPEN}cue{USER_CUE_CLOSE}"
        with pytest.raises(ValueError, match="system_prompt"):
            parse_rubric_prompt(rubric)

    def test_missing_user_cue_raises(self) -> None:
        rubric = f"{SYSTEM_PROMPT_OPEN}sys{SYSTEM_PROMPT_CLOSE}"
        with pytest.raises(ValueError, match="user_cue"):
            parse_rubric_prompt(rubric)

    def test_real_rubric_file_parses(self) -> None:
        # Smoke test against the actual rubric checked into the repo.
        rubric_path = Path("docs/release/llm_critique_prompt.md")
        if not rubric_path.exists():
            pytest.skip("rubric file not present in this checkout")
        sys_prompt, cue = parse_rubric_prompt(rubric_path.read_text(encoding="utf-8"))
        assert "Output contract" in sys_prompt
        assert "Apply the rubric above" in cue


# ---------------------------------------------------------------------------
# Input-bundle builder — determinism + sync with leakage_probes constants
# ---------------------------------------------------------------------------


class TestBuildInputBundle:
    def test_deterministic_same_input(self, tmp_path: Path) -> None:
        release_dir = _write_minimal_release(tmp_path)
        a = build_input_bundle(release_dir, tier="intermediate")
        b = build_input_bundle(release_dir, tier="intermediate")
        assert a.sha256 == b.sha256
        assert a.bundle_hashes == b.bundle_hashes
        assert a.render() == b.render()

    def test_block_order_is_pinned(self, tmp_path: Path) -> None:
        release_dir = _write_minimal_release(tmp_path)
        bundle = build_input_bundle(release_dir, tier="intermediate")
        names = [b.name for b in bundle.blocks]
        # Pinned: README first, break-me guide last; in between, the
        # other nine blocks in the order the rubric expects.
        assert names[0] == "release/README.md"
        assert names[-1].startswith("docs/release/break_me_guide.md")
        # The eleven blocks the design doc commits to.
        assert len(names) == 11

    def test_diff_summary_lists_every_banned_constant(self, tmp_path: Path) -> None:
        # The whole point of live-referencing leakage_probes constants
        # is that the diff summary stays in sync.  Pin that explicitly.
        release_dir = _write_minimal_release(tmp_path)
        bundle = build_input_bundle(release_dir, tier="intermediate")
        diff_block = next(b for b in bundle.blocks if "diff summary" in b.name)
        for col in BANNED_LEAD_COLUMNS:
            assert f"`{col}`" in diff_block.body
        for col in BANNED_OPP_COLUMNS:
            assert f"`{col}`" in diff_block.body
        for table in BANNED_TABLES:
            assert f"`{table}`" in diff_block.body

    def test_test_split_sample_renders_describe_and_head(self, tmp_path: Path) -> None:
        release_dir = _write_minimal_release(tmp_path, n_test_rows=5)
        bundle = build_input_bundle(release_dir, tier="intermediate", n_test_sample_rows=3)
        block = next(b for b in bundle.blocks if "test.parquet" in b.name)
        # Both sections are present: per-column statistics and a row head.
        assert "## Per-column statistics (df.describe)" in block.body
        assert "## First 3 rows (df.head)" in block.body
        # The head's CSV header lists the columns.
        assert "lead_id,industry,converted_within_90_days" in block.body

    def test_missing_input_raises_filenotfound(self, tmp_path: Path) -> None:
        release_dir = _write_minimal_release(tmp_path)
        # Remove a required input.
        (release_dir / "README.md").unlink()
        with pytest.raises(FileNotFoundError, match="README.md"):
            build_input_bundle(release_dir, tier="intermediate")

    def test_per_file_hashes_carry_each_input(self, tmp_path: Path) -> None:
        release_dir = _write_minimal_release(tmp_path)
        bundle = build_input_bundle(release_dir, tier="intermediate")
        # Eleven hashes, one per logical block.
        assert len(bundle.bundle_hashes) == 11
        assert all(len(digest) == 64 for digest in bundle.bundle_hashes.values()), (
            "expected sha256 hex digests"
        )

    def test_mechanism_summary_tracks_requested_tier(self, tmp_path: Path) -> None:
        # COPILOT-1 fix: --tier advanced must produce an "advanced tier"
        # mechanism block, not a hardcoded "intermediate tier" header.
        release_dir = tmp_path / "release"
        for tier in ("intermediate", "advanced"):
            (release_dir / tier).mkdir(parents=True, exist_ok=True)
        # Write all required inputs for both tiers; the only thing
        # that differs is the per-tier dir name.
        _write_minimal_release(tmp_path, tier="intermediate")
        _write_minimal_release(tmp_path, tier="advanced")
        intermediate = build_input_bundle(release_dir, tier="intermediate")
        advanced = build_input_bundle(release_dir, tier="advanced")
        intermediate_summary = next(
            b for b in intermediate.blocks if b.name == "public-safe mechanism summary"
        )
        advanced_summary = next(
            b for b in advanced.blocks if b.name == "public-safe mechanism summary"
        )
        assert "(intermediate tier)" in intermediate_summary.body
        assert "(advanced tier)" in advanced_summary.body
        # Sanity: the two tiers produce different mechanism blocks
        # (the header alone makes them differ).
        assert intermediate_summary.body != advanced_summary.body

    def test_real_release_dir_smoke(self) -> None:
        # Smoke test against the real ``release/`` artefacts on disk:
        # all eleven source files resolve, every block has a non-empty
        # body, and re-running the builder produces identical hashes.
        # Skipped when the release dir isn't present (CI on a fresh
        # checkout, or the in-package test run).
        release_dir = Path("release")
        if not (release_dir / "intermediate" / "manifest.json").exists():
            pytest.skip("release/intermediate/ not present in this checkout")
        if not (release_dir / "validation" / "validation_report.json").exists():
            pytest.skip("release/validation/ not present in this checkout")
        bundle = build_input_bundle(release_dir, tier="intermediate")
        # Eleven blocks with non-empty bodies.
        assert len(bundle.blocks) == 11
        for block in bundle.blocks:
            assert block.body.strip(), f"block {block.name!r} has empty body"
        # Determinism on the real artefacts: re-build, same hashes.
        rerun = build_input_bundle(release_dir, tier="intermediate")
        assert bundle.bundle_hashes == rerun.bundle_hashes
        assert bundle.sha256 == rerun.sha256


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------


def _parse_payload(payload: dict, *, run_timestamp: str = "2026-05-08T12:00:00Z") -> CritiqueResult:
    """Convenience wrapper for the validator under test."""
    return parse_critique_response(
        json.dumps(payload),
        model="claude-opus-4-7",
        effort="high",
        thinking_mode=DEFAULT_THINKING_MODE,
        bundle_hashes={"release/README.md": "abc"},
        input_bundle_sha256="def",
        run_timestamp=run_timestamp,
    )


class TestSchemaValidator:
    def test_well_formed_payload_round_trips(self) -> None:
        result = _parse_payload(_well_formed_response_payload())
        assert isinstance(result, CritiqueResult)
        assert result.overall_score == 7
        assert len(result.findings) == 1
        assert result.findings[0].severity == "medium"
        assert result.findings[0].rubric_dimension == "D1"

    def test_missing_required_top_level_field(self) -> None:
        payload = _well_formed_response_payload()
        del payload["overall_score"]
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("overall_score" in p for p in excinfo.value.problems)

    def test_invalid_severity(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"][0]["severity"] = "catastrophic"
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("severity" in p and "catastrophic" in p for p in excinfo.value.problems)

    def test_invalid_category(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"][0]["category"] = "vibes"
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("category" in p and "vibes" in p for p in excinfo.value.problems)

    def test_invalid_rubric_dimension(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"][0]["rubric_dimension"] = "D99"
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("D99" in p for p in excinfo.value.problems)

    def test_finding_id_collision(self) -> None:
        payload = _well_formed_response_payload()
        # Append a duplicate-id second finding.
        dup = dict(payload["findings"][0])
        payload["findings"].append(dup)
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("collide" in p for p in excinfo.value.problems)

    def test_findings_must_be_list(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"] = "not a list"
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("findings" in p for p in excinfo.value.problems)

    def test_top_level_non_object(self) -> None:
        with pytest.raises(CritiqueValidationError) as excinfo:
            parse_critique_response(
                json.dumps([1, 2, 3]),
                model="m",
                effort="high",
                thinking_mode=DEFAULT_THINKING_MODE,
                bundle_hashes={},
                input_bundle_sha256="",
            )
        assert any("object" in p for p in excinfo.value.problems)

    def test_non_json_response(self) -> None:
        with pytest.raises(CritiqueValidationError) as excinfo:
            parse_critique_response(
                "Sure, here's my critique:\nThe dataset looks fine!",
                model="m",
                effort="high",
                thinking_mode=DEFAULT_THINKING_MODE,
                bundle_hashes={},
                input_bundle_sha256="",
            )
        assert any("not valid JSON" in p for p in excinfo.value.problems)

    def test_strips_outer_code_fence(self) -> None:
        # Defensive: even though the rubric forbids fences, a single
        # outer fence shouldn't hard-fail.
        payload = _well_formed_response_payload()
        wrapped = "```json\n" + json.dumps(payload) + "\n```"
        result = parse_critique_response(
            wrapped,
            model="m",
            effort="high",
            thinking_mode=DEFAULT_THINKING_MODE,
            bundle_hashes={},
            input_bundle_sha256="",
        )
        assert result.overall_score == payload["overall_score"]

    def test_overall_score_out_of_range(self) -> None:
        payload = _well_formed_response_payload()
        payload["overall_score"] = 11
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("[1, 10]" in p for p in excinfo.value.problems)

    def test_empty_findings_list_is_valid(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"] = []
        result = _parse_payload(payload)
        assert result.findings == []

    def test_wrong_release_id_rejected(self) -> None:
        # Strict release_id check — silent drift would defeat the
        # audit-artifact-sync contract the design doc commits to.
        payload = _well_formed_response_payload()
        payload["release_id"] = "leadforge-xyz"
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("release_id" in p and "leadforge-xyz" in p for p in excinfo.value.problems)

    def test_non_string_prose_field_rejected(self) -> None:
        # Silent str() coercion would let an int "claim" land on disk
        # as the string "5" with no audit trail.
        payload = _well_formed_response_payload()
        payload["findings"][0]["claim"] = 42
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("claim must be a string" in p for p in excinfo.value.problems)

    def test_non_string_missing_section_rejected(self) -> None:
        payload = _well_formed_response_payload()
        payload["missing_sections"] = ["ok", 42]
        with pytest.raises(CritiqueValidationError) as excinfo:
            _parse_payload(payload)
        assert any("missing_sections" in p for p in excinfo.value.problems)


# ---------------------------------------------------------------------------
# Severity policy
# ---------------------------------------------------------------------------


class TestSeverityPolicy:
    def test_high_severity_flagged(self) -> None:
        result = _parse_payload(_well_formed_response_payload(severity="high"))
        assert has_unresolved_high_severity(result) is True

    def test_medium_severity_does_not_flag(self) -> None:
        result = _parse_payload(_well_formed_response_payload(severity="medium"))
        assert has_unresolved_high_severity(result) is False

    def test_no_findings_does_not_flag(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"] = []
        result = _parse_payload(payload)
        assert has_unresolved_high_severity(result) is False


# ---------------------------------------------------------------------------
# Constants alignment
# ---------------------------------------------------------------------------


class TestVocabulariesAlignWithBreakMeGuide:
    def test_categories_match_break_me_guide(self) -> None:
        # The break-me guide is the source of truth for the triage label
        # vocabulary; assert in lockstep.
        guide_path = Path("docs/release/break_me_guide.md")
        if not guide_path.exists():
            pytest.skip("break-me guide not present in this checkout")
        guide_text = guide_path.read_text(encoding="utf-8")
        for category in VALID_CATEGORIES:
            assert f"`{category}`" in guide_text, (
                f"category {category!r} not mentioned in break_me_guide.md; vocabulary has drifted"
            )

    def test_rubric_dimensions_are_d1_through_d13(self) -> None:
        assert VALID_RUBRIC_DIMENSIONS == {f"D{i}" for i in range(1, 14)}

    def test_severities_are_three_values(self) -> None:
        assert VALID_SEVERITIES == frozenset({"high", "medium", "low"})


# ---------------------------------------------------------------------------
# Round-tripping result_to_dict / result_to_json
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_result_to_dict_round_trip(self) -> None:
        result = _parse_payload(_well_formed_response_payload())
        d = result_to_dict(result)
        assert d["overall_score"] == 7
        assert isinstance(d["findings"], list)
        assert d["findings"][0]["id"] == "F001"

    def test_result_to_json_is_stable(self) -> None:
        result = _parse_payload(_well_formed_response_payload())
        a = result_to_json(result)
        b = result_to_json(result)
        assert a == b
        assert json.loads(a) == result_to_dict(result)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


class TestMarkdownSummary:
    def test_renders_findings_grouped_by_severity(self) -> None:
        payload = _well_formed_response_payload()
        # Add one high-severity finding too.
        payload["findings"].append(
            {
                "id": "F002",
                "severity": "high",
                "category": "critical-leakage",
                "rubric_dimension": "D2",
                "claim": "Undocumented join path reconstructs the label.",
                "evidence": "...",
                "reproducer": "...",
                "suggested_fix": "...",
            }
        )
        result = _parse_payload(payload)
        md = render_markdown_summary(result)
        assert "Severity: high (1)" in md
        assert "Severity: medium (1)" in md
        assert "F001" in md
        assert "F002" in md
        # Bundle hashes table renders.
        assert "Bundle hashes (audit)" in md

    def test_no_findings_shows_placeholder(self) -> None:
        payload = _well_formed_response_payload()
        payload["findings"] = []
        result = _parse_payload(payload)
        md = render_markdown_summary(result)
        assert "*No findings reported.*" in md


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------


class TestOutputPaths:
    def test_raw_path_includes_timestamp(self, tmp_path: Path) -> None:
        ts = "2026-05-08T12:00:00.123456Z"
        p = raw_output_path(tmp_path, ts)
        assert p.name == "llm_critique_raw_20260508T120000.123456Z.json"
        assert p.parent == tmp_path

    def test_raw_path_with_tag(self, tmp_path: Path) -> None:
        ts = "2026-05-08T12:00:00.123456Z"
        p = raw_output_path(tmp_path, ts, tag="adj1")
        assert p.name == "llm_critique_raw_20260508T120000.123456Z_adj1.json"

    def test_summary_path_canonical(self, tmp_path: Path) -> None:
        p = summary_output_path(tmp_path)
        assert p.name == "llm_critique_summary.md"

    def test_microsecond_precision_avoids_collision(self) -> None:
        # Two timestamps that differ only in the microsecond field
        # must produce different filenames so adjacent runs in the
        # same wall-clock second don't clobber the raw JSON history.
        ts1 = "2026-05-08T12:00:00.000001Z"
        ts2 = "2026-05-08T12:00:00.000002Z"
        assert raw_output_path(Path("."), ts1) != raw_output_path(Path("."), ts2)


# ---------------------------------------------------------------------------
# LLMCritiqueClient protocol — mocked end-to-end through parse_critique_response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CannedCritiqueClient:
    """Protocol-conforming fake that returns a checked-in JSON string."""

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
        # Sanity-check the protocol contract: the driver must pass
        # non-empty values for the four prompt-shape arguments.
        assert system_prompt
        assert input_bundle_text
        assert user_cue
        return self.canned


class TestProtocolWiring:
    def test_canned_client_satisfies_protocol(self) -> None:
        client: LLMCritiqueClient = _CannedCritiqueClient(canned="{}")
        # Protocol structural typing check: this assignment is the test.
        assert client is not None

    def test_full_round_trip_with_mock(self) -> None:
        canned = json.dumps(_well_formed_response_payload())
        client: LLMCritiqueClient = _CannedCritiqueClient(canned=canned)
        raw = client.run(
            system_prompt="sys",
            input_bundle_text="bundle",
            user_cue="cue",
            model="claude-opus-4-7",
            max_tokens=16000,
            effort="high",
        )
        result = parse_critique_response(
            raw,
            model="claude-opus-4-7",
            effort="high",
            thinking_mode=DEFAULT_THINKING_MODE,
            bundle_hashes={"x": "y"},
            input_bundle_sha256="z",
        )
        assert result.overall_score == 7
        assert isinstance(result.findings[0], Finding)

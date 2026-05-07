"""LLM critique module for ``leadforge-lead-scoring-v1`` release candidates.

PR 7.1's structured-critique core: builds the deterministic input
bundle that the rubric prompt is fed against, calls the LLM provider
through a single-implementation protocol abstraction, validates the
returned JSON against the v1 critique schema, and renders a human-
readable Markdown summary.

Companion files:

* :mod:`scripts.run_llm_critique` — the driver (CLI + filesystem
  glue).
* ``docs/release/llm_critique_prompt.md`` — the rubric the driver
  feeds to this module.
* ``docs/release/llm_critique_design.md`` — the load-bearing design
  decisions, referenced from the rubric and the v2 decision log.

Out of scope here:

* Live API calls in tests (the test suite mocks the
  :class:`LLMCritiqueClient` protocol; see
  ``tests/validation/test_llm_critique.py``).
* Multi-provider support (single-provider for v1; the protocol is
  the seam for a future provider, not an inline switch).
* Bundle regeneration (``BUNDLE_SCHEMA_VERSION`` does not change in
  PR 7.1).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import pandas as pd

from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    SNAPSHOT_FILTERED_TABLES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default release-id stamped into the critique result.  Mirrors the
#: dataset-tag constant in the platform packagers; keeping a copy here
#: keeps this module's import graph free of ``scripts/_release_common.py``.
RELEASE_ID: Final[str] = "leadforge-lead-scoring-v1"

#: Env var the Anthropic SDK reads.  We honour the same name so a
#: machine that already has the SDK working needs zero extra setup.
ANTHROPIC_API_KEY_ENV: Final[str] = "ANTHROPIC_API_KEY"

#: Default model.  Chosen at PR 7.1; bumped via the ``--model`` flag
#: on :mod:`scripts.run_llm_critique` without rebuilding this module.
DEFAULT_MODEL: Final[str] = "claude-opus-4-7"

#: Effort level for the critique pass.  Per the ``claude-api`` skill's
#: Opus 4.7 guidance, ``high`` is the recommended minimum for
#: intelligence-sensitive work; we use it as the default.
DEFAULT_EFFORT: Final[str] = "high"

#: Adaptive thinking is the only mode supported on Opus 4.7 (manual
#: ``budget_tokens`` returns 400).  ``display="summarized"`` opts back
#: into visible reasoning so the Markdown summary can quote it.
DEFAULT_THINKING_MODE: Final[str] = "adaptive"
DEFAULT_THINKING_DISPLAY: Final[str] = "summarized"

#: Generous output budget: the structured response is ~30 fields plus
#: a list of findings, and Opus 4.7's token-counting shift means we
#: stay generous to avoid mid-thought truncation.
DEFAULT_MAX_TOKENS: Final[int] = 16000

#: Valid severity vocabulary.  Mirrors the rubric's contract.
VALID_SEVERITIES: Final[frozenset[str]] = frozenset({"high", "medium", "low"})

#: Valid category vocabulary.  Lifted verbatim from
#: ``docs/release/break_me_guide.md`` so findings can route to the
#: existing issue-template labels without translation.  Add or remove
#: entries here ONLY in lockstep with the break-me guide.
VALID_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "critical-leakage",
        "realism",
        "difficulty",
        "documentation",
        "platform",
        "notebook",
        "pedagogy",
        "v2-idea",
        "out-of-scope-v1",
    }
)

#: Rubric dimensions defined in ``docs/release/llm_critique_prompt.md``.
#: The validator uses this set to confirm every finding cites a known
#: dimension; new dimensions land in lockstep with the rubric.
VALID_RUBRIC_DIMENSIONS: Final[frozenset[str]] = frozenset({f"D{i}" for i in range(1, 15)})

#: Tier whose artefacts the input bundle is built from.  See the design
#: doc — feeding all three tiers triples context for marginal value.
DEFAULT_TIER: Final[str] = "intermediate"

#: How many rows of the test split to sample into the input bundle.
#: 100 rows × ~40 columns is small enough not to drown the model in
#: tabular data, large enough to surface obvious distribution issues.
TEST_SAMPLE_ROWS: Final[int] = 100

#: Section markers in the rubric prompt.  The driver splits on these
#: to extract the system prompt and the user-turn cue.  Renaming
#: requires updating ``docs/release/llm_critique_prompt.md`` AND the
#: regex below in lockstep.
SYSTEM_PROMPT_OPEN: Final[str] = "<system_prompt>"
SYSTEM_PROMPT_CLOSE: Final[str] = "</system_prompt>"
USER_CUE_OPEN: Final[str] = "<user_cue>"
USER_CUE_CLOSE: Final[str] = "</user_cue>"

_SYSTEM_PROMPT_RE: Final[re.Pattern[str]] = re.compile(
    rf"{re.escape(SYSTEM_PROMPT_OPEN)}\s*(.*?)\s*{re.escape(SYSTEM_PROMPT_CLOSE)}",
    re.DOTALL,
)
_USER_CUE_RE: Final[re.Pattern[str]] = re.compile(
    rf"{re.escape(USER_CUE_OPEN)}\s*(.*?)\s*{re.escape(USER_CUE_CLOSE)}",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Result dataclasses — JSON-primitive so they round-trip cleanly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One critique finding.

    Field names and the ``severity`` / ``category`` enums are part of
    the public output contract — downstream tooling (issue-template
    drafts, the v2 decision log auto-import) reads JSON keyed by these
    exact strings.  Add fields only at the bottom; never rename.
    """

    id: str
    severity: Literal["high", "medium", "low"]
    category: str  # one of VALID_CATEGORIES
    rubric_dimension: str  # one of VALID_RUBRIC_DIMENSIONS
    claim: str
    evidence: str
    reproducer: str
    suggested_fix: str


@dataclass(frozen=True)
class CritiqueResult:
    """Structured result of one critique pass.

    Carries the full provenance triple (model + effort + thinking mode)
    plus the input-bundle hash, so the audit-artifact-sync test can
    detect when a committed result has gone stale relative to the
    current release artefacts on disk.
    """

    release_id: str
    model: str
    effort: str
    thinking_mode: str
    run_timestamp: str
    bundle_hashes: dict[str, str]
    input_bundle_sha256: str
    overall_score: int
    overall_assessment: str
    findings: list[Finding] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    questions_for_maintainer: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputBundleBlock:
    """One named text block in the LLM's input bundle.

    The driver renders these as ``# <name>\\n\\n<body>`` separated by
    horizontal rules; the rubric refers to block names verbatim.
    """

    name: str
    body: str


@dataclass(frozen=True)
class InputBundle:
    """The full ordered input bundle the driver feeds to the LLM."""

    blocks: tuple[InputBundleBlock, ...]
    sha256: str
    bundle_hashes: dict[str, str]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CritiqueValidationError(ValueError):
    """Raised when an LLM response fails schema validation.

    Carries ``problems`` — the structured list of malformations — so the
    driver can render every issue rather than just the first one.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        rendered = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"LLM response failed critique-schema validation "
            f"({len(self.problems)} problem(s)):\n{rendered}"
        )


class MissingCredentialsError(RuntimeError):
    """Raised by :func:`api_key_or_skip` when ``--no-execute`` wants a key."""


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class LLMCritiqueClient(Protocol):
    """Protocol every critique-provider implementation satisfies.

    The driver only ever calls :meth:`run` — it passes a fully-rendered
    system prompt, the input-bundle text, and the user cue, and gets
    back the raw JSON string the provider produced.  Schema validation
    is the driver's responsibility, not the provider's.
    """

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
        """Send the prompt to the model and return the raw response text."""
        ...


def build_anthropic_client() -> LLMCritiqueClient:
    """Construct the default Anthropic critique client.

    Imports the SDK lazily so this module imports cleanly even on
    machines that don't have ``anthropic`` installed.  The skip-cleanly
    path in the driver returns before this is called; the
    ``--no-execute`` smoke path calls this purely to confirm the SDK
    is importable.
    """

    import anthropic  # noqa: PLC0415 — lazy import is intentional

    return _AnthropicCritiqueClient(anthropic.Anthropic())


@dataclass(frozen=True)
class _AnthropicCritiqueClient:
    """Default :class:`LLMCritiqueClient` backed by the Anthropic SDK.

    Caching strategy (per the design doc, §3):

    * Breakpoint 1 — end of the system prompt.  Frozen across runs.
    * Breakpoint 2 — end of the input-bundle blocks.  Frozen across
      re-runs of the same RC; only the rubric tweak path invalidates
      breakpoint 1.

    Volatile content (the user cue) goes after both breakpoints.
    Re-running the critique on the same RC — the common adjudication
    workflow — should hit cache on both breakpoints.
    """

    client: Any

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
        # Stream so the underlying httpx client doesn't trip the 10-min
        # idle-connection timeout on long adaptive-thinking responses;
        # ``.get_final_message()`` re-assembles the streamed chunks
        # into a complete Message object.
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={
                "type": DEFAULT_THINKING_MODE,
                "display": DEFAULT_THINKING_DISPLAY,
            },
            output_config={"effort": effort},
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_bundle_text,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": user_cue},
                    ],
                }
            ],
        ) as stream:
            message = stream.get_final_message()
        for block in message.content:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        raise RuntimeError(
            "Anthropic response contained no text block — got "
            f"types={[getattr(b, 'type', '?') for b in message.content]}"
        )


# ---------------------------------------------------------------------------
# Credential gate — the skip-cleanly path
# ---------------------------------------------------------------------------


def has_anthropic_credentials(env: dict[str, str] | None = None) -> bool:
    """Return True iff ``ANTHROPIC_API_KEY`` is set and non-empty.

    "Set and non-empty" matters because shells routinely set
    ``ANTHROPIC_API_KEY=""`` (e.g. ``env -i`` or stale ``.envrc``
    files), and the SDK would fail with a confusing 401 rather than the
    clean skip the driver expects.  ``os.environ`` is the default
    source; an explicit ``env`` argument is for tests.
    """

    source = env if env is not None else os.environ
    raw = source.get(ANTHROPIC_API_KEY_ENV, "")
    return raw.strip() != ""


def api_key_or_skip(env: dict[str, str] | None = None) -> str:
    """Return the API key or raise :class:`MissingCredentialsError`.

    Used by ``--no-execute`` (which wants a hard error if creds are
    missing — that's the gate's whole point).  The skip-cleanly path
    in the driver uses :func:`has_anthropic_credentials` directly so
    it can exit 0 cleanly without needing a try/except.
    """

    source = env if env is not None else os.environ
    raw = source.get(ANTHROPIC_API_KEY_ENV, "")
    key = raw.strip()
    if not key:
        raise MissingCredentialsError(
            f"{ANTHROPIC_API_KEY_ENV} is not set or is empty after strip; "
            "set it to run the critique."
        )
    return key


# ---------------------------------------------------------------------------
# Rubric prompt parsing
# ---------------------------------------------------------------------------


def parse_rubric_prompt(text: str) -> tuple[str, str]:
    """Extract the system prompt and user cue from a rubric file.

    The rubric file (``docs/release/llm_critique_prompt.md``) is a
    parseable document with ``<system_prompt>`` and ``<user_cue>``
    sections; surrounding prose is informational and ignored here.

    Returns ``(system_prompt, user_cue)`` with whitespace trimmed.
    Raises :class:`ValueError` when either marker is missing — that's
    a malformed rubric, not a recoverable degraded mode.
    """

    sys_match = _SYSTEM_PROMPT_RE.search(text)
    if sys_match is None:
        raise ValueError(
            f"rubric prompt is missing the {SYSTEM_PROMPT_OPEN} ... {SYSTEM_PROMPT_CLOSE} block"
        )
    cue_match = _USER_CUE_RE.search(text)
    if cue_match is None:
        raise ValueError(f"rubric prompt is missing the {USER_CUE_OPEN} ... {USER_CUE_CLOSE} block")
    return sys_match.group(1).strip(), cue_match.group(1).strip()


# ---------------------------------------------------------------------------
# Input bundle assembly
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    """Read a UTF-8 text file, raising a clean error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"required input-bundle file missing: {path}")
    return path.read_text(encoding="utf-8")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _hash_bytes(text.encode("utf-8"))


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _render_test_split_sample(bundle_dir: Path, n_rows: int) -> str:
    """Render the first ``n_rows`` of the test split as CSV.

    Reads ``tasks/converted_within_90_days/test.parquet`` (the canonical
    public-facing split).  Renders deterministically via
    ``DataFrame.to_csv(index=False)`` — the parquet bytes themselves
    aren't byte-stable across pyarrow patch versions, but the *rendered
    CSV* is.
    """

    split_path = bundle_dir / "tasks" / "converted_within_90_days" / "test.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"test split missing at {split_path}; bundle is incomplete")
    df = pd.read_parquet(split_path)
    head = df.head(n_rows)
    # ``to_csv`` defaults are stable across pandas versions for pure
    # data; ``lineterminator="\n"`` keeps the rendered text identical
    # across OSes (pandas defaults to ``os.linesep`` otherwise).
    # ``to_csv(path_or_buf=None, ...)`` returns ``str`` at runtime, but
    # the stub's union widens to ``str | None``; cast pins the type so
    # mypy doesn't complain about returning Any.
    rendered: str = head.to_csv(index=False, lineterminator="\n")  # type: ignore[assignment]
    return rendered


def _render_public_instructor_diff() -> str:
    """Render the public/instructor diff summary as Markdown.

    Sources of truth are the constants in
    :mod:`leadforge.validation.leakage_probes` — :data:`BANNED_LEAD_COLUMNS`,
    :data:`BANNED_OPP_COLUMNS`, :data:`BANNED_TABLES`, and
    :data:`SNAPSHOT_FILTERED_TABLES`.  Live-referenced (not duplicated)
    so the diff stays in sync when the leakage contract changes.
    """

    lines: list[str] = []
    lines.append("## Public/instructor diff — what's redacted from `student_public`")
    lines.append("")
    lines.append("Single source of truth: `leadforge/validation/leakage_probes.py`.")
    lines.append("")
    lines.append("### Columns dropped from public `leads.parquet`")
    lines.append("")
    for col in BANNED_LEAD_COLUMNS:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("### Columns dropped from public `opportunities.parquet`")
    lines.append("")
    for col in BANNED_OPP_COLUMNS:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("### Tables omitted from public bundles entirely")
    lines.append("")
    lines.append("These tables exist only for converted leads — their mere")
    lines.append("presence reconstructs the label.")
    lines.append("")
    for table in BANNED_TABLES:
        lines.append(f"- `{table}`")
    lines.append("")
    lines.append("### Tables filtered per-lead by snapshot window")
    lines.append("")
    lines.append("Each public-table row is kept only if its timestamp")
    lines.append("column is `<= lead_created_at + snapshot_day`.")
    lines.append("")
    lines.append("| Table | Timestamp column |")
    lines.append("|---|---|")
    for table, ts_col in SNAPSHOT_FILTERED_TABLES:
        lines.append(f"| `{table}` | `{ts_col}` |")
    return "\n".join(lines) + "\n"


def _render_public_safe_mechanism_summary(repo_root: Path) -> str:
    """Render the public-safe mechanism summary.

    Names the motif families and difficulty-profile knobs WITHOUT
    leaking latent-trait weights, mechanism parameters, or the hidden
    graph structure.  Same redaction posture as the ``student_public``
    mode itself.

    Pulls the difficulty-profile descriptions from the recipe YAML
    when available so the summary stays in sync with the recipe;
    falls back to a static description if the YAML is unreadable
    (the LLM critique should still run on a partial bundle).
    """

    motif_families = (
        "fit_dominant",
        "intent_dominant",
        "sales_execution_sensitive",
        "demo_trial_mediated",
        "buying_committee_friction",
    )

    lines: list[str] = []
    lines.append("## Public-safe mechanism summary")
    lines.append("")
    lines.append(
        "This summary describes the *shape* of the underlying data-"
        "generating process at a level that matches the public bundle's"
        " documentation. It deliberately does NOT include latent-trait"
        " weights, mechanism parameters, or the hidden DAG — those are"
        " redacted from `student_public` and from this critique input"
        " for the same reason."
    )
    lines.append("")
    lines.append("### Motif families")
    lines.append("")
    lines.append(
        "Each generated world is sampled from one of five motif "
        "families. Each family produces a different conversion-driver "
        "structure; difficulty profiles select the family and modulate "
        "its strength."
    )
    lines.append("")
    for family in motif_families:
        lines.append(f"- `{family}`")
    lines.append("")
    lines.append("### Difficulty profile (intermediate tier)")
    lines.append("")
    yaml_path = (
        repo_root / "leadforge" / "recipes" / "b2b_saas_procurement_v1" / "difficulty_profiles.yaml"
    )
    if yaml_path.exists():
        # Safe-load and render only the structural keys; never the
        # numeric mechanism params (those would leak).
        try:
            from leadforge.core.serialization import load_yaml  # noqa: PLC0415

            payload = load_yaml(yaml_path)
            knobs = _safe_difficulty_knobs(payload, "intermediate")
        except Exception:
            knobs = []
        if knobs:
            for knob in knobs:
                lines.append(f"- `{knob}`")
        else:
            lines.append("- (knob list unavailable; consult the recipe YAML)")
    else:
        lines.append("- (difficulty-profile YAML not found at expected path)")
    return "\n".join(lines) + "\n"


def _safe_difficulty_knobs(payload: Any, tier: str) -> list[str]:
    """Extract the *names* of difficulty knobs without leaking values.

    The point is the LLM should know ``noise_level`` exists as a knob
    on this tier; the LLM should NOT be told that the knob is set to
    ``0.7`` (that's mechanism truth).  Returns a sorted list of knob
    names, or an empty list if the YAML doesn't match the shape we
    know how to redact safely.
    """

    if not isinstance(payload, dict):
        return []
    profiles = payload.get("profiles") or payload.get("difficulty_profiles") or payload
    if not isinstance(profiles, dict):
        return []
    tier_block = profiles.get(tier)
    if not isinstance(tier_block, dict):
        return []
    knobs: set[str] = set()
    for k, v in tier_block.items():
        if isinstance(v, dict | list):
            knobs.add(str(k))
        else:
            knobs.add(str(k))
    return sorted(knobs)


def build_input_bundle(
    release_dir: Path,
    *,
    tier: str = DEFAULT_TIER,
    repo_root: Path | None = None,
    n_test_sample_rows: int = TEST_SAMPLE_ROWS,
) -> InputBundle:
    """Assemble the full input bundle the driver feeds to the LLM.

    Pure: same ``release_dir`` / ``tier`` / ``repo_root`` →
    byte-identical output.  Same input → same ``sha256``.  No
    ``datetime.now()``, no random, no env reads beyond the static
    constants in this module.

    Block order is part of the contract — the rubric refers to block
    names verbatim and a re-order would invalidate the prompt cache.

    The ``bundle_hashes`` field carries per-tier-file SHA256s for the
    audit-artifact-sync test: a re-run of this builder against the
    same release dir must produce hashes byte-identical to the
    committed result's ``bundle_hashes``.

    :param release_dir: the ``release/`` directory at repo root.
    :param tier: which tier's per-tier artefacts to include.  The
        default (``intermediate``) matches the recommended HF entry
        point and minimises context usage.
    :param repo_root: repository root; used to read ancillary docs
        (``docs/release/generation_method.md``, ``break_me_guide.md``,
        the recipe YAML).  Defaults to ``release_dir.parent``.
    :param n_test_sample_rows: how many rows of the test split to
        sample in.  Default ``TEST_SAMPLE_ROWS``.
    """

    if repo_root is None:
        repo_root = release_dir.parent

    bundle_dir = release_dir / tier
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"tier directory missing: {bundle_dir}; is {release_dir} a leadforge release directory?"
        )

    # Read the eleven block sources.  Each call raises FileNotFoundError
    # with a clean message if the artefact is missing.
    readme = _read_text(release_dir / "README.md")
    dataset_card = _read_text(bundle_dir / "dataset_card.md")
    generation_method = _read_text(repo_root / "docs" / "release" / "generation_method.md")
    manifest_text = _read_text(bundle_dir / "manifest.json")
    feature_dict = _read_text(bundle_dir / "feature_dictionary.csv")
    validation_md = _read_text(release_dir / "validation" / "validation_report.md")
    validation_json = _read_text(release_dir / "validation" / "validation_report.json")
    test_sample = _render_test_split_sample(bundle_dir, n_test_sample_rows)
    public_instructor_diff = _render_public_instructor_diff()
    mechanism_summary = _render_public_safe_mechanism_summary(repo_root)
    break_me_guide = _read_text(repo_root / "docs" / "release" / "break_me_guide.md")

    # Per-source-file hashes for audit-artifact-sync.  Use raw bytes
    # for files (catches BOM / line-ending drift), text-hash for
    # rendered blocks (the dataframe-to-csv path).
    bundle_hashes = {
        "release/README.md": _hash_file(release_dir / "README.md"),
        f"release/{tier}/dataset_card.md": _hash_file(bundle_dir / "dataset_card.md"),
        "docs/release/generation_method.md": _hash_file(
            repo_root / "docs" / "release" / "generation_method.md"
        ),
        f"release/{tier}/manifest.json": _hash_file(bundle_dir / "manifest.json"),
        f"release/{tier}/feature_dictionary.csv": _hash_file(bundle_dir / "feature_dictionary.csv"),
        "release/validation/validation_report.md": _hash_file(
            release_dir / "validation" / "validation_report.md"
        ),
        "release/validation/validation_report.json": _hash_file(
            release_dir / "validation" / "validation_report.json"
        ),
        f"release/{tier}/tasks/test.parquet[head{n_test_sample_rows}]": _hash_text(test_sample),
        "public_instructor_diff": _hash_text(public_instructor_diff),
        "public_safe_mechanism_summary": _hash_text(mechanism_summary),
        "docs/release/break_me_guide.md": _hash_file(
            repo_root / "docs" / "release" / "break_me_guide.md"
        ),
    }

    blocks = (
        InputBundleBlock("release/README.md", readme),
        InputBundleBlock(f"release/{tier}/dataset_card.md", dataset_card),
        InputBundleBlock("docs/release/generation_method.md", generation_method),
        InputBundleBlock(f"release/{tier}/manifest.json", manifest_text),
        InputBundleBlock(f"release/{tier}/feature_dictionary.csv", feature_dict),
        InputBundleBlock("release/validation/validation_report.md", validation_md),
        InputBundleBlock("release/validation/validation_report.json", validation_json),
        InputBundleBlock(
            f"release/{tier}/tasks/converted_within_90_days/test.parquet "
            f"(first {n_test_sample_rows} rows, rendered as CSV)",
            test_sample,
        ),
        InputBundleBlock(
            "public/instructor diff summary (live-derived from leakage_probes constants)",
            public_instructor_diff,
        ),
        InputBundleBlock("public-safe mechanism summary", mechanism_summary),
        InputBundleBlock(
            "docs/release/break_me_guide.md (existing patterns — do not re-derive)",
            break_me_guide,
        ),
    )

    rendered = render_input_bundle_text(blocks)
    return InputBundle(
        blocks=blocks,
        sha256=_hash_text(rendered),
        bundle_hashes=bundle_hashes,
    )


def render_input_bundle_text(blocks: Iterable[InputBundleBlock]) -> str:
    """Render an input bundle as a single text payload.

    Format: each block is ``# <name>\\n\\n<body>``, blocks separated by
    a Markdown horizontal rule.  The trailing newline is deterministic.
    """

    parts: list[str] = []
    for block in blocks:
        parts.append(f"# {block.name}\n\n{block.body.rstrip()}\n")
    return "\n---\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


_REQUIRED_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "release_id",
    "overall_score",
    "overall_assessment",
    "findings",
    "missing_sections",
    "questions_for_maintainer",
)

_REQUIRED_FINDING_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "severity",
    "category",
    "rubric_dimension",
    "claim",
    "evidence",
    "reproducer",
    "suggested_fix",
)


def parse_critique_response(
    raw_text: str,
    *,
    model: str,
    effort: str,
    thinking_mode: str,
    bundle_hashes: dict[str, str],
    input_bundle_sha256: str,
    run_timestamp: str | None = None,
) -> CritiqueResult:
    """Parse and validate the LLM's raw response into a :class:`CritiqueResult`.

    Raises :class:`CritiqueValidationError` on any malformation; the
    error carries every detected problem so the driver can render a
    full report rather than fixing them one at a time.

    Required fields are pinned in the rubric prompt's "Output contract"
    section.  Add new fields to that contract AND to the validator
    in lockstep — silent drift between the two is the failure mode
    this validator exists to catch.
    """

    problems: list[str] = []

    # Step 1: parse JSON.  The rubric explicitly says no Markdown code
    # fences, no preamble — we strip a leading code fence defensively
    # but don't tolerate any other framing.
    cleaned = raw_text.strip()
    cleaned = _strip_code_fence(cleaned)
    try:
        payload: Any = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CritiqueValidationError(
            [f"response is not valid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"]
        ) from exc

    if not isinstance(payload, dict):
        raise CritiqueValidationError(
            [f"top-level value must be a JSON object; got {type(payload).__name__}"]
        )

    # Step 2: required top-level fields present.
    for name in _REQUIRED_TOP_LEVEL_FIELDS:
        if name not in payload:
            problems.append(f"missing required top-level field: {name!r}")

    # Step 3: types of top-level fields.
    overall_score = payload.get("overall_score")
    if not isinstance(overall_score, int) or isinstance(overall_score, bool):
        problems.append(
            "overall_score must be an integer; "
            f"got {type(overall_score).__name__} ({overall_score!r})"
        )
    elif not 1 <= overall_score <= 10:
        problems.append(f"overall_score must be in [1, 10]; got {overall_score}")

    overall_assessment = payload.get("overall_assessment", "")
    if not isinstance(overall_assessment, str) or not overall_assessment.strip():
        problems.append("overall_assessment must be a non-empty string")

    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        problems.append(f"findings must be a list; got {type(raw_findings).__name__}")
        raw_findings = []

    raw_missing = payload.get("missing_sections", [])
    if not isinstance(raw_missing, list) or any(not isinstance(s, str) for s in raw_missing):
        problems.append("missing_sections must be a list of strings")
        raw_missing = []

    raw_questions = payload.get("questions_for_maintainer", [])
    if not isinstance(raw_questions, list) or any(not isinstance(s, str) for s in raw_questions):
        problems.append("questions_for_maintainer must be a list of strings")
        raw_questions = []

    # Step 4: validate each finding.
    findings: list[Finding] = []
    seen_ids: set[str] = set()
    for idx, raw in enumerate(raw_findings):
        if not isinstance(raw, dict):
            problems.append(f"findings[{idx}] must be an object; got {type(raw).__name__}")
            continue

        for fname in _REQUIRED_FINDING_FIELDS:
            if fname not in raw:
                problems.append(f"findings[{idx}] missing required field: {fname!r}")

        fid = raw.get("id")
        if not isinstance(fid, str) or not fid.strip():
            problems.append(f"findings[{idx}].id must be a non-empty string")
            fid = f"_anon_{idx}"
        if fid in seen_ids:
            problems.append(f"findings[{idx}].id={fid!r} collides with an earlier finding")
        seen_ids.add(fid)

        severity = raw.get("severity")
        if severity not in VALID_SEVERITIES:
            problems.append(
                f"findings[{idx}].severity={severity!r} is not in {sorted(VALID_SEVERITIES)}"
            )

        category = raw.get("category")
        if category not in VALID_CATEGORIES:
            problems.append(
                f"findings[{idx}].category={category!r} is not in {sorted(VALID_CATEGORIES)}"
            )

        rubric_dim = raw.get("rubric_dimension")
        if rubric_dim not in VALID_RUBRIC_DIMENSIONS:
            problems.append(
                f"findings[{idx}].rubric_dimension={rubric_dim!r} is not in "
                f"{sorted(VALID_RUBRIC_DIMENSIONS)}"
            )

        # If the structural problems above already invalidate the
        # finding, don't construct it — it would carry placeholder
        # values that aren't load-bearing.  ``problems`` already
        # carries the report.
        if (
            severity in VALID_SEVERITIES
            and category in VALID_CATEGORIES
            and rubric_dim in VALID_RUBRIC_DIMENSIONS
            and isinstance(fid, str)
        ):
            findings.append(
                Finding(
                    id=fid,
                    severity=severity,  # type: ignore[arg-type]
                    category=str(category),
                    rubric_dimension=str(rubric_dim),
                    claim=str(raw.get("claim", "")),
                    evidence=str(raw.get("evidence", "")),
                    reproducer=str(raw.get("reproducer", "")),
                    suggested_fix=str(raw.get("suggested_fix", "")),
                )
            )

    if problems:
        raise CritiqueValidationError(problems)

    timestamp = run_timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return CritiqueResult(
        release_id=str(payload.get("release_id", RELEASE_ID)),
        model=model,
        effort=effort,
        thinking_mode=thinking_mode,
        run_timestamp=timestamp,
        bundle_hashes=dict(bundle_hashes),
        input_bundle_sha256=input_bundle_sha256,
        overall_score=int(overall_score) if isinstance(overall_score, int) else 0,
        overall_assessment=str(overall_assessment),
        findings=findings,
        missing_sections=list(raw_missing),
        questions_for_maintainer=list(raw_questions),
    )


def _strip_code_fence(text: str) -> str:
    """Strip a single leading/trailing Markdown code fence if present.

    Defensive: the rubric explicitly forbids code fences, but a model
    that ignores that instruction once shouldn't hard-fail the run.
    Anything beyond a single outer fence is treated as malformed.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # Drop the first line (``` or ```json) and the last fence.
    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    if lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


# ---------------------------------------------------------------------------
# Result serialisation
# ---------------------------------------------------------------------------


def result_to_dict(result: CritiqueResult) -> dict[str, Any]:
    """Convert a :class:`CritiqueResult` to a plain dict."""

    return dataclasses.asdict(result)


def result_to_json(result: CritiqueResult, *, indent: int = 2) -> str:
    """Serialise a :class:`CritiqueResult` deterministically.

    Sorted keys, fixed indent.  The audit-artifact-sync test diffs
    against this exact output, so any drift is caught.
    """

    return json.dumps(result_to_dict(result), indent=indent, sort_keys=True)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def render_markdown_summary(result: CritiqueResult) -> str:
    """Render a human-readable Markdown summary of a critique result.

    Single canonical filename (``llm_critique_summary.md``) — the most
    recent run overwrites it so the dataset card's link stays fresh.
    The full history lives in the timestamped raw JSON files; this is
    the "latest run, at a glance" surface.
    """

    lines: list[str] = []
    lines.append("# LLM critique summary — `leadforge-lead-scoring-v1`")
    lines.append("")
    lines.append(f"- **Release:** `{result.release_id}`")
    lines.append(
        f"- **Model:** `{result.model}` "
        f"(effort: `{result.effort}`, thinking: `{result.thinking_mode}`)"
    )
    lines.append(f"- **Run timestamp:** {result.run_timestamp}")
    lines.append(f"- **Input-bundle SHA256:** `{result.input_bundle_sha256}`")
    lines.append(f"- **Overall score:** {result.overall_score}/10")
    lines.append("")
    lines.append("## Overall assessment")
    lines.append("")
    lines.append(result.overall_assessment.strip())
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not result.findings:
        lines.append("*No findings reported.*")
    else:
        by_severity: dict[str, list[Finding]] = {"high": [], "medium": [], "low": []}
        for f in result.findings:
            by_severity.setdefault(f.severity, []).append(f)
        for severity in ("high", "medium", "low"):
            bucket = by_severity.get(severity, [])
            if not bucket:
                continue
            lines.append(f"### Severity: {severity} ({len(bucket)})")
            lines.append("")
            for f in bucket:
                lines.append(f"#### {f.id} — `{f.category}` / `{f.rubric_dimension}`")
                lines.append("")
                lines.append(f"**Claim.** {f.claim}")
                lines.append("")
                lines.append(f"**Evidence.** {f.evidence}")
                lines.append("")
                lines.append(f"**Reproducer.** {f.reproducer}")
                lines.append("")
                lines.append(f"**Suggested fix.** {f.suggested_fix}")
                lines.append("")
    lines.append("## Missing sections")
    lines.append("")
    if not result.missing_sections:
        lines.append("*None reported.*")
    else:
        for s in result.missing_sections:
            lines.append(f"- {s}")
    lines.append("")
    lines.append("## Questions for the maintainer")
    lines.append("")
    if not result.questions_for_maintainer:
        lines.append("*None reported.*")
    else:
        for q in result.questions_for_maintainer:
            lines.append(f"- {q}")
    lines.append("")
    lines.append("## Bundle hashes (audit)")
    lines.append("")
    lines.append("| File / block | SHA256 |")
    lines.append("|---|---|")
    for path, digest in sorted(result.bundle_hashes.items()):
        lines.append(f"| `{path}` | `{digest[:12]}…` |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------


def raw_output_path(out_dir: Path, run_timestamp: str, *, tag: str | None = None) -> Path:
    """Return the timestamped raw-JSON output path.

    Timestamp is folded into the filename so re-runs accumulate without
    clobber.  ``tag``, when provided, suffixes the filename so
    adjudication runs (re-run after fixing finding F003) don't shadow
    the canonical run.
    """

    safe_ts = run_timestamp.replace(":", "").replace("-", "")
    suffix = f"_{tag}" if tag else ""
    return out_dir / f"llm_critique_raw_{safe_ts}{suffix}.json"


def summary_output_path(out_dir: Path) -> Path:
    """Return the canonical Markdown summary path.

    Single filename — overwritten on each run.  Pair with the raw JSON
    history when you need to look at a specific run.
    """

    return out_dir / "llm_critique_summary.md"


# ---------------------------------------------------------------------------
# Severity policy — how the driver maps findings to exit codes
# ---------------------------------------------------------------------------


def has_unresolved_high_severity(result: CritiqueResult) -> bool:
    """Return True iff the result carries any high-severity findings.

    Adjudication (resolving in code OR logging to v2_decision_log.md)
    happens *after* the critique runs and outside this module's scope.
    The driver uses this signal to set its exit code to 1 — a real
    high-severity finding blocks the release-candidate gate until the
    maintainer either fixes it or documents the disposition.
    """

    return any(f.severity == "high" for f in result.findings)


__all__ = [
    "ANTHROPIC_API_KEY_ENV",
    "DEFAULT_EFFORT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_THINKING_DISPLAY",
    "DEFAULT_THINKING_MODE",
    "DEFAULT_TIER",
    "RELEASE_ID",
    "TEST_SAMPLE_ROWS",
    "VALID_CATEGORIES",
    "VALID_RUBRIC_DIMENSIONS",
    "VALID_SEVERITIES",
    "CritiqueResult",
    "CritiqueValidationError",
    "Finding",
    "InputBundle",
    "InputBundleBlock",
    "LLMCritiqueClient",
    "MissingCredentialsError",
    "api_key_or_skip",
    "build_anthropic_client",
    "build_input_bundle",
    "has_anthropic_credentials",
    "has_unresolved_high_severity",
    "parse_critique_response",
    "parse_rubric_prompt",
    "raw_output_path",
    "render_input_bundle_text",
    "render_markdown_summary",
    "result_to_dict",
    "result_to_json",
    "summary_output_path",
]

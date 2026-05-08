#!/usr/bin/env python3
"""LLM critique driver for ``leadforge-lead-scoring-v1``.

PR 7.1's CLI + filesystem glue. Wraps :mod:`leadforge.validation.llm_critique`
to:

1. Load the rubric prompt from ``docs/release/llm_critique_prompt.md``.
2. Build the deterministic input bundle from ``release/<tier>/`` and
   surrounding docs.
3. Call the Anthropic Claude critique provider (skip-cleanly when
   ``ANTHROPIC_API_KEY`` is unset).
4. Schema-validate the response.
5. Write timestamped raw JSON + canonical Markdown summary under
   ``release/validation/``.
6. Translate findings to an exit code (0 pass / 1 high-severity
   surfaced / 2 pre-flight error).

CLI shape mirrors ``scripts/validate_release_candidate.py`` — same
``--release-dir`` / ``--out-dir`` / exit-code conventions so the
maintainer's muscle memory works.

Usage examples::

    # Full critique against the canonical intermediate bundle.
    python scripts/run_llm_critique.py

    # Build the input bundle and write it to disk for inspection;
    # don't call the API.
    python scripts/run_llm_critique.py --dry-run

    # Confirm SDK + creds are wired up; don't actually run the
    # critique. CI smoke gate.
    python scripts/run_llm_critique.py --no-execute

    # Adjudication re-run after fixing a finding — stamp the new
    # output filename so it doesn't shadow the original.
    python scripts/run_llm_critique.py --out-tag adj1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from leadforge.validation.llm_critique import (
    ANTHROPIC_API_KEY_ENV,
    DEFAULT_EFFORT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_THINKING_MODE,
    DEFAULT_TIER,
    CritiqueResult,
    CritiqueValidationError,
    LLMCritiqueClient,
    MissingCredentialsError,
    api_key_or_skip,
    build_anthropic_client,
    build_input_bundle,
    has_anthropic_credentials,
    has_unresolved_high_severity,
    parse_critique_response,
    parse_rubric_prompt,
    raw_output_path,
    render_markdown_summary,
    result_to_json,
    summary_output_path,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_DIR: Path = Path("release")
DEFAULT_OUT_DIR: Path = Path("release/validation")
DEFAULT_PROMPT: Path = Path("docs/release/llm_critique_prompt.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the driver CLI.

    Free function so integration tests can construct a Namespace via
    this exact path without exec-ing the script — matches
    ``validate_release_candidate.py``'s posture.
    """

    parser = argparse.ArgumentParser(
        prog="run_llm_critique",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=(
            "Release directory; expected to contain per-tier bundles "
            "and validation/. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where to write the raw JSON and Markdown summary. Default: %(default)s",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Rubric prompt file. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Anthropic model id. Default: %(default)s",
    )
    parser.add_argument(
        "--tier",
        default=DEFAULT_TIER,
        help=("Tier whose per-tier artefacts feed the input bundle. Default: %(default)s"),
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        help="Effort level passed to the model. Default: %(default)s",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="max_tokens for the critique response. Default: %(default)s",
    )
    parser.add_argument(
        "--out-tag",
        default=None,
        help=(
            "Optional suffix for the raw-JSON filename so adjudication "
            "re-runs don't clobber the canonical one. Example: --out-tag adj1"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Build the input bundle and write it to <out-dir>/"
            "llm_critique_input_<ts>.md; do not call the API."
        ),
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help=(
            "Confirm the SDK is importable and ANTHROPIC_API_KEY is set; "
            "do not call the API or write any output. CI smoke gate."
        ),
    )
    parser.add_argument(
        "--require-execute",
        action="store_true",
        help=(
            "Convert the skip-cleanly path to a hard failure when "
            "ANTHROPIC_API_KEY is unset. Set this in release-readiness CI "
            "where 'no critique ran' must not silently pass the gate."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Driver config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverConfig:
    """Resolved driver settings — produced from CLI args, consumed by run()."""

    release_dir: Path
    out_dir: Path
    prompt: Path
    model: str
    tier: str
    effort: str
    max_tokens: int
    out_tag: str | None
    dry_run: bool
    no_execute: bool
    require_execute: bool


def _config_from_args(args: argparse.Namespace) -> DriverConfig:
    return DriverConfig(
        release_dir=args.release_dir,
        out_dir=args.out_dir,
        prompt=args.prompt,
        model=args.model,
        tier=args.tier,
        effort=args.effort,
        max_tokens=args.max_tokens,
        out_tag=args.out_tag,
        dry_run=args.dry_run,
        no_execute=args.no_execute,
        require_execute=args.require_execute,
    )


@dataclass(frozen=True)
class DriverResult:
    """Materialised outputs of one critique run.

    ``result`` is None for the skip-cleanly, dry-run, and no-execute
    paths; otherwise carries the structured critique.  ``written_files``
    lists every path the driver wrote, in order, so tests can assert
    against it without re-deriving the timestamp suffix.
    """

    result: CritiqueResult | None
    written_files: tuple[Path, ...]
    skipped: bool
    skip_reason: str | None


# ---------------------------------------------------------------------------
# Driver — pre-flight, dispatch, write
# ---------------------------------------------------------------------------


def _utc_iso_timestamp() -> str:
    """Render the current UTC instant for the raw-output filename.

    Microsecond precision so two adjacent runs in the same wall-clock
    second don't clobber each other's raw JSON — the design doc commits
    to "raw JSON files are append-only history".  ``--out-tag`` is the
    user-facing way to disambiguate adjudication runs; this is the
    just-in-case for unattended scripted runs.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _preflight(config: DriverConfig) -> tuple[Path, Path]:
    """Resolve and validate input paths; return the rubric path and the bundle dir."""

    if not config.release_dir.exists():
        raise FileNotFoundError(f"--release-dir {config.release_dir} does not exist")
    if not config.prompt.exists():
        raise FileNotFoundError(
            f"--prompt {config.prompt} does not exist; expected docs/release/llm_critique_prompt.md"
        )
    bundle_dir = config.release_dir / config.tier
    if not bundle_dir.exists():
        raise FileNotFoundError(
            f"tier directory missing: {bundle_dir}; "
            f"--tier={config.tier} requires {bundle_dir}/manifest.json"
        )
    return config.prompt, bundle_dir


def run_critique(
    config: DriverConfig,
    *,
    client: LLMCritiqueClient | None = None,
    env: dict[str, str] | None = None,
) -> DriverResult:
    """Execute the critique pipeline.

    Pure of side effects only on the skip-cleanly and no-execute paths;
    every other path writes timestamped output under ``config.out_dir``.

    Tests inject ``client`` to mock the Anthropic call; production runs
    leave it as ``None`` and let :func:`build_anthropic_client`
    construct the default Anthropic implementation lazily.

    The skip-cleanly path triggers BEFORE any I/O — no rubric read,
    no bundle build, no out-dir write. Tests pin this with a no-side-
    effects check.
    """

    # --no-execute: confirm creds + SDK importability and exit.  Runs
    # BEFORE any pre-flight I/O so the CI smoke gate is fast and
    # doesn't read the bundle.  Raises MissingCredentialsError if the
    # key is absent — the smoke gate is supposed to fail loud here.
    if config.no_execute:
        api_key_or_skip(env)
        if client is None:
            # Lazy import; fails fast if the SDK isn't installed.
            # Construction is enough to prove the SDK is present —
            # we don't make an API call.
            build_anthropic_client()
        return DriverResult(
            result=None,
            written_files=(),
            skipped=True,
            skip_reason="--no-execute: SDK + credentials verified; API not called.",
        )

    # Skip-cleanly: ANTHROPIC_API_KEY unset or empty-after-strip.
    # ``--dry-run`` deliberately bypasses the cred check (the bundle
    # builder is the whole point of the dry run; no API is called).
    # ``--require-execute`` converts the skip into a hard failure so
    # release-readiness CI doesn't silently pass when the gate didn't
    # actually run.
    if not config.dry_run and not has_anthropic_credentials(env):
        if config.require_execute:
            raise MissingCredentialsError(
                f"{ANTHROPIC_API_KEY_ENV} is not set; --require-execute "
                "demands the critique actually run."
            )
        return DriverResult(
            result=None,
            written_files=(),
            skipped=True,
            skip_reason=("ANTHROPIC_API_KEY is not set or is empty; skipping critique pass."),
        )

    # Pre-flight: verify paths exist before doing anything else.
    prompt_path, _ = _preflight(config)

    # Build the input bundle.  Pure; same release_dir → identical bytes.
    bundle = build_input_bundle(
        config.release_dir,
        tier=config.tier,
    )
    bundle_text = bundle.render()

    # Parse the rubric prompt.
    rubric_text = prompt_path.read_text(encoding="utf-8")
    system_prompt, user_cue = parse_rubric_prompt(rubric_text)

    timestamp = _utc_iso_timestamp()

    # --dry-run: write the input bundle for human inspection, no API call.
    if config.dry_run:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        safe_ts = timestamp.replace(":", "").replace("-", "")
        dry_path = config.out_dir / f"llm_critique_input_{safe_ts}.md"
        dry_path.write_text(bundle_text, encoding="utf-8")
        return DriverResult(
            result=None,
            written_files=(dry_path,),
            skipped=True,
            skip_reason=(f"--dry-run: input bundle written to {dry_path}; API not called."),
        )

    # Live path: confirm creds, construct the client, run the critique.
    api_key_or_skip(env)
    if client is None:
        client = build_anthropic_client()

    raw_text = client.run(
        system_prompt=system_prompt,
        input_bundle_text=bundle_text,
        user_cue=user_cue,
        model=config.model,
        max_tokens=config.max_tokens,
        effort=config.effort,
    )

    # Validate.  A malformed response raises and the driver translates
    # to exit code 2 — we don't try to "salvage" partial JSON.
    result = parse_critique_response(
        raw_text,
        model=config.model,
        effort=config.effort,
        thinking_mode=DEFAULT_THINKING_MODE,
        bundle_hashes=bundle.bundle_hashes,
        input_bundle_sha256=bundle.sha256,
        run_timestamp=timestamp,
    )

    # Write outputs: timestamped raw JSON + canonical Markdown summary.
    config.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_output_path(config.out_dir, timestamp, tag=config.out_tag)
    summary_path = summary_output_path(config.out_dir, tag=config.out_tag)
    raw_path.write_text(result_to_json(result) + "\n", encoding="utf-8")
    summary_path.write_text(render_markdown_summary(result) + "\n", encoding="utf-8")

    return DriverResult(
        result=result,
        written_files=(raw_path, summary_path),
        skipped=False,
        skip_reason=None,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_summary(driver_result: DriverResult) -> str:
    """Single-line summary suitable for stdout."""

    if driver_result.skipped:
        return f"run_llm_critique: SKIPPED — {driver_result.skip_reason}"
    result = driver_result.result
    if result is None:
        # Defensive — should never happen on a non-skipped path.
        return "run_llm_critique: ERROR — no result and not skipped"
    n_findings = len(result.findings)
    n_high = sum(1 for f in result.findings if f.severity == "high")
    n_medium = sum(1 for f in result.findings if f.severity == "medium")
    n_low = sum(1 for f in result.findings if f.severity == "low")
    status = "FAIL" if has_unresolved_high_severity(result) else "PASS"
    return (
        f"run_llm_critique: {status} — score {result.overall_score}/10; "
        f"{n_findings} finding(s) [high={n_high}, medium={n_medium}, low={n_low}]; "
        f"output: {', '.join(str(p) for p in driver_result.written_files)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _config_from_args(args)

    try:
        driver_result = run_critique(config)
    except FileNotFoundError as exc:
        print(f"run_llm_critique: pre-flight error: {exc}", file=sys.stderr)
        return 2
    except MissingCredentialsError as exc:
        # ``--no-execute`` fails loud here when the key is absent;
        # other paths skip cleanly via has_anthropic_credentials.
        print(f"run_llm_critique: pre-flight error: {exc}", file=sys.stderr)
        return 2
    except CritiqueValidationError as exc:
        print(
            "run_llm_critique: schema-validation error on LLM response:",
            file=sys.stderr,
        )
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as exc:
        # Malformed rubric, malformed bundle, etc.  Surface cleanly.
        print(f"run_llm_critique: malformed input: {exc}", file=sys.stderr)
        return 2

    print(format_summary(driver_result))

    # Loud warning when the credential gate skipped — release-readiness
    # CI must not silently pass on a skipped critique.  ``--require-execute``
    # already converts that case to MissingCredentialsError above; this
    # is the local-dev / non-CI surface.
    if (
        driver_result.skipped
        and driver_result.skip_reason
        and ("ANTHROPIC_API_KEY" in driver_result.skip_reason)
    ):
        print(
            "run_llm_critique: WARNING — critique was skipped because "
            f"{ANTHROPIC_API_KEY_ENV} is unset; release-readiness gate has "
            "NOT been evaluated. Set --require-execute in CI to fail loud.",
            file=sys.stderr,
        )

    # Exit-code policy:
    #   0 — pass (skip-cleanly counts as pass; no high-severity findings).
    #   1 — high-severity finding(s) present and unresolved at the
    #       critique-output level.  Adjudication (resolve in code OR
    #       log to v2_decision_log.md) happens *after* this exit code,
    #       outside the driver — the next critique run is the gate.
    #   2 — pre-flight or schema-validation error (handled above).
    if driver_result.skipped or driver_result.result is None:
        return 0
    if has_unresolved_high_severity(driver_result.result):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

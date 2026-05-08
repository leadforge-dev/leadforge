# PR 7.1 — `llm_critique` design notes

Working notes for the LLM critique module
(`leadforge/validation/llm_critique.py`), its rubric prompt
(`docs/release/llm_critique_prompt.md`), and its driver
(`scripts/run_llm_critique.py`). Captured before implementation; kept
short on purpose.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | Single-provider (Anthropic Claude) via an `LLMCritiqueClient` protocol; no preemptive OpenAI / Gemini stubs. | Multi-provider is post-v1 (`post_v1_roadmap.md`). The protocol gives a future provider a seam without paying for it now. |
| 2 | `ANTHROPIC_API_KEY` env var. "Absent" = unset OR empty after `.strip()`. On absent: skip cleanly, exit 0, no I/O. `--require-execute` flag converts the skip into exit 2 for release-readiness CI. | Roadmap acceptance criterion: live API not required to pass `pytest`. Empty-after-strip handles `env -i` / stale `.envrc`. The CI gate needs an opt-in to fail loud. |
| 3 | Model `claude-opus-4-7`, `thinking={"type": "adaptive", "display": "summarized"}`, `effort="high"`, `messages.create()` with explicit 600s timeout, single prompt-cache breakpoint at end of input bundle. | Adaptive is the only mode on Opus 4.7 (manual `budget_tokens` 400s). `summarized` so the Markdown summary can quote reasoning. `high` is the recommended minimum for intelligence-sensitive work. One breakpoint suffices: system content sits inside the cached prefix anyway, and any rubric edit invalidates the bundle cache, so a second breakpoint buys nothing and burns a slot. |
| 4 | Frozen-dataclass schema (no pydantic). `category` vocabulary lifted **verbatim** from `break_me_guide.md` (the nine triage labels). `rubric_dimension` (D1–D14) required on every finding. Strict `release_id` equality check. Provenance triple (`model` / `effort` / `thinking_mode`) plus per-source-file `bundle_hashes` and assembled `input_bundle_sha256` carried for audit. | Matches the rest of the codebase (no pydantic anywhere). Locked vocabulary = findings route to existing labels without translation. Requiring `rubric_dimension` lets reviewers audit clustering. Strict `release_id` so silent drift can't defeat the audit gate. |
| 5 | Eleven-block input bundle, intermediate tier only: README, per-tier dataset card, generation method, manifest, feature dictionary, validation report `.{md,json}`, test-split `df.describe()` + 20-row head, public/instructor diff (live-derived from `BANNED_*` constants in `leakage_probes.py`), public-safe mechanism summary (motif family names + difficulty knob *names*, no values), break-me guide verbatim. | Each block earns its place. Live-derived diff = single source of truth, sync-tested. Mechanism summary names-only matches the `student_public` redaction posture. `df.describe()` carries the per-column statistics raw rows can't. All-three-tiers would triple context for marginal value (cross-tier spread is in the validation report already). |
| 6 | No fake determinism (Opus 4.7 doesn't accept `temperature`). Provenance instead: model + effort + thinking + bundle hashes recorded on every result. Timestamped raw JSON accumulates per run; canonical Markdown summary overwrites in place. | Reviewer concern is "could a different maintainer get a different result" — yes. Mitigation is provenance, not fake `temperature=0`. |
| 7 | CLI mirrors `scripts/validate_release_candidate.py`: free-function `parse_args`, frozen `DriverConfig`, `run_critique(config) -> DriverResult`, `main(argv) -> int`. Exit codes 0 / 1 / 2. Three modes alongside the live path: `--dry-run` writes the input bundle for inspection (no API call); `--no-execute` validates SDK + creds and exits (CI smoke gate, fails loud on absent creds); `--out-tag` suffixes both raw JSON *and* summary filenames for adjudication re-runs. | Maintainer muscle memory + small surface. `--out-tag` suffixes both files because the summary is the at-a-glance entry point — clobbering the canonical run's summary on adjudication is the bug. |
| 8 | Tests: no live API. Mocked `LLMCritiqueClient` protocol with a small in-process canned-response fake. Sync tests pin (a) every `VALID_CATEGORIES` entry appears in `break_me_guide.md`, (b) `VALID_RUBRIC_DIMENSIONS` is exactly D1–D14, (c) the live-derived public/instructor diff names every banned-column / banned-table constant. Smoke test exercises `build_input_bundle` against the real `release/intermediate/` artefacts when present. | Roadmap acceptance: live API not required. Sync tests are the cheap-but-load-bearing guards against vocabulary drift. |
| 9 | First live run is maintainer-driven. Outputs land at `release/validation/llm_critique_raw_<UTC-iso>.json` + `release/validation/llm_critique_summary.md`. Hand-adjudicate: resolve high-severity findings in code OR log to `docs/release/v2_decision_log.md` with verdict (`accepted-for-v2` / `deferred` / `wont-fix` / `needs-investigation`). | Adjudication is human work. The next critique's exit code is the gate. |

## What this PR does not touch

- `BUNDLE_SCHEMA_VERSION` stays at 5.
- `release/validation/validation_report.{json,md}` does not regenerate.
- PR 7.2 (Kaggle/HF mock-page preview) and PR 7.3 (publish + tag) are separate PRs.
- Multi-provider abstraction beyond the protocol seam.
- CI integration of the critique gate (post-v1 unless `--require-execute` lands in a workflow this PR or later).

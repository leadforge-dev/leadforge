# PR 7.1 — `llm_critique` design decisions

This file captures the load-bearing decisions for the LLM critique
module (`leadforge/validation/llm_critique.py`), its rubric prompt
(`docs/release/llm_critique_prompt.md`), and its driver
(`scripts/run_llm_critique.py`). Recorded *before* implementation, so
reviewers — human or LLM — can audit the call against the choice.

The roadmap entry is `docs/release/v1_release_roadmap.md` Phase 7;
the foundation it sits on is the existing release-quality
(`leadforge/validation/release_quality.py`), driver
(`scripts/validate_release_candidate.py`), and adversarial framing
(`docs/release/break_me_guide.md`, `docs/release/v2_decision_log.md`).

## 1. Provider abstraction shape

**Decision.** Single-provider for v1 — Anthropic Claude, via the
official `anthropic` Python SDK. One `LLMCritiqueClient` protocol
with one Anthropic implementation. **No** OpenAI / Gemini stubs.

**Rationale.** The roadmap (Phase 7 work-items) leaves room for a
future provider via env var, but actually wiring more than one
costs reviewer attention and dependency surface for zero v1 benefit.
Multi-provider critique is explicitly listed as out-of-scope in
`v1_release_roadmap.md` ("Out-of-scope" section) and post-v1 in
`post_v1_roadmap.md`. The protocol gives us a clean seam for a
future provider without paying for it now.

**SDK posture.** `pip install anthropic` is gated behind a new
`[critique]` extra so the default `dev` install isn't burdened with
a network-tier dependency. The module imports `anthropic` lazily
inside the Anthropic implementation — module import succeeds
without the SDK installed (skip-cleanly path needs to work even on
machines that don't have `anthropic`).

## 2. Skip-cleanly behaviour

**Decision.** Env var: `ANTHROPIC_API_KEY` (the SDK convention).
"Absent" means unset OR empty-string-after-strip. When absent:
- Print one line to stderr: `run_llm_critique: ANTHROPIC_API_KEY
  not set; skipping critique pass.`
- Exit 0. **Not** a failure — the rest of CI must keep working.
- **Do not** write a stub output file. If a previous critique ran
  succeeded, those committed outputs stay; if not, the directory
  stays empty. A stub file would lie about the bundle's audit state.

**Rationale.** PR 5.2 already established the "publish-extra-gated"
posture for SDK-bearing tests (`load_dataset()` smoke). This is the
same shape: optional, non-failing absence. Roadmap acceptance
criterion: "Test posture: live API not required to pass `pytest`."

The empty-strip check matters because shells routinely set
`ANTHROPIC_API_KEY=""` (e.g. `env -i` or stale `.envrc` files), and
the SDK would fail with a confusing 401 rather than the clean skip.

The skip path triggers **before** any I/O — no input-bundle build,
no API client construction. Tests pin this with a no-side-effects
check.

## 3. Model + caching + thinking

**Decision.**
- **Model:** `claude-opus-4-7` (Default per `claude-api` skill +
  the system context's `currentDate=2026-05-08`. Latest Opus.)
- **Thinking:** `thinking={"type": "adaptive"}` with
  `display="summarized"`. Adaptive lets Claude allocate effort by
  finding density; `summarized` so the rendered Markdown summary
  can quote the model's reasoning instead of an empty pause.
- **Effort:** `output_config={"effort": "high"}`. Critique is an
  intelligence-sensitive task; per the skill's Opus 4.7 guidance,
  `high` is the recommended minimum for that class.
- **Temperature:** *cannot* be set on Opus 4.7 (removed; would 400).
  Reproducibility comes from the rubric being deterministic and
  the input bundle being byte-stable; we don't try to fake
  determinism via `temperature=0`.
- **Prompt caching:** **two breakpoints** —
  1. End of the system prompt (the rubric — frozen across runs).
  2. End of the input-bundle blocks (the release artefacts —
     identical across re-runs of the same RC).
  Volatile content (the user-turn "now produce the critique" cue)
  goes after both breakpoints. Re-running the critique on the same
  RC — common during adjudication — should hit cache on both
  breakpoints. Re-running with a tweaked rubric only invalidates
  breakpoint 2; breakpoint 1 still hits.
- **Streaming:** yes. `max_tokens=16000` for the structured-output
  response. Streaming protects against the 10-min idle-connection
  timeout on a large adaptive-thinking response, and lets the
  driver print a progress dot per chunk so the maintainer doesn't
  stare at a blank terminal.

**Rationale.** Re-runs are a real workflow — adjudicate a finding,
fix the bundle, re-run. Two breakpoints (rubric, bundle) match the
stability tiers per the skill's `prompt-caching.md` placement
patterns. Single-block caching would force a rebuild on every rubric
tweak; no caching would burn cost on adjudication loops.

The Opus 4.7 token-counting shift (skill warning) means we stay
generous on `max_tokens=16000` — the structured output schema is
~30 fields with arrays of findings, so it could legitimately run
long.

## 4. Output schema

**Decision.** Pydantic-model-shaped, but implemented as **frozen
`@dataclass` with explicit field-by-field validation** rather than
pydantic. `leadforge` already uses dataclasses everywhere (per the
CLAUDE.md "typed dataclasses/models" invariant) and avoiding a new
runtime dependency on pydantic for one module is the cheaper call.

**Top-level shape (matches `v1_release_roadmap.md` Phase 7
work-items, with the additions called out in the brief):**

```
CritiqueResult
├── release_id: str           # "leadforge-lead-scoring-v1" (recipe + dataset name)
├── bundle_hashes: dict[tier→sha]  # for audit-artifact-sync
├── model: str                # "claude-opus-4-7" (echoed for provenance)
├── effort: str               # "high"
├── thinking_mode: str        # "adaptive"
├── run_timestamp: str        # ISO 8601, UTC
├── input_bundle_sha256: str  # hash of the assembled input bundle
├── overall_score: int        # 1-10, rubric-defined
├── overall_assessment: str   # one paragraph summary
├── findings: list[Finding]
├── missing_sections: list[str]
└── questions_for_maintainer: list[str]

Finding
├── id: str                   # "F001" .. — stable within a run for adjudication
├── severity: Literal["high", "medium", "low"]
├── category: Literal[...]    # 9-value vocabulary, see below
├── claim: str
├── evidence: str             # JSON path / notebook §, free-form quote
├── reproducer: str           # code snippet OR shell command
├── suggested_fix: str
└── rubric_dimension: str     # which of the 10-14 dimensions surfaced this
```

**Category vocabulary — locked-in, lifted verbatim from the
`break_me_guide.md` triage labels** so reporters/maintainers/critique
share one taxonomy:

```
critical-leakage | realism | difficulty | documentation | platform |
notebook | pedagogy | v2-idea | out-of-scope-v1
```

This is the intentional vocabulary alignment the brief calls out;
keeping it identical to the issue-template auto-applied label
(`needs-triage` is set by the issue templates) means an LLM finding
can be auto-converted into a draft issue with the right label
without translation.

**Rubric dimension on every finding.** The brief asks for 10-14
rubric dimensions; without `rubric_dimension` on each finding, we
can't audit "did the rubric get applied uniformly or did the model
cluster on dimension 3 and ignore 8-12?" Cheap to require, high
audit value.

**Validation.** Schema validator runs on the model's JSON output
before it lands on disk. Unknown fields → drop silently (the
rubric is the contract; extra fields are tolerated). Missing
required fields → exit code 2 (treated as a model malfunction,
not a finding). `release_id` not equal to `RELEASE_ID` → exit
code 2 (silent drift would defeat the audit-artifact-sync
contract). Severity outside the 3-value set → exit code 2.
Unknown category → exit code 2. Unknown rubric dimension → exit
code 2. The validator collects every problem in one
`CritiqueValidationError` so the driver can render the full
report instead of fixing them one at a time.

**Rationale.** Roadmap pins the shape (release_id, model,
run_timestamp, overall_score, findings[severity/category/claim/
evidence/reproducer/suggested_fix], missing_sections,
questions_for_maintainer). The additions
(`bundle_hashes`/`input_bundle_sha256`/`rubric_dimension`/
`finding.id`/`temperature`/`effort`/`thinking_mode`) are for
audit-artifact-sync: re-running on the same RC should produce the
same bundle hashes and input-bundle hash; the model-config triple
is provenance for the v2 decision log to cite.

## 5. Input bundle composition

**Decision.** Inline text blocks, not Files API. The total bundle
is ~50-80KB once the parquet head is rendered as CSV — well below
any reasonable inline limit, and prompt caching makes re-runs free
on the bundle blocks.

The bundle is built as an ordered list of `(name, body)` pairs by
`build_input_bundle(release_dir, tier)`, exactly as the roadmap
specifies, with the additions stated in the brief:

1. `release/README.md` — the dataset card.
2. `release/<tier>/dataset_card.md` — the per-tier card.
3. `docs/release/generation_method.md` — DGP summary.
4. `release/<tier>/manifest.json` — provenance.
5. `release/<tier>/feature_dictionary.csv` — column spec.
6. `release/validation/validation_report.md` — release-quality.
7. `release/validation/validation_report.json` — machine-readable
   metrics so the LLM can cite JSON paths in `evidence`.
8. **First 100 rows** of `release/<tier>/tasks/converted_within_90_days/test.parquet`
   rendered as CSV. (`test.parquet` over `lead_scoring.csv` because the
   CSV is the same data and we want to feed the LLM the exact split
   it would compute lift on.)
9. **Public/instructor diff summary** — derived live from
   `BANNED_LEAD_COLUMNS`, `BANNED_OPP_COLUMNS`, `BANNED_TABLES`,
   `SNAPSHOT_FILTERED_TABLES` in `leadforge/validation/leakage_probes.py`.
   Rendered as a Markdown table — what's dropped, why each is
   dropped. Single source of truth, auto-stays-in-sync.
10. **Public-safe mechanism summary** — motif families
    (`fit_dominant`, `intent_dominant`, `sales_execution_sensitive`,
    `demo_trial_mediated`, `buying_committee_friction`) +
    difficulty-profile knob explanations from
    `recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml`.
    Critically: **NO latent-trait weights**, NO hidden-graph edges,
    NO mechanism parameters. Same redaction posture as the
    `student_public` mode. (If the LLM critique needs the hidden
    truth, it should ask via `questions_for_maintainer` rather than
    receive it.)
11. **`break_me_guide.md`** — included verbatim. The roadmap's
    "avoid re-deriving" guidance: the 9 cataloged patterns are the
    floor, the LLM should be looking for novel ones.

**Tier choice.** `--tier intermediate` is the default. The brief
lists it explicitly; intermediate is the recommended downstream
entry point per `package_hf_release.py` (`default: true` config),
and feeding the LLM all three tiers would multiply context by ~3×
without commensurate value (the validation report's cross-tier
spread is already in the input bundle).

**Determinism.** `build_input_bundle` is pure (no `now()`, no
`uuid()`, no env). The same input → identical output bytes. A
sync-test re-runs it and diffs against a checked-in fixture path
to catch drift. (Audit-artifact-sync pattern.)

## 6. Determinism vs creativity

**Decision.** Opus 4.7 doesn't accept `temperature` (would 400).
We don't try to fake determinism. Instead:

- The rubric is fully deterministic (no "be creative" prompts).
- The input bundle is byte-stable.
- The model + thinking + effort triple is recorded in
  `CritiqueResult` for provenance.
- The committed outputs are versioned by **timestamp** in the
  filename (`llm_critique_raw_<UTC-iso>.json`) so re-runs accumulate
  rather than overwrite — the maintainer can compare two runs and
  decide which is the source of truth for the current release.
- The `audit-artifact-sync` test pins the **input-bundle hash** and
  the **schema validator** as deterministic; the LLM's text output
  is intentionally not pinned (would force a re-run of every test
  every time the rubric or model changed).

**Rationale.** The reviewer concern is "could a different
maintainer run this and get a different result?" Yes — the model
output is non-deterministic. The mitigation is provenance, not fake
determinism. The schema validator and the input-bundle builder are
where we enforce reproducibility.

## 7. CLI flags for `run_llm_critique.py`

**Decision.** Mirror `validate_release_candidate.py`'s posture
(argparse, free-function `parse_args` for testability, `DriverConfig`
dataclass, `run_critique(config) -> DriverResult`, `main(argv)`
returning an exit code).

```
--release-dir release/                    # default
--out-dir release/validation/             # default
--prompt docs/release/llm_critique_prompt.md  # default
--model claude-opus-4-7                   # default
--tier intermediate                       # default
--effort high                             # default
--max-tokens 16000                        # default
--dry-run                                 # build the bundle, write it
                                          # to <out>/llm_critique_input_<ts>.md,
                                          # don't call the API
--no-execute                              # check creds + format, don't run
                                          # — for CI smoke
--out-tag                                 # optional suffix on output filename
                                          # so adjudication runs don't
                                          # clobber each other
```

**Exit codes.**
- `0` — pass (no unresolved high-severity findings *and* schema
  validation passed *and* (`ANTHROPIC_API_KEY` skip → 0 too)).
- `1` — critique surfaced unresolved high-severity findings. The
  adjudicator must either fix in code OR log to v2_decision_log.md
  before the gate flips to 0. (Adjudication is **maintainer-driven**
  in this PR; PR 7.3 wires the gate into a release-readiness check.)
- `2` — pre-flight error (missing release dir, malformed prompt
  file, schema-validation failure on the LLM response, network
  exhaustion).

**Rationale.** PR 5.2 / 5.1 / 4.1 / 3.3 all use this shape. Mirroring
it means the maintainer's muscle memory works
(`--no-rebuild`-equivalent is `--dry-run` here, since this script
doesn't rebuild bundles).

`--no-execute` separately from `--dry-run`: the former checks the
SDK is installed and the key is set without burning a real API
call (CI smoke); the latter writes the input bundle to disk for
manual inspection without calling the API. Different jobs.

## 8. Test posture

**Decision.** No live API calls in `pytest`. Tests live under
`tests/validation/test_llm_critique.py` and `tests/scripts/test_run_llm_critique.py`.

Coverage:

1. `build_input_bundle` is deterministic — same release dir →
   identical bytes. Fixture-driven (a small synthetic bundle under
   `tests/fixtures/llm_critique/`).
2. `build_input_bundle` references `BANNED_*` constants live (not
   string-duplicated) — sync test asserts the diff summary contains
   every banned column from the constants.
3. `parse_critique_response` accepts a well-formed payload, rejects
   the pinned malformations (missing required field, wrong severity
   value, wrong category value, wrong rubric dimension, non-JSON
   output, top-level non-object, finding.id collision, findings
   non-list, score out of range, wrong release_id, non-string
   `missing_sections` / `questions_for_maintainer` entry, defensive
   single-outer-code-fence stripping). `run_timestamp` is
   driver-generated (not LLM-supplied), so it has no malformation
   surface to validate.
4. `run_critique` skip-cleanly path: with `ANTHROPIC_API_KEY` unset,
   exit 0, no I/O, single stderr line. Spot-check this writes
   nothing to `--out-dir`.
5. `run_critique` skip-cleanly path: with `ANTHROPIC_API_KEY=""`
   (empty after strip), same behavior as unset.
6. Mocked-client happy path: monkey-patch the Anthropic
   implementation to return a canned JSON response → assert the
   driver writes both files, exit 0, hash matches.
7. Mocked-client high-severity path: canned response with one
   `severity=high` finding → exit 1, summary still rendered.
8. Mocked-client malformed path: canned response with extra
   non-JSON prose → exit 2, error message specific to the malformation.
9. Output filename includes ISO-8601 timestamp; two consecutive
   runs produce two files (no clobber).
10. `--dry-run` writes the input-bundle file and skips the API
    call; `--no-execute` validates creds without writing anything.

Mocked client is a small Protocol-conforming class that returns a
fixture response; not a `unittest.mock.MagicMock`, which would
encourage testing implementation details. The fixture response is
itself checked-in JSON under `tests/fixtures/llm_critique/`.

## 9. The first critique run

**Sequencing.** Module + driver + rubric land first as a separate
commit. Then run the critique once locally (with the user's real
key — agent does NOT have access; the brief flags this as a
"first actions" step the maintainer or the agent runs at the end
of the work). Adjudicate any high-severity findings:
- Fix in code in **this** PR if the fix is small and uncontroversial.
- Otherwise, log to `docs/release/v2_decision_log.md` with
  verdict per the schema (`accepted-for-v2` / `deferred` /
  `wont-fix` / `needs-investigation`).

**Output filenames.** Per the brief:
- `release/validation/llm_critique_raw_<UTC-iso>.json`
- `release/validation/llm_critique_summary.md`

The `<UTC-iso>` timestamp lets re-runs accumulate without clobber.
The Markdown summary is a single canonical file (overwritten per
run) so the dataset card's link doesn't rot. The raw JSON files
are append-only history.

**Audit-artifact-sync.** A separate test asserts the
**input-bundle builder** is in sync with the **release artefacts
on disk**: `build_input_bundle("release/", "intermediate")` →
hash matches the `input_bundle_sha256` field in the most-recent
committed `llm_critique_raw_*.json`. If the bundle changes, the
test fails — flagging that the LLM critique is stale and needs
re-running before the next release-candidate gate.

The LLM's text output itself is **not** pinned. The schema validator
proves the structure is sound; the freshness gate proves the input
was current; the model output is intentionally one-shot per
release-candidate.

## Out of scope (logged so reviewers don't ask)

- Multi-provider abstraction (post-v1).
- CI integration of the critique gate (post-v1; this PR is local-only).
- Quantitative semantic-diversity validator (post-v1; recommendation
  #12's post-v1 scope, see `recommendations_pass.md`).
- All three tiers in one critique (only intermediate; cross-tier is
  in the validation report already).
- Streaming the LLM output to the human in real-time (we stream the
  API call to avoid timeouts but consume to completion before
  writing — simpler, no UI cost).

## What this PR does not touch

- `BUNDLE_SCHEMA_VERSION` stays at 5.
- `release/validation/validation_report.{json,md}` does not
  regenerate (nothing in this PR changes the metrics).
- PR 7.2's preview tooling and PR 7.3's publish scripts are
  separate PRs.

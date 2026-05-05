# v1 Release Design

Architectural decisions specific to `leadforge-lead-scoring-v1`. The
existing `docs/leadforge_design_doc.md` and
`docs/leadforge_architecture_spec.md` remain authoritative for the
framework; this document captures only the decisions that arise from the
v1 dataset release and that diverge from or extend the existing design.

## Naming and versioning decoupling

The leadforge **package** stays at `1.x` in `pyproject.toml`. The curated
public dataset release is named **`leadforge-lead-scoring-v1`** and is
versioned independently of the package. Future iterations of the dataset
(after fixing leakage / rebuilding channel signal / etc.) bump the
*dataset* version (`v1` → `v2`), not the package.

**Rationale:** the alpha shipped while the package was already at `1.0.0` /
Production/Stable. Conflating the two confuses users — a v0.x dataset
could not exist while the package is v1, and a "v1 of the dataset" implies
a coordinated bump of the framework that is not actually planned.

**Implication for releases:**
- Kaggle dataset slug: `leadforge-lead-scoring-v1` (or `<owner>/leadforge-lead-scoring-v1`).
- Hugging Face repo: `<owner>/leadforge-lead-scoring-v1`.
- GitHub release tag and Release name: `leadforge-lead-scoring-v1`. Distinct from any package tags.

## Dataset family architecture

The release is a *family*, not a single CSV.

### Public family (Kaggle / HF)
- **intro** — easiest tier (intro 41.5% conversion in alpha; AUC ~0.89).
- **intermediate** — middle tier (~20.1% conversion; AUC ~0.88).
- **advanced** — hardest tier (~7.9% conversion; AUC ~0.87).

Each tier is a complete bundle:
- Flat task splits as Parquet at `tasks/<task_id>/{train,valid,test}.parquet` (current alpha contract) plus a single joined `lead_scoring.csv` with a `split` column. Phase 5 platform packaging will additionally emit `{train,valid,test}.csv` exports for Kaggle/HF consumers who prefer flat CSV; filenames mirror the parquet split names (i.e. `valid.csv`, not `validation.csv`, to keep one canonical name).
- **Snapshot-safe relational tables** (see below): `accounts.parquet`, `contacts.parquet`, `leads.parquet`, `touches.parquet`, `sessions.parquet`, `sales_activities.parquet`, `opportunities.parquet`. **No `customers` or `subscriptions` tables in public bundles.**
- `manifest.json`, `feature_dictionary.csv`, `dataset_card.md`.

### Instructor / research companion (separate artifact)
Exists at the `intermediate_instructor` tier only (matches alpha pattern). Contents:
- Full hidden world graph (`metadata/graph.json`, `graph.graphml`).
- Latent registry (`metadata/latent_registry.json`).
- World spec (`metadata/world_spec.json`).
- Mechanism summary (`metadata/mechanism_summary.json`).
- **Full-horizon relational tables** including `customers.parquet`, `subscriptions.parquet`, `leads.parquet` with `converted_within_90_days` + `conversion_timestamp`, `opportunities.parquet` with `close_outcome` + `closed_at`.
- Leakage-trap features explicitly marked (`__leakage__*` naming convention from v6/v7).

### Where the companion lives
The instructor companion ships as a **separate** GitHub Release artifact and a **separate** Hugging Face repo (`<owner>/leadforge-lead-scoring-v1-instructor`). It is **not** uploaded to Kaggle.

**Rationale:** Kaggle's dataset model assumes one dataset per repo and surfaces all files alike. Hidden truth and leakage-trap features should not be one click away from student-facing files. A separate repo also lets us require explicit acceptance (e.g., HF gated repo) before download if needed for academic settings.

## Snapshot-safe relational export — new architectural component

The single most important architectural change in this release.

### Problem
The v0.1.0-alpha public `student_public` bundles include relational tables that allow target reconstruction with 100% accuracy via joins:
- `tables/leads.parquet` retains `converted_within_90_days` and `conversion_timestamp`.
- `tables/opportunities.parquet.close_outcome == "closed_won"` perfectly distinguishes converted leads.
- `customers` and `subscriptions` tables exist *only* for converted leads — their presence is the label.

Verified in a 500-lead `student_public` smoke bundle by ChatGPT v2 reviewer (chatgpt_report_v2.md §0).

### Decision
A new module `leadforge/render/relational_snapshot_safe.py` produces a **snapshot-safe** relational export for `student_public` bundles. Properties:

1. **Event tables** are filtered per-table to their snapshot-relative timestamp column: `touches.touch_timestamp`, `sessions.session_timestamp`, `sales_activities.activity_timestamp` — each must satisfy `<= lead_created_at + snapshot_day`. Same temporal boundary used for flat-CSV features.
2. **`leads.parquet`** drops `converted_within_90_days` and `conversion_timestamp`. The label only lives in the task splits, where it is the explicit y-column.
3. **`opportunities.parquet`** is filtered to `created_at <= lead_created_at + snapshot_day` and drops `close_outcome` and `closed_at`.
4. **`customers.parquet` and `subscriptions.parquet`** are omitted from public bundles entirely.
5. **Account- and contact-level tables** are not filtered (they are static firmographic/personographic features).

The full-horizon relational export remains in `leadforge/render/relational.py` and is used unchanged for the instructor companion.

### Bundle schema bump: v4 → v5
The `BUNDLE_SCHEMA_VERSION` constant moves from 4 to 5. The manifest gains a `relational_snapshot_safe: bool` field (true for `student_public`, false for `research_instructor`). This makes consumers self-describing — a tool reading a v5 bundle can tell from the manifest whether the relational tables are snapshot-safe or full-horizon.

### New validator `leadforge/validation/relational_leakage.py`
Three categories of probe:
- **Structural**: assert no banned columns appear in public `leads`/`opportunities`; assert `customers`/`subscriptions` absent from public; assert event-table timestamps ≤ snapshot.
- **Probabilistic**: train a lightweight model using only public relational features and joinable keys; assert reconstructed-target AUC/accuracy is below tolerance.
- **Schema-vs-manifest**: assert the manifest's `relational_snapshot_safe` flag matches the actual table contents (catches misconfigured exposure routes).

Wired into `leadforge/validation/bundle_checks.py:validate_bundle()` so any bundle that violates these contracts fails validation by default.

## Release validation — new architectural component

The framework already has `leadforge/validation/{bundle_checks,realism,difficulty,drift,lead_scoring}.py`. The v1 release adds a higher-level **release-grade** layer that consumes those primitives and produces a single reproducible report.

### New modules
- `leadforge/validation/release_quality.py` — orchestrates the metric panel.
- `leadforge/validation/leakage_probes.py` — direct / time-window / relational / split / model-realism probes (per recommendations Guid §8).
- `leadforge/validation/reporting.py` — renders `validation_report.{json,md}` and figures.
- `scripts/validate_release_candidate.py` — driver script.

### Output contract
```
release/validation/
  validation_report.json     # machine-readable; fields per v1_acceptance_gates.md
  validation_report.md       # human-readable
  figures/
    lift_curve_intro.png
    lift_curve_intermediate.png
    lift_curve_advanced.png
    calibration_intermediate.png
    leakage_delta.png
    cohort_shift.png
    value_capture.png
```

### Difficulty bands
The current `validation/difficulty.py` validates conversion-rate ranges. v1 expands the check to:
- **AP** band per tier
- **P@K** band per tier
- **GBM-vs-LR delta** band (model-family delta — pedagogically meaningful)
- **Calibration** (Brier score) band per tier
- **Cohort-shift AUC degradation** band

Concrete numeric ranges are set in `v1_acceptance_gates.md` once Phase 3 produces baseline numbers.

## LLM critique — new architectural component (minimal v1 scope)

A new validation module `leadforge/validation/llm_critique.py` provides a structured one-shot LLM review.

### Scope decisions
- **Single provider** in v1 (Anthropic Claude as default). Multi-provider adjudication is post-v1 work.
- **Skips cleanly** without credentials — env var absence yields a clear "skipped: no credentials" message, not a failure.
- **Output schema** is fixed (per `recommendations_pass.md` §11; mirrors Guid §12):
  ```
  {
    "release_id": "leadforge-lead-scoring-v1",
    "model": "anthropic/claude-opus-4-7/...",
    "run_timestamp": "ISO-8601",
    "overall_score": <float>,
    "findings": [
      { "severity": "critical|high|medium|low|nit",
        "category": "leakage|realism|documentation|platform|ethics|pedagogy|code",
        "claim": "...",
        "evidence": "file/path:line or artifact ref",
        "reproducer": "optional cmd",
        "suggested_fix": "..." }
    ],
    "missing_sections": [],
    "questions_for_maintainer": []
  }
  ```
- **Adjudication is manual** in v1 — high-severity findings are resolved by hand or filed in `v2_decision_log.md` if intentional-and-accepted. CI auto-fail on high severity is post-v1.

### Rubric dimensions for v1
- **Logical coherence** (G1, G2): does the lead trajectory make sense?
- **Behavioral plausibility** (G1, G2): are events consistent with firmographics?
- **Effective semantic diversity** (G2): does the cohort cover the firmographic / behavioral space?
- **Syntax validity** (G2): are categorical fields free of hallucinatory artifacts?
- **Documentation completeness** (Datasheets / Data Cards Playbook): is the dataset card complete?
- **Leakage flagging** (C2): does the documentation make the leakage policy clear?
- **Pedagogical clarity** (C2): does a student have a clear entry point?

### Bias mitigation (G2)
- Forced-rationale prompts: judge must emit step-by-step analysis before assigning a numerical score.
- Explicit instruction not to favor verbose responses or self-similar outputs.

## Module landscape (new in v1 release work)

```
leadforge/
  render/
    relational_snapshot_safe.py     # NEW — Phase 2
  validation/
    relational_leakage.py           # NEW — Phase 2
    release_quality.py              # NEW — Phase 3
    leakage_probes.py               # NEW — Phase 3
    reporting.py                    # NEW — Phase 3
    llm_critique.py                 # NEW — Phase 7

scripts/
  audit_channel_signal.py           # NEW — Phase 4
  validate_release_candidate.py     # NEW — Phase 3
  package_kaggle_release.py         # NEW — Phase 5
  package_hf_release.py             # NEW — Phase 5
  run_llm_critique.py               # NEW — Phase 7
  publish_kaggle.py                 # NEW — Phase 7
  publish_hf.py                     # NEW — Phase 7

docs/release/
  v1_release_roadmap.md             # this PR
  post_v1_roadmap.md                # this PR
  v1_release_design.md              # this PR (this file)
  v1_acceptance_gates.md            # this PR
  v1_current_state_audit.md         # Phase 1
  channel_signal_audit.md           # Phase 4
  generation_method.md              # Phase 4
  feature_dictionary.md             # Phase 4
  break_me_guide.md                 # Phase 6
  v2_decision_log.md                # Phase 6 (starts empty)
  llm_critique_prompt.md            # Phase 7
  v1_release_notes.md               # Phase 7

release/
  kaggle/                           # NEW — Phase 5 (generated)
    dataset-metadata.json
  huggingface/                      # NEW — Phase 5 (generated)
    README.md
  validation/                       # NEW — Phase 3+ (generated)
    validation_report.{json,md}
    figures/*.png
    llm_critique_*.{json,md}
  notebooks/
    01_baseline_lead_scoring.ipynb  # updated — Phase 6
    02_relational_feature_engineering.ipynb   # NEW — Phase 6
    03_leakage_and_time_windows.ipynb         # NEW — Phase 6
    04_lift_calibration_value_ranking.ipynb   # NEW — Phase 6
  dataset-cover-image.png           # NEW — Phase 5

.github/
  ISSUE_TEMPLATE/
    dataset_breakage_report.yml     # NEW — Phase 6
    realism_feedback.yml            # NEW — Phase 6
```

## What this design does NOT change

- The seven-layer design (narrative / schema / structure / mechanism / simulation / render / validation / exposure) remains.
- Determinism, RNG roots, motif sampling, hidden-graph DAG construction, and exposure modes are unchanged.
- The flat-CSV path is unchanged at the feature level (it was already snapshot-safe via windowed snapshot in `BUNDLE_SCHEMA_VERSION` 4).
- `Generator.from_recipe(...).generate(...)` API surface is unchanged.
- CLI commands `generate`, `inspect`, `validate`, `list-recipes` are unchanged in shape (some grow `--json` output post-v1).

## Risks captured

- **Bundle schema v4 → v5 break.** Consumers of v0.1.0-alpha bundles may need to re-read against the new schema. Mitigated by retaining the schema version field in manifest and documenting the v4→v5 contract in `v1_release_notes.md`.
- **Snapshot-safe export may eliminate features that students actually want.** Mitigated by keeping the *flat task path* feature-rich (it was always snapshot-safe) and providing the relational FE notebook (Phase 6 #02) to demonstrate legitimate joins.
- **LLM critique false-positives.** Mitigated by manual adjudication in v1; fully automated gate deferred to post-v1.
- **Cover image sourcing TBD.** Captured as open question in roadmap.

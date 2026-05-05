# v1 Acceptance Gates

Concrete, machine-checkable criteria for "v1 ready". A release candidate
that satisfies every gate below can be tagged and published. Numeric bands
prefixed with `TBD` are placeholders set in Phase 3 of the v1 release
roadmap; a release candidate cannot ship until all `TBD`s are resolved.

This file is the operational definition of done for the v1 release. It is
read by `scripts/validate_release_candidate.py` and by humans before tag.

## Naming and versioning gate

- **G1.1** Dataset release name: `leadforge-lead-scoring-v1`. Locked in Phase 1.
- **G1.2** Kaggle slug: `leadforge-lead-scoring-v1`.
- **G1.3** Hugging Face repo: `leadforge-lead-scoring-v1` (public family) and `leadforge-lead-scoring-v1-instructor` (companion).
- **G1.4** Bundle `package_version` reflects the leadforge package at build time.
- **G1.5** Bundle `bundle_schema_version == 5`.

## Reproducibility gate

- **G2.1** Two independent builds with the same `--generation-timestamp` produce byte-identical bundles modulo timestamp-derived fields. Verified by `scripts/verify_hash_determinism.py`.
- **G2.2** All file SHA-256 hashes recorded in `manifest.json` match the actual files at validation time.
- **G2.3** A clean-environment regeneration on a different machine produces identical bundles to the developer's build (if not literally identical, deviations must be explainable solely by the timestamp field).

## Structural gate

- **G3.1** Every bundle in the family contains `manifest.json`, `dataset_card.md`, `feature_dictionary.csv`, `tables/`, `tasks/`.
- **G3.2** Every required relational table for the bundle's mode is present and non-empty.
- **G3.3** All foreign-key constraints in `ALL_CONSTRAINTS` hold.
- **G3.4** All task splits (`train`, `valid`, `test`) are non-empty and disjoint.

## Relational leakage gate (the v1 critical gate)

This is the gate that motivates the v1 release. Failures here are blockers.

- **G4.1** Public `tables/leads.parquet` does **not** contain `converted_within_90_days` or `conversion_timestamp`.
- **G4.2** Public `tables/opportunities.parquet` does **not** contain `close_outcome` or `closed_at`.
- **G4.3** Public bundles do **not** contain `tables/customers.parquet` or `tables/subscriptions.parquet`.
- **G4.4** Public event tables (`touches`, `sessions`, `sales_activities`) contain no rows where `event_timestamp > lead_created_at + snapshot_day`.
- **G4.5** Probabilistic relational reconstruction probe: a model trained using only public relational features (joined on `lead_id`/`account_id`/`contact_id`) achieves AUC ≤ TBD-G4.5 against `converted_within_90_days`. Threshold derived during Phase 3 from honest-feature baseline.
- **G4.6** Manifest field `relational_snapshot_safe == true` for `student_public` bundles; `false` for `research_instructor`.

## Direct leakage gate

- **G5.1** Models trained using only post-snapshot aggregate features cannot reconstruct the target above tolerance TBD-G5.1.
- **G5.2** Models trained using only suspect-stage columns (`current_stage`, `is_sql`) cannot reconstruct the target above tolerance TBD-G5.2.
- **G5.3** ID-only models (using only `lead_id`/`account_id`/`contact_id`) achieve AUC ≤ 0.5 + ε.
- **G5.4** No public feature derives from events with timestamp > `lead_created_at + snapshot_day` (audited at the `FeatureSpec` level — recipe must declare provenance).

## Split leakage gate

- **G6.1** Account-overlap audit: same `account_id` in train + test is documented as intentional or absent.
- **G6.2** Contact-overlap audit: same `contact_id` in train + test is documented as intentional or absent.
- **G6.3** Near-duplicate row detection: no rows with feature-vector cosine similarity > 0.99 across splits.
- **G6.4** Cohort-time-shift split exists: AUC degradation under cohort split ≥ TBD-G6.4 (lower bound — cohort split should be meaningfully harder than random) and ≤ TBD-G6.4-upper (upper bound — but not catastrophic).

## Performance gates (per tier)

Bands set in Phase 3 from baseline measurements; written here as the contract.

### Intro tier
- **G7.1.1** Conversion rate within [TBD, TBD]
- **G7.1.2** LR AUC within [TBD, TBD]
- **G7.1.3** GBM AUC within [TBD, TBD]
- **G7.1.4** GBM-vs-LR AUC delta ≥ TBD-G7.1.4
- **G7.1.5** AP within [TBD, TBD]
- **G7.1.6** P@100 within [TBD, TBD]
- **G7.1.7** Brier score within [TBD, TBD]
- **G7.1.8** Calibration max-bin error ≤ TBD-G7.1.8

### Intermediate tier
- **G7.2.1**–**G7.2.8** mirroring intro, with bands shifted to reflect higher difficulty (lower AP, lower P@K, similar AUC, similar GBM-vs-LR delta).

### Advanced tier
- **G7.3.1**–**G7.3.8** mirroring intro, with hardest bands.

### Cross-tier ordering
- **G7.4.1** AP ordering: intro > intermediate > advanced.
- **G7.4.2** P@K ordering: intro > intermediate > advanced.
- **G7.4.3** Conversion-rate ordering: intro > intermediate > advanced.
- **G7.4.4** GBM-vs-LR delta is positive in every tier (sophistication is rewarded).

## Cross-seed stability gate

- **G8.1** Run N=5 seeds per tier; each metric in G7 falls within ±TBD-G8.1 of the reported median.
- **G8.2** No degenerate seeds (conversion rate < 1% or > 99% in any seed).

## Public/instructor diff gate

- **G9.1** Every public/instructor difference is intentional and listed in `release/EXPOSURE_DELTA.md`.
- **G9.2** Manifest `redacted_columns` field matches the actual public bundle's column omissions.
- **G9.3** Instructor-companion-only artifacts (`metadata/`, leakage-trap features, full-horizon tables) are absent from public bundles.

## Documentation gate

- **G10.1** `release/README.md` (the dataset card) passes a Datasheets-for-Datasets / Data Cards Playbook checklist:
  - Provenance (who, when, why)
  - Motivation
  - Composition (entities, features, label, splits)
  - Collection / generation method
  - Preprocessing and transformations
  - Recommended uses
  - Out-of-scope uses
  - Known limitations and biases
  - Maintenance plan
- **G10.2** `docs/release/generation_method.md` exists and is readable as a standalone document.
- **G10.3** `docs/release/feature_dictionary.md` covers every feature in the snapshot CSV with description, dtype, source, leakage flag, and recommended-for-modeling flag.
- **G10.4** `docs/release/break_me_guide.md` exists and links from `release/README.md`.
- **G10.5** `docs/release/v1_release_notes.md` exists and is human-readable.
- **G10.6** Every claim made in the dataset card about realism, calibration, or difficulty has a backing reference in `release/validation/validation_report.md`.

## Platform packaging gate

### Kaggle
- **G11.1** `release/kaggle/dataset-metadata.json` exists and validates against current Kaggle schema:
  - `title` length 6-50 chars
  - `subtitle` length 20-80 chars
  - `id` slug 3-50 chars
  - exactly one entry in `licenses`
  - `expectedUpdateFrequency` from approved values (`never` for v1)
  - all `resources[].schema.fields` listed in column order
- **G11.2** `release/dataset-cover-image.png` exists with dimensions ≥ 560 × 280.
- **G11.3** Kaggle dry-run package builds without error: `kaggle datasets create -p release/kaggle --dir-mode zip` (in `--dry-run` if available, or shape-validate without).

### Hugging Face
- **G12.1** `release/huggingface/README.md` exists with valid YAML metadata: `pretty_name`, `license`, `language: en`, `task_categories: [tabular-classification]`, `size_categories`, `tags`, `configs`.
- **G12.2** Exactly one config has `default: true`.
- **G12.3** Local `load_dataset(release/huggingface, "intro")` succeeds; same for `intermediate`, `advanced`.
- **G12.4** Companion repo (`leadforge-lead-scoring-v1-instructor`) packages independently and loads via `load_dataset()` for at least one config.

## Notebook gate

- **G13.1** All four notebooks in `release/notebooks/` execute top-to-bottom from a clean environment without errors.
- **G13.2** Each notebook's printed metrics match the validation report within tolerance TBD-G13.2.
- **G13.3** Each notebook explicitly distinguishes the public path from the instructor companion path; instructor-only artifacts are not loaded by the public notebooks.

## LLM critique gate

- **G14.1** `scripts/run_llm_critique.py` runs successfully when credentials are present.
- **G14.2** The critique produces a structured findings JSON conforming to the schema in `v1_release_design.md` §"LLM critique".
- **G14.3** No unresolved high-severity findings remain. Each high-severity finding is either:
  - resolved in code (with a backing PR), or
  - documented in `docs/release/v2_decision_log.md` as intentional-and-accepted with rationale.
- **G14.4** Raw LLM outputs are archived under `release/validation/llm_critique_raw_*.json` for audit.

## Adversarial framing gate

- **G15.1** GitHub issue templates (`dataset_breakage_report.yml`, `realism_feedback.yml`) render correctly.
- **G15.2** `docs/release/break_me_guide.md` is linked from `release/README.md`, the Kaggle description, and the HF README.
- **G15.3** `docs/release/v2_decision_log.md` exists (may be empty at launch).

## Out-of-scope acknowledgment

The following are explicitly NOT release blockers for v1; they live in `post_v1_roadmap.md`:

- Channel-conditional MQL→SQL rates (audit only in v1).
- Log-normal sales-cycle distributions.
- Demographic noise injection.
- Quantitative semantic-diversity validator.
- Multi-provider LLM critique CI integration.
- LTV labels as first-class outputs.
- Second vertical / per-vertical calibration.
- Leaderboard mini-site.

## Definition of green

A release candidate is **green** (ready to publish) when:
- All gates G1–G15 pass.
- All `TBD-*` placeholders have been resolved with concrete numeric values during Phase 3.
- The validation report explicitly cites the gate that justifies each metric band.
- A human signs off on `v2_decision_log.md` entries for any accepted-with-rationale findings.

A release candidate is **blocked** if any of:
- G4.* relational leakage gate fails.
- G5.* direct leakage gate fails.
- G7.4.4 GBM-vs-LR delta is non-positive in any tier (the dataset doesn't reward sophistication).
- G14.3 has unresolved high-severity findings.
- Any `TBD-*` remains unresolved at tag time.

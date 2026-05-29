---
sidebar_position: 4
title: Acceptance gates
---

# v1 Acceptance Gates

Concrete, machine-checkable criteria for "v1 ready". A release candidate
that satisfies every gate below can be tagged and published.

This file is the human-readable contract.  Numeric bands are tuned in
the companion YAML (`v1_acceptance_gates_bands.yaml`) — that file is
loaded by `scripts/validate_release_candidate.py` and is the single
source of truth for the per-band numbers.  This document records the
medians and rationale.

Initial calibration: 2026-05-06 from the PR 3.3 N=5 sweep on the
regenerated PR 2.2 bundles (BUNDLE_SCHEMA_VERSION 5; see
`release/validation/validation_report.json`).  Re-tune when the recipe,
mechanism layer, or difficulty profiles change.

## Naming and versioning gate

- **G1.1** Dataset release name: `leadforge-lead-scoring-v1`. Locked in Phase 1 (PR #61 milestone rename + roadmap edits; reaffirmed in PR 1.1's `docs/release/v1_current_state_audit.md`).
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
- **G4.4** Public event tables contain no rows past the snapshot: no `touches` row with `touch_timestamp > lead_created_at + snapshot_day`, no `sessions` row with `session_timestamp > lead_created_at + snapshot_day`, no `sales_activities` row with `activity_timestamp > lead_created_at + snapshot_day`. Public `opportunities` rows must satisfy `created_at <= lead_created_at + snapshot_day`.
- **G4.5** Probabilistic relational reconstruction probe: a model trained using only public relational features (joined on `lead_id`/`account_id`/`contact_id`) achieves AUC ≤ **0.65** against `converted_within_90_days`.  Threshold matches the existing `scripts/probe_relational_leakage.py --max-accuracy 0.65` posture used for the structural sweep on the alpha bundles; honest relational features (per-lead opportunity counts and ACV aggregates) carry signal but should not solo-dominate the task.
- **G4.6** Manifest field `relational_snapshot_safe == true` for `student_public` bundles; `false` for `research_instructor`.

## Direct leakage gate

- **G5.1** Models trained using only post-snapshot aggregate features (`total_touches_all`, the v1 leakage trap) achieve AUC ≤ **0.95** on the test split.  Observed median across seeds: ~0.54–0.55 per tier (max ~0.62).  The trap is *meant* to be predictive — the band only flags total-domination scenarios.
- **G5.2** Models trained using only suspect-stage columns (`current_stage`, `is_sql`) achieve AUC ≤ **0.95** when present.  Both columns are redacted under the `student_public` exposure mode; the gate is therefore effectively skipped on public bundles, but the band is declared for the instructor companion's full-horizon export.
- **G5.3** ID-only models (using only `lead_id`/`account_id`/`contact_id`) achieve AUC ≤ **0.60**.  Observed median per tier ~0.49–0.51 (max ~0.56); the 0.60 ceiling admits stratified-CV variance without green-lighting genuine ID-encoded leakage.
- **G5.4** No public feature derives from events with timestamp > `lead_created_at + snapshot_day` (audited at the `FeatureSpec` level — recipe must declare provenance).

## Split leakage gate

- **G6.1** Account-overlap audit: same `account_id` in train + test is documented as intentional or absent.
- **G6.2** Contact-overlap audit: same `contact_id` in train + test is documented as intentional or absent.
- **G6.3** Near-duplicate row detection: no rows with feature-vector cosine similarity > 0.99 across splits.
- **G6.4** Cohort-time-shift split exists: AUC degradation under cohort split lies within **[-0.05, 0.10]**.  Observed range across tiers is roughly [-0.02, 0.02] — v1's bundles are roughly IID-balanced over the 90-day horizon (no time-of-year drift baked in), so the gate is *informational* in v1 rather than discriminating.  v2 will explicitly inject seasonality / quarterly close cycles to make the gate bite; the lower bound stays loose for v1.

## Performance gates (per tier)

Bands fitted to the PR 3.3 N=5 sweep on `release/{intro,intermediate,advanced}/`.
All numeric bands live in `v1_acceptance_gates_bands.yaml`; medians and
rationale follow.

> **These bands are regression fences, not realism thresholds.**
> They are calibrated to the observed five-seed spread for this DGP and
> recipe configuration. A band being "wide" does not mean any value within
> it is equally realistic — it means the validator will not flag a new
> bundle as broken unless a metric drifts *outside* that window. The medians
> in each gate note are the meaningful targets; bands only fire on
> substantial unintended regressions. Tightening the bands is expected work
> when the DGP is redesigned for v2.

### Intro tier
- **G7.1.1** Conversion rate within **[0.24, 0.61]**.  Median 0.4267.
- **G7.1.2** LR AUC within **[0.82, 0.94]**.  Median 0.8788.
- **G7.1.3** GBM AUC within **[0.82, 0.92]**.  Median 0.8729.
- **G7.1.4** GBM-vs-LR AUC delta within **[-0.05, 0.05]**.  Median -0.0045.  *See G7.4.4 for the cross-tier sign concern.*
- **G7.1.5** Average Precision (LR) within **[0.62, 0.90]**.  Median 0.7608.
- **G7.1.6** P@100 within **[0.65, 0.95]**.  Median 0.80.
- **G7.1.7** Brier score ≤ **0.17**.  Median 0.1301.
- **G7.1.8** Calibration max-bin error ≤ **0.65**.  Median 0.2497.  Calibration metrics are noisy at small per-bin n; the band reflects observed spread, not a tightness claim.

### Intermediate tier
- **G7.2.1** Conversion rate within **[0.12, 0.31]**.  Median 0.2160.
- **G7.2.2** LR AUC within **[0.84, 0.93]**.  Median 0.8859.
- **G7.2.3** GBM AUC within **[0.82, 0.93]**.  Median 0.8755.
- **G7.2.4** GBM-vs-LR AUC delta within **[-0.04, 0.03]**.  Median -0.0072.
- **G7.2.5** Average Precision (LR) within **[0.40, 0.75]**.  Median 0.5752.
- **G7.2.6** P@100 within **[0.45, 0.75]**.  Median 0.59.
- **G7.2.7** Brier score ≤ **0.14**.  Median 0.1096.
- **G7.2.8** Calibration max-bin error ≤ **0.90**.  Median 0.2490.

### Advanced tier
- **G7.3.1** Conversion rate within **[0.04, 0.12]**.  Median 0.0840.
- **G7.3.2** LR AUC within **[0.81, 0.97]**.  Median 0.8861.
- **G7.3.3** GBM AUC within **[0.84, 0.91]**.  Median 0.8726.
- **G7.3.4** GBM-vs-LR AUC delta within **[-0.06, 0.04]**.  Median -0.0133.
- **G7.3.5** Average Precision (LR) within **[0.19, 0.52]**.  Median 0.3514.
- **G7.3.6** P@100 within **[0.20, 0.55]**.  Median 0.34.
- **G7.3.7** Brier score ≤ **0.09**.  Median 0.0611.
- **G7.3.8** Calibration max-bin error ≤ **1.0**.  Median 0.5234.  Class imbalance inflates per-bin variance; the band admits the observed range without green-lighting total miscalibration.

### Cross-tier ordering
- **G7.4.1** AP ordering: intro > intermediate > advanced.  *Holds.*
- **G7.4.2** P@K ordering: intro > intermediate > advanced.  *Holds.*
- **G7.4.3** Conversion-rate ordering: intro > intermediate > advanced.  *Holds.*
- **G7.4.4** GBM-vs-LR delta is positive in every tier (sophistication is rewarded).  **Known finding (v1 → v2).**  Observed median delta is slightly *negative* in every tier (intro -0.0045, intermediate -0.0072, advanced -0.0133): v1's snapshot is dominated by linear features (engagement aggregates + firmographics) and a HistGBM does not consistently beat a regularised logistic regression at this signal level.  The PR 3.3 driver gates on the per-tier `gbm_minus_lr_auc` bands (G7.1.4 / G7.2.4 / G7.3.4) rather than the cross-tier sign check; v2 will introduce non-linear interactions in the simulator (saturation curves, threshold effects) so the gate bites.  Tracked in the post-v1 roadmap.

## Cross-seed stability gate

- **G8.1** Run N=5 seeds per tier; the max-min spread of each headline metric stays under the per-metric ceiling: LR/GBM AUC ≤ 0.06; GBM−LR delta ≤ 0.05; LR Average Precision ≤ 0.13; Brier score ≤ 0.04; conversion rate ≤ 0.15.  Calibration max-bin error is intentionally not bounded here — its per-bin-n noise dominates the cross-seed signal at v1's class balances.
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
- **G13.2** Each notebook's printed metrics match the validation report within tolerance **±0.05** on AUC / AP / P@K and **±0.05** on Brier (out of scope for PR 3.3; set when notebooks land in Phase 6).
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
- The validation report explicitly cites the gate that justifies each metric band.
- A human signs off on `v2_decision_log.md` entries for any accepted-with-rationale findings.

A release candidate is **blocked** if any of:
- G4.* relational leakage gate fails.
- G5.* direct leakage gate fails.
- G7.4.4 GBM-vs-LR delta is non-positive in *every* tier *and* the per-tier `gbm_minus_lr_auc` bands have not been re-tuned to fit the new dataset (i.e. the dataset has degraded; v1's known-finding posture is not a free pass for future regressions).
- G14.3 has unresolved high-severity findings.

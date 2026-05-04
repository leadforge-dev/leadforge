# Changelog

All notable changes to leadforge are documented here.
Format inspired by [Keep a Changelog](https://keepachangelog.com/).

---

## Unreleased

### Bundle schema v3

`bundle_schema_version` bumped from `"2"` to `"3"`.  Two structural-leakage
fixes follow up on PR #56 (issue #57):

- **`is_mql` removed from the canonical feature list.**  Every lead is
  initialised at MQL stage in the simulator, making the column constant
  `True` and zero-variance.  It carried no information for modelling.
  The `LeadRow.is_mql` field is retained on the relational `leads.parquet`
  for now; only the snapshot/task-split column and feature-dictionary row
  are removed.  Affects all exposure modes.
- **`is_sql` redacted in `student_public` mode.**  Measured on the v3
  bundles: P(converted | is_sql=False) ≈ 0.04 / 0.015 / 0.006 across
  intro / intermediate / advanced.  At advanced tier this is effectively
  deterministic for the negative class — practically a one-rule
  classifier.  `is_sql` remains in `research_instructor` exports for
  DGP-aware research.

### New automated check

`validate_bundle()` now flags any zero-variance feature in the published
student_public task split (excluding ID columns and the target).

### Bundle column counts (v3)

- `student_public/{intro,intermediate,advanced}` — 32 columns (down from
  34 in v2): `is_mql` removed, `is_sql` redacted; `current_stage`
  redaction from PR #56 retained.
- `research_instructor/intermediate_instructor` — 34 columns (down from
  35): `is_mql` removed; `current_stage` and `is_sql` retained.

### Open follow-up

Issue #57 sub-item 1 remains open: event-aggregate features
(`touch_count`, `session_count`, `pricing_page_views`, ...) are still
computed over the same 90-day window the label resolves in.  The
structural fix is a windowed snapshot rebuild and is deferred to its
own PR.

---

## v1.0.0 — 2026-05-02

First stable release. All milestones (M0–M13) complete.

### Highlights

- **Full end-to-end generation**: recipe → simulated world → relational bundle with deterministic reproducibility.
- **90-day daily-step simulation engine** with churn, stage advancement, conversion hazards, and touch emission.
- **5 motif families** (fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, buying-committee-friction) with stochastic rewiring.
- **Exposure filtering**: `student_public` (leakage-safe) and `research_instructor` (full truth) modes.
- **CLI**: `generate`, `inspect`, `validate`, `list-recipes` — all fully wired.
- **Validation harness**: determinism, exposure monotonicity, realism bounds, difficulty profiles, cross-seed drift detection.
- **v4–v6 dataset pipelines**: progressive dataset versions with leakage traps, student/instructor splits, value-aware scoring, and GBM improvement validation.

### Post-milestone improvements (since M13)

- **Direct conversion bypass** (PR #45): pre-SQL leads can now convert via a rare direct path, fixing the deterministic `is_sql → converts` invariant.
- **Configurable label window** (PR #43): `label_window_days` controls conversion label derivation in the simulation.
- **Generalized task support** (PR #40, #42): `primary_task` threaded through bundle, validation, and pipelines; dataset card prose adapts to non-conversion tasks.
- **Pipeline extraction** (PR #29, #34): build pipeline functions extracted into `leadforge.pipelines` with proper RNG conventions.
- **Latent-aware touch intensity** (PR #31): `LatentDecayIntensity` mechanism creates causal link between latent traits and touch patterns.
- **Canonical validation module** (PR #26): reusable lead scoring validation with sklearn pipeline.

### Development history

<details>
<summary>Milestone changelog (no intermediate versions were published)</summary>

#### Milestone 0.5.0 — Validation Harness & CLI Complete (2026-04-29)

- Full validation harness: determinism checks, exposure monotonicity, realism bounds, difficulty validation, cross-seed drift detection.
- `leadforge validate` command with artifact checks, FK integrity, leakage detection, and task split validation.
- Parquet metadata used for row counts (no full table reads during validation).

#### Milestone 0.4.0 — Simulation Engine & End-to-End Generation (2026-04-28)

- 90-day daily-step simulation engine with churn, stage advancement, conversion hazards, and touch emission.
- Population generation: accounts (3 latent traits), contacts (4 traits), leads (1 trait) with motif-family biases.
- Full render pipeline: 9-table relational output, leakage-free lead snapshots, deterministic train/valid/test splits.
- Exposure filtering: `student_public` and `research_instructor` modes with truth redaction.
- CLI commands: `generate`, `inspect`, `validate`, `list-recipes` — all fully wired.
- Bundle manifest with provenance, row counts, and SHA-256 file hashes.

#### Milestone 0.3.0 — World Structure & Mechanisms (2026-04-25)

- Hidden world graph (DAG) with 5 motif families: fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, buying-committee-friction.
- Stochastic graph rewiring: optional-node dropping, edge-weight jitter, latent-confounder injection.
- Mechanism layer: latent scores, conversion hazards, stage transitions, Poisson intensities, categorical influences, noisy proxies.
- Motif-aware mechanism assignment policies.

#### Milestone 0.2.0 — Config, Recipes & Narrative (2026-04-20)

- Typed `GenerationConfig`, `Recipe`, `WorldSpec` models with full precedence resolution (kwargs > override > recipe > defaults).
- Seeded RNG with SHA-256-derived named substreams for reproducibility.
- Narrative layer: company, product, market, GTM motion, personas, funnel stages — loaded from recipe YAML.
- Schema layer: 9 entity dataclasses, 10 FK constraints, 29 snapshot features, feature dictionary writer.
- Dataset card renderer from narrative + world spec.

#### Milestone 0.1.0 — Project Foundation (2026-04-18)

- Package skeleton, CLI entry point (`leadforge list-recipes`), CI pipeline.
- Recipe registry with `b2b_saas_procurement_v1` recipe.
- GitHub Actions: lint, typecheck, test matrix (Python 3.11 + 3.12).

</details>

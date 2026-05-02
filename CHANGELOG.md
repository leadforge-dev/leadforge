# Changelog

All notable changes to leadforge are documented here.
Format inspired by [Keep a Changelog](https://keepachangelog.com/).

---

## Unreleased

- **Direct conversion bypass** (PR #45): pre-SQL leads can now convert via a rare direct path, fixing the deterministic `is_sql → converts` invariant.
- **Configurable label window** (PR #43): `label_window_days` controls conversion label derivation in the simulation.
- **Generalized task support** (PR #40, #42): `primary_task` threaded through bundle, validation, and pipelines; dataset card prose adapts to non-conversion tasks.
- **Pipeline extraction** (PR #29, #34): build pipeline functions extracted into `leadforge.pipelines` with proper RNG conventions.
- **Latent-aware touch intensity** (PR #31): `LatentDecayIntensity` mechanism creates causal link between latent traits and touch patterns.
- **Canonical validation module** (PR #26): reusable lead scoring validation with sklearn pipeline.
- **v4–v6 dataset pipelines**: progressive dataset versions with leakage traps, student/instructor splits, value-aware scoring, and GBM improvement validation.

---

## Milestone 0.5.0 — Validation Harness & CLI Complete (2026-04-29)

- Full validation harness: determinism checks, exposure monotonicity, realism bounds, difficulty validation, cross-seed drift detection.
- `leadforge validate` command with artifact checks, FK integrity, leakage detection, and task split validation.
- Parquet metadata used for row counts (no full table reads during validation).

## Milestone 0.4.0 — Simulation Engine & End-to-End Generation (2026-04-28)

- 90-day daily-step simulation engine with churn, stage advancement, conversion hazards, and touch emission.
- Population generation: accounts (3 latent traits), contacts (4 traits), leads (1 trait) with motif-family biases.
- Full render pipeline: 9-table relational output, leakage-free lead snapshots, deterministic train/valid/test splits.
- Exposure filtering: `student_public` and `research_instructor` modes with truth redaction.
- CLI commands: `generate`, `inspect`, `validate`, `list-recipes` — all fully wired.
- Bundle manifest with provenance, row counts, and SHA-256 file hashes.

## Milestone 0.3.0 — World Structure & Mechanisms (2026-04-25)

- Hidden world graph (DAG) with 5 motif families: fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, buying-committee-friction.
- Stochastic graph rewiring: optional-node dropping, edge-weight jitter, latent-confounder injection.
- Mechanism layer: latent scores, conversion hazards, stage transitions, Poisson intensities, categorical influences, noisy proxies.
- Motif-aware mechanism assignment policies.

## Milestone 0.2.0 — Config, Recipes & Narrative (2026-04-20)

- Typed `GenerationConfig`, `Recipe`, `WorldSpec` models with full precedence resolution (kwargs > override > recipe > defaults).
- Seeded RNG with SHA-256-derived named substreams for reproducibility.
- Narrative layer: company, product, market, GTM motion, personas, funnel stages — loaded from recipe YAML.
- Schema layer: 9 entity dataclasses, 10 FK constraints, 29 snapshot features, feature dictionary writer.
- Dataset card renderer from narrative + world spec.

## Milestone 0.1.0 — Project Foundation (2026-04-18)

- Package skeleton, CLI entry point (`leadforge list-recipes`), CI pipeline.
- Recipe registry with `b2b_saas_procurement_v1` recipe.
- GitHub Actions: lint, typecheck, test matrix (Python 3.11 + 3.12).

# Changelog

All notable changes to leadforge are documented here.
Format inspired by [Keep a Changelog](https://keepachangelog.com/).

---

## Unreleased

### CLI surfaces v4 fields

- `leadforge inspect` now prints `Primary task`, `Label window`,
  `Snapshot day`, and `Redactions` for v3+ bundles, immediately after
  `Schema ver`.  Lines are omitted entirely on older v2 bundles —
  no `?` placeholders.  Snapshot day prints `(full horizon, no
  windowing)` only when the manifest stores `null`; numeric values
  (including `snapshot_day == horizon_days`) are printed verbatim.
- `leadforge inspect --json` / `-j` emits the parsed `manifest.json`
  to stdout — the output is byte-equivalent JSON to the on-disk
  manifest, suitable for `jq` pipelines.
- `leadforge generate` adds `--snapshot-day`, `--primary-task`, and
  `--label-window-days` flags, threading directly to existing
  `Generator.from_recipe()` kwargs.  Recipe defaults still apply when
  the flags are omitted.

### Bundle schema v4

`bundle_schema_version` bumped from `"3"` to `"4"`.  Closes the final
sub-item of issue #57: event-aggregate features are no longer computed
over the same 90-day window the label resolves in.

- **Windowed snapshot.**  `GenerationConfig.snapshot_day` (also exposed
  as a recipe-level field and an explicit kwarg on
  `Generator.from_recipe()`) now controls the feature aggregation
  window.  When set, `build_snapshot()` filters touches, sessions,
  sales activities, and opportunities to events with timestamp
  ≤ `lead_created_at + snapshot_day`.  The
  `b2b_saas_procurement_v1` recipe pins `snapshot_day: 30` —
  measurements at seed 42, n_leads=5000 across all three difficulty
  tiers showed day 30 keeps LR AUC in [0.85, 0.86] (challenging but
  modelable) while preserving a meaningful trap gap of ~3 touches
  with 54–77% of leads showing any divergence between
  `total_touches_all` (full-horizon) and `touch_count` (windowed).
- **Conversion rates unchanged.**  The label is event-derived from
  `label_window_days` in the simulator and is independent of
  `snapshot_day`, so the published rates stay at 41.5% / 20.1% / 7.9%
  (intro / intermediate / advanced) — well inside the declared
  `difficulty_profiles.yaml` ranges.
- **`manifest.snapshot_day` recorded.**  The published bundle
  declares its windowing contract; consumers can distinguish
  full-horizon (legacy v2/v3) bundles from windowed (v4) bundles
  without inspecting package internals.  Column SET is unchanged
  from v3, but column VALUES are no longer full-horizon — a contract
  shift that v3 consumers would not detect from schema alone.
- **Schema contract test.**  `tests/render/test_bundle_schema_v3_contract.py`
  renamed to `test_bundle_schema_v4_contract.py` and gains a
  `snapshot_day == 30` assertion alongside the existing column-set
  pinning.
- **Trap invariant guard.**  New `tests/render/test_windowed_bundle_trap.py`
  asserts `total_touches_all >= touch_count` for every lead and
  `>` for at least some — guarding against a future refactor that
  silently widens `touch_count` back to the full horizon and
  collapses the pedagogical gap.

### Bundle schema v3

`bundle_schema_version` bumped from `"2"` to `"3"`.  Three structural
changes follow up on PR #56 (issue #57):

- **`is_mql` fully removed.**  Every lead is initialised at MQL stage in
  the simulator, making the field constant `True` and zero-variance.
  It carried no information for modelling and is now removed from the
  `LeadRow` entity, the relational `leads.parquet`, the snapshot, the
  task splits, and the feature dictionary — in all exposure modes.
- **`is_sql` redacted in `student_public` mode.**  Measured across 5
  seeds on full-size bundles: P(converted | is_sql=False) =
  0.061 ± 0.026 (intro) / 0.020 ± 0.010 (intermediate) /
  0.011 ± 0.004 (advanced).  At advanced tier this is essentially
  deterministic for the negative class — practically a one-rule
  classifier.  `is_sql` remains in `research_instructor` exports for
  DGP-aware research.
- **Redaction now applies to relational tables too.**  In v2, the
  exposure-layer redaction only stripped columns from the snapshot /
  task splits; users following the README's "Option 3" (feature
  engineering off the raw `tables/leads.parquet`) could trivially
  rejoin redacted columns.  In v3, `redacted_columns_for(mode)` is
  applied uniformly to every published parquet under both `tables/`
  and `tasks/`.  In `student_public` bundles, `tables/leads.parquet`
  no longer carries `current_stage` or `is_sql`.

### New automated checks

- `tests/render/test_bundle_schema_v3_contract.py` pins the exact
  column set per mode for v3 — any future change that touches the
  feature spec or redaction policy without updating the contract
  fails this test, forcing an explicit version coordination.
- `test_no_zero_variance_features` in `tests/exposure/test_redaction.py`
  asserts no constant or near-constant columns in the published
  student_public task split (1% rare-class threshold on bundles
  large enough for the threshold to be statistically meaningful).

### Bundle column counts (v3)

- `student_public/{intro,intermediate,advanced}`: **32** task split
  columns (down from 34 in v2); **9** columns in `tables/leads.parquet`
  (down from 12).
- `research_instructor/intermediate_instructor`: **34** task split
  columns (down from 35); **11** columns in `tables/leads.parquet`
  (down from 12 — `is_mql` removed).

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

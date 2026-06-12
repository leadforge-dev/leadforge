# LTV Workstream Roadmap — `b2b_saas_ltv_v1`

> Milestone/PR breakdown for the predictive-lifetime-value (pLTV) workstream.
> Design rationale lives in `design.md` (single source of truth). Update the
> checkboxes as work lands.

## Planning notation

Work items use a deliberate **`LTV-` prefix** scheme so they never collide
with GitHub PR numbers (`#NNN`):

- **Milestones** — `LTV-M0` … `LTV-M8`. A milestone is a coherent capability.
- **PRs** — `LTV-Pa`, `LTV-Pb`, … (sequential letters, globally unique across
  the whole roadmap). Each PR maps to exactly one milestone.

When a PR is opened on GitHub, reference its planning code in the title, e.g.
`feat(schema): lifecycle entity rows [LTV-Pb]`, and the GitHub number (`#NNN`)
is recorded back here on merge. The two namespaces stay distinct.

GitHub milestone: **`dataset: leadforge-ltv-v1`** (#8) — all LTV PRs assign to
it.

## Architecture context (peer generation schemes)

Per `design.md` §2.5, leadforge becomes a platform hosting **two parallel
generation schemes** (`lead_scoring`, `lifecycle`) behind a `GenerationScheme`
protocol + registry, with the package physically reorganized into
`leadforge/schemes/{lead_scoring,lifecycle}/`. Two consequences for sequencing:

- **`LTV-M2`** extracts the scheme abstraction **against the known-good
  lead-scoring path first** (output byte-identical) and performs the physical
  reorg — it lands early, before lifecycle internals are built, so those
  internals are written directly in their new home.
- The reorg touches a **published 1.x package**; every reorg PR keeps
  `verify_hash_determinism` + the full suite green and the public API stable.

---

## Milestone overview

| Milestone | Capability | PRs | GitHub PRs |
|-----------|------------|-----|------------|
| `LTV-M0` | Planning + design lock | `LTV-Pa` | #102, #103 (+ scheme reframe) |
| `LTV-M1` | Lifecycle schema foundation | `LTV-Pb`, `LTV-Pc` | #104 (Pb) |
| `LTV-M2` | Generation-scheme architecture + physical reorg | `LTV-Pd`, `LTV-Pe`, `LTV-Pf`, `LTV-Pg` | #107 (Pd), #108 (Pe), #109 (Pf.1), #110 (Pf.2), #111 (Pg.1), #112 (Pg.2) |
| `LTV-M3` | Customer population + lifecycle world | `LTV-Ph`, `LTV-Pi` | #113 (Ph) |
| `LTV-M4` | Lifecycle simulation engine | `LTV-Pj`, `LTV-Pk` | #117 (Pj), #118 (Pk) |
| `LTV-M5` | Customer snapshots + pLTV targets (both regimes) | `LTV-Pl`, `LTV-Pm` | #119 (Pl) |
| `LTV-M6` | Register LifecycleScheme + recipe + manifest/version | `LTV-Pn`, `LTV-Po` | |
| `LTV-M7` | Validation + regression-metric calibration | `LTV-Pp` | |
| `LTV-M8` | CLI, notebooks, publish | `LTV-Pq`, `LTV-Pr`, `LTV-Ps` | |

Total: ~19 PRs across 9 milestones.

---

## `LTV-M0` — Planning + design lock

- [x] **`LTV-Pa`** — planning. `docs/ltv/design.md` + `docs/ltv/roadmap.md`;
  milestone #8 + label. **Merged #102**, reframed to pLTV regression in
  **#103**, then reframed again to peer generation schemes (design.md §2.5).
  - Labels: `type: docs`

---

## `LTV-M1` — Lifecycle schema foundation

- [x] **`LTV-Pb`** — `feat(schema): lifecycle entity rows` (**PR #104**).
  `SubscriptionEventRow`, `HealthSignalRow`, `InvoiceRow` + dedicated
  `CustomerLifecycleRow`/`SubscriptionLifecycleRow`; separate
  `LIFECYCLE_ROW_TYPES` / `LIFECYCLE_CONSTRAINTS` registries; new ID prefixes.
  Lead-scoring catalog untouched. (These rows relocate into
  `schemes/lifecycle/` during `LTV-M2`.)
  - Labels: `type: feature`, `layer: schema`
- [~] **`LTV-Pc`** — `feat(schema): pLTV feature spec + regression task specs`.
  **Feature-catalog half discharged in `LTV-Pl` (#119):**
  `CUSTOMER_SNAPSHOT_FEATURES` (three `ltv_revenue_{90,365,730}d` targets, the
  secondary `churned_within_180d`, the `mrr_change_full_period` trap) is
  authored in `schemes/lifecycle/features.py` (post-reorg home, per the
  `LTV-M2` note above) because the snapshot builder needs it.  **Remaining
  scope (folds into `LTV-Pn`):** regression task specs + a `task_type`
  (`regression` | `classification`) on the task model — they belong with the
  task-split writer's continuous-target path.
  - Tests: feature-spec invariants ✓ (#119); regression task-spec shape → `LTV-Pn`.
  - Labels: `type: feature`, `layer: schema`

---

## `LTV-M2` — Generation-scheme architecture + physical reorg

> The foundational refactor. Extract the abstraction against the shipped
> lead-scoring path, then move both schemes into `leadforge/schemes/`. Each PR
> keeps lead-scoring output byte-identical (hash-determinism) and the public
> API stable.

- [x] **`LTV-Pd`** — `refactor(api): GenerationScheme protocol + registry`
  (**PR #107**). Added `schemes/base.py` (`GenerationScheme` protocol +
  `SCHEME_REGISTRY`) and `schemes/lead_scoring/` wrapping the existing pipeline
  *in place*; `Generator.generate()` routes through the registry; `Recipe` and
  `WorldSpec` gain a `scheme` field (default `lead_scoring`). Verified
  byte-identical (all 14 files of a pinned-timestamp bundle hash identically,
  main vs branch).
  - Labels: `type: refactor`, `layer: api`, `layer: core`
- [x] **`LTV-Pe`** — `refactor(render): scheme owns bundle rendering` (**PR #108**). Complete
  the **second half** of the seam against the known-good lead-scoring path:
  add `write_bundle` to the `GenerationScheme` protocol; move the
  `api/bundle.py` orchestration body into `LeadScoringScheme.write_bundle`
  (reusing the already-modular shared helpers — `build_manifest`,
  `apply_exposure`, `get_filter`); `api/bundle.py::write_bundle` becomes a thin
  dispatcher on `bundle.spec.scheme`, so `WorldBundle.save()` delegates to the
  producing scheme. Also harden scheme registration so resolution no longer
  depends on import order (the side-effect-registration footgun). Verified
  byte-identical. **Sequenced before the physical move** so the file move
  relocates a *complete* (both-halves) scheme and `bundle.py`'s call sites
  change only once. (Reorder rationale: the render path is where schemes
  diverge most; design it against lead-scoring with byte-identity as the oracle
  before building lifecycle.)
  - Tests: render dispatch, determinism through `save()`, unknown-scheme on
    `save`, base-direct resolution (footgun guard), full suite green.
  - Labels: `type: refactor`, `layer: render`, `layer: api`
- [ ] **`LTV-Pf`** — `refactor: move lead-scoring pipeline to schemes/lead_scoring/`.
  Physically relocate the (now fully scheme-owned) lead-scoring modules under
  `schemes/lead_scoring/`; leave shared primitives in `schema/` and the
  `render/` envelope. **Hard break, no shims** (decision D12): old internal
  import paths are removed and all in-repo callers updated; the
  `leadforge-datasets-private` build scripts must update in lockstep (tracked
  via a breakage issue there). Public API (`leadforge.api`, CLI) unchanged;
  package stays `1.x` with a CHANGELOG "Moved" note. Split into two PRs to keep
  each reviewable and byte-identical:
  - [x] **`LTV-Pf.1`** — compute core: `simulation/` + `mechanisms/` +
    `structure/` moved as whole directories (21 file renames, all callers
    rewritten). Verified byte-identical; full suite green. (**PR #109**)
  - [x] **`LTV-Pf.2`** — render: relocated `render/{snapshots,relational_snapshot_safe,tasks}`
    under `schemes/lead_scoring/render/`, and split `render/relational.py` so the
    shared `write_relational_tables` stays in the envelope while the 9-table
    `to_dataframes` moved. Verified byte-identical; full suite green. (**PR #110**)
    (The lead-scoring `schema` specs split lands with `LTV-Pg`.)
  - Tests: full suite + hash-determinism green; public API imports unchanged.
  - Labels: `type: refactor`, `layer: schema`, `layer: simulation`, `layer: render`
- [ ] **`LTV-Pg`** — `refactor: scaffold schemes/lifecycle/ + split lead-scoring schema`.
  Split into two PRs to keep each tractable:
  - [x] **`LTV-Pg.1`** — scaffold `schemes/lifecycle/`: moved the lifecycle
    entity rows + `LIFECYCLE_ROW_TYPES`/`LIFECYCLE_TABLE_NAMES` (from #104) into
    `schemes/lifecycle/entities.py` and `LIFECYCLE_CONSTRAINTS` into
    `schemes/lifecycle/relationships.py`; registered a stub `LifecycleScheme`
    (`build_world`/`write_bundle` raise `NotImplementedError` until M3–M6).
    Shared primitives (`EntityRowProtocol`, `_empty_df`, `AccountRow`,
    `FKConstraint`) stay in `schema/` and are imported. Byte-identical;
    full suite green. (**PR #111**)
  - [x] **`LTV-Pg.2`** — split the **lead-scoring** schema (**PR #112**): move the
    lead-scoring entity rows / `ALL_ROW_TYPES` / `ALL_CONSTRAINTS` /
    `LEAD_SNAPSHOT_FEATURES` / task specs into `schemes/lead_scoring/`, leaving
    only genuinely shared primitives in `schema/`. (The lifecycle `LTV-Pc`
    feature/task specs are authored directly in `schemes/lifecycle/` when M1's
    `LTV-Pc` lands.)
  - Tests: lifecycle registry imports from new home; lead-scoring unaffected.
  - Labels: `type: refactor`, `layer: schema`

---

## `LTV-M3` — Customer population + lifecycle world

> Built directly under `schemes/lifecycle/`.

- [x] **`LTV-Ph`** — `feat(lifecycle): customer population builder` (**PR #113**). Customer
  entities, 5 new latent traits, **staggered start dates** ending at the
  absolute `observation_date` (D4); seam for future chained generation (D3).
  - Tests: determinism, latent distributions, staggered-start spread, FK
    integrity, acquisition-window boundary.
  - Labels: `type: feature`, `layer: simulation`
- [x] **`LTV-Pi`** — `feat(lifecycle): motif families + mechanism policies` (**PR #116**). 5
  retention motif families; `assign_lifecycle_mechanisms()` mapping motif →
  churn/expansion/payment params.
  - Tests: per-motif param tables, dispatch, determinism.
  - Labels: `type: feature`, `layer: mechanisms`

---

## `LTV-M4` — Lifecycle simulation engine

- [x] **`LTV-Pj`** — `feat(lifecycle): churn / expansion / payment hazards` (**PR #117**).
  Weibull churn hazard with renewal-date spike, expansion propensity (the
  heavy-tail generator for pLTV), payment failure + dunning.
  - Tests: hazard shape over tenure, renewal spike, dunning escalation,
    expansion MRR-delta bounds.
  - Labels: `type: feature`, `layer: mechanisms`
- [ ] **`LTV-Pk`** — `feat(lifecycle): weekly simulation engine` (**PR #118**).
  `simulate_lifecycle()`: weekly loop per customer through `observation_date +
  730d (+ early-regime buffer)` so all three windows are fully simulated (D6);
  emits `subscription_events`, `health_signals`, `invoices`; updates terminal
  state.
  - Tests: determinism, churn-rate bounds per difficulty, still-active fraction,
    weekly health cadence, monthly invoice cadence, full-window coverage.
  - Labels: `type: feature`, `layer: simulation`

---

## `LTV-M5` — Customer snapshots + pLTV targets (both regimes)

- [x] **`LTV-Pl`** — `feat(lifecycle): calendar-anchored customer snapshot`
  (**PR #119**). `schemes/lifecycle/snapshots.py`:
  `build_customer_snapshot(cutoff=…)` — one row per active-at-cutoff customer;
  at-cutoff subscription state reconstructed from the event chain (not the
  terminal row); last-12-week health aggregates + whole-history `last_nps_score`;
  `mrr_change_at_snapshot` (valid) + `mrr_change_full_period` (trap, all modes,
  distortion-exempt); `ltv_revenue_{90,365,730}d` (gross = paid + recovered
  invoices, attributed by issuance date) + `churned_within_180d`.
  `CUSTOMER_SNAPSHOT_FEATURES` catalog in `schemes/lifecycle/features.py`
  (discharges the `LTV-Pc` catalog half).  Difficulty distortions extracted to
  the scheme-agnostic `render/distortions.py` (lead-scoring delegates;
  verified byte-identical).  **Deliberately omitted from the catalog:**
  `current_plan` (no plan-change mechanism → exact duplicate of
  `initial_plan`) and `downgrade_count` (no downgrade mechanism →
  zero-variance); re-add only with the mechanism.
  - Tests (43): censoring-based leakage probe (features identical when all
    post-cutoff events are deleted); target derivation vs the invoice table;
    failed/written-off exclusion (D7); ZILN target shape; trap-divergence
    invariant; trap + targets exempt from distortion; weeks_to_next_renewal
    agrees with `is_renewal_week`.
  - Self-review hardening: `LifecycleSimulationResult` records its
    `forward_window_days`/`early_tenure_weeks` and the builder rejects sims
    whose horizon cannot cover the 730d/180d target windows (silent-censoring
    guard); anniversary boundary single-sourced via public
    `hazards.next_renewal_week`; population/sim mismatch raises a real error.
  - **Deferred to `LTV-Pn` (difficulty wiring):** the design.md §7 secondary
    advanced-tier trap `last_health_signal_post_obs` — it is tier-conditional,
    so it belongs with the difficulty-profile plumbing, not the builder.
  - **Deferred to `LTV-Pn` (bundle writer):** an opt-in dtype-preserving
    missingness mode for `render/distortions.py` (`pd.NA` into nullable
    `Int64` instead of the Float64 conversion) — the lead-scoring default is
    byte-identity-locked, but the lifecycle scheme has no shipped bundles yet
    and can pick the cleaner semantics when its parquet schemas are fixed
    (Copilot review suggestion on #119).
  - Labels: `type: feature`, `layer: render`
- [ ] **`LTV-Pm`** — `feat(lifecycle): early-pLTV (tenure-anchored) task family`.
  Reuse the snapshot builder with a per-customer relative cutoff
  (`customer_start + early_tenure_weeks`) to emit the cold-start snapshot +
  recomputed targets (D8); separate task directory.
  - Tests: per-customer cutoff correctness, short-tenure sparsity, target parity,
    no post-cutoff leakage.
  - Labels: `type: feature`, `layer: render`

---

## `LTV-M6` — Register LifecycleScheme + recipe + manifest/version

- [ ] **`LTV-Pn`** — `feat(lifecycle): complete LifecycleScheme + manifest/version`.
  Fill in the `LifecycleScheme` pipeline methods (population→sim→render→tasks);
  add `n_customers` + lifecycle config (windows, early-tenure, observation
  anchor) to `GenerationConfig`; record `generation_scheme` + `observation_date`
  + windows in the manifest; bump `BUNDLE_SCHEMA_VERSION` 5 → 6 (D5); teach the
  task-split writer the continuous-target path. Extend `CLAUDE.md` hard
  constraints with the lifecycle snapshot-safety clause + the schemes/ layout.
  - **Layering cleanup (carried debt, see `Known deferred cleanups` below):**
    generalise `build_manifest` (drop the lead-scoring `world_graph` param) and
    `apply_exposure` (stop hard-coding the lead-scoring hidden graph + latent
    registry) so they are scheme-agnostic; with that done, remove the
    `core.models` / `render.relational` **TYPE_CHECKING** back-references to
    `leadforge.schemes.lead_scoring.*` introduced in `LTV-Pf.1` (a core→scheme
    layering inversion), and lift the shared render orchestration out of each
    scheme's `write_bundle` (the decomposition deferred in `LTV-Pe`).
  - Tests: dispatch, lead-scoring path unaffected, manifest fields, regression
    split writer, exposure filtering for new tables.
  - Labels: `type: feature`, `layer: api`, `layer: render`
- [ ] **`LTV-Po`** — `feat(recipes): b2b_saas_ltv_v1 recipe assets`. The three
  recipe YAMLs (`scheme: lifecycle`); register in the recipe registry;
  end-to-end `Generator.from_recipe("b2b_saas_ltv_v1").generate()` smoke test.
  - Tests: recipe loads, full round-trip, determinism, all task splits (3
    windows × 2 regimes + secondary churn), public/instructor split.
  - Labels: `type: feature`, `layer: recipes`

---

## `LTV-M7` — Validation + regression-metric calibration

- [ ] **`LTV-Pp`** — `feat(validation): lifecycle leakage probes + pLTV metric bands`.
  Scheme-aware leakage probes (cutoff window check; banned terminal
  columns/tables; banned forward-window target columns); regression evaluation
  (Spearman, normalized Gini, decile calibration, total-pred-vs-actual, value
  capture) + per-tier × per-window bands; trap-invariant guard; cross-seed
  drift; lifecycle dataset-card renderer.
  - Tests: probe coverage, regression bands, cross-seed stability.
  - Labels: `type: feature`, `layer: validation`

---

## `LTV-M8` — CLI, notebooks, publish

- [ ] **`LTV-Pq`** — `feat(cli): lifecycle generate flags + scheme-aware inspect`.
  `--n-customers`, observation/early-tenure flags; `inspect` dispatches on the
  bundle's `generation_scheme`.
  - Labels: `type: feature`, `layer: cli`
- [ ] **`LTV-Pr`** — `docs(notebooks): pLTV teaching sequence`. ZILN-vs-MSE
  baseline; discrimination/calibration metrics; the `mrr_change_full_period`
  leakage demo; early/cold-start pLTV; value-aware ranking; right-censoring note.
  - Labels: `type: docs`, `layer: render`
- [ ] **`LTV-Ps`** — `feat(release): package + publish b2b_saas_ltv_v1`. Kaggle
  + HF packaging (reuse Phase-5 packagers, scheme-aware), LLM critique, dataset
  card, release notes, tag. Publishes under the live `leadforge` Kaggle org.
  - Labels: `type: feature`, `layer: validation`

---

## Known deferred cleanups (tech debt carried by M2, paid down in M6)

The peer-schemes reorg deliberately defers a few cleanups to keep each M2 PR
byte-identical and reviewable. They are tracked here and discharged in
**`LTV-Pn`** (M6), where the manifest/exposure generalization makes them clean:

1. **Shared render orchestration** — `LTV-Pe` left each scheme owning its full
   `write_bundle`; only `write_relational_tables` is shared. A shared bundle
   orchestrator with scheme render hooks lands once there are two schemes.
2. **`build_manifest` / `apply_exposure` are lead-scoring-coupled** —
   `build_manifest` takes a `world_graph`; `apply_exposure` writes the
   lead-scoring hidden graph + latent registry. Generalize both to be
   scheme-agnostic.
3. **core→scheme layering inversion** — `LTV-Pf.1` introduced
   `TYPE_CHECKING`-only imports of `leadforge.schemes.lead_scoring.*` in
   `core.models` (`WorldBundle.world_graph: WorldGraph | None`) and
   `render.relational`. Harmless at runtime (no eager import), but `core`/shared
   `render` should not reference a scheme. Remove once (2) makes
   `WorldBundle` hold scheme-agnostic artifacts.

---

## Dependencies

```
LTV-M0 (plan)
  └─ LTV-M1 (lifecycle schema)
       └─ LTV-M2 (scheme abstraction + physical reorg)   ← refactor vs known-good path
            └─ LTV-M3 (population + motifs, in schemes/lifecycle/)
                 └─ LTV-M4 (engine)
                      └─ LTV-M5 (snapshots + pLTV targets, both regimes)
                           └─ LTV-M6 (register scheme + recipe + manifest/v6)  ← first e2e bundle
                                └─ LTV-M7 (validation)
                                     └─ LTV-M8 (publish)
```

`LTV-M2` can begin in parallel with `LTV-M1` finishing — it only touches the
existing lead-scoring path. `LTV-M6` is the first point where `leadforge
generate --recipe b2b_saas_ltv_v1` produces a bundle end-to-end.

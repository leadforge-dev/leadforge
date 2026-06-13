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
| `LTV-M5` | Customer snapshots + pLTV targets (both regimes) | `LTV-Pl`, `LTV-Pm` | #119 (Pl), #120 (Pm) |
| `LTV-M6` | Register LifecycleScheme + recipe + manifest/version | `LTV-Pn.1…4`, `LTV-Po` | #121 (Pn.1), #122 (Pn.2), #124 (Pn.3) |
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
- [x] **`LTV-Pc`** — `feat(schema): pLTV feature spec + regression task specs`.
  **Feature-catalog half discharged in `LTV-Pl` (#119):**
  `CUSTOMER_SNAPSHOT_FEATURES` (three `ltv_revenue_{90,365,730}d` targets, the
  secondary `churned_within_180d`, the `mrr_change_full_period` trap) is
  authored in `schemes/lifecycle/features.py` (post-reorg home, per the
  `LTV-M2` note above) because the snapshot builder needs it.  **Remaining
  scope (folds into `LTV-Pn`):** regression task specs + a `task_type`
  (`regression` | `classification`) on the task model — they belong with the
  task-split writer's continuous-target path.
  - Tests: feature-spec invariants ✓ (#119); regression task-spec shape ✓
    (#124, `LTV-Pn.3`).  **`LTV-Pc` fully discharged.**
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
- [x] **`LTV-Pm`** — `feat(lifecycle): early-pLTV (tenure-anchored) snapshot`
  (**PR #120**). `build_early_pltv_snapshot(early_tenure_weeks=…)` in
  `schemes/lifecycle/snapshots.py`: per-customer relative cutoff at
  `customer_start + early_tenure_weeks` (D8).  The calendar and early builders
  now share one per-customer-cutoff core (`_assemble_snapshot` + cutoff-map
  aggregation helpers), so feature derivations, the trap, target attribution,
  and distortions are defined once; the calendar regime's output is unchanged
  (LTV-Pl tests pass as-is).  Eligibility = survival to the anchor (drops
  onboarding churners, keeps late starters / post-anchor churners); forward
  windows are fully simulated relative to each customer's own start, so the
  anchor may legitimately land after `observation_date`.
  - Tests (19): tenure constant at the anchor; eligibility = survival to
    anchor; cohort difference vs calendar (post-anchor pre-obs churners);
    per-customer censoring leakage probe; targets recomputed per-customer
    cutoff vs the invoice table; cold-start sparsity (NPS all-null at 4w);
    anchor-validation (`>= 1`, `<= sim.early_tenure_weeks`), short-window /
    mismatch / missing-obs guards; distortions leave targets + trap intact.
  - **Known degenerate columns at a short anchor (deferred to `LTV-Pp`
    validation):** by cadence math, several catalog columns are structurally
    dead in the early table — `tenure_weeks` (constant = anchor),
    `renewal_count` (0 for anchor < 52w), `last_nps_score` (all-null for
    anchor < 13w), and near-degenerate `weeks_since_last_payment_failure`.
    The catalog is shared with the calendar regime by design, so the
    no-zero-variance / no-all-null checks must exempt these for the early task
    family; whether to drop them from the early feature set instead is open for
    `LTV-Pn`.
  - **Deferred to `LTV-Pn` (bundle/task writer):** the actual early-pLTV
    *task directory* + train/valid/test split export (`render/tasks.py`,
    design.md §536) — this PR delivers the snapshot + recomputed targets only,
    matching how `LTV-Pl` deferred the calendar task-split writer.
  - Labels: `type: feature`, `layer: render`

---

## `LTV-M6` — Register LifecycleScheme + recipe + manifest/version

`LTV-Pn` was too large for one reviewable, byte-identity-guarded PR (envelope
generalization + 3 carried cleanups + config + task model + the lifecycle
pipeline + schema bump).  Split into four sub-PRs in dependency order:

- [x] **`LTV-Pn.1`** — `refactor(render): scheme-agnostic build_manifest +
  schema v6` (**PR #121**).  `build_manifest` no longer takes the lead-scoring
  `world_graph`: it takes `generation_scheme: str`, `motif_family: str | None`,
  and an `extra_fields` mapping for scheme-specific keys.  Every manifest now
  records `generation_scheme`; `BUNDLE_SCHEMA_VERSION` bumped 5 → 6.  Removes
  the `manifests.py` → `lead_scoring.structure.graph` TYPE_CHECKING back-ref
  (part of cleanup #3).  **Lead-scoring data files byte-identical** (tables/,
  tasks/); only `manifest.json` changes (new field + version).  Schema
  contract test renamed v5 → v6.
  - Labels: `type: refactor`, `layer: render`
- [x] **`LTV-Pn.2`** — `refactor: scheme-agnostic WorldBundle + exposure hook`
  (**PR #122**).  `WorldBundle` now holds only `spec` + an opaque
  `artifacts: Any` (scheme-owned; lead-scoring stores `LeadScoringArtifacts`),
  finishing cleanup #3 — the `core.models` lead-scoring type imports are gone.
  `apply_exposure` is scheme-agnostic: it writes the generic `world_spec.json`
  and dispatches hidden-truth files to the producing scheme's new
  `GenerationScheme.write_metadata` hook (cleanup #2); the lead-scoring graph /
  latent registry / mechanism summary writers moved out of `exposure/` into the
  lead-scoring scheme.  Lead-scoring bundle **byte-identical** across both
  exposure modes (full SHA-256 harness).
  - **Re-scoped:** the shared bundle orchestrator (cleanup #1) moves to
    `LTV-Pn.4` — per this file's own note it is best designed *with the second
    scheme's `write_bundle` in hand*; building it now against one scheme would
    guess the hook shape.
  - Labels: `type: refactor`, `layer: api`, `layer: core`, `layer: render`
- [x] **`LTV-Pn.3`** — `feat: lifecycle config + regression task model`
  (**PR #124**).  `GenerationConfig` gains validated lifecycle fields
  (`n_customers`, `forward_windows_days`, `early_tenure_weeks`,
  `observation_date`).  `TaskManifest` gains a validated `task_type`
  (`VALID_TASK_TYPES = {binary_classification, regression}`) and target-agnostic
  docs.  The deterministic split writer is lifted to the shared envelope
  (`leadforge/render/tasks.py`, byte-identical; lead-scoring delegates) so it
  serves continuous pLTV targets.  `schemes/lifecycle/tasks.py` defines the
  per-regime task families (3 `pltv_revenue_*` regression + `churned_within_180d`
  classification, `early_`-prefixed for the tenure regime) — **completing the
  `LTV-Pc` regression-task-spec deferral**.  Data definitions only; wiring is
  Pn.4.  Lead-scoring data byte-identical (only `world_spec.json` gains the new
  config fields, by design).
  - Labels: `type: feature`, `layer: api`, `layer: schema`, `layer: render`
- [ ] **`LTV-Pn.4`** — `feat(lifecycle): complete LifecycleScheme + e2e bundle`.
  Implement `LifecycleScheme.build_world` (population → sim) and `write_bundle`
  (lifecycle relational tables; both regime snapshots → two task families ×
  3 windows + secondary churn; dataset card; manifest `observation_date` +
  windows via `extra_fields`; lifecycle `write_metadata` hidden-truth hook).
  With both schemes' `write_bundle` in hand, **lift the shared bundle
  orchestrator with scheme render hooks** out of the two implementations
  (carried cleanup #1).  First end-to-end lifecycle bundle (programmatic;
  recipe wiring is `LTV-Po`).  Extend `CLAUDE.md` hard constraints with the
  lifecycle snapshot-safety clause + the `schemes/` layout.  Carries the
  LTV-Pp validation flags: early-regime degenerate-column exemptions; the
  dtype-preserving missingness opt-in.
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
**`LTV-Pn.1`/`LTV-Pn.2`** (M6), where the manifest/exposure generalization makes them clean:

1. **Shared render orchestration** — `LTV-Pe` left each scheme owning its full
   `write_bundle`; only `write_relational_tables` is shared. A shared bundle
   orchestrator with scheme render hooks lands in **`LTV-Pn.4`**, once the
   lifecycle `write_bundle` exists to reveal the real shared shape.
2. ~~**`build_manifest` / `apply_exposure` are lead-scoring-coupled**~~ —
   **Done** (`build_manifest` in `LTV-Pn.1`; `apply_exposure` in `LTV-Pn.2` via
   the `write_metadata` scheme hook).
3. ~~**core→scheme layering inversion**~~ — **Done.** `LTV-Pn.1` removed the
   `render.manifests` back-ref; `LTV-Pn.2` removed the `core.models.WorldBundle`
   lead-scoring type imports (it now holds an opaque `artifacts: Any`).  Only a
   `DEFAULT_SCHEME = "lead_scoring"` string default and doc-comment
   cross-references remain — neither is an import/type inversion.

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

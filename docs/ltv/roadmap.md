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
| `LTV-M2` | Generation-scheme architecture + physical reorg | `LTV-Pd`, `LTV-Pe`, `LTV-Pf`, `LTV-Pg` | #107 (Pd) |
| `LTV-M3` | Customer population + lifecycle world | `LTV-Ph`, `LTV-Pi` | |
| `LTV-M4` | Lifecycle simulation engine | `LTV-Pj`, `LTV-Pk` | |
| `LTV-M5` | Customer snapshots + pLTV targets (both regimes) | `LTV-Pl`, `LTV-Pm` | |
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
- [ ] **`LTV-Pc`** — `feat(schema): pLTV feature spec + regression task specs`.
  `CUSTOMER_SNAPSHOT_FEATURES` (three `ltv_revenue_{90,365,730}d` targets, the
  secondary `churned_within_180d`, the `mrr_change_full_period` trap); regression
  task specs + a `task_type` (`regression` | `classification`) on the task model.
  - Tests: feature-spec invariants, regression task-spec shape.
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
- [ ] **`LTV-Pe`** — `refactor(render): scheme owns bundle rendering`. Complete
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
  Physically relocate the (now fully scheme-owned) lead-scoring population/
  engine/state/mechanisms/structure/snapshot/relational/task modules + its
  entity/feature/task specs under `schemes/lead_scoring/`; leave shared
  primitives in `schema/`, `render/` envelope, etc. Add back-compat import
  shims where `scripts/` or the sibling datasets repo reference internal paths.
  - Tests: full suite + hash-determinism green; public API imports unchanged;
    shim coverage.
  - Labels: `type: refactor`, `layer: schema`, `layer: simulation`, `layer: render`
- [ ] **`LTV-Pg`** — `refactor: scaffold schemes/lifecycle/ + relocate LTV-Pb/Pc specs`.
  Create `schemes/lifecycle/`; move the lifecycle entity rows (from #104) and
  the `LTV-Pc` feature/task specs into it; register a stub `LifecycleScheme`
  (pipeline methods raise `NotImplementedError` until M3–M6). Split any
  remaining shared schema primitives out cleanly.
  - Tests: lifecycle registry imports from new home; lead-scoring unaffected.
  - Labels: `type: refactor`, `layer: schema`

---

## `LTV-M3` — Customer population + lifecycle world

> Built directly under `schemes/lifecycle/`.

- [ ] **`LTV-Ph`** — `feat(lifecycle): customer population builder`. Customer
  entities, 5 new latent traits, **staggered start dates** ending at the
  absolute `observation_date` (D4); seam for future chained generation (D3).
  - Tests: determinism, latent distributions, staggered-start spread, FK
    integrity, acquisition-window boundary.
  - Labels: `type: feature`, `layer: simulation`
- [ ] **`LTV-Pi`** — `feat(lifecycle): motif families + mechanism policies`. 5
  retention motif families; `assign_lifecycle_mechanisms()` mapping motif →
  churn/expansion/payment params.
  - Tests: per-motif param tables, dispatch, determinism.
  - Labels: `type: feature`, `layer: mechanisms`

---

## `LTV-M4` — Lifecycle simulation engine

- [ ] **`LTV-Pj`** — `feat(lifecycle): churn / expansion / payment hazards`.
  Weibull churn hazard with renewal-date spike, expansion propensity (the
  heavy-tail generator for pLTV), payment failure + dunning.
  - Tests: hazard shape over tenure, renewal spike, dunning escalation,
    expansion MRR-delta bounds.
  - Labels: `type: feature`, `layer: mechanisms`
- [ ] **`LTV-Pk`** — `feat(lifecycle): weekly simulation engine`.
  `simulate_lifecycle()`: weekly loop per customer through `observation_date +
  730d (+ early-regime buffer)` so all three windows are fully simulated (D6);
  emits `subscription_events`, `health_signals`, `invoices`; updates terminal
  state.
  - Tests: determinism, churn-rate bounds per difficulty, still-active fraction,
    weekly health cadence, monthly invoice cadence, full-window coverage.
  - Labels: `type: feature`, `layer: simulation`

---

## `LTV-M5` — Customer snapshots + pLTV targets (both regimes)

- [ ] **`LTV-Pl`** — `feat(lifecycle): calendar-anchored customer snapshot`.
  `build_customer_snapshot(cutoff=observation_date)`: last-12-week health
  aggregates; `mrr_change_at_snapshot` (valid) + `mrr_change_full_period`
  (trap); the three `ltv_revenue_{90,365,730}d` gross-revenue targets +
  `churned_within_180d`; difficulty distortions.
  - Tests: no post-cutoff data in windowed columns; ZILN target shape; trap
    invariant; target derivation; trap exempt from distortion.
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

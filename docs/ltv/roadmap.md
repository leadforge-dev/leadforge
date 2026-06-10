# LTV Workstream Roadmap — `b2b_saas_ltv_v1`

> Milestone/PR breakdown for the customer-lifetime-value workstream. Design
> rationale lives in `design.md` (single source of truth). Update the
> checkboxes as work lands.

## Planning notation

Work items use a deliberate **`LTV-` prefix** scheme so they never collide
with GitHub PR numbers (`#NNN`):

- **Milestones** — `LTV-M1` … `LTV-M7`. A milestone is a coherent capability.
- **PRs** — `LTV-Pa`, `LTV-Pb`, … (sequential letters, globally unique across
  the whole roadmap). Each PR maps to exactly one milestone.

When a PR is opened on GitHub, reference its planning code in the title, e.g.
`feat(schema): lifecycle entity rows [LTV-Pa]`, and the GitHub number (`#NNN`)
is recorded back here on merge. The two namespaces stay distinct: `LTV-Pa`
(plan) ↔ `#123` (GitHub).

GitHub milestone: **`dataset: leadforge-ltv-v1`** — all LTV PRs assign to it.
Default labels per PR: a `type:` label, relevant `layer:` labels, and
`dataset: leadforge-ltv-v1`.

---

## Milestone overview

| Milestone | Capability | PRs | GitHub PRs |
|-----------|------------|-----|------------|
| `LTV-M0` | Planning + design lock | `LTV-Pa` | _this PR_ |
| `LTV-M1` | Schema foundation | `LTV-Pb`, `LTV-Pc` | |
| `LTV-M2` | Customer population + lifecycle world | `LTV-Pd`, `LTV-Pe` | |
| `LTV-M3` | Lifecycle simulation engine | `LTV-Pf`, `LTV-Pg` | |
| `LTV-M4` | Customer snapshot + leakage trap | `LTV-Ph` | |
| `LTV-M5` | Recipe wiring + framework dispatch | `LTV-Pi`, `LTV-Pj` | |
| `LTV-M6` | Validation + difficulty calibration | `LTV-Pk` | |
| `LTV-M7` | CLI, notebooks, publish | `LTV-Pl`, `LTV-Pm`, `LTV-Pn` | |

Total: ~13 PRs across 8 milestones (LTV-M0 = planning). Comparable in scope to
the original M4–M9 framework build.

---

## `LTV-M0` — Planning + design lock

- [ ] **`LTV-Pa`** — _this PR._ Land `docs/ltv/design.md` + `docs/ltv/roadmap.md`;
  create the `dataset: leadforge-ltv-v1` GitHub milestone + label; record the
  five locked design decisions (D1–D5). No package code.
  - Labels: `type: docs`
  - Deliverable: design doc, roadmap, milestone/label scaffolding.

---

## `LTV-M1` — Schema foundation

- [ ] **`LTV-Pb`** — `feat(schema): lifecycle entity rows`. Add
  `SubscriptionEventRow`, `HealthSignalRow`, `InvoiceRow` to `entities.py`;
  extend `CustomerRow` / `SubscriptionRow` with nullable lifecycle fields
  (lead-scoring output unchanged). Register in `ALL_ROW_TYPES`. Add FK
  constraints to `relationships.py`. Add ID prefixes (`subev_`, `hsig_`,
  `inv_`).
  - Tests: row round-trips, empty-dataframe dtypes, FK constraint registration,
    lead-scoring schema unaffected.
  - Labels: `type: feature`, `layer: schema`
- [ ] **`LTV-Pc`** — `feat(schema): customer snapshot feature spec + tasks`. Add
  `CUSTOMER_SNAPSHOT_FEATURES` to `features.py` (incl. the `mrr_change_full_period`
  trap with `leakage_risk=True`); add `CHURN_WITHIN_180_DAYS` and
  `LTV_BUCKET_6M` task specs to `tasks.py`.
  - Tests: feature-spec invariants (single target, trap flagged, no
    zero-variance by construction), task-spec shape.
  - Labels: `type: feature`, `layer: schema`

---

## `LTV-M2` — Customer population + lifecycle world

- [ ] **`LTV-Pd`** — `feat(simulation): customer population builder`.
  `build_customer_population()` in `customer_population.py`: customer entities,
  5 new latent traits, **staggered start dates** within an acquisition window
  ending at the absolute `observation_date` (D4). Keep a seam for future
  chained generation (D3). Reuse `RNGRoot` named-substream convention.
  - Tests: determinism under seed, latent distributions, staggered-start
    spread, FK integrity, acquisition-window boundary.
  - Labels: `type: feature`, `layer: simulation`
- [ ] **`LTV-Pe`** — `feat(mechanisms): lifecycle motif families + policies`. 5
  retention motif families with latent-mean biases; `assign_lifecycle_mechanisms()`
  policy mapping motif → churn/expansion/payment mechanism params.
  - Tests: per-motif param tables, policy dispatch, determinism.
  - Labels: `type: feature`, `layer: mechanisms`

---

## `LTV-M3` — Lifecycle simulation engine

- [ ] **`LTV-Pf`** — `feat(mechanisms): churn / expansion / payment hazards`.
  `lifecycle_hazards.py`: Weibull-shaped churn hazard with renewal-date spike,
  expansion propensity, payment-failure + dunning. Built on `LatentScore` +
  per-step Bernoulli.
  - Tests: hazard shape over tenure, renewal spike, dunning escalation,
    expansion MRR delta bounds.
  - Labels: `type: feature`, `layer: mechanisms`
- [ ] **`LTV-Pg`** — `feat(simulation): weekly lifecycle engine`.
  `simulate_lifecycle()` in `lifecycle.py`: weekly loop (D2) per customer from
  staggered start through `observation_date + 180d + buffer`; emits
  `subscription_events`, `health_signals`, `invoices`; updates
  customer/subscription terminal state. RNG substreams
  `lifecycle_transitions` / `lifecycle_events` / `lifecycle_post_sim`.
  - Tests: determinism, churn-rate bounds per difficulty, still-active
    fraction, weekly health-signal cadence, monthly invoice cadence,
    every customer simulated through the full label window.
  - Labels: `type: feature`, `layer: simulation`

---

## `LTV-M4` — Customer snapshot + leakage trap

- [ ] **`LTV-Ph`** — `feat(render): customer snapshot builder`.
  `build_customer_snapshot()` in `customer_snapshots.py`: **absolute
  `observation_date` cutoff** (not relative `snapshot_day`); aggregate health
  / events / invoices over last-12-weeks windows; compute
  `mrr_change_at_snapshot` (valid) and `mrr_change_full_period` (trap); derive
  `churned_within_180_days` and `ltv_bucket_6m`; difficulty distortions.
  - Tests: no post-`observation_date` data in windowed columns; trap-invariant
    (full ≠ snapshot for non-trivial fraction); label derivation; difficulty
    distortion exemption for the trap.
  - Labels: `type: feature`, `layer: render`

---

## `LTV-M5` — Recipe wiring + framework dispatch

- [ ] **`LTV-Pi`** — `feat(api,core): recipe_type dispatch + lifecycle config`.
  Add `n_customers` + lifecycle fields to `GenerationConfig`; parse
  `recipe_type` + `lifecycle:` section in `recipes.py`; dispatch the lifecycle
  path in `Generator.generate()`; bump `BUNDLE_SCHEMA_VERSION` 5 → 6 (D5);
  record `observation_date` in the manifest. Extend `CLAUDE.md` hard
  constraints with the lifecycle snapshot-safety clause.
  - Tests: config precedence, dispatch on recipe_type, lead-scoring path
    unaffected, manifest schema-version + observation_date, exposure filtering
    for new tables.
  - Labels: `type: feature`, `layer: api`, `layer: core`, `layer: render`
- [ ] **`LTV-Pj`** — `feat(recipes): b2b_saas_ltv_v1 recipe assets`. The three
  recipe YAMLs (recipe/narrative/difficulty_profiles); register in the recipe
  registry; end-to-end `Generator.from_recipe("b2b_saas_ltv_v1").generate()`
  smoke test producing a saved bundle.
  - Tests: recipe loads, full generation round-trip, determinism under same
    seed, both task splits written, public/instructor exposure split.
  - Labels: `type: feature`, `layer: recipes`

---

## `LTV-M6` — Validation + difficulty calibration

- [ ] **`LTV-Pk`** — `feat(validation): lifecycle leakage probes + realism bands`.
  Lifecycle leakage probes (absolute-cutoff window check; banned terminal
  columns/tables; deterministic reconstruction); difficulty calibration
  (churn-rate / expansion-rate / still-active bands); trap-invariant guard;
  cross-seed drift. Dataset-card renderer for the lifecycle narrative.
  - Tests: probe coverage, band checks per tier, cross-seed stability.
  - Labels: `type: feature`, `layer: validation`

---

## `LTV-M7` — CLI, notebooks, publish

- [ ] **`LTV-Pl`** — `feat(cli): lifecycle generate flags + inspect surfacing`.
  `--n-customers`, `--observation-date` (or derived) flags; `inspect` surfaces
  lifecycle manifest fields (observation_date, task inventory).
  - Labels: `type: feature`, `layer: cli`
- [ ] **`LTV-Pm`** — `docs(notebooks): survival + churn teaching sequence`.
  Notebooks: churn baseline, survival-analysis intro (right-censoring on the
  relational tables), the `mrr_change_full_period` leakage demo, value-aware
  LTV ranking.
  - Labels: `type: docs`, `layer: render`
- [ ] **`LTV-Pn`** — `feat(release): package + publish b2b_saas_ltv_v1`. Kaggle
  + HF packaging (reuse Phase-5 packagers), LLM critique run, dataset card,
  release notes, tag. Publishes under the now-live `leadforge` Kaggle org.
  - Labels: `type: feature`, `layer: validation`

---

## Dependencies

```
LTV-M0 (plan)
  └─ LTV-M1 (schema)
       └─ LTV-M2 (population + motifs)
            └─ LTV-M3 (engine)
                 └─ LTV-M4 (snapshot)
                      └─ LTV-M5 (wiring + recipe)  ← first end-to-end bundle
                           └─ LTV-M6 (validation)
                                └─ LTV-M7 (publish)
```

`LTV-M5` is the first point where `leadforge generate --recipe
b2b_saas_ltv_v1` produces a bundle end-to-end. Everything before it is
bottom-up framework construction; everything after is quality + delivery.

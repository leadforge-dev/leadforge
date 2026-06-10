# LTV Workstream Roadmap — `b2b_saas_ltv_v1`

> Milestone/PR breakdown for the predictive-lifetime-value (pLTV) workstream.
> Design rationale lives in `design.md` (single source of truth). Update the
> checkboxes as work lands.

## Planning notation

Work items use a deliberate **`LTV-` prefix** scheme so they never collide
with GitHub PR numbers (`#NNN`):

- **Milestones** — `LTV-M0` … `LTV-M7`. A milestone is a coherent capability.
- **PRs** — `LTV-Pa`, `LTV-Pb`, … (sequential letters, globally unique across
  the whole roadmap). Each PR maps to exactly one milestone.

When a PR is opened on GitHub, reference its planning code in the title, e.g.
`feat(schema): lifecycle entity rows [LTV-Pb]`, and the GitHub number (`#NNN`)
is recorded back here on merge. The two namespaces stay distinct: `LTV-Pb`
(plan) ↔ `#123` (GitHub).

GitHub milestone: **`dataset: leadforge-ltv-v1`** (#8) — all LTV PRs assign to
it. Default labels per PR: a `type:` label, relevant `layer:` labels, and
`dataset: leadforge-ltv-v1`.

---

## Milestone overview

| Milestone | Capability | PRs | GitHub PRs |
|-----------|------------|-----|------------|
| `LTV-M0` | Planning + design lock | `LTV-Pa` | #102 (+ pLTV reframe) |
| `LTV-M1` | Schema foundation | `LTV-Pb`, `LTV-Pc` | |
| `LTV-M2` | Customer population + lifecycle world | `LTV-Pd`, `LTV-Pe` | |
| `LTV-M3` | Lifecycle simulation engine | `LTV-Pf`, `LTV-Pg` | |
| `LTV-M4` | Customer snapshots + pLTV targets (both regimes) | `LTV-Ph`, `LTV-Pi` | |
| `LTV-M5` | Recipe wiring + framework dispatch | `LTV-Pj`, `LTV-Pk` | |
| `LTV-M6` | Validation + regression-metric calibration | `LTV-Pl` | |
| `LTV-M7` | CLI, notebooks, publish | `LTV-Pm`, `LTV-Pn`, `LTV-Po` | |

Total: ~15 PRs across 8 milestones (LTV-M0 = planning).

---

## `LTV-M0` — Planning + design lock

- [x] **`LTV-Pa`** — planning. Land `docs/ltv/design.md` + `docs/ltv/roadmap.md`;
  create the `dataset: leadforge-ltv-v1` GitHub milestone + label; record the
  locked design decisions. **Merged as #102.** A follow-up docs PR corrected
  the target framing from churn classification to **pLTV regression** (ZILN;
  multiple windows; gross revenue; first-class early-pLTV variant; churn kept
  secondary) — see `design.md` §2.2. No package code.
  - Labels: `type: docs`

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
- [ ] **`LTV-Pc`** — `feat(schema): pLTV feature spec + regression task specs`.
  Add `CUSTOMER_SNAPSHOT_FEATURES` to `features.py` — including the three
  continuous targets (`ltv_revenue_{90,365,730}d`), the secondary
  `churned_within_180d`, and the `mrr_change_full_period` trap
  (`leakage_risk=True`). Add **regression** task specs
  (`LTV_REVENUE_{90,365,730}D`) + the secondary `CHURN_WITHIN_180D` to
  `tasks.py`; extend the task-spec model to carry `task_type`
  (`regression` | `classification`).
  - Tests: feature-spec invariants (multiple targets allowed, trap flagged,
    no zero-variance by construction), regression task-spec shape.
  - Labels: `type: feature`, `layer: schema`

---

## `LTV-M2` — Customer population + lifecycle world

- [ ] **`LTV-Pd`** — `feat(simulation): customer population builder`.
  `build_customer_population()` in `customer_population.py`: customer entities,
  5 new latent traits, **staggered start dates** within an acquisition window
  ending at the absolute `observation_date` (D4). Keep a seam for future
  chained generation (D3). Reuse the `RNGRoot` named-substream convention.
  - Tests: determinism under seed, latent distributions, staggered-start
    spread, FK integrity, acquisition-window boundary.
  - Labels: `type: feature`, `layer: simulation`
- [ ] **`LTV-Pe`** — `feat(mechanisms): lifecycle motif families + policies`. 5
  retention motif families with latent-mean biases; `assign_lifecycle_mechanisms()`
  mapping motif → churn/expansion/payment params.
  - Tests: per-motif param tables, policy dispatch, determinism.
  - Labels: `type: feature`, `layer: mechanisms`

---

## `LTV-M3` — Lifecycle simulation engine

- [ ] **`LTV-Pf`** — `feat(mechanisms): churn / expansion / payment hazards`.
  `lifecycle_hazards.py`: Weibull-shaped churn hazard with renewal-date spike,
  expansion propensity (the heavy-tail generator for pLTV), payment failure +
  dunning. Built on `LatentScore` + per-step Bernoulli.
  - Tests: hazard shape over tenure, renewal spike, dunning escalation,
    expansion MRR-delta bounds.
  - Labels: `type: feature`, `layer: mechanisms`
- [ ] **`LTV-Pg`** — `feat(simulation): weekly lifecycle engine`.
  `simulate_lifecycle()` in `lifecycle.py`: weekly loop (D2) per customer from
  staggered start through `observation_date + 730d (+ buffer for the early
  regime)` so **all three target windows are fully simulated** (D6); emits
  `subscription_events`, `health_signals`, `invoices`; updates
  customer/subscription terminal state. RNG substreams
  `lifecycle_transitions` / `lifecycle_events` / `lifecycle_post_sim`.
  - Tests: determinism, churn-rate bounds per difficulty, still-active
    fraction, weekly health cadence, monthly invoice cadence, every customer
    simulated through the longest forward window.
  - Labels: `type: feature`, `layer: simulation`

---

## `LTV-M4` — Customer snapshots + pLTV targets (both regimes)

- [ ] **`LTV-Ph`** — `feat(render): calendar-anchored customer snapshot`.
  `build_customer_snapshot(cutoff=observation_date)` in `customer_snapshots.py`:
  **absolute `observation_date` cutoff**; aggregate health / events / invoices
  over last-12-weeks windows; compute `mrr_change_at_snapshot` (valid) and
  `mrr_change_full_period` (trap); compute the three forward-window gross-revenue
  targets `ltv_revenue_{90,365,730}d` (D6/D7) and the secondary
  `churned_within_180d`; difficulty distortions.
  - Tests: no post-cutoff data in windowed feature columns; ZILN target shape
    (positive zero-mass + heavy tail; mass grows with window); trap-invariant;
    label/target derivation; trap exempt from distortion.
  - Labels: `type: feature`, `layer: render`
- [ ] **`LTV-Pi`** — `feat(render): early-pLTV (tenure-anchored) task family`.
  Reuse `build_customer_snapshot` with a **per-customer relative cutoff**
  (`customer_start + early_tenure_weeks`, e.g. 4w) to emit the cold-start
  snapshot + the same three forward-window targets recomputed off that cutoff
  (D8). Exported under a separate task directory.
  - Tests: per-customer cutoff correctness, short-tenure feature sparsity,
    target recomputation parity, no post-cutoff leakage.
  - Labels: `type: feature`, `layer: render`

---

## `LTV-M5` — Recipe wiring + framework dispatch

- [ ] **`LTV-Pj`** — `feat(api,core,render): recipe_type dispatch + regression
  task splits`. Add `n_customers` + lifecycle config (windows, early-tenure,
  observation anchor) to `GenerationConfig`; parse `recipe_type` + `lifecycle:`
  in `recipes.py`; dispatch the lifecycle path in `Generator.generate()`; bump
  `BUNDLE_SCHEMA_VERSION` 5 → 6 (D5); record `observation_date` + windows in the
  manifest; teach the task-split writer a **continuous-target** path. Extend
  `CLAUDE.md` hard constraints with the lifecycle snapshot-safety clause.
  - Tests: config precedence, dispatch on recipe_type, lead-scoring path
    unaffected, manifest schema-version + observation_date, regression split
    writer, exposure filtering for new tables.
  - Labels: `type: feature`, `layer: api`, `layer: core`, `layer: render`
- [ ] **`LTV-Pk`** — `feat(recipes): b2b_saas_ltv_v1 recipe assets`. The three
  recipe YAMLs; register in the recipe registry; end-to-end
  `Generator.from_recipe("b2b_saas_ltv_v1").generate()` smoke test producing a
  saved bundle with both task families.
  - Tests: recipe loads, full generation round-trip, determinism under seed,
    all task splits written (3 windows × 2 regimes + secondary churn),
    public/instructor exposure split.
  - Labels: `type: feature`, `layer: recipes`

---

## `LTV-M6` — Validation + regression-metric calibration

- [ ] **`LTV-Pl`** — `feat(validation): lifecycle leakage probes + pLTV metric
  bands`. Lifecycle leakage probes (cutoff window check; banned terminal
  columns/tables; banned forward-window target columns in relational tables);
  **regression** evaluation (Spearman, normalized Gini, decile calibration,
  total-pred-vs-actual, value capture) and per-tier/per-window bands;
  trap-invariant guard; cross-seed drift. Dataset-card renderer for the
  lifecycle narrative.
  - Tests: probe coverage, regression metric bands per tier × window,
    cross-seed stability.
  - Labels: `type: feature`, `layer: validation`

---

## `LTV-M7` — CLI, notebooks, publish

- [ ] **`LTV-Pm`** — `feat(cli): lifecycle generate flags + inspect surfacing`.
  `--n-customers`, observation/early-tenure flags; `inspect` surfaces lifecycle
  manifest fields (observation_date, windows, task inventory).
  - Labels: `type: feature`, `layer: cli`
- [ ] **`LTV-Pn`** — `docs(notebooks): pLTV teaching sequence`. Notebooks:
  ZILN vs MSE regression baseline, discrimination/calibration metrics
  (Spearman / normalized Gini / decile charts), the `mrr_change_full_period`
  leakage demo, **early/cold-start pLTV** (predict long-horizon value from a
  short window), value-aware ranking, and a right-censoring note on total LTV.
  - Labels: `type: docs`, `layer: render`
- [ ] **`LTV-Po`** — `feat(release): package + publish b2b_saas_ltv_v1`. Kaggle
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
                 └─ LTV-M4 (snapshots + pLTV targets, both regimes)
                      └─ LTV-M5 (wiring + recipe)  ← first end-to-end bundle
                           └─ LTV-M6 (validation)
                                └─ LTV-M7 (publish)
```

`LTV-M5` is the first point where `leadforge generate --recipe
b2b_saas_ltv_v1` produces a bundle end-to-end. Everything before it is
bottom-up framework construction; everything after is quality + delivery.

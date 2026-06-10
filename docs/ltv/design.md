# LTV Dataset Design — `b2b_saas_ltv_v1`

> **Single source of truth** for the customer-lifetime-value workstream. The
> companion `roadmap.md` breaks this design into milestones and PRs. Update
> both as the workstream progresses.

**Status:** planning (pre-implementation)
**Owner:** @shaypalachy
**Vertical:** same Veridian Technologies procurement/AP SaaS world as
`b2b_saas_procurement_v1`, but **post-conversion**: subscriptions, renewals,
expansion, churn, and lifetime value.

---

## 1. Goal

Add a customer-lifecycle dataset family to leadforge that picks up where the
lead-scoring dataset leaves off. The lead-scoring recipe ends at `closed_won`;
the LTV recipe begins there and simulates the subscription lifecycle.

**Teaching goals (analogous to lead scoring):**

- Predict churn / lifetime value for active customers from a point-in-time
  snapshot.
- Teach **survival analysis** (right-censoring discipline), **cohort
  analysis**, and **expansion-revenue modelling**.
- Reproduce realistic confusions: leakage traps around future expansion
  events, right-censoring of still-active customers, and time-window
  discipline on health-signal features.

This is a **new recipe on the existing framework**, not a fork. The CLI,
bundle format, RNG system, exposure modes, manifest schema, and validation
harness are reused. A single dispatch hook in `Generator.generate()` selects
the lifecycle simulation path when the recipe declares `recipe_type:
lifecycle`.

---

## 2. Locked design decisions

These five questions were resolved by the maintainer on 2026-06-10. They are
load-bearing for everything below.

| # | Question | Decision |
|---|----------|----------|
| D1 | Primary task for the first shipped recipe | **`churn_within_180_days`** (binary). `ltv_bucket_6m` ships as a secondary task in the same bundle. |
| D2 | Simulation time resolution | **Weekly steps** — gives granular health-signal trend curves for teaching trend features. Billing/renewal events resolve to the enclosing week. |
| D3 | Independent vs chained generation | **Independent for v1.** The customer population is generated self-contained, not chained from a lead-scoring bundle. Optional chaining is designed-for-later but not built now. |
| D4 | Observation cohort | **Staggered start dates, fixed observation date.** Customers are acquired across an acquisition window; all are observed at one absolute calendar date, so tenure-at-observation varies (cold-start customers vs mature customers). |
| D5 | Bundle schema version | **Bump** `BUNDLE_SCHEMA_VERSION` (currently `5` → `6`). The table inventory and the customers/subscriptions semantics change substantially. |

### 2.1 Consequence of D4 — absolute observation anchor

This is the most important framework divergence. The lead-scoring path filters
events by a **per-entity relative** cutoff (`lead_created_at + snapshot_day`).
With staggered starts + a single fixed observation date (D4), the lifecycle
path filters by an **absolute calendar** cutoff (`observation_date`, the same
for every customer).

Therefore the lifecycle path does **not** reuse `snapshot_day`. It introduces
an `observation_date` concept derived deterministically from the world
calendar (acquisition-window end) and recorded in the manifest. The customer
snapshot builder is a **separate function** (`build_customer_snapshot`) so it
can apply absolute-cutoff filtering without touching the lead-scoring
`build_snapshot` path.

### 2.2 Consequence of D1 — right-censoring is a property, not a label problem

Because the primary task is `churn_within_180_days`, every customer gets a
**definite** binary label as long as the simulation runs through
`observation_date + 180 days` for all customers. A customer still active at
`observation_date + 180d` is a clean negative, not a censored row.

Right-censoring is still taught — but as a property of the **relational
tables** and the **secondary LTV task**: a customer still active at the end of
the simulation horizon is right-censored for *total* lifetime value. The
notebooks teach censoring discipline on the relational data and on
`ltv_bucket_6m`, while the headline binary task stays clean. This is a cleaner
split than making censoring a label-derivation hazard.

---

## 3. Entities and tables

### 3.1 New entity rows (in `leadforge/schema/entities.py`)

**`SubscriptionEventRow`** — the lifecycle backbone; one row per state change.

| column | dtype | notes |
|--------|-------|-------|
| `event_id` | string | opaque ID, `subev_000001` |
| `subscription_id` | string | FK → subscriptions |
| `customer_id` | string | FK → customers |
| `event_timestamp` | string | ISO-8601 (week-resolved) |
| `event_type` | string | `renewal` / `expansion` / `downgrade` / `churn` / `payment_failure` / `payment_recovered` |
| `mrr_before` | Int64 | MRR (USD) before the event |
| `mrr_after` | Int64 | MRR (USD) after; `0` on churn |
| `contract_term_months_new` | Int64 \| null | set on `renewal` only |

**`HealthSignalRow`** — weekly product-usage telemetry; the core predictive signal.

| column | dtype | notes |
|--------|-------|-------|
| `signal_id` | string | opaque ID |
| `customer_id` | string | FK → customers |
| `period_start` | string | ISO-8601 first day of the signal week |
| `active_users` | Int64 | weekly active users |
| `feature_depth_score` | Float64 | 0–1, latent-derived breadth-of-use |
| `support_tickets` | Int64 | tickets opened that week |
| `nps_score` | Int64 \| null | quarterly survey; null most weeks |

**`InvoiceRow`** — monthly billing; drives the payment-failure mechanism.

| column | dtype | notes |
|--------|-------|-------|
| `invoice_id` | string | opaque ID |
| `customer_id` | string | FK → customers |
| `invoice_date` | string | ISO-8601 |
| `amount_usd` | Int64 | |
| `payment_status` | string | `paid` / `failed` / `recovered` / `written_off` |

### 3.2 Extended existing entity rows

The current `CustomerRow` (4 fields) and `SubscriptionRow` (5 fields,
`subscription_status` hardcoded `"active"`) are shells. The lifecycle recipe
fills them out. **The lead-scoring recipe keeps the thin versions** — the new
fields are nullable/optional so the procurement recipe's output is unchanged.

`CustomerRow` gains: `initial_mrr`, `initial_plan`, `contract_term_months`,
`csm_rep_id`.

`SubscriptionRow` gains: `current_mrr`, `subscription_end_at`, `churn_at`,
`churn_reason`, `renewal_count`, `expansion_count`.

### 3.3 Public lifecycle table inventory

| table | public (`student_public`) | instructor (`research_instructor`) |
|-------|---------------------------|-------------------------------------|
| accounts | ✓ | ✓ |
| customers | ✓ (terminal fields redacted) | ✓ full |
| subscriptions | ✓ (terminal fields redacted) | ✓ full |
| subscription_events | ✓ (filtered to `<= observation_date`) | ✓ full horizon |
| health_signals | ✓ (filtered to `<= observation_date`) | ✓ full horizon |
| invoices | ✓ (filtered to `<= observation_date`) | ✓ full horizon |

Contacts/leads/touches/etc. from the lead-scoring world are **not** part of
the LTV bundle (independent generation, D3).

---

## 4. Snapshot-safety contract (lifecycle)

Analogous to the lead-scoring hard constraint in `CLAUDE.md`, but anchored on
the **absolute** `observation_date`:

- Every timestamp column in public event tables
  (`subscription_events.event_timestamp`, `health_signals.period_start`,
  `invoices.invoice_date`) must satisfy `<= observation_date`.
- No terminal/outcome fields in public `customers` / `subscriptions`:
  `churn_at`, `churn_reason`, `subscription_end_at`, and the derived
  `churned_within_180_days` label are banned from the public relational
  tables.
- `subscription_events` rows with `event_type == "churn"` after
  `observation_date` are excluded from public bundles (they encode the label).
- No flat snapshot feature may use events after `observation_date`, **except**
  the deliberately-retained leakage trap (§6).

This contract is enforced by lifecycle-specific leakage probes (see roadmap
`LTV-M6`) and recorded in the manifest. `CLAUDE.md`'s hard-constraints section
gains a lifecycle clause when the recipe wiring lands (`LTV-M5`).

---

## 5. Simulation mechanisms

Three new mechanism types, none of which exist today. All reuse the existing
`LatentScore` + per-step Bernoulli pattern from
`leadforge/mechanisms/hazards.py`.

**Churn hazard (post-conversion).** Weekly probability driven by health
signals + latent traits, structurally different from the flat pre-conversion
`_DAILY_CHURN_RATE`:

- Weibull-shaped over tenure: elevated in weeks 1–12 (onboarding instability),
  low in the steady middle.
- **Renewal-date spike**: at each contract anniversary the churn hazard is
  ~10× the background rate (discrete renewal decision).
- Drivers: `latent_product_fit` (background), `latent_champion_strength`
  (renewal-date), `feature_depth_score` (leading indicator),
  unrecovered payment failures (financial trigger).

**Expansion propensity.** Weekly probability of a plan upgrade / seat add:

- Drivers: `latent_adoption_velocity`, `feature_depth_score`, active-user
  growth, `employee_band` (expansion ceiling).
- Expansion MRR delta: `randint(0.25·mrr, 1.0·mrr)`.

**Payment failure.** Monthly billing event with a failure probability:

- Driver: `latent_budget_stability`.
- Dunning: 3 months of `failed` before escalation to `recovered` or
  `written_off`; unrecovered → forced churn.

### 5.1 Lifecycle motif families (new)

The 5 lead-scoring motifs describe *buying*; the LTV world needs *retention*
motifs:

| family | retention driver |
|--------|------------------|
| `product_led_retention` | `latent_product_fit` dominant; health signals strongly predictive |
| `relationship_led_retention` | `latent_champion_strength` dominant; health weaker |
| `expansion_led_growth` | low churn, high upsell; LTV variance from expansion |
| `payment_fragile` | financially-triggered churn; `latent_budget_stability` dominant |
| `churner_dominated` | high background churn; strong early-warning signals (teaching tier) |

### 5.2 New customer latent traits

`latent_product_fit`, `latent_adoption_velocity`, `latent_budget_stability`,
`latent_champion_strength`, `latent_organizational_stability`. Sampled from the
same clipped-Gaussian `_sample_latent` helper, with motif-family mean biases
mirroring `_MOTIF_LATENT_BIAS`.

---

## 6. Leakage trap

**Primary trap: `mrr_change_full_period` vs `mrr_change_at_snapshot`.**

- `mrr_change_at_snapshot` = `current_mrr − initial_mrr` measured at
  `observation_date`. **Valid.**
- `mrr_change_full_period` = MRR delta from start to **end of simulation**.
  **Leaks** — future expansions (which correlate with high LTV / low churn)
  inflate it. Retained in all modes (`leakage_risk=True,
  redact_in_modes=frozenset()`), documented in the feature dictionary and
  release notes, exactly like `total_touches_all`.

Why it's a good trap: both columns are "MRR delta", differing only by window;
standalone AUC is moderate (looks useful, not obviously inflated); tree models
extract more from it than LR (reproduces the NB03 lesson); removing it causes a
measurable-but-not-catastrophic drop.

**Secondary trap (advanced tier): `last_health_signal_post_obs`** — a
health-signal reading from *after* `observation_date`, named to look like a
current-state feature. More subtle because the column name doesn't reveal the
time shift.

A trap-invariant test (analogous to `test_windowed_bundle_trap.py`) asserts
`mrr_change_full_period` diverges from `mrr_change_at_snapshot` for a
non-trivial fraction of customers and never contradicts it in sign for
expansion-only customers.

---

## 7. Customer snapshot features (at `observation_date`)

Grouped like `LEAD_SNAPSHOT_FEATURES`. New constant
`CUSTOMER_SNAPSHOT_FEATURES` in `leadforge/schema/features.py`.

**Account** (from `AccountRow`): `industry`, `region`, `employee_band`,
`estimated_revenue_band`.

**Customer/subscription:** `tenure_weeks`, `initial_plan`, `current_plan`,
`initial_mrr`, `current_mrr`, `mrr_change_at_snapshot`, `renewal_count`,
`expansion_count`, `downgrade_count`, `contract_term_months`,
`weeks_to_next_renewal`.

**Health (aggregated over last 12 weeks before `observation_date`):**
`avg_active_users_l12w`, `active_user_trend_l12w` (slope),
`avg_feature_depth_l12w`, `support_ticket_count_l12w`, `last_nps_score`
(nullable).

**Financial:** `payment_failure_count`, `weeks_since_last_payment_failure`
(nullable).

**Leakage trap (all modes):** `mrr_change_full_period`.

**Target (primary, `churn_within_180_days`):** `churned_within_180_days`
(boolean) — True iff a `churn` event falls in
`[observation_date, observation_date + 180d]`.

**Target (secondary, `ltv_bucket_6m`):** quartile (`low`/`medium`/`high`/`top`)
of revenue collected (paid invoices) in the 6 months after `observation_date`.

---

## 8. Framework changes inventory

### New files

| file | purpose |
|------|---------|
| `leadforge/simulation/lifecycle.py` | `simulate_lifecycle()` — weekly-step subscription simulator |
| `leadforge/simulation/customer_population.py` | `build_customer_population()` — customer entities + latents + staggered starts |
| `leadforge/render/customer_snapshots.py` | `build_customer_snapshot()` — per-customer row at absolute `observation_date` |
| `leadforge/mechanisms/lifecycle_hazards.py` | churn hazard, expansion propensity, payment failure |
| `leadforge/recipes/b2b_saas_ltv_v1/{recipe,narrative,difficulty_profiles}.yaml` | new recipe |

### Modified files

| file | change |
|------|--------|
| `leadforge/schema/entities.py` | add 3 rows; extend `CustomerRow`/`SubscriptionRow` |
| `leadforge/schema/features.py` | add `CUSTOMER_SNAPSHOT_FEATURES` |
| `leadforge/schema/tasks.py` | add `CHURN_WITHIN_180_DAYS`, `LTV_BUCKET_6M` |
| `leadforge/schema/relationships.py` | FK constraints for new tables |
| `leadforge/core/models.py` | add `n_customers: int \| None`; lifecycle config fields |
| `leadforge/api/recipes.py` | parse `recipe_type` + `lifecycle:` section |
| `leadforge/api/generator.py` | dispatch on `recipe.recipe_type` |
| `leadforge/render/manifests.py` | record `observation_date`, bump schema version to 6 |
| `leadforge/validation/*` | lifecycle leakage probes + realism bands |
| `CLAUDE.md` | lifecycle snapshot-safety hard-constraint clause; reference docs |

### Unchanged (reused as-is)

CLI commands/flags, `WorldBundle`/`WorldSpec`, RNG architecture, exposure-mode
dispatch, `manifest.json` envelope (additive), determinism/monotonicity
invariants.

---

## 9. Difficulty profiles

| dimension | intro | intermediate | advanced |
|-----------|-------|--------------|----------|
| `signal_strength` | 0.90 | 0.70 | 0.50 |
| `noise_scale` | 0.10 | 0.30 | 0.55 |
| `missing_rate` | 0.02 | 0.08 | 0.18 |
| `annual_churn_rate_range` | [0.10, 0.20] | [0.20, 0.35] | [0.30, 0.50] |
| `expansion_rate_range` | [0.15, 0.30] | [0.10, 0.20] | [0.05, 0.15] |
| `still_active_fraction` (≈ right-censored for total LTV) | ~0.40 | ~0.60 | ~0.75 |
| secondary trap `last_health_signal_post_obs` | off | off | on |

Primary task is `churn_within_180_days` on all three tiers; difficulty is a
prevalence + noise + calibration axis (matching the v1 reframe — *not* a flat
AUC-vs-tier promise).

---

## 10. Open items deferred past planning

- **Chained generation** (D3 later): an interface to seed the customer
  population from a lead-scoring bundle's converted leads. Designed-for but not
  built; the customer-population builder will keep its acquisition logic behind
  a seam so a "from converted leads" source can be slotted in.
- **Continuous-time engine**: weekly steps are sufficient for v1; not coupled
  to this dataset.
- **LTV regression label** (vs bucket): bucket is the secondary v1 task; a
  continuous LTV regression target is a later addition.
- **CLAUDE.md hard-constraint edit**: the lifecycle snapshot-safety clause is
  added when `LTV-M5` wiring lands, not in the planning PR.

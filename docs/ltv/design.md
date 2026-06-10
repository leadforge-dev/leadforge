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

Add a **predictive-lifetime-value (pLTV) regression** dataset family to
leadforge that picks up where the lead-scoring dataset leaves off. The
lead-scoring recipe ends at `closed_won`; the LTV recipe begins there and
simulates the subscription lifecycle, then asks the canonical pLTV question:

> **Given what we know about a customer at an observation point, how much
> monetary value will they generate over the next N days?**

This is a **regression** task, not classification. It follows the framing in
Google's [`lifetime_value`](https://github.com/google/lifetime_value) (the
ZILN paper, [arXiv:1912.07753](https://arxiv.org/abs/1912.07753)) and the
commercial pLTV framing used by [Voyantis](https://www.voyantis.ai/):
continuous future-value prediction, optimised for value-based decisions
(acquisition, bidding, retention prioritisation), often under **early /
cold-start** conditions where a customer's true value isn't yet clear.

**Teaching goals:**

- Predict continuous future value for customers from a point-in-time snapshot —
  a target that is **zero-inflated and heavy-tailed** by construction.
- Teach the **ZILN** decomposition (`P(value>0) × E[value | value>0]`) and why
  MSE is the wrong loss for heavy-tailed value.
- Teach pLTV evaluation: **discrimination** (Spearman / normalized Gini /
  decile lift) and **calibration** (decile charts, total-predicted-vs-actual) —
  not AUC.
- Teach **early / cold-start prediction**: predict long-horizon value from a
  short observation window.
- Reproduce realistic confusions: leakage traps around future expansion
  events, right-censoring of still-active customers, and time-window discipline
  on health-signal features.

This is a **new recipe on the existing framework**, not a fork. The CLI,
bundle format, RNG system, exposure modes, manifest schema, and validation
harness are reused. A single dispatch hook in `Generator.generate()` selects
the lifecycle simulation path when the recipe declares `recipe_type:
lifecycle`.

---

## 2. Locked design decisions

### 2.1 First pass — 2026-06-10 (workstream shape)

| # | Question | Decision |
|---|----------|----------|
| D2 | Simulation time resolution | **Weekly steps** — granular health-signal trend curves. Billing/renewal events resolve to the enclosing week. |
| D3 | Independent vs chained generation | **Independent for v1.** Customer population is self-contained, not chained from a lead-scoring bundle. Chaining is designed-for-later. |
| D4 | Observation cohort | **Staggered start dates, fixed observation date.** Customers acquired across an acquisition window; all observed at one absolute calendar date, so tenure-at-observation varies. |
| D5 | Bundle schema version | **Bump** `BUNDLE_SCHEMA_VERSION` 5 → 6. |

### 2.2 Second pass — 2026-06-10 (target reframe to pLTV regression)

The original D1 ("primary task = `churn_within_180_days`") was a
mis-framing — it pattern-matched onto the lead-scoring binary classifier
instead of the actual pLTV literature. Corrected:

| # | Question | Decision |
|---|----------|----------|
| D1 | Primary task type | **Continuous pLTV regression.** Target = future gross revenue over a forward window. ZILN-shaped (zero mass + lognormal tail). The LTV-bucket multiclass idea is dropped. |
| D6 | Target horizon(s) | **Multiple windows: 90 / 365 / 730 days.** Three regression targets per customer. Zero-inflation and tail-heaviness grow with the window, giving a built-in difficulty gradient. |
| D7 | Value basis | **Gross revenue** = sum of paid invoice amounts (`payment_status ∈ {paid, recovered}`) inside the window. Matches the MRR mechanics directly. |
| D8 | Early/cold-start emphasis | **First-class early-pLTV task variant.** A tenure-anchored observation regime (observe each customer at a fixed short tenure, predict long-horizon value) ships alongside the calendar-anchored standard regime. |
| D9 | Churn task | **Kept as a secondary/auxiliary task** (`churn_within_180_days`), exposing the ZILN zero-inflation indicator. Not the headline. |

### 2.3 Consequence of D4 — absolute observation anchor

The lead-scoring path filters events by a **per-entity relative** cutoff
(`lead_created_at + snapshot_day`). With staggered starts + a single fixed
observation date (D4), the **calendar-anchored** lifecycle path filters by an
**absolute calendar** cutoff (`observation_date`, the same for every customer),
derived deterministically from the world calendar and recorded in the manifest.

The **tenure-anchored** early-pLTV regime (D8) does the opposite — a per-customer
relative cutoff at `customer_start + early_tenure_weeks` — which is structurally
the same relative-window logic the lead-scoring snapshot already uses. So both
window types have precedent in the codebase; the customer snapshot builder
supports both via an explicit cutoff argument.

### 2.4 Consequence of D6 — fully-simulated windows, clean targets

Because every target window is **fully simulated** (the engine runs each
customer through `observation_date + 730d` for the longest standard window, and
through `customer_start + early_tenure + 730d` for the early variant), all three
regression targets are **complete, not right-censored**. A customer who churns
mid-window simply has low/zero forward revenue — that *is* the zero-inflation,
not censoring.

Right-censoring is still taught, but as a property of *total* lifetime value on
the **relational tables** (a customer still active at sim-end has censored total
LTV) and in the notebooks — never as a hazard in the headline fixed-window
targets.

---

## 3. The pLTV target (ZILN)

For each customer and each window `W ∈ {90, 365, 730}` days:

```
ltv_revenue_{W}d = Σ amount_usd  for invoices with
    payment_status ∈ {paid, recovered}
    AND  cutoff < invoice_date <= cutoff + W days
```

where `cutoff` is the observation anchor (absolute `observation_date` for the
standard regime; `customer_start + early_tenure_weeks` for the early-pLTV
regime).

**Distribution shape (the whole point):**

- **Zero / near-zero mass** — customers who churn early in the window generate
  little/no forward revenue. Mass grows with `W` (more time to churn) and with
  difficulty tier (higher churn rate).
- **Lognormal tail** — retained + expanding customers compound MRR; top
  accounts dominate total value.

This is exactly the ZILN setting: model `P(value>0)` and the lognormal `(μ, σ)`
of the positive part. The dataset is built so a ZILN/two-part model beats a
plain MSE regressor — a live lesson, not a claim.

Why three windows: `90d` is near-deterministic (almost everyone pays ~3
invoices, low zero mass) → an easy warm-up. `365d` introduces a renewal
decision and meaningful zero mass. `730d` is strongly ZILN (heavy zero mass +
compounded expansion tail) → the hard target. Students can watch a model's
discrimination/calibration degrade as the horizon lengthens.

### 3.1 Two observation regimes

| regime | cutoff | tenure at cutoff | use case |
|--------|--------|------------------|----------|
| **calendar-anchored** (standard) | absolute `observation_date` | varies (cold→mature) | "score my current book of business" |
| **tenure-anchored** (early-pLTV) | `customer_start + early_tenure_weeks` (e.g. 4w) | fixed & short | "predict new-customer value early" (Voyantis acquisition framing) |

Both regimes are emitted from the **same simulated world** — two snapshot
tables + two task families. The early-pLTV regime is the genuine cold-start
hard case: only a few weeks of health signal exist at the cutoff.

---

## 4. Entities and tables

### 4.1 New entity rows (in `leadforge/schema/entities.py`)

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

**`InvoiceRow`** — monthly billing; drives payment failure **and the pLTV target**.

| column | dtype | notes |
|--------|-------|-------|
| `invoice_id` | string | opaque ID |
| `customer_id` | string | FK → customers |
| `invoice_date` | string | ISO-8601 |
| `amount_usd` | Int64 | the unit of pLTV value (§3) |
| `payment_status` | string | `paid` / `failed` / `recovered` / `written_off` |

### 4.2 Richer customer / subscription rows (lifecycle-only)

The current `CustomerRow` (4 fields) and `SubscriptionRow` (5 fields,
`subscription_status` hardcoded `"active"`) are thin shells that only record
conversion in the procurement world. The lifecycle bundle needs much richer
versions.

**Implementation note (decided in LTV-Pb):** these are added as *dedicated*
classes — `CustomerLifecycleRow` and `SubscriptionLifecycleRow` (both reusing
the logical table names `customers` / `subscriptions`) — kept in a separate
`LIFECYCLE_ROW_TYPES` registry, **not** by extending the existing classes in
place. The reason: `EntityRow.to_dict()` emits *every* dataclass field, so
adding fields to `CustomerRow`/`SubscriptionRow` would silently change the
lead-scoring instructor bundle's parquet schema. Dedicated classes keep the
lead-scoring catalog (`ALL_ROW_TYPES`, `TABLE_NAMES`, `ALL_CONSTRAINTS`) and
its output byte-for-byte unchanged. The two shapes never co-occur in one
bundle.

`CustomerLifecycleRow` carries: `customer_id`, `account_id`,
`customer_start_at`, `initial_plan`, `initial_mrr`, `contract_term_months`,
`csm_rep_id`, and a nullable `opportunity_id` (always `None` under independent
generation; reserved for future chaining).

`SubscriptionLifecycleRow` carries: `subscription_id`, `customer_id`,
`plan_name`, `subscription_status`, `subscription_start_at`, `current_mrr`,
`contract_term_months`, `renewal_count`, `expansion_count`, and the
public-redacted terminal fields `subscription_end_at`, `churn_at`,
`churn_reason`.

### 4.3 Public lifecycle table inventory

| table | public (`student_public`) | instructor (`research_instructor`) |
|-------|---------------------------|-------------------------------------|
| accounts | ✓ | ✓ |
| customers | ✓ (terminal fields redacted) | ✓ full |
| subscriptions | ✓ (terminal fields redacted) | ✓ full |
| subscription_events | ✓ (filtered to `<= cutoff`) | ✓ full horizon |
| health_signals | ✓ (filtered to `<= cutoff`) | ✓ full horizon |
| invoices | ✓ (filtered to `<= cutoff`) | ✓ full horizon |

Contacts/leads/touches/etc. from the lead-scoring world are **not** part of the
LTV bundle (independent generation, D3).

---

## 5. Snapshot-safety contract (lifecycle)

Analogous to the lead-scoring hard constraint in `CLAUDE.md`, anchored on the
relevant `cutoff` (absolute `observation_date`, or per-customer tenure cutoff
for the early regime):

- Every timestamp column in public event tables
  (`subscription_events.event_timestamp`, `health_signals.period_start`,
  `invoices.invoice_date`) must satisfy `<= cutoff`.
- No terminal/outcome fields in public `customers` / `subscriptions`:
  `churn_at`, `churn_reason`, `subscription_end_at`.
- No pLTV target column (`ltv_revenue_*`) or the secondary churn label may
  appear in the public relational tables — they are forward-window aggregates
  by construction.
- No flat snapshot feature may use events after `cutoff`, **except** the
  deliberately-retained leakage trap (§7).

Enforced by lifecycle-specific leakage probes (roadmap `LTV-M6`) and recorded
in the manifest. `CLAUDE.md`'s hard-constraints section gains a lifecycle
clause when the recipe wiring lands (`LTV-M5`).

---

## 6. Simulation mechanisms

Three new mechanism types, none of which exist today. All reuse the existing
`LatentScore` + per-step Bernoulli pattern from
`leadforge/mechanisms/hazards.py`.

**Churn hazard (post-conversion).** Weekly probability driven by health
signals + latent traits:

- Weibull-shaped over tenure: elevated in weeks 1–12 (onboarding instability),
  low in the steady middle.
- **Renewal-date spike**: ~10× background at each contract anniversary.
- Drivers: `latent_product_fit` (background), `latent_champion_strength`
  (renewal-date), `feature_depth_score` (leading indicator), unrecovered
  payment failures (financial trigger).

**Expansion propensity.** Weekly probability of a plan upgrade / seat add:

- Drivers: `latent_adoption_velocity`, `feature_depth_score`, active-user
  growth, `employee_band` (expansion ceiling).
- Expansion MRR delta: `randint(0.25·mrr, 1.0·mrr)`. This is the heavy-tail
  generator for the pLTV target.

**Payment failure.** Monthly billing event with a failure probability:

- Driver: `latent_budget_stability`.
- Dunning: 3 months `failed` before escalation to `recovered` / `written_off`;
  unrecovered → forced churn. Failed/written-off invoices do **not** count
  toward gross-revenue pLTV (D7).

### 6.1 Lifecycle motif families (new)

| family | retention driver |
|--------|------------------|
| `product_led_retention` | `latent_product_fit` dominant; health signals strongly predictive |
| `relationship_led_retention` | `latent_champion_strength` dominant; health weaker |
| `expansion_led_growth` | low churn, high upsell; pLTV variance from the expansion tail |
| `payment_fragile` | financially-triggered churn; `latent_budget_stability` dominant |
| `churner_dominated` | high background churn; strong early-warning signals (teaching tier) |

### 6.2 New customer latent traits

`latent_product_fit`, `latent_adoption_velocity`, `latent_budget_stability`,
`latent_champion_strength`, `latent_organizational_stability`. Sampled from the
clipped-Gaussian `_sample_latent` helper, with motif-family mean biases
mirroring `_MOTIF_LATENT_BIAS`.

---

## 7. Leakage trap

**Primary trap: `mrr_change_full_period` vs `mrr_change_at_snapshot`.**

- `mrr_change_at_snapshot` = `current_mrr − initial_mrr` at the cutoff. **Valid.**
- `mrr_change_full_period` = MRR delta from start to **end of simulation**.
  **Leaks** — future expansions (which directly drive the pLTV target) inflate
  it. Even more natural against a value target than it was against the
  lead-scoring label. Retained in all modes (`leakage_risk=True,
  redact_in_modes=frozenset()`), documented in the feature dictionary and
  release notes.

**Secondary trap (advanced tier): `last_health_signal_post_obs`** — a
health-signal reading from *after* the cutoff, named to look like a
current-state feature.

A trap-invariant test (analogous to `test_windowed_bundle_trap.py`) asserts
`mrr_change_full_period` diverges from `mrr_change_at_snapshot` for a
non-trivial fraction of customers.

---

## 8. Customer snapshot features (at `cutoff`)

Grouped like `LEAD_SNAPSHOT_FEATURES`. New constant
`CUSTOMER_SNAPSHOT_FEATURES` in `leadforge/schema/features.py`. The same
feature set serves both observation regimes (only the cutoff differs).

**Account** (from `AccountRow`): `industry`, `region`, `employee_band`,
`estimated_revenue_band`.

**Customer/subscription:** `tenure_weeks`, `initial_plan`, `current_plan`,
`initial_mrr`, `current_mrr`, `mrr_change_at_snapshot`, `renewal_count`,
`expansion_count`, `downgrade_count`, `contract_term_months`,
`weeks_to_next_renewal`.

**Health (aggregated over last 12 weeks before `cutoff`):**
`avg_active_users_l12w`, `active_user_trend_l12w` (slope),
`avg_feature_depth_l12w`, `support_ticket_count_l12w`, `last_nps_score`
(nullable).

**Financial:** `payment_failure_count`, `weeks_since_last_payment_failure`
(nullable).

**Leakage trap (all modes):** `mrr_change_full_period`.

**Targets:**

| column | type | task |
|--------|------|------|
| `ltv_revenue_90d` | Float64 | primary regression (warm-up horizon) |
| `ltv_revenue_365d` | Float64 | primary regression (standard horizon) |
| `ltv_revenue_730d` | Float64 | primary regression (hard horizon) |
| `churned_within_180d` | boolean | secondary / ZILN zero-inflation indicator |

For the **early-pLTV** task family the same target columns are recomputed off
the tenure-anchored cutoff and exported under a separate task directory.

---

## 9. Evaluation & difficulty

### 9.1 pLTV metrics (not AUC)

Regression + ranking + calibration, following the ZILN paper:

- **Discrimination:** Spearman rank correlation; **normalized Gini** (a.k.a.
  the value-weighted lift / Lorenz curve); decile lift.
- **Calibration:** decile chart (predicted vs actual mean per decile);
  total-predicted-vs-total-actual ratio.
- **Value capture:** top-K / top-decile share of realised value captured (reuses
  the lead-scoring `expected_acv`-capture machinery in `release_quality.py`).
- **Point error (reported, not optimised):** MAE / RMSE on `log1p(value)` —
  raw-scale MSE is shown as the *anti-pattern*.

### 9.2 Difficulty profiles

| dimension | intro | intermediate | advanced |
|-----------|-------|--------------|----------|
| `signal_strength` | 0.90 | 0.70 | 0.50 |
| `noise_scale` | 0.10 | 0.30 | 0.55 |
| `missing_rate` | 0.02 | 0.08 | 0.18 |
| `annual_churn_rate_range` | [0.10, 0.20] | [0.20, 0.35] | [0.30, 0.50] |
| `expansion_rate_range` | [0.15, 0.30] | [0.10, 0.20] | [0.05, 0.15] |
| `still_active_fraction` (≈ censored total LTV) | ~0.40 | ~0.60 | ~0.75 |
| secondary trap `last_health_signal_post_obs` | off | off | on |

Two orthogonal difficulty axes: the **tier** (signal/noise/prevalence) and the
**horizon** (90 < 365 < 730 → growing zero-mass + tail). Per-tier calibration
bands are fit on the regression metrics, not AUC.

---

## 10. Framework changes inventory

### New files

| file | purpose |
|------|---------|
| `leadforge/simulation/lifecycle.py` | `simulate_lifecycle()` — weekly-step subscription simulator |
| `leadforge/simulation/customer_population.py` | `build_customer_population()` — customer entities + latents + staggered starts |
| `leadforge/render/customer_snapshots.py` | `build_customer_snapshot(cutoff=…)` — per-customer row at a cutoff; serves both regimes |
| `leadforge/mechanisms/lifecycle_hazards.py` | churn hazard, expansion propensity, payment failure |
| `leadforge/recipes/b2b_saas_ltv_v1/{recipe,narrative,difficulty_profiles}.yaml` | new recipe |

### Modified files

| file | change |
|------|--------|
| `leadforge/schema/entities.py` | add 5 lifecycle rows + `LIFECYCLE_ROW_TYPES` registry (3 event tables + dedicated `CustomerLifecycleRow`/`SubscriptionLifecycleRow`); lead-scoring catalog untouched |
| `leadforge/schema/features.py` | add `CUSTOMER_SNAPSHOT_FEATURES` (3 regression targets + secondary churn) |
| `leadforge/schema/tasks.py` | add `LTV_REVENUE_{90,365,730}D` regression task specs + `CHURN_WITHIN_180D` |
| `leadforge/schema/relationships.py` | FK constraints for new tables |
| `leadforge/core/models.py` | add `n_customers`; lifecycle config (windows, early-tenure, observation anchor) |
| `leadforge/api/recipes.py` | parse `recipe_type` + `lifecycle:` section |
| `leadforge/api/generator.py` | dispatch on `recipe.recipe_type` |
| `leadforge/render/manifests.py` | record `observation_date` + windows; bump schema version to 6 |
| `leadforge/render/tasks.py` | regression task splits (continuous target) + early-pLTV task dir |
| `leadforge/validation/*` | lifecycle leakage probes + **regression** realism/metric bands |
| `CLAUDE.md` | lifecycle snapshot-safety hard-constraint clause; reference docs |

### Unchanged (reused as-is)

CLI commands/flags surface, `WorldBundle`/`WorldSpec`, RNG architecture,
exposure-mode dispatch, `manifest.json` envelope (additive),
determinism/monotonicity invariants. The task-split writer needs a continuous
target path (today it assumes a classification label).

---

## 11. Open items deferred past planning

- **Chained generation** (D3 later): seed the customer population from a
  lead-scoring bundle's converted leads. The population builder keeps a seam
  for a "from converted leads" source.
- **Continuous-time engine**: weekly steps suffice for v1.
- **Explicit ZILN baseline model** shipped in the package: notebooks
  demonstrate ZILN; a first-class `leadforge`-side ZILN baseline is a later
  addition.
- **Contribution-margin value basis** (vs gross revenue, D7): would add a cost
  model; deferred.
- **CLAUDE.md hard-constraint edit**: the lifecycle snapshot-safety clause is
  added when `LTV-M5` wiring lands, not in a planning PR.

# v4 Lead Scoring Dataset — Design Document

> Single source of truth for the v4 dataset: requirements, contract, engine changes, and implementation plan.
> Validation checks are in the companion `validation_spec.md`.

---

## Prior versions and lessons

| Version | Key issue | Lesson |
|---|---|---|
| v1 | `funnel_stage` contained `closed_won`/`closed_lost` — perfect leakage | Must validate that no single feature determines the target |
| v2 | Snapshot at day 90 with 90-day target — post-mortem, not prediction | Snapshot must be strictly earlier than outcome horizon |
| v2 | `reached_sql=0` → 0% conversion (n=127); `has_opportunity=1` → 0% (n=235) | Binary proxies from engine invariants create deterministic groups |
| v3 | Day-21 snapshot + non-deterministic proxies — clean but AUC only 0.62 | Engine's intro difficulty produces flat category effects; early features lack signal |

---

## Requirements

### R1 — Operational decision framing (expected ACV)

Include an `expected_acv` numeric feature so students can compute `P(conversion) × expected_acv` and practice value-aware ranking.

**ACV derivation (single source of truth):**

| Condition | Value |
|---|---|
| Opportunity created by snapshot day | Opportunity's `estimated_acv` |
| No opportunity, `estimated_revenue_band` known | Band midpoint (see table below) |
| No opportunity, band unknown | NaN |

**Revenue band → ACV midpoint mapping:**

| Band | Midpoint ($k) |
|---|---|
| $1M–$10M | 25 |
| $10M–$50M | 55 |
| $50M–$200M | 85 |
| $200M+ | 140 |

These midpoints are derived from the engine's `_EMPLOYEE_ACV_RANGES` in `simulation/engine.py`, which maps employee bands to ACV ranges. Since the dataset exposes `estimated_revenue_band` (not employee band), the midpoints approximate the overlap between revenue bands and the engine's ACV sampling.

### R2 — Safe temporal / momentum features

- `touches_week_1`: touches in days 0–7 after lead creation. Strictly pre-snapshot.
- `days_since_first_touch`: `snapshot_day - first_touch_day`. NaN if no touches.

### R3 — Structured missingness (MAR, not only MCAR)

Three patterns, each with a pedagogical rationale:

| Column | Pattern | Rates | Rationale |
|---|---|---|---|
| `days_since_last_touch` | Structural | NaN when no touches by snapshot | Natural — no event to measure from |
| `web_sessions` | Source-conditional | ~15% `sdr_outbound`, ~2% `inbound`, ~5% `partner` | CRM web tracking often not configured for outbound leads |
| `seniority` | Source-conditional | ~8% `partner_referral`, ~1% others | Referral partners don't always provide full contact details |
| `days_since_last_touch` | Additional MCAR | ~3% on top of structural | Random CRM logging gaps |

**Why these specific rates:** They are chosen to be detectable at n≈1000 with a chi-squared test at p<0.01 (the outbound/inbound ratio for `web_sessions` is ~7.5×, well above the 3× detection threshold), but not so extreme that imputation becomes trivial. These are tunable parameters, not ground truth — the validation spec checks the *ratio* (>3×), not the exact rates.

### R4 — Deliberate leakage trap

`total_touches_all` counts ALL touches over the full 90-day window, violating the snapshot boundary. It is strongly predictive but not deterministic. Must be labeled in release notes and feature dictionary but NOT revealed in student-facing `BACKGROUND.md`.

### R5 — Reduce redundancy

Drop `total_touches` (= `inbound_touches + outbound_touches`). Keep the breakdown.

### R6 — Stronger category signal

See "Engine change 1" below. Target: ≥15% spread for at least two category features; baseline LR AUC 0.65–0.90.

### R7 — Robust automated validation

See `validation_spec.md`.

---

## Target column set

| # | Column | Type | Source | Notes |
|---|---|---|---|---|
| 1 | `industry` | categorical | account | 4 values |
| 2 | `region` | categorical | account | US, UK |
| 3 | `company_size` | categorical | account | 4 bands |
| 4 | `company_revenue` | categorical | account | 4 bands |
| 5 | `contact_role` | categorical | contact | 4 roles |
| 6 | `seniority` | categorical | contact | 5 levels (~8% missing for partner_referral) |
| 7 | `lead_source` | categorical | lead | 3 channels |
| 8 | `opportunity_created` | binary 0/1 | derived | Opp opened by snapshot day |
| 9 | `demo_completed` | binary 0/1 | derived | Demo done by snapshot day |
| 10 | `expected_acv` | numeric | derived | See R1 ACV derivation table |
| 11 | `inbound_touches` | integer | events ≤ snapshot | Inbound touchpoints |
| 12 | `outbound_touches` | integer | events ≤ snapshot | Outbound touchpoints |
| 13 | `touches_week_1` | integer | events ≤ day 7 | First-week touch intensity |
| 14 | `web_sessions` | integer | events ≤ snapshot | Sessions (~15% missing for outbound) |
| 15 | `sales_activities` | integer | events ≤ snapshot | Sales activities count |
| 16 | `days_since_last_touch` | float | events ≤ snapshot | Natural NaN when no touches |
| 17 | `total_touches_all` | integer | **ALL events** | Leakage trap — full 90-day window |
| 18 | `converted` | binary 0/1 | target | Converted within 90 days |

Total: 17 features + 1 target = 18 columns.

---

## Snapshot contract

- **Snapshot day:** 21 (configurable).
- **Observation window:** Days 0–21 inclusive.
- **Prediction horizon:** Days 22–90.
- **Temporal guarantee:** No feature except `total_touches_all` uses post-snapshot data.

| Data source | Temporal gate |
|---|---|
| Account/contact/lead attributes | Static — always valid |
| Touch/session/activity events | `timestamp ≤ lead_created_at + snapshot_day` |
| Opportunity records | `opportunity.created_at ≤ lead_created_at + snapshot_day` |

### Target definition

```
converted = 1  if lead reached closed_won within 90 days of lead_created_at
converted = 0  otherwise (including closed_lost, still in funnel, churned)
```

Derived from simulated events, never directly sampled.

### Subsampling

- Source bundle: 5,000 leads, `b2b_saas_procurement_v1`, seed 42, difficulty intro.
- Stratified subsampling to 1,000 rows at ~30% conversion.
- Subsampling uses `np.random.RandomState(seed)` for reproducibility.

---

## Engine changes

### Change 1: Stronger category signal via population-level correlation

#### Problem

Observable categories (seniority, revenue band, lead source) are drawn **independently** from latent traits in `population.py`. The conversion hazard uses only latent traits (via `LatentScore`). Therefore category → conversion correlation is near-zero by construction.

The v3 dataset confirms this: category spreads are 2–11% and LR AUC is 0.62.

Note: `CategoricalInfluence` exists in `mechanisms/categorical.py` but is **never wired** into `assign_mechanisms()` or the simulation loop. The `MechanismContext` only passes latent traits, stage, time, and dwell days — not observable categories.

#### Solution

Correlate observable categories with latent traits during population generation. This is a population-layer change — no simulation loop modifications needed.

Add a `category_latent_correlations` mapping to the difficulty profile, applied in `build_population()` after initial latent sampling:

| Observable | Latent trait | Boost per value |
|---|---|---|
| `seniority` | `latent_contact_authority` | individual_contributor: −0.27, manager: −0.09, director: +0.09, vp: +0.22, c_suite: +0.36 |
| `estimated_revenue_band` | `latent_account_fit` | $1M–$10M: −0.18, $10M–$50M: 0.0, $50M–$200M: +0.18, $200M+: +0.32 |
| `lead_source` | `latent_engagement_propensity` | sdr_outbound: −0.14, inbound_marketing: +0.09, partner_referral: +0.22 |

These are the scale=1.8 boosts from the spike experiment (`scripts/spike_category_signal.py`).

#### Spike experiment results (seed 42, 5000 leads, fit_dominant motif)

| Setting | AUC | seniority spread | revenue spread | role_function spread |
|---|---|---|---|---|
| Baseline (no correlation) | 0.663 | 9.5% | 10.8% | 11.0% |
| Scale 1.0 | 0.650 | 5.5% | 10.3% | 3.7% |
| **Scale 1.8** | **0.694** | **22.1%** | **15.2%** | 1.7% |
| Scale 2.5 | 0.701 | 11.9% | 15.1% | 3.2% |

**Observations:**
- Scale 1.8 gives AUC 0.694, within the [0.65, 0.90] target.
- `seniority` and `estimated_revenue_band` exceed the 15% spread target.
- `role_function` gets **no boost** from this approach because there is no natural latent trait to correlate it with. The spike shows role_function spread is driven entirely by noise and varies widely across runs (1.7%–11%).
- The `fit_dominant` motif gives zero weight to `latent_contact_authority`, so the seniority boost only works through its indirect correlation with other traits at population level. Different motif families will produce different spread profiles.
- At scale 2.5, seniority spread *decreases* (11.9%) due to [0, 1] clamp saturation.

#### Important caveats

1. The spike tested only `fit_dominant` motif (seed 42). Other motifs weight different latent traits, so the same boosts will produce different category spreads. The implementation should test across all 5 motif families.
2. `role_function` signal remains weak. If role_function spread ≥15% is required, a separate mechanism is needed (either role-specific latent biases in `population.py` or wiring `CategoricalInfluence` into the conversion score, which would require sim loop changes). For v4, we accept that not all categories will have strong signal.
3. The boost values are empirical, not principled. They should be treated as starting points, not final values.

#### Files affected

- `leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml` — add `category_latent_correlations`
- `leadforge/simulation/population.py` — apply correlations after initial latent sampling
- `tests/simulation/test_population.py` — test correlation application, backward compat

### Change 2: Windowed snapshot (`snapshot_day` parameter)

Add `snapshot_day: int | None` to `build_snapshot()`. When set, all event aggregations filter to `timestamp ≤ lead_created_at + snapshot_day`. Default `None` preserves current behavior (full horizon).

New features computed in the snapshot:
- `touches_week_1` — count touches where `days_after_creation ≤ 7`
- `days_since_first_touch` — `snapshot_day - first_touch_day` (NaN if no touches)
- `expected_acv` — see R1 derivation table above
- `total_touches_all` — full-horizon touch count (ignoring snapshot gate)

#### Files affected

- `leadforge/render/snapshots.py` — add `snapshot_day`, windowed filtering, new features
- `leadforge/schema/features.py` — add `FeatureSpec` entries
- `tests/render/test_snapshots.py` — test windowed aggregation

### Change 3: Structured missingness + leakage trap (build script only)

These are NOT engine changes. The build script (`scripts/build_v4_snapshot.py`) applies missingness injection and computes the leakage trap after snapshot construction.

#### Files affected

- `scripts/build_v4_snapshot.py` (new)
- No changes to `leadforge/` core for missingness

---

## Known limitations and workarounds

### `is_sql=False → never converts` (engine invariant)

The simulation engine requires leads to pass through SQL stage before converting. This creates a deterministic group: `is_sql=False` → 0% conversion. v4 works around this by:

- **Excluding** `is_sql` and `reached_sql` from the column set entirely
- Using `opportunity_created` and `demo_completed` as non-deterministic binary proxies instead

A proper fix would modify the conversion hazard in `engine.py` to allow rare direct conversions. This is tracked as a deferred item — it would benefit v5+ but is out of scope for v4.

### `role_function` lacks signal

The population-level correlation approach provides no mechanism for `role_function` to influence conversion (there is no natural latent trait to map it to). role_function spread in the v4 dataset will be noise-driven (2–11%). This is acceptable for an intro course but should be addressed in a future engine revision.

---

## Tuning protocol

If validation checks fail during implementation, use these adjustments:

| Failure | Adjustment |
|---|---|
| AUC < 0.65 | Increase boost scale (try 2.0, 2.5, 3.0) |
| AUC > 0.90 | Decrease boost scale or add noise to latent correlations |
| Leakage trap boost < 0.03 | Widen snapshot gap (try day 14 instead of 21) to increase information delta |
| Subsampling destroys signal | Increase `n_leads` from 5000 to 10000 before subsampling |
| Category spread < 15% for seniority/revenue | Increase individual boost magnitudes for that feature |
| Deterministic group detected | Check which feature/value, adjust boost or drop the feature |

---

## Implementation plan

### Milestone structure

v4 work is split into **two implementation milestones** plus the planning PR:

```
v4-M0 (this PR — planning + spike)
  └── v4-M1: engine + build pipeline (single PR)
        └── v4-M2: dataset generation + release docs
```

v4-M1 merges the engine changes and build pipeline into one milestone because:
- The engine change (population-level correlations) cannot be validated without the build script
- The build script depends on the snapshot_day parameter from the engine change
- A single PR with both, validated end-to-end, is more reviewable

### v4-M1: Engine + build pipeline

**Deliverables:**
1. `difficulty_profiles.yaml` — `category_latent_correlations` for intro profile
2. `simulation/population.py` — apply correlations during population generation
3. `render/snapshots.py` — `snapshot_day` parameter, windowed aggregation, new features
4. `schema/features.py` — new `FeatureSpec` entries
5. `scripts/build_v4_snapshot.py` — day-21 snapshot + missingness + leakage trap + subsampling
6. `scripts/validate_v4_dataset.py` — validation per `validation_spec.md`
7. Tests for all changes

**Acceptance criteria:**
- [ ] No correlation (`category_latent_correlations` absent or empty) → identical output to current engine
- [ ] Scale 1.8 correlations → seniority and revenue_band spread ≥15%
- [ ] `snapshot_day=21` correctly filters events
- [ ] `touches_week_1` counts only days 0–7
- [ ] `expected_acv` uses ACV derivation table
- [ ] Build script produces 1000 rows × 18 columns at 30% conversion
- [ ] Validation script passes all mandatory checks
- [ ] LR AUC in [0.65, 0.90] without trap; ≥0.03 boost with trap
- [ ] All existing tests pass
- [ ] Reproducible with seed 42

### v4-M2: Documentation + release

**Deliverables (in leadforge-datasets-private):**
1. `lead_scoring_intro/lead_scoring_intro_v4.csv`
2. `lead_scoring_intro/RELEASE_v4.md`
3. Updated README

**Deliverables (in leadforge):**
1. Updated `.agent-plan.md`

**Acceptance criteria:**
- [ ] CSV passes all validation checks
- [ ] RELEASE_v4.md documents snapshot day, target definition, changes from v3, leakage trap
- [ ] Previous versions marked as superseded

### Relationship to existing roadmap

| Existing milestone | v4 interaction |
|---|---|
| M0–M11 | Complete, no changes |
| M12 (CLI polish) | **Deferred** — low priority vs v4 |
| M14 (Sample datasets + notebooks) | **Absorbed** — v4 dataset IS the sample |
| M15 (Docs polish + v1.0 RC) | **Deferred** — do after v4 |

Discarded: M14 notebooks 3–4 (no current audience).

---

## Non-goals

- v4 does NOT modify the simulation loop (`engine.py` daily step logic).
- v4 does NOT change the relational bundle format or task splits.
- v4 does NOT add new recipes.
- v4 does NOT change exposure modes.
- v4 does NOT fix the `is_sql=False → never converts` invariant (deferred).

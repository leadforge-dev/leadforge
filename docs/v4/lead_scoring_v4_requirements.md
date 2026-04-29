# Lead Scoring Dataset v4 — Requirements

## Purpose

This document defines the requirements for the **v4 lead scoring intro dataset**, the primary pedagogical output of leadforge for a BA-level intro ML course. It is informed by three prior dataset iterations (v1–v3) and the lessons learned from each.

## Prior version history and lessons

| Version | Key issue | What we learned |
|---|---|---|
| v1 | `funnel_stage` contained `closed_won`/`closed_lost` — perfect leakage | Must validate that no single feature determines the target |
| v2 | Snapshot at day 90 with 90-day target — post-mortem, not prediction | Snapshot must be strictly earlier than outcome horizon |
| v2 | `reached_sql=0` → 0% conversion (n=127); `has_opportunity=1` → 0% (n=235) | Binary proxies from engine invariants create deterministic groups |
| v3 | Day-21 snapshot + non-deterministic proxies — clean but AUC only 0.62 | Engine's intro difficulty produces flat category effects; early features lack signal |

## v4 requirements

### R1 — Operational decision framing (capacity + value)

**Problem:** v1–v3 frame lead scoring as pure classification. Real lead scoring is a **decision tool** — ranking leads by expected value, not just probability.

**Requirement:**
- Include an `expected_acv` numeric feature (estimated annual contract value) available at snapshot time.
- The feature must be derived from the opportunity table (for leads with an opportunity by snapshot) or from account-level heuristics (employee band → ACV range midpoint) for leads without one.
- This enables students to compute `expected_value = P(conversion) × expected_acv` and practice ranking/top-K selection.

**Engine change needed:** The snapshot builder must join opportunity ACV data gated by snapshot day, with a fallback to account-band heuristic ACV.

### R2 — Safe temporal / momentum features

**Problem:** v1–v3 engagement features are cumulative counts with no temporal shape. Real lead scoring uses recency and momentum signals.

**Requirement:**
- Include exactly one momentum feature: `touches_week_1` (touches in days 0–7 after lead creation).
- This is strictly pre-snapshot (snapshot is at day 21+) and gives students a "first-week intensity" signal to compare against total touches.
- Additionally, `days_since_first_touch` (snapshot_day minus day of first touch) provides a lead-age signal.

**Engine change needed:** The snapshot builder must compute windowed aggregates from event timestamps.

### R3 — Structured missingness (not only MCAR)

**Problem:** v1–v3 inject missingness randomly (MCAR). Real CRM data has structured gaps.

**Requirement:** Implement three missingness patterns:
1. **Natural (structural):** `days_since_last_touch` is NaN when `total_touches == 0` (no touches recorded). Already exists but must be preserved.
2. **Conditional on source:** `web_sessions` is missing for ~15% of `sdr_outbound` leads (CRM tracking often not set up for outbound-sourced leads) but only ~2% of `inbound_marketing` leads.
3. **Role data gap:** `seniority` is missing for ~8% of `partner_referral` leads (referral partners don't always provide full contact details).

**Engine change needed:** Missingness injection in the snapshot builder, conditioned on feature values.

### R4 — Deliberate leakage trap

**Problem:** Students need to practice identifying leakage, but v1–v3 either have accidental leakage (bad) or none at all (missed teaching opportunity).

**Requirement:**
- Include one feature `total_touches_all` that counts **all** touches over the full 90-day window, not just up to snapshot.
- This feature is strongly predictive (uses future data) but not perfectly deterministic (it correlates with but doesn't fully determine conversion).
- The feature MUST be clearly labeled as "intentionally invalid — included for leakage discussion" in `RELEASE_v4.md` and the feature dictionary.
- The validation script must flag it, but the v4 build script intentionally includes it.
- The `BACKGROUND.md` / student instructions must NOT reveal the trap — students should discover it through EDA.

**Engine change needed:** The snapshot builder computes a second touch count using the full horizon.

### R5 — Reduce redundancy

**Problem:** `total_touches = inbound_touches + outbound_touches` is a perfect linear dependency. Students may be confused by it, or models waste a degree of freedom.

**Requirement:**
- Drop `total_touches` from v4. Keep `inbound_touches` and `outbound_touches` as the touch breakdown.
- Note: `total_touches_all` (the leakage trap from R4) is a different feature and is kept.
- Document this as a teaching point: "you can derive total from inbound + outbound."

### R6 — Stronger category signal

**Problem:** At intro difficulty, category conversion rates span only 2–11%. This makes the dataset nearly impossible to model well (AUC ~0.62).

**Requirement:**
- The engine must produce category-level conversion rate spreads of at least 15–25% for key features (`contact_role`, `company_revenue`, `seniority`).
- Target baseline LR AUC: **0.70–0.85** (after snapshot + subsampling).
- This requires engine changes to the difficulty profile or mechanism weights, not just post-hoc manipulation.

**Engine change needed:** Adjust intro difficulty profile or mechanism policy to produce wider category effects.

### R7 — Robust automated validation

**Requirement:** The v4 dataset must pass all of the following automated checks:

| Check | Criterion |
|---|---|
| No banned columns | No `current_stage`, `funnel_stage`, `conversion_timestamp`, `is_sql` |
| No deterministic groups | For every feature value with n≥50: conversion rate in [2%, 98%] |
| Conversion rate | In [15%, 40%] |
| Baseline LR AUC | In [0.65, 0.90] (all features except leakage trap) |
| Leakage trap AUC boost | AUC with trap > AUC without trap by ≥0.03 |
| Missingness per column | Each column with nulls: 1–15% missing |
| Missingness structure | `web_sessions` missing rate for `sdr_outbound` > 3× rate for `inbound_marketing` |
| Row count | Exactly 1,000 |
| Column count | 16–18 (features + target) |
| Reproducibility | Same seed → identical output |

## v4 target column set

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
| 10 | `expected_acv` | numeric | derived | Opp ACV if available, else band midpoint (R1) |
| 11 | `inbound_touches` | integer | events ≤ snapshot | Inbound touchpoints |
| 12 | `outbound_touches` | integer | events ≤ snapshot | Outbound touchpoints |
| 13 | `touches_week_1` | integer | events ≤ day 7 | First-week touch intensity (R2) |
| 14 | `web_sessions` | integer | events ≤ snapshot | Sessions (~15% missing for outbound, ~2% inbound) |
| 15 | `sales_activities` | integer | events ≤ snapshot | Sales activities count |
| 16 | `days_since_last_touch` | float | events ≤ snapshot | Natural NaN when no touches |
| 17 | `total_touches_all` | integer | **ALL events** | ⚠️ LEAKAGE TRAP — uses full 90-day window |
| 18 | `converted` | binary 0/1 | target | Converted within 90 days |

Total: 17 features + 1 target = 18 columns.

## Non-goals for v4

- v4 does NOT require engine changes to the simulation loop itself (stage transitions, churn, conversion hazard).
- v4 does NOT change the relational bundle format or task splits.
- v4 does NOT require a new recipe — it uses `b2b_saas_procurement_v1` with adjusted difficulty tuning.
- v4 does NOT need to change the `student_public` / `research_instructor` exposure modes.

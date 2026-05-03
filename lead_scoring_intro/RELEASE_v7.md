# RELEASE v7 — Lead Scoring Intro Dataset

## Overview

v7 is the seventh iteration of the lead scoring intro dataset, designed for **3–4 lectures** on applied ML for lead scoring. The key change from v6 is a **purely causal leakage trap** — no label-conditioned noise is injected. The trap signal comes entirely from shared latent drivers in the simulation, making it a realistic example of temporal leakage.

### Key changes from v6

| Change | v6 | v7 |
|---|---|---|
| Leakage trap | Causal touches + **Poisson(3) boost conditional on `converted`** | **Purely causal** — post-snapshot touches only, no label injection |
| Trap mechanism | `boost_leakage_trap()` added label-correlated noise | Removed entirely; trap is raw simulated event count |
| Engine: follow-up ramp | Not present | **LatentDecayIntensity follow-up boost** — sales teams increase engagement with high-quality leads after day 20, using different latent dimensions (budget_readiness, process_maturity) than pre-snapshot features |
| Documentation | Mismatches (snapshot day, region list, seniority levels) | **Exact alignment** with generated data, validated programmatically |
| Trap delta thresholds | mean >= 0.03 (calibrated for label-boosted trap) | **mean >= 0.008** (calibrated for purely causal trap) |

---

## Snapshot definition

- **Snapshot day**: 20 (features computed from events on days 0–20 after lead creation)
- **Prediction horizon**: 90 days from lead creation
- **Target**: `converted` — 1 if a `closed_won` event occurred within 90 days, 0 otherwise
- **Rows**: 1,000 (stratified subsample at 30% conversion rate from 5,000 generated leads)

**What "as-of snapshot" means**: all legitimate features use information available at or before day 20 of each lead's lifecycle. No feature uses events from day 21 onward. The instructor trap column explicitly violates this rule (it counts touches from days 21–90), which is why it constitutes leakage.

---

## Student vs instructor files

| File | Columns | Use |
|---|---|---|
| `lead_scoring_intro_v7.csv` | 20 (19 features + 1 target) | Student-safe — no leakage |
| `lead_scoring_intro_v7_instructor.csv` | 21 (same 20 + 1 trap) | Instructor only — contains deliberate temporal leakage |

Both files have identical rows in identical order. The only difference is the instructor file includes one extra column: `__leakage__touches_post_snapshot_21_90`.

---

## Column dictionary

### Categorical features (8)

| Column | Type | Unique values | Description |
|---|---|---|---|
| `industry` | string | healthcare_non_clinical, logistics, manufacturing, professional_services | Industry vertical of the buying organization |
| `region` | string | UK, US | Geographic region |
| `company_size` | string | 200-499, 500-999, 1000-1999, 2000+ | Employee headcount band |
| `company_revenue` | string | $1M-$10M, $10M-$50M, $50M-$200M, $200M+ | Estimated annual revenue band |
| `contact_role` | string | ap_manager, it_director, procurement_manager, vp_finance | Functional area of primary contact |
| `seniority` | string | c_suite, director, individual_contributor, manager, vp | Seniority level |
| `lead_source` | string | inbound_marketing, partner_referral, sdr_outbound | Origination channel |
| `acquisition_wave` | string | A, B, C | Cohort label — roughly chronological |

### Binary features (2)

| Column | Type | Description |
|---|---|---|
| `opportunity_created` | int (0/1) | Whether an opportunity existed by snapshot day |
| `demo_completed` | int (0/1) | Whether demo page was viewed (proxy for demo) |

### Numeric features (9)

| Column | Type | Description | Missingness |
|---|---|---|---|
| `expected_acv` | float | Expected ACV in USD ($18k–$120k) | 1.5% MCAR |
| `inbound_touches` | int | Inbound marketing touches (days 0–20) | — |
| `outbound_touches` | int | Outbound sales touches (days 0–20) | — |
| `touches_week_1` | int | Touches in first 7 days | — |
| `touches_last_7_days` | int | Touches in last 7 days of snapshot window (days 14–20) | — |
| `days_since_first_touch` | float | Days from first touch to snapshot cutoff | Structural (no touches) + 2.5% MCAR |
| `web_sessions` | float | Web sessions within snapshot window | 6.6% MAR by lead_source |
| `sales_activities` | int | Sales rep activities within snapshot window | — |
| `days_since_last_touch` | float | Days since last touch to snapshot cutoff | Structural (no touches) + 1.2% MCAR |

### Target (1)

| Column | Type | Description |
|---|---|---|
| `converted` | int (0/1) | 1 if closed_won within 90 days, 0 otherwise |

### Instructor-only leakage trap (1)

| Column | Type | Description |
|---|---|---|
| `__leakage__touches_post_snapshot_21_90` | int | Touch count in days 21–90 (post-snapshot) — purely from simulated events, NO label injection |

---

## Missingness patterns

| Column | Count | Rate | Pattern |
|---|---|---|---|
| `seniority` | 28 | 2.8% | MAR: partner referral 8%, others 1% |
| `expected_acv` | 15 | 1.5% | MCAR: uniform 2% |
| `days_since_first_touch` | 40 | 4.0% | Structural (no touches) + MCAR 2% |
| `web_sessions` | 66 | 6.6% | MAR: SDR outbound 15%, inbound 2%, partner 5% |
| `days_since_last_touch` | 42 | 4.2% | Structural (no touches) + MCAR 3% |

Total: 191 missing values across 5 columns.

---

## Dataset statistics

### Expected ACV distribution

| Statistic | Value |
|---|---|
| Min | $18,000 |
| Mean | $59,945 |
| Median | $55,000 |
| P95 | $117,537 |
| P99 | $119,463 |
| Max | $119,937 |
| At max (pile-up) | 0.1% |

---

## Baseline metrics

Evaluated using the canonical sklearn pipeline:
```
Numeric:  SimpleImputer(median) → StandardScaler
Categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
Model: LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
Split: 70/30 stratified hold-out
```

### Logistic Regression baseline (seed 42)

| Metric | Value |
|---|---|
| ROC-AUC | 0.671 |
| PR-AUC | 0.426 |
| Base rate | 30.0% |
| Precision@25 | 0.440 (Lift: 1.47x) |
| Precision@50 | 0.420 (Lift: 1.40x) |

### Tree model comparison (5-seed average, seeds 42–46)

| Model | Mean AUC | vs LR |
|---|---|---|
| Logistic Regression | 0.650 | — |
| GBM (100 trees) | 0.721 | +0.072 |

GBM reliably outperforms LR due to nonlinear interactions in the DGP (latent trait interactions with engagement patterns, opportunity × momentum, seniority × engagement).

### Value-aware ranking (seed 42)

| K | By P(convert) | By expected value | Uplift |
|---|---|---|---|
| 25 | $822,099 | $932,505 | +13.4% |
| 50 | $1,528,789 | $1,839,009 | +20.3% |

---

## Leakage trap evaluation

The v7 trap is **purely causal**: `__leakage__touches_post_snapshot_21_90` counts actual simulated touch events in days 21–90. **No Poisson boost or label-conditioned noise is applied.** The trap is predictive because:

1. Latent traits (budget_readiness, process_maturity, account_fit) drive both conversion probability and post-snapshot follow-up intensity
2. Sales teams increase engagement with high-quality leads after the initial assessment period (day 20), modeled via the `LatentDecayIntensity` follow-up ramp mechanism
3. The follow-up weights emphasize *different* latent dimensions than pre-snapshot features, giving the trap genuinely marginal information

### Why v7 trap delta is smaller than v6

v6 applied `Poisson(3) * converted` to the trap — injecting signal that was directly correlated with the label, not merely through shared latent confounders. v7 removes this entirely. A purely causal trap provides smaller marginal AUC improvement because pre-snapshot features already capture most latent signal.

### Per-seed trap delta table

Pipeline: canonical LR (full feature set) with and without trap column.

| Seed | Delta |
|---|---|
| 42 | +0.0193 |
| 43 | +0.0177 |
| 44 | +0.0151 |
| 45 | +0.0092 |
| 46 | +0.0151 |
| 47 | +0.0144 |
| 48 | +0.0139 |
| 49 | +0.0087 |
| 50 | +0.0112 |
| 51 | +0.0005 |

| Statistic | Value |
|---|---|
| Mean delta | +0.0125 |
| Median delta | +0.0141 |
| Min delta | +0.0005 |
| Max delta | +0.0193 |
| Positive seeds | 10/10 |

Computed with:
```bash
python scripts/validate_v7_dataset.py \
  lead_scoring_intro/lead_scoring_intro_v7.csv \
  lead_scoring_intro/lead_scoring_intro_v7_instructor.csv \
  --out-json validation_v7_report.json
```

---

## Cohort split evaluation

The `acquisition_wave` feature enables a distribution-shift lecture:
- **Random split** approximates IID evaluation (train and test drawn from same distribution)
- **Cohort split** approximates future generalization: train on waves A+B (earlier leads), test on wave C (latest leads)

| Split | AUC | PR-AUC |
|---|---|---|
| Random (70/30, seed 42) | 0.683 | 0.446 |
| Cohort (A+B → C) | 0.594 | 0.349 |
| **AUC gap (random − cohort)** | **0.089** | |

The 0.089 AUC gap demonstrates that model performance degrades on future cohorts — a key lesson in real-world ML deployment.

---

## Known limitations

1. **Two regions only** (US, UK) — limits geographic diversity. Driven by the v1 narrative vertical.
2. **Purely causal trap delta is modest** (mean +0.013) compared to v6's label-boosted trap (mean +0.061). This is the honest result of removing label injection. For a more dramatic leakage demo, instructors can note that in practice, leakage features often have much larger effects because they can be near-perfectly correlated with the outcome.
3. **Small dataset** (1,000 rows) creates variance in hold-out metrics across random seeds.

---

## Teaching guidance

### Lecture 1: Pipeline + Evaluation

**Goal**: Students build their first ML pipeline and learn proper evaluation.

- Load `lead_scoring_intro_v7.csv`
- Handle missing values (5 columns have NaN — discuss structural vs MCAR)
- Build a baseline logistic regression with train/test split
- Evaluate: AUC (~0.67), PR-AUC, confusion matrix
- Discuss class imbalance (30% positive rate)

### Lecture 2: Top-K + Expected Value Ranking

**Goal**: Students learn decision-oriented evaluation.

- Precision@K and Lift@K: "If sales can contact 50 leads, how many convert?"
- Expected value ranking: `P(convert) * expected_acv`
- Demonstrate that EV ranking captures 13% more ACV at K=25 than probability ranking
- Discuss when value-aware scoring matters (heterogeneous deal sizes)

### Lecture 3: Feature Engineering + Error Slicing

**Goal**: Students learn to improve models through feature understanding.

- Examine `touches_last_7_days` (momentum) vs `touches_week_1` (early signal)
- Error analysis by `region`, `company_size`, `lead_source`
- Missing value patterns: why is `web_sessions` missing more for SDR outbound?
- Feature interactions: `opportunity_created` x `touches_last_7_days`

### Lecture 4: Trees/GBM + Nonlinearity (+ optional cohort shift)

**Goal**: Students see why tree models outperform linear models.

- Train GBM, compare AUC vs LR (+0.072 on average)
- Feature importance from GBM
- Discuss nonlinear interactions captured by trees
- **Optional**: use `acquisition_wave` for cohort split (train A/B, test C)
  - Random split AUC: 0.683, Cohort split AUC: 0.594 (AUC gap: 0.089)
  - Demonstrates distribution shift and evaluation realism

### Instructor note: Leakage detection exercise

Use `lead_scoring_intro_v7_instructor.csv` for a leakage detection exercise:
- Students train with all columns including `__leakage__touches_post_snapshot_21_90`
- AUC improves by ~0.013 on average (subtler than v6's label-boosted trap)
- Challenge: identify which column is leaking and explain *why* it's invalid at scoring time
- The trap is **purely causal** — future engagement correlates with conversion via shared latent drivers, not because the label was injected. This makes it a realistic and pedagogically honest example of temporal leakage.
- Teaching point: "at scoring time (day 20), you cannot know how many touches the lead will receive in the future. Using this feature would require time-traveling."

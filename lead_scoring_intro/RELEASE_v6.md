# RELEASE v6 — Lead Scoring Intro Dataset

## Overview

v6 is the sixth iteration of the lead scoring intro dataset, designed for **3–4 lectures** on applied ML for lead scoring. It introduces causally-grounded leakage detection, value-aware ranking, nonlinear interaction structure, and cohort-based evaluation.

### Key changes from v5

| Change | v5 | v6 |
|---|---|---|
| Snapshot day | 10 | **20** |
| Leakage trap | Label-noise boost (`total_touches_90d`) | **Causal** post-snapshot touches (days 21–90) + Poisson(3) boost |
| Student/instructor split | Single file | **Two files**: student-safe + instructor |
| Momentum features | `touches_week_1`, `days_since_first_touch` | + **`touches_last_7_days`** |
| Cohort feature | — | **`acquisition_wave`** (A/B/C) |
| Value column | `expected_acv` (hard clip) | `expected_acv` (soft winsorize) |
| Tree improvement | Not validated | **Validated**: GBM > LR |
| MCAR injection | `days_since_last_touch` | + `expected_acv` (2%) |

---

## Snapshot definition

- **Snapshot day**: 20 (features computed from events on days 0–20 after lead creation)
- **Horizon**: 90 days (label derived from events through day 90)
- **Target**: `converted` — 1 if a `closed_won` event occurred within 90 days, 0 otherwise
- **Rows**: 1,000 (stratified subsample at 30% conversion rate)

---

## Column dictionary

### Categorical features (8)

| Column | Type | Description |
|---|---|---|
| `industry` | string | Industry vertical of the buying organization |
| `region` | string | Geographic region (US, UK, EU, APAC) |
| `company_size` | string | Employee headcount band (200–499, 500–999, etc.) |
| `company_revenue` | string | Estimated annual revenue band |
| `contact_role` | string | Functional area of primary contact |
| `seniority` | string | Seniority level (IC, manager, director, VP) |
| `lead_source` | string | Origination channel (inbound_marketing, sdr_outbound, partner_referral) |
| `acquisition_wave` | string | Cohort label (A, B, C) — roughly chronological |

### Binary features (2)

| Column | Type | Description |
|---|---|---|
| `opportunity_created` | int (0/1) | Whether an opportunity existed by snapshot day |
| `demo_completed` | int (0/1) | Whether demo page was viewed (proxy for demo) |

### Numeric features (9)

| Column | Type | Description | Missingness |
|---|---|---|---|
| `expected_acv` | float | Expected ACV in USD ($18k–$120k) | 2% MCAR |
| `inbound_touches` | int | Inbound marketing touches (days 0–20) | — |
| `outbound_touches` | int | Outbound sales touches (days 0–20) | — |
| `touches_week_1` | int | Touches in first 7 days | — |
| `touches_last_7_days` | int | Touches in last 7 days of snapshot window (days 14–20) | — |
| `days_since_first_touch` | float | Days from first touch to snapshot cutoff | Structural (no touches) + 2% MCAR |
| `web_sessions` | int | Web sessions within snapshot window | MAR by lead_source |
| `sales_activities` | int | Sales rep activities within snapshot window | — |
| `days_since_last_touch` | float | Days since last touch to snapshot cutoff | Structural (no touches) + 3% MCAR |

### Target (1)

| Column | Type | Description |
|---|---|---|
| `converted` | int (0/1) | 1 if closed_won within 90 days, 0 otherwise |

### Instructor-only leakage trap (1)

| Column | Type | Description |
|---|---|---|
| `__leakage__touches_post_snapshot_21_90` | int | Touch count in days 21–90 (post-snapshot) + Poisson(3) boost for converted leads |

---

## Missingness patterns

| Pattern | Column(s) | Type | Rate |
|---|---|---|---|
| Structural | `days_since_last_touch` | No touches → NaN | ~2% |
| Structural | `days_since_first_touch` | No touches → NaN | ~1% |
| MAR | `web_sessions` | SDR outbound: 15%, inbound: 2%, partner: 5% | ~7% overall |
| MAR | `seniority` | Partner referral: 8%, others: 1% | ~2% overall |
| MCAR | `expected_acv` | Uniform 2% | ~2% |
| MCAR | `days_since_last_touch` | Additional 3% on top of structural | ~3% |
| MCAR | `days_since_first_touch` | Additional 2% on top of structural | ~2% |

Total missingness: ~192 values across 5 columns (~19 values per 1000 rows per column on average).

---

## Baseline metrics

Evaluated on a 70/30 stratified hold-out split (seed 42).

### Logistic Regression baseline

| Metric | Value |
|---|---|
| ROC-AUC | 0.676 |
| PR-AUC | 0.439 |
| Base rate | 30.0% |
| Precision@25 | 0.480 (Lift: 1.60x) |
| Precision@50 | 0.420 (Lift: 1.40x) |

### Tree model comparison (5-seed average)

| Model | Mean AUC | vs LR |
|---|---|---|
| Logistic Regression | 0.651 | — |
| GBM (100 trees) | 0.721 | +0.070 |

GBM reliably outperforms LR due to nonlinear interactions in the DGP (latent trait interactions with engagement patterns).

### Value-aware ranking

| K | By P(convert) | By expected value | Uplift |
|---|---|---|---|
| 25 | $907,099 | $1,017,505 | +12.2% |
| 50 | $1,588,789 | $1,839,009 | +15.7% |

### Leakage trap evaluation (instructor dataset)

| Metric | Value |
|---|---|
| Column | `__leakage__touches_post_snapshot_21_90` |
| Seeds | 10 (42–51) |
| Mean AUC delta | 0.061 |
| Min AUC delta | 0.027 |
| Max AUC delta | 0.080 |

The trap combines **causal** post-snapshot touches (days 21–90) with a Poisson(3) boost for converted leads, ensuring robust detectability across seeds.

---

## Teaching guidance

### Lecture 1: Pipeline + Evaluation

**Goal**: Students build their first ML pipeline and learn proper evaluation.

- Load `lead_scoring_intro_v6.csv`
- Handle missing values (5 columns have NaN — discuss structural vs MCAR)
- Build a baseline logistic regression with train/test split
- Evaluate: AUC, PR-AUC, confusion matrix
- Discuss class imbalance (30% positive rate)

### Lecture 2: Top-K + Expected Value Ranking

**Goal**: Students learn decision-oriented evaluation.

- Precision@K and Lift@K: "If sales can contact 50 leads, how many convert?"
- Expected value ranking: `P(convert) * expected_acv`
- Demonstrate that EV ranking captures 12–16% more ACV than probability ranking
- Discuss when value-aware scoring matters (heterogeneous deal sizes)

### Lecture 3: Feature Engineering + Error Slicing

**Goal**: Students learn to improve models through feature understanding.

- Examine `touches_last_7_days` (momentum) vs `touches_week_1` (early signal)
- Error analysis by `region`, `company_size`, `lead_source`
- Missing value patterns: why is `web_sessions` missing more for SDR outbound?
- Feature interactions: `opportunity_created` x `touches_last_7_days`

### Lecture 4: Trees/GBM + Nonlinearity (+ optional cohort shift)

**Goal**: Students see why tree models outperform linear models.

- Train GBM, compare AUC vs LR (+0.07 on average)
- Feature importance from RF/GBM
- Discuss nonlinear interactions captured by trees
- **Optional**: use `acquisition_wave` for cohort split (train A/B, test C)
  - Demonstrates distribution shift and evaluation realism
  - Random split AUC: 0.687, Cohort split AUC: 0.594 (AUC drop: +0.093 — demonstrates real distribution shift)

### Instructor note: Leakage detection exercise

Use `lead_scoring_intro_v6_instructor.csv` for a leakage detection exercise:
- Students train with all columns including `__leakage__touches_post_snapshot_21_90`
- AUC jumps by ~0.061 on average
- Challenge: identify which column is leaking and explain why
- The trap is causally grounded (future engagement correlates with conversion via shared latent traits), making it a realistic example of temporal leakage

# Feature dictionary — `leadforge-lead-scoring-v1`

Narrative companion to the per-tier `feature_dictionary.csv` shipped
inside each public bundle. The CSV is the authoritative
machine-readable spec (column / dtype / description / category /
target flag / leakage flag); this document groups features by
analytical role and adds the prose explanation, modelling
recommendations, and pedagogical caveats that don't fit a CSV row.

The grouping below covers every feature in the public student-facing
snapshot — the same 31 columns ship in `intro`, `intermediate`, and
`advanced` bundles. The instructor companion adds the hidden truth
in `metadata/`; it does not change the feature list.

| Category | Columns | Modelling default |
|---|---|---|
| Lead identity & timing | 4 | drop `lead_id`; keep `lead_created_at` for cohort splits, drop for production |
| Lead source & channel | 1 | keep |
| Firmographics | 5 | keep all |
| Personographics | 3 | keep all (categorical encoders welcome) |
| Engagement (snapshot-window) | 10 | keep all |
| Funnel & sales-process | 4 | keep all |
| Value | 2 | keep all |
| Leakage trap | 1 | **drop** unless deliberately demonstrating leakage |
| Target | 1 | label — never used as a feature |

## Lead identity and timing

| Column | Dtype | Source | Modelling notes |
|---|---|---|---|
| `lead_id` | string | identity | Opaque, deterministic per run; not informative. Use as a join key or row index, never as a feature. |
| `account_id` | string | identity | Foreign key into `tables/accounts.parquet`. Out-of-sample accounts may appear in test; if you fit account-level features, watch for cold-start. |
| `contact_id` | string | identity | Foreign key into `tables/contacts.parquet`. Same warning. |
| `lead_created_at` | string (ISO-8601) | simulation clock | Lead birthday; useful for cohort/time-shift evaluation (see `docs/release/v1_acceptance_gates.md` G6.4). Drop or bin it for production models — feeding raw timestamps to a linear model is rarely what you want. |

## Lead source and channel

One column describes how each lead entered the funnel. It is
populated from the recipe's GTM-motion mix
(`inbound_marketing` 45%, `sdr_outbound` 35%, `partner_referral` 20%).

| Column | Dtype | Why it might matter |
|---|---|---|
| `lead_source` | string | Origination channel; one of `inbound_marketing` / `sdr_outbound` / `partner_referral`. |

**Note.** `first_touch_channel` was removed from the task snapshot in PR 8.1: in v1
it is byte-identical to `lead_source` (both are set to the same origination value), so
it adds no information. It still appears in the relational `tables/leads.parquet` for
post-v1 use cases where origination and first-touch can diverge.

**Caveat.** Per [`docs/release/channel_signal_audit.md`](channel_signal_audit.md),
v1's channel signal is weak: per-channel rate spread ≤ 0.043 and
univariate AUC ≤ 0.521 across all tiers, well below the G2 /
Gemini v2 industry MQL→SQL band (SEO ~51%, PPC ~26%, Email <1%).
Expect modest feature importance from these columns; do not expect
channel to be a top-tier predictor in v1.

## Firmographics (account-level)

These describe the buying organisation. They come from the recipe's
narrative spec (industry, region, employee bands, revenue bands)
and from latent traits sampled per account. Five columns plus the
`account_id` foreign key listed under "Lead identity and timing"
above; all five are fair to use as features.

| Column | Dtype | Why it might matter |
|---|---|---|
| `industry` | string | Categorical mix is fixed by the recipe (`manufacturing`, `logistics`, `professional_services`, `healthcare_non_clinical`); motif-family latent biases create modest cross-industry conversion-rate differences. |
| `region` | string | `US` / `UK`. Currently a low-signal axis — the simulator does not model channel-by-region interactions. |
| `employee_band` | string | Bands are aligned with the ICP range (200–2,000 employees, plus tails). Larger accounts trend toward higher expected ACV. |
| `estimated_revenue_band` | string | Bands span `$1M-$10M` to `$200M+`; correlated with `employee_band` by design. |
| `process_maturity_band` | string | A discretisation of the latent `process_maturity` trait — *visible* signal of `motif_family.fit_dominant`'s "fit beats engagement" story. |

## Personographics (contact-level)

These describe the primary contact attached to the lead. Three
categorical features (the `contact_id` foreign key is listed
under "Lead identity and timing"); all three are fair to use.

| Column | Dtype | Why it might matter |
|---|---|---|
| `role_function` | string | Functional area: `finance`, `ops`, `it`, `procurement`. Drives demo-page views and the demo/trial path through `motif_family.demo_trial_mediated`. |
| `seniority` | string | `c_suite` / `vp` / `director` / `manager` / `individual_contributor`. Strongly correlated with the latent `contact_authority` trait that gates `motif_family.buying_committee_friction`. |
| `buyer_role` | string | `economic_buyer`, `champion`, `technical_evaluator`, `end_user`. Hand-mapped from `role_function` × `seniority`. |

## Engagement (snapshot-window aggregates)

Ten engagement features computed strictly over events on days
`[0, snapshot_day]` (with `snapshot_day = 30` for v1). The simulator
emits touches, sessions, and page views every day from
`lead_created_at` onward; the renderer aggregates them up to but
not past day 30. The 90-day label window resolves separately, so
features cannot encode events that drove the late-window outcome.

| Column | Dtype | What it captures |
|---|---|---|
| `touch_count` | Int64 | All marketing/sales touches in the snapshot window. |
| `inbound_touch_count` | Int64 | Inbound touches only. |
| `outbound_touch_count` | Int64 | Outbound touches only. |
| `session_count` | Int64 | Web/trial session count. |
| `pricing_page_views` | Int64 | Cumulative pricing-page views across sessions. |
| `demo_page_views` | Int64 | Cumulative demo-page views across sessions. |
| `total_session_duration_seconds` | Int64 | Cumulative seconds across all sessions. |
| `touches_days_0_7` | Int64 | Touches in days 0–7 inclusive (early urgency proxy). Renamed from `touches_week_1` in PR 8.1 for precision: the window covers 8 day values (0, 1, …, 7). |
| `touches_last_7_days` | Int64 | Touches in the last 7 days of the snapshot window — for `snapshot_day=30`, days 24–30 inclusive (the snapshot builder uses `_day > snapshot_day - 7`). |
| `days_since_first_touch` | Float64 | NaN if the lead has had zero touches by snapshot day. |

## Funnel and sales-process

The funnel state at snapshot day, exposed via four columns. None of
these are terminal stages — `current_stage` (which can encode
`closed_won` / `closed_lost`) is redacted from public bundles via
the exposure layer.

| Column | Dtype | What it captures |
|---|---|---|
| `activity_count` | Int64 | Sales-activity events (calls, demos, follow-ups) in the snapshot window. |
| `days_since_last_touch` | Float64 | Recency of the most recent touch; NaN if zero touches. |
| `opportunity_created` | boolean | Whether *any* opportunity was created by snapshot day, regardless of state. |
| `has_open_opportunity` | boolean | Whether an opportunity existed in an open stage at snapshot day. |

## Value

Two value features. Both are useful as inputs to value-aware
ranking (`expected_acv × P(convert)`); see notebook 4 once Phase 6
ships.

| Column | Dtype | What it captures |
|---|---|---|
| `opportunity_estimated_acv` | Float64 | Estimated ACV of the most recent open opportunity at snapshot day; NaN if no opportunity. |
| `expected_acv` | Float64 | Falls back to a revenue-band midpoint heuristic when no opportunity exists, so it has fewer NaNs than `opportunity_estimated_acv`. |

## Leakage trap (deliberate)

| Column | Dtype | Why it ships |
|---|---|---|
| `total_touches_all` | Int64 | Counts touches across the full 90-day horizon — not the snapshot window. Flagged `leakage_risk=True` in the CSV (the per-bundle dictionary has columns `name,dtype,description,category,is_target,leakage_risk`); documented in `release/README.md`. The gap `total_touches_all − touch_count` carries label-correlated signal because high-converting leads accumulate more late-window touches in the simulator. **Drop this column from your features unless you are explicitly demonstrating leakage detection.** |

## Target

| Column | Dtype | Definition |
|---|---|---|
| `converted_within_90_days` | boolean | True iff a `closed_won` event occurred within 90 days of `lead_created_at`. Derived from simulated events; never sampled directly. |

## Difficulty modulation

Difficulty profiles distort the same feature set with different
parameters; columns and dtypes are identical across tiers. The
distortions are applied in `leadforge/render/snapshots.py` via
`_apply_difficulty_distortions()`:

- **Gaussian noise** on float features. `intro` 0.10, `intermediate`
  0.30, `advanced` 0.55 (multipliers applied to per-feature
  standard deviations).
- **MCAR missingness.** `intro` 2%, `intermediate` 8%,
  `advanced` 18%.
- **Outlier injection** at the same per-tier rate as missingness.
- **Signal strength.** Latent-score weights are multiplied by 0.90
  (`intro`), 0.70 (`intermediate`), and 0.50 (`advanced`),
  weakening the link between latent traits and conversion as
  difficulty rises.

The conversion-rate band for each tier is recipe-defined; observed
medians across the canonical seed sweep (42–46) are
0.4267 (`intro`), 0.2160 (`intermediate`), 0.0840 (`advanced`).
See `release/validation/validation_report.md` for the full
cross-seed × cross-tier metrics panel.

## Recommended modelling defaults

A short opinionated checklist for a first model. Note: the flat
`lead_scoring.csv` and the per-task Parquet splits ship every column
in the table above, including the IDs — the recommendation is what to
**use as features**, not what's in the file.

1. **Identifiers — drop before fitting.** `lead_id` is opaque and
   carries no signal; drop it. `account_id` / `contact_id` are joinable
   keys, useful only when you're computing cross-table aggregates;
   drop from the feature matrix unless you actually use them. Drop or
   bin `lead_created_at` — feeding raw timestamps to a linear model
   is rarely what you want; use it as the cohort key for time-shift
   evaluation instead.
2. **Trap — drop.** `total_touches_all` is the deliberate leakage
   trap. Drop unless you're demonstrating leakage detection.
3. **Categoricals — encode.** One-hot or target-encode `industry`,
   `region`, `employee_band`, `estimated_revenue_band`,
   `process_maturity_band`, `role_function`, `seniority`,
   `buyer_role`, `lead_source`.
   (`first_touch_channel` was removed from the snapshot in PR 8.1 — it
   was byte-identical to `lead_source` in v1; it still exists in
   `tables/leads.parquet` but not in the task splits.)
4. **Engagement and funnel — keep all.** The `Float64` columns carry
   NaN for "no event in window", which is itself a signal — encode
   missingness explicitly rather than imputing to zero blindly.
5. **Value-aware ranking.** Use `expected_acv` over
   `opportunity_estimated_acv`; the latter is missing for leads
   without an opportunity. Multiply by your model's predicted
   probability for a default value-weighted ranker.
6. **Cohort evaluation.** Sort by `lead_created_at` and split
   chronologically; the random-split AUC is *not* the right number to
   report if your downstream use is forecasting.

## See also

- `release/{intro,intermediate,advanced}/feature_dictionary.csv` —
  the authoritative machine-readable spec, regenerated with each
  bundle.
- `release/README.md` — the dataset card.
- `docs/release/generation_method.md` — how the underlying
  events are generated.
- `docs/release/channel_signal_audit.md` — how strongly each
  channel column signals conversion in v1.
- `release/validation/validation_report.md` — calibration, lift,
  P@K, model-family deltas, cross-seed bands.

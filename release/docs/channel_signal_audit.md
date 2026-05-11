# Channel-signal audit — leadforge-lead-scoring-v1

Audit produced by `scripts/audit_channel_signal.py`; see `channel_signal_audit.json` for the machine-readable form.

**Scope.** For every tier we compute per-channel conversion rates on the train split and the univariate AUC of channel against `converted_within_90_days`, scored as the empirical positive rate per channel (a 1-D Bayes classifier). Two AUCs are reported: an **in-sample** number (train rates → train labels — biased upward by construction) and an **out-of-sample** number (train rates → test labels — directly comparable to the `source_only` baselines in `release/validation/validation_report.json`).

**Caveat on the industry benchmark.** The G2 / Gemini v2 numbers below are single-step **MQL→SQL** rates (recommendation #8 in `docs/external_review/summaries/recommendations_pass.md`). v1's label is **90-day closed-won**, the entire funnel resolved. The two metrics are not directly comparable; the table is reproduced for context only.

## Industry benchmark (context, not target)

| Channel | MQL→SQL conversion rate |
|---|---|
| Email | 0.50% |
| PPC | 26.00% |
| SEO | 51.00% |

## Tier: `intro`

`n_train = 3500` (90-day conversion rate 41.46%); `n_test = 750` (rate 42.67%).

### Columns: `lead_source`, `first_touch_channel` (audit values identical)

Per-channel rate spread (max − min): **0.0433**  ·  In-sample univariate AUC: **0.5200**  ·  Out-of-sample univariate AUC: **0.5014**

| Channel | n (train) | Share (train) | Converted (train) | Train rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 682 | 43.44% |
| `partner_referral` | 698 | 19.94% | 273 | 39.11% |
| `sdr_outbound` | 1232 | 35.20% | 496 | 40.26% |

## Tier: `intermediate`

`n_train = 3500` (90-day conversion rate 20.14%); `n_test = 750` (rate 22.27%).

### Columns: `lead_source`, `first_touch_channel` (audit values identical)

Per-channel rate spread (max − min): **0.0365**  ·  In-sample univariate AUC: **0.5212**  ·  Out-of-sample univariate AUC: **0.5139**

| Channel | n (train) | Share (train) | Converted (train) | Train rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 334 | 21.27% |
| `partner_referral` | 698 | 19.94% | 123 | 17.62% |
| `sdr_outbound` | 1232 | 35.20% | 248 | 20.13% |

## Tier: `advanced`

`n_train = 3500` (90-day conversion rate 7.91%); `n_test = 750` (rate 7.87%).

### Columns: `lead_source`, `first_touch_channel` (audit values identical)

Per-channel rate spread (max − min): **0.0056**  ·  In-sample univariate AUC: **0.5083**  ·  Out-of-sample univariate AUC: **0.5226**

| Channel | n (train) | Share (train) | Converted (train) | Train rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 128 | 8.15% |
| `partner_referral` | 698 | 19.94% | 53 | 7.59% |
| `sdr_outbound` | 1232 | 35.20% | 96 | 7.79% |

## Discussion

The numbers above answer one question: *how strongly does channel alone signal 90-day conversion in v1?* They do not answer *whether v1 matches industry channel performance*, since the benchmarks measure a different funnel transition (single MQL→SQL step) and v1 measures the entire funnel resolved over 90 days. Treat the v1 numbers as an internal description of the simulator's channel signal.

Two empirical observations a reader can make from the numbers above:

1. **The out-of-sample univariate AUC is the comparable number** for any external baseline. It uses train-derived rates scored against held-out test labels — the same shape as the `source_only` HistGBM baseline reported in `release/validation/validation_report.json`, which is built on the same task splits with `lead_source` + `first_touch_channel` as the only features. The in-sample number is biased upward by construction — small at v1's N but visible — and is reported here for transparency rather than comparison.
2. **The numerical conclusion is bundle-specific.** When the per-channel rate spread is small and the OOS univariate AUC is close to chance, channel alone is a weak feature for the bundle this audit was run against. v1's bundles currently produce that outcome (see the per-tier sections above) — consistent with the design: the simulator drives conversion through motif-family hazards keyed off latent traits, not channel-conditional probabilities. Channel-conditional encoding is tracked as post-v1 work in `docs/release/post_v1_roadmap.md`.

# Channel-signal audit — leadforge-lead-scoring-v1

Audit produced by `scripts/audit_channel_signal.py`; see also `docs/release/channel_signal_audit.json` for the machine-readable form.

**Scope.** For every tier we compute per-channel conversion rates and the univariate AUC of channel against `converted_within_90_days`, scored as the empirical positive rate per channel (a 1-D Bayes classifier, equivalent to a saturated logistic regression on one-hot channel features). Compared against the G2 / Gemini v2 industry MQL→SQL benchmark band (SEO ~51%, PPC ~26%, Email <1%, surfaced in `docs/external_review/summaries/recommendations_pass.md` recommendation #8).

**Caveat.** Industry benchmarks are MQL→SQL rates, not 90-day closed-won rates. They are the closest public anchor for *how much* channel ought to matter; use them as a band of reference, not a hard target.

## Industry benchmark band

| Channel | MQL→SQL conversion rate |
|---|---|
| Email | 0.50% |
| PPC | 26.00% |
| SEO | 51.00% |

## Tier: `intro`

`n_leads = 3500`, overall 90-day conversion rate 41.46%.

### Column: `lead_source`

Univariate AUC: **0.5200**  ·  Per-channel rate spread (max − min): **0.0433**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 682 | 43.44% |
| `partner_referral` | 698 | 19.94% | 273 | 39.11% |
| `sdr_outbound` | 1232 | 35.20% | 496 | 40.26% |

### Column: `first_touch_channel`

Univariate AUC: **0.5200**  ·  Per-channel rate spread (max − min): **0.0433**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 682 | 43.44% |
| `partner_referral` | 698 | 19.94% | 273 | 39.11% |
| `sdr_outbound` | 1232 | 35.20% | 496 | 40.26% |

## Tier: `intermediate`

`n_leads = 3500`, overall 90-day conversion rate 20.14%.

### Column: `lead_source`

Univariate AUC: **0.5212**  ·  Per-channel rate spread (max − min): **0.0365**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 334 | 21.27% |
| `partner_referral` | 698 | 19.94% | 123 | 17.62% |
| `sdr_outbound` | 1232 | 35.20% | 248 | 20.13% |

### Column: `first_touch_channel`

Univariate AUC: **0.5212**  ·  Per-channel rate spread (max − min): **0.0365**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 334 | 21.27% |
| `partner_referral` | 698 | 19.94% | 123 | 17.62% |
| `sdr_outbound` | 1232 | 35.20% | 248 | 20.13% |

## Tier: `advanced`

`n_leads = 3500`, overall 90-day conversion rate 7.91%.

### Column: `lead_source`

Univariate AUC: **0.5083**  ·  Per-channel rate spread (max − min): **0.0056**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 128 | 8.15% |
| `partner_referral` | 698 | 19.94% | 53 | 7.59% |
| `sdr_outbound` | 1232 | 35.20% | 96 | 7.79% |

### Column: `first_touch_channel`

Univariate AUC: **0.5083**  ·  Per-channel rate spread (max − min): **0.0056**  ·  Verdict: **weak signal**

| Channel | n | Share | Converted | Conversion rate |
|---|---:|---:|---:|---:|
| `inbound_marketing` | 1570 | 44.86% | 128 | 8.15% |
| `partner_referral` | 698 | 19.94% | 53 | 7.59% |
| `sdr_outbound` | 1232 | 35.20% | 96 | 7.79% |

## Verdict

v1's channel signal is **weak**: across all tiers and both channel columns the largest per-channel conversion-rate spread is 0.043 and the largest univariate AUC is 0.521. That is well below the G2 / Gemini v2 industry MQL→SQL benchmark band, where SEO leads convert 50 percentage points more than Email leads. v1 drives conversion through motif-family hazards keyed off latent traits, not channel-conditional probabilities, so this is the expected outcome; channel-conditional encoding is tracked as post-v1 work in `docs/release/post_v1_roadmap.md`.

# v4 Dataset Contract

## Snapshot definition

- **Snapshot day:** Day 21 after `lead_created_at` (configurable, default 21).
- **Observation window:** Days 0–21 inclusive. All features computed from events in this window only.
- **Prediction horizon:** Days 22–90. The target `converted` reflects whether `closed_won` occurs in the full 90-day window.
- **Temporal guarantee:** No feature (except the explicitly marked leakage trap) uses information from after the snapshot day.

## What is pre-snapshot (valid for features)

| Data source | Temporal gate |
|---|---|
| Account attributes | Static — always valid |
| Contact attributes | Static — always valid |
| Lead metadata (source, etc.) | Lead creation — always valid |
| Touch events | `touch_timestamp ≤ lead_created_at + snapshot_day` |
| Session events | `session_timestamp ≤ lead_created_at + snapshot_day` |
| Sales activity events | `activity_timestamp ≤ lead_created_at + snapshot_day` |
| Opportunity records | `opportunity.created_at ≤ lead_created_at + snapshot_day` |
| ACV estimates | From opportunity if available by snapshot; else account heuristic |

## What is post-snapshot (invalid for features)

| Data | Why invalid |
|---|---|
| `current_stage` at day 90 | Contains `closed_won` / `closed_lost` — outcome data |
| `is_sql` (final state flag) | Engine invariant: `is_sql=False` → never converts. Deterministic. |
| `conversion_timestamp` | Direct outcome information |
| Touch/session/activity events after snapshot day | Future data |
| Opportunity close outcome | Post-outcome |
| `total_touches_all` | ⚠️ Intentional leakage trap — counts full 90-day touches |

## Leakage trap contract

The feature `total_touches_all` deliberately violates the snapshot boundary:
- It counts touches over the **full 90-day simulation**, not just up to snapshot.
- It is included to teach students about temporal leakage detection.
- It must be clearly marked in the feature dictionary and release notes.
- The validation script must detect it and flag it (but not fail the build).
- Removing this feature should drop AUC by ≥0.03.

## Missingness contract

| Column | Pattern | Rate | Condition |
|---|---|---|---|
| `days_since_last_touch` | Structural | Natural | NaN when `total touches == 0` by snapshot |
| `web_sessions` | Source-conditional | ~15% for `sdr_outbound`, ~2% for `inbound_marketing`, ~5% for `partner_referral` | CRM tracking gaps |
| `seniority` | Source-conditional | ~8% for `partner_referral`, ~1% for others | Referral partners omit contact details |
| `days_since_last_touch` | Additional MCAR | ~3% | Random CRM logging gaps (on top of structural) |

## Target definition

```
converted = 1  if lead reached closed_won within 90 days of lead_created_at
converted = 0  otherwise (including closed_lost, still in funnel, churned)
```

The target is derived from simulated events, never directly sampled.

## Subsampling contract

- Source bundle: 5,000 leads generated with `b2b_saas_procurement_v1`, seed 42, difficulty intro.
- Stratified subsampling to 1,000 rows at ~30% conversion rate.
- All negatives retained (up to 700); positives downsampled.
- Subsampling preserves within-class feature distributions.

## Reproducibility

- Seed: 42 (or documented if changed).
- All stochastic operations use `np.random.RandomState(seed)` or derived substreams.
- Same (seed, recipe, leadforge version) → byte-identical CSV output.

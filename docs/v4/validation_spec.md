# v4 Validation Specification

## Overview

v4 validation operates at two levels:
1. **Engine-level validation** (existing `leadforge validate` harness) — structural checks on bundles.
2. **Dataset-level validation** (new `scripts/validate_v4_dataset.py`) — checks specific to the simplified CSV output.

This document specifies the dataset-level validation for v4.

---

## Mandatory checks

### Check 1: No banned columns

The CSV must NOT contain any of:
- `current_stage`, `funnel_stage` — outcome-stage leakage
- `conversion_timestamp` — direct outcome
- `is_sql` — engine invariant creates deterministic groups
- `is_mql` — zero variance
- `lead_created_at` — timestamp that could be used to reverse-engineer temporal info
- Any column containing `_id` suffix (opaque identifiers, not features)

**Implementation:** Set intersection check on column names.

### Check 2: No deterministic feature groups

For every feature (categorical AND binary), for every value with n ≥ 50:
- Conversion rate must be in [0.02, 0.98].

This catches:
- `reached_sql=0` → 0% (caught in v2)
- `has_opportunity=1` → 0% (caught in v2)
- Any future deterministic pattern

**Implementation:** `groupby(feature)[target].agg(['mean', 'count'])`, filter to count ≥ 50, check bounds.

### Check 3: Conversion rate realism

- Overall conversion rate must be in [0.15, 0.40].

### Check 4: Baseline model AUC (without leakage trap)

- Train a logistic regression on all features EXCEPT `total_touches_all`.
- AUC must be in [0.65, 0.90].
- If AUC < 0.65: features lack signal (category effects too flat).
- If AUC > 0.90: likely residual leakage.

### Check 5: Leakage trap effectiveness

- Train a logistic regression with ALL features including `total_touches_all`.
- AUC must be at least 0.03 higher than the clean model from Check 4.
- If the trap doesn't boost AUC, it's not an effective teaching tool.

### Check 6: Missingness structure

- `web_sessions` must have nulls.
- Missing rate for `web_sessions` among `sdr_outbound` leads must be > 3× the rate among `inbound_marketing` leads.
- `seniority` must have nulls.
- Missing rate for `seniority` among `partner_referral` leads must be > 3× the rate among non-`partner_referral` leads.
- `days_since_last_touch` must have nulls.
- No column should have > 20% missing.

### Check 7: Shape constraints

- Exactly 1,000 rows.
- 18 columns (17 features + 1 target).

### Check 8: Reproducibility

- Running the build script twice with the same seed produces identical output (byte-level CSV comparison).

---

## Warning checks (non-fatal)

### Warning 1: Leakage trap is labeled

- Check that the feature dictionary (if present) marks `total_touches_all` with `leakage_risk: True`.

### Warning 2: Column redundancy

- Warn if `inbound_touches + outbound_touches` correlates > 0.99 with any other column.

### Warning 3: Low-variance features

- Warn if any feature has < 3 unique values (excluding binary features).

---

## Integration with existing validation

The engine-level `leadforge validate` harness (`validation/bundle_checks.py`) continues to validate the full Parquet bundle. The v4 dataset validator is a separate script for the simplified CSV output.

If engine changes add new features to `LEAD_SNAPSHOT_FEATURES`, the existing `validation/realism.py` checks (non-negative counts, valid booleans, stage diversity) automatically cover them.

---

## Validator script interface

```bash
python scripts/validate_v4_dataset.py lead_scoring_intro/lead_scoring_intro_v4.csv
```

Exit code 0 = all mandatory checks pass. Exit code 1 = at least one failure.

Output format: structured report showing each check name, status (PASS/FAIL/WARN), and details.

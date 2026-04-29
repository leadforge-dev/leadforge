# v4 Engine Changes Specification

## Overview

v4 requires **two categories** of changes to the leadforge codebase:
1. **Mechanism / difficulty tuning** — make intro difficulty produce stronger category-level signal.
2. **Snapshot builder enhancements** — compute windowed aggregates, ACV derivation, structured missingness, and the leakage trap feature.

Neither category requires changes to the simulation loop itself (`engine.py`'s daily step logic). The simulation produces the same event stream; we change how features are derived from it.

---

## Change 1: Stronger category signal at intro difficulty

### Problem

The current mechanism policy (`mechanisms/policies.py`) produces conversion rates that are nearly uniform across categories at intro difficulty. For example, `contact_role` spreads only 11% (25.6%–36.7% after subsampling). This yields LR AUC ~0.62, which is too low for a useful teaching dataset.

### Root cause

The `assign_mechanisms()` function builds a `LatentScore` with weights that are quite flat across categories. The intro difficulty profile specifies `signal_strength: 0.90` but this controls noise scale, not the magnitude of category effects.

### Solution

Add **category effect multipliers** to the difficulty profile YAML:

```yaml
intro:
  # ... existing fields ...
  category_effect_scale: 1.8  # amplify category → latent score effects
```

In `mechanisms/policies.py`, scale the `CategoricalInfluence` weights by `category_effect_scale` when building the `LatentScore`. This widens the gap between, say, `vp_finance` and `it_director` conversion rates without changing the overall noise structure.

### Target outcome

After this change + subsampling to 30%, category spreads should be:
- `contact_role`: ≥15% spread
- `company_revenue`: ≥12% spread
- `seniority`: ≥10% spread
- Baseline LR AUC: 0.70–0.85

### Files affected

- `leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml` — add `category_effect_scale`
- `leadforge/mechanisms/policies.py` — use `category_effect_scale` when building categorical influences
- `tests/mechanisms/test_policies.py` — test that different scales produce different spread

### Risk

Low. The change is additive (new config field with a default of 1.0 for backward compatibility). Existing tests continue to pass at `category_effect_scale=1.0`.

---

## Change 2: Snapshot builder — windowed aggregates and new features

### Problem

The current `render/snapshots.py` computes all aggregates over the full simulation horizon. v4 needs aggregates gated by a configurable snapshot day, plus new derived features.

### Solution

Add a new function or extend `build_snapshot()` to accept a `snapshot_day` parameter:

```python
def build_snapshot(
    result: SimulationResult,
    population: PopulationResult,
    horizon_days: int = 90,
    snapshot_day: int | None = None,  # NEW — default None means use horizon_days
) -> pd.DataFrame:
```

When `snapshot_day` is set, all event aggregations filter to events within `[lead_created_at, lead_created_at + snapshot_day]`.

### New features to compute

| Feature | Computation | Notes |
|---|---|---|
| `touches_week_1` | Count touches where `days_after_creation ≤ 7` | Momentum signal |
| `days_since_first_touch` | `snapshot_day - first_touch_day` (NaN if no touches) | Lead age signal |
| `expected_acv` | Opportunity ACV if opp created by snapshot; else employee_band midpoint | Value feature |
| `total_touches_all` | Count of ALL touches over full horizon (ignoring snapshot gate) | Leakage trap |

### Files affected

- `leadforge/render/snapshots.py` — add `snapshot_day` parameter, windowed filtering, new feature computations
- `leadforge/schema/features.py` — add new `FeatureSpec` entries for the new columns
- `tests/render/test_snapshots.py` — test windowed aggregation correctness

### Risk

Medium. The snapshot builder is well-tested but core to correctness. The `snapshot_day` parameter should be additive (default `None` preserves existing behavior). New features are computed alongside existing ones.

---

## Change 3: Structured missingness injection

### Problem

Current missingness is MCAR (random injection). v4 needs conditional missingness.

### Solution

Add a missingness injection step to the v4 build script (NOT to the engine's `build_snapshot`). This keeps the engine's output clean and makes missingness a dataset-packaging concern.

The build script (`scripts/build_v4_snapshot.py`) applies missingness after snapshot construction:

```python
def inject_missingness(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    # 1. web_sessions: 15% missing for sdr_outbound, 2% inbound, 5% partner
    for source, rate in [("sdr_outbound", 0.15), ("inbound_marketing", 0.02), ("partner_referral", 0.05)]:
        mask = (df["lead_source"] == source) & (rng.random(len(df)) < rate)
        df.loc[mask, "web_sessions"] = np.nan

    # 2. seniority: 8% missing for partner_referral, 1% for others
    partner_mask = (df["lead_source"] == "partner_referral") & (rng.random(len(df)) < 0.08)
    other_mask = (df["lead_source"] != "partner_referral") & (rng.random(len(df)) < 0.01)
    df.loc[partner_mask | other_mask, "seniority"] = np.nan

    # 3. days_since_last_touch: additional 3% MCAR on top of structural NaN
    dslt_mask = rng.random(len(df)) < 0.03
    df.loc[dslt_mask, "days_since_last_touch"] = np.nan

    return df
```

### Files affected

- `scripts/build_v4_snapshot.py` (new) — missingness injection
- No changes to `leadforge/` core modules for missingness

### Risk

Low. Missingness is applied post-generation, outside the engine.

---

## Change 4: Leakage trap feature

### Problem

Students need a feature that looks valid but violates temporal boundaries.

### Solution

The v4 build script computes `total_touches_all` by counting ALL touches in the full 90-day window (not gated by snapshot day). This is computed alongside the snapshot but uses different temporal filtering.

### Files affected

- `scripts/build_v4_snapshot.py` — compute `total_touches_all` from full event stream
- Feature dictionary and release notes — mark as leakage trap

### Risk

None to the engine. The trap is a build-script concern.

---

## Summary of engine vs. script changes

| Change | Where | Risk |
|---|---|---|
| Category effect scaling | `leadforge/` core (mechanisms, difficulty profiles) | Low |
| Snapshot `snapshot_day` parameter | `leadforge/` core (render/snapshots) | Medium |
| New features (ACV, momentum, first_touch) | `leadforge/` core (render/snapshots, schema/features) | Medium |
| Structured missingness | `scripts/` (build script only) | Low |
| Leakage trap | `scripts/` (build script only) | None |

Total engine-side changes: ~200–400 lines across 4–5 files. Build script: ~250 lines new.

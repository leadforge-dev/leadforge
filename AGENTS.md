# AGENTS.md — leadforge

Agent-specific conventions layered on top of CLAUDE.md.

---

## PR Review Comment Workflow

When addressing PR review comments (Copilot, human reviewers, or otherwise):

1. Triage each comment — recommend one of: resolve as irrelevant, accept and implement, open a separate issue and resolve as out-of-scope, accept a different solution, or resolve as already treated.
2. After the user confirms decisions, implement accepted changes and push **all changes in a single commit** to the PR branch.
3. **After the commit lands, resolve the corresponding GitHub review threads** using the GraphQL API:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "PRRT_..."}) { thread { isResolved } } }'
```

Resolve every addressed thread — whether the action was "implement", "already treated", or "irrelevant/out-of-scope". Unresolved threads indicate open work; resolved threads mean the discussion is closed.

Do **not** leave threads unresolved after the commit is pushed.

---

## Branch & PR Conventions

See CLAUDE.md for the full mandatory branch/PR workflow (branch → commit → update `.agent-plan.md` → open PR).

---

## v4 Implementation Guide

### What is v4?

A pedagogically improved lead scoring dataset (single CSV) for an intro ML course. The engine changes are small and targeted. See `docs/v4/` for full specs.

### Implementation order

```
v4-M0 (planning PR — already done)
  └── v4-M1: engine changes (category signal + windowed snapshots)
        └── v4-M2: build pipeline + validation scripts
              └── v4-M3: dataset generation + release docs
```

### Key files to modify per milestone

**v4-M1 (engine):**
- `leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml` — add `category_effect_scale`
- `leadforge/mechanisms/policies.py` — apply scale to categorical influences
- `leadforge/render/snapshots.py` — add `snapshot_day` param, windowed aggregation
- `leadforge/schema/features.py` — add new FeatureSpec entries
- Tests in `tests/mechanisms/` and `tests/render/`

**v4-M2 (build pipeline):**
- `scripts/build_v4_snapshot.py` (new) — snapshot builder with missingness + leakage trap
- `scripts/validate_v4_dataset.py` (new) — dataset-level validation
- These live in the leadforge repo (not datasets-private)

**v4-M3 (release):**
- Work in `leadforge-datasets-private` repo
- `lead_scoring_intro/lead_scoring_intro_v4.csv`
- `lead_scoring_intro/RELEASE_v4.md`

### Coding conventions for v4

1. **Backward compatibility:** All engine changes must default to current behavior. New parameters must have defaults that produce identical output when unset.
2. **No simulation loop changes:** Do not modify the daily step logic in `engine.py`. v4 changes are in mechanism weights and snapshot rendering only.
3. **Temporal correctness:** Every feature computation must be explicitly gated by snapshot day. Use `event_timestamp <= lead_created_at + snapshot_day` — never `<`.
4. **Test coverage:** Every new parameter and feature must have unit tests. Test both `snapshot_day=None` (backward compat) and `snapshot_day=21` (v4 mode).
5. **Determinism:** All new stochastic operations must use seeded RNG. Verify with a determinism test (same seed → identical output).

### Validation checklist for v4 dataset

Before declaring v4-M2 complete, the dataset must pass:

- [ ] 1,000 rows, 18 columns
- [ ] 30% conversion rate (±1%)
- [ ] No deterministic groups (n≥50 at 0% or 100% conversion)
- [ ] LR AUC 0.65–0.90 (without leakage trap)
- [ ] LR AUC boost ≥0.03 when leakage trap included
- [ ] `web_sessions` missingness: outbound rate > 3× inbound rate
- [ ] `seniority` missingness: partner_referral rate > 3× others
- [ ] Reproducible with seed 42
- [ ] `total_touches_all` uses full 90-day data (confirmed by AUC boost)

### How to test engine changes locally

```bash
# Quick smoke test: generate a small bundle and inspect
leadforge generate --recipe b2b_saas_procurement_v1 --seed 42 --difficulty intro --n-leads 1000 --out /tmp/test_bundle
leadforge validate /tmp/test_bundle

# Check category signal spread
python -c "
import pandas as pd
df = pd.read_parquet('/tmp/test_bundle/tasks/converted_within_90_days/train.parquet')
for col in ['role_function', 'seniority', 'estimated_revenue_band']:
    rates = df.groupby(col)['converted_within_90_days'].mean()
    print(f'{col}: spread={rates.max()-rates.min():.1%}')
"
```

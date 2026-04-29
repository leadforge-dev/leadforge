# v4 Implementation Plan

## Overview

This plan implements the v4 lead scoring dataset in 4 milestones across 4–6 PRs. Each milestone produces testable artifacts and has explicit acceptance criteria.

## Relationship to existing roadmap

v4 work slots into the existing leadforge roadmap as follows:

| Existing milestone | Status | v4 interaction |
|---|---|---|
| M0–M11 | ✅ Complete | No changes needed |
| M12 (CLI polish) | ⬜ Planned | **Deferred** — low priority vs v4 dataset needs. Integrate after v4. |
| M13 (Validation harness) | ✅ Implemented as M11 | v4 extends with dataset-level validation |
| M14 (Sample datasets + notebooks) | ⬜ Planned | **Absorbed into v4-M3** — v4 dataset IS the sample dataset |
| M15 (Docs polish + v1.0 RC) | ⬜ Planned | **Deferred** — do after v4 ships |

### Explicitly discarded items

| Item | Rationale |
|---|---|
| M12 `--json` flag for inspect/validate | Nice-to-have; no dataset consumer needs it yet. Can add later. |
| M12 `--strict` flag for validate | Validation strictness is better controlled per-check, not globally. |
| M14 Notebook 3 (public vs instructor comparison) | No current audience for this; instructor mode is not used in the course. |
| M14 Notebook 4 (recipe customization walkthrough) | Premature — recipe system is stable but not user-facing yet. |

### Explicitly kept / integrated items

| Item | How it maps to v4 |
|---|---|
| M14 Sample bundle generation | v4-M2 generates the source bundle |
| M14 Lead-scoring baseline notebook | v4-M3 includes a validation notebook or script |
| M15 Docs audit | v4-M0 updates CLAUDE.md and AGENTS.md; v4-M3 produces RELEASE_v4.md |

---

## v4 Milestones

### v4-M0: Requirements, contract, and agent instructions

**Goal:** Establish the v4 dataset contract and update repo documentation so implementation can begin immediately.

**Deliverables:**
- `docs/v4/lead_scoring_v4_requirements.md` — full requirements
- `docs/v4/dataset_contract.md` — schema contract, temporal gates, missingness
- `docs/v4/validation_spec.md` — automated check specifications
- `docs/v4/engine_changes_spec.md` — what changes where and why
- `docs/v4/implementation_plan.md` — this file
- Updated `CLAUDE.md` — repository map, generation/validation commands
- Updated `AGENTS.md` — implementation conventions for v4 work
- Updated `.agent-plan.md` — reflects v4 as next work

**Acceptance criteria:**
- [ ] All docs are internally consistent
- [ ] CLAUDE.md contains repo map and commands
- [ ] .agent-plan.md points to v4 milestones
- [ ] No contradictions with existing architecture docs

**PR:** This PR (the planning PR).

---

### v4-M1: Engine — category signal tuning + snapshot enhancements

**Goal:** Make the engine produce datasets with stronger category signal and support windowed snapshot computation.

**Deliverables:**
1. `difficulty_profiles.yaml` — add `category_effect_scale: 1.8` to intro profile
2. `mechanisms/policies.py` — apply `category_effect_scale` to categorical influence weights
3. `render/snapshots.py` — add optional `snapshot_day` parameter for windowed aggregation
4. `schema/features.py` — add `FeatureSpec` entries for new columns (`touches_week_1`, `days_since_first_touch`, `expected_acv`)
5. Tests for all changes

**Acceptance criteria:**
- [ ] `category_effect_scale=1.0` produces identical output to current engine (backward compat)
- [ ] `category_effect_scale=1.8` produces category spreads ≥15% for `contact_role`
- [ ] `snapshot_day=21` correctly filters events to first 21 days
- [ ] `touches_week_1` counts only days 0–7 touches
- [ ] `expected_acv` uses opportunity ACV when available, else band midpoint
- [ ] All existing tests pass
- [ ] New tests cover the new parameters

**Estimated size:** ~400 lines diff across 5 files + tests.

**PR:** Single PR: `feat: v4 engine — category signal tuning + windowed snapshots`

---

### v4-M2: Build pipeline — v4 snapshot builder + structured missingness

**Goal:** Create the v4 build script that transforms a generated bundle into the final CSV.

**Deliverables:**
1. `scripts/build_v4_snapshot.py` — snapshot builder with:
   - Day-21 windowed features
   - Leakage trap feature (`total_touches_all`)
   - Structured missingness injection
   - Stratified subsampling to 1,000 rows / 30% conversion
   - Column selection and renaming
2. `scripts/validate_v4_dataset.py` — validation script per validation spec
3. Generated `lead_scoring_intro_v4.csv` (in datasets repo, not leadforge)

**Acceptance criteria:**
- [ ] Build script produces 1,000 rows × 18 columns
- [ ] Conversion rate is 30% (±1%)
- [ ] `total_touches_all` uses full 90-day data (leakage trap)
- [ ] `web_sessions` missing rate for outbound > 3× inbound rate
- [ ] `seniority` missing rate for partner_referral > 3× others
- [ ] `days_since_last_touch` has structural + injected NaNs
- [ ] Validation script passes all mandatory checks
- [ ] Baseline LR AUC (without trap) in [0.65, 0.90]
- [ ] LR AUC boost with trap ≥ 0.03
- [ ] No deterministic groups (n≥50 at 0% or 100%)
- [ ] Reproducible with seed 42

**Estimated size:** ~350 lines (build script) + ~200 lines (validator).

**PR:** Single PR: `feat: v4 build pipeline + validation`

---

### v4-M3: Documentation + release

**Goal:** Produce the final dataset files and release documentation.

**Deliverables (in leadforge-datasets-private repo):**
1. `lead_scoring_intro/lead_scoring_intro_v4.csv`
2. `lead_scoring_intro/RELEASE_v4.md`
3. Updated `lead_scoring_intro/BACKGROUND.md` (if needed for v4 framing)
4. Updated `README.md` (dataset index)

**Deliverables (in leadforge repo):**
1. Updated `.agent-plan.md` reflecting completion

**Acceptance criteria:**
- [ ] CSV passes all validation checks
- [ ] RELEASE_v4.md documents snapshot day, target definition, changes from v3, leakage trap
- [ ] README in datasets repo marks v4 as recommended
- [ ] Previous versions marked as superseded

**PR:** Two PRs (one per repo).

---

## Dependency graph

```
v4-M0 (this PR)
  └── v4-M1 (engine changes)
        └── v4-M2 (build pipeline + validation)
              └── v4-M3 (docs + release)
```

Strictly sequential — each milestone depends on the previous.

---

## Timeline estimate

Not providing time estimates per project convention. The work is 4 PRs of moderate size (~300–500 lines each).

---

## What this plan does NOT do

- Does not change the simulation loop (`engine.py` daily step logic)
- Does not change the relational bundle format
- Does not change exposure modes
- Does not add new recipes
- Does not implement M12 (CLI polish) — deferred
- Does not implement the engine fix for `is_sql=False → never converts` (deferred to a separate issue; v4 avoids `is_sql` entirely)

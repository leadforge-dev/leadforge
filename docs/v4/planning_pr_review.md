# PR #19 Self-Review — Critical Assessment

## Point 1: Unvalidated numbers (`category_effect_scale`, AUC target)

**Issue:** `category_effect_scale: 1.8` appears in three docs as a known-good value, but has zero empirical backing. Similarly, `snapshot_day=21` gave AUC 0.622 in v3 — below the 0.65 floor mandated here. The plan has a circular dependency: AUC target assumes the engine change works, engine change spec assumes AUC target is reachable.

**Treatment:** Run a spike experiment before merging. Patch `category_effect_scale` into mechanism policy, generate a 5000-lead bundle at day-21 snapshot, measure category spread + LR AUC. Record results in engine_changes_spec. Adjust the number before it's enshrined across multiple documents.

---

## Point 2: Five overlapping spec documents

**Issue:** Five docs (~500 lines) to describe ~550 lines of implementation. Significant content overlap — "18 columns" appears in requirements, contract, AND validation spec. Missingness rates appear in three places. Changing one number means updating 3 files.

**Treatment:** Consolidate into two docs: one design doc (`docs/v4/design.md`) covering requirements, contract, and engine changes; one validation spec (`docs/v4/validation_spec.md`). Use a single source of truth for shared constants (column list, missingness rates, AUC bounds).

---

## Point 3: "No sim loop changes" masks a known bug

**Issue:** Repeatedly emphasizing "no changes to engine.py" as a feature, when the actual problem — `is_sql=False → 0% conversion` creating deterministic groups — lives there. The plan designs around a bug it refuses to fix, but doesn't state that clearly.

**Treatment:** Add an explicit "Known Limitations & Workarounds" section to the design doc. State plainly: v4's column set excludes `reached_sql` and `has_opportunity` because `is_sql=False → 0% conversion` is a simulation invariant we're choosing not to fix here. Link to a tracked issue for the future engine fix. Don't hide it in a deferred-items table.

---

## Point 4: Arbitrary missingness rates

**Issue:** Missingness rates (15%/2%/5% for web_sessions, 8%/1% for seniority) are unjustified. The validation check ("outbound > 3× inbound") is a tautology given the hardcoded rates.

**Treatment:** State the pedagogical rationale: rates must be detectable at n=1000 with a chi-squared test at p<0.01, but not so extreme students can't impute. Validate this claim in the spike experiment. Acknowledge these are tunable, not ground truth.

---

## Point 5: No failure mode handling

**Issue:** The plan assumes every parameter works on the first try. No guidance for what to do when AUC is too low/high, leakage trap doesn't boost, or subsampling destroys signal. For a dataset on v4 because v1–v3 had unforeseen problems, this is remarkably optimistic.

**Treatment:** Add a "Tuning Protocol" decision tree:
- AUC < 0.65 → increase `category_effect_scale` (try 2.0/2.5/3.0)
- AUC > 0.90 → decrease scale or add noise
- Leakage trap boost < 0.03 → widen snapshot window gap (day 14 instead of 21)
- Subsampling destroys signal → increase n_leads from 5000 to 10000

---

## Point 6: AGENTS.md will rot

**Issue:** v4-specific content (file-per-milestone lists, validation checklist, testing commands) hardcoded into a permanent repo doc. Becomes stale noise after v4 ships.

**Treatment:** Move v4-specific content into `docs/v4/` only. AGENTS.md keeps durable conventions plus a single pointer: "For v4 implementation details, see `docs/v4/`." Delete v4 content from AGENTS.md after v4 ships — or don't put it there in the first place.

---

## Point 7: `expected_acv` underspecified

**Issue:** "Opportunity ACV if opp created by snapshot; else band midpoint" — but what's the midpoint of "$100M+"? What if band is null? One table row in the longest spec doc.

**Treatment:** Define band→midpoint mapping as an explicit lookup table in the design doc. Specify null-band behavior (population median or NaN).

---

## Point 8: M1/M2 coupling is too rigid

**Issue:** M1 (engine knob) and M2 (build pipeline) are tightly coupled — M1 can't be validated without M2's build script. "Strictly sequential" milestones pretend otherwise. In practice, both will be developed with feedback loops.

**Treatment:** Merge M1 and M2 into a single milestone with two deliverables. One PR with both the engine knob and the build script, validated end-to-end, is more honest and more reviewable.

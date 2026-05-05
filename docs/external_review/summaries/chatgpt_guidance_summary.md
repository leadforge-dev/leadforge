# Summary — leadforge_second_attempt_guidance.md

**Source:** `docs/external_review/chatgpt/leadforge_second_attempt_guidance.md` (1167 lines, ~39 KB)
**Author:** ChatGPT (guidance document attached to the v2 retry as a constraint file)
**Verdict in one line:** The methodology spec that produced chatgpt_v2 — also independently useful for its dataset-forensics requirements (leakage probes) and required release-tree structure.

## Document role

A "how to do the v2 attempt correctly" instruction document. Required mandatory phases, file-paths-to-inspect tables, commands to run, web queries to issue, citation standards, and a rubric for accepting the result. Treated as a constraint file by the v2 author.

## Top points

1. **Mandatory methodology (7 phases):**
   - Phase 0 — extract Repomix, build evidence inventory
   - Phase 1 — static code audit by repo area (table maps area → files-to-inspect → questions-to-answer)
   - Phase 2 — dynamic reproducibility audit (install, run pytest, run the CLI, run the build script, record exit codes)
   - Phase 3 — alpha dataset forensic audit (manifest schema, row counts, splits, leakage probes)
   - Phase 4 — external research expedition (public dataset census, B2B realism, synthetic-data evaluation, platform requirements)
   - Phase 5 — gap matrix (area / current evidence / gap / severity / recommended fix / files+commands / acceptance)
   - Phase 6 — roadmap with files, commands, deliverables, acceptance, risks
2. **Dataset-forensics required probes (the v2 leakage finding came out of this):**
   - 8.1 Direct leakage: train w/ all features vs w/o suspect cols vs IDs vs post-snapshot aggregates; compare deltas.
   - 8.2 Time-window leakage: every public feature must derive from events ≤ `lead_created_at + snapshot_day`; label resolution uses full horizon but not as a feature source (except documented teaching traps).
   - 8.3 **Relational leakage**: opportunity status, customer/subscription rows that exist only for conversions, sales activities after snapshot, stage tables, join paths reconstructing `is_sql` / `current_stage` / terminal states. ← This is the probe class that surfaced THE blocker in v2.
   - 8.4 Split leakage: same account in train/test, same contact in train/test, near-duplicates across splits, temporal-split overlap.
   - 8.5 Model realism: AUC, PR-AUC, Brier, calibration, lift, P@K, R@K, top-decile, expected-value-at-K.
3. **Required release-tree structure** (the v1 release should look like a release, not a code dump): `dataset-cover-image.png`, `docs/{DATASET_CARD,GENERATION_METHOD,VALIDATION_REPORT,FEATURE_DICTIONARY,BREAK_ME_GUIDE,INSTRUCTOR_GUIDE}`, `data/{intro,intermediate,advanced}/{train,validation,test}.csv` + `relational/`, `instructor_companion/intermediate_instructor/`, `validation/` with figures, `notebooks/{01_baseline,02_relational,03_leakage,04_lift_calibration_value}`, `kaggle/dataset-metadata.json` + README, `huggingface/README.md`.
4. **LLM critique loop schema** (output JSON: release_id, model, run_timestamp, overall_score, findings[severity/category/claim/evidence/reproducer/suggested_fix], missing_sections[], questions_for_maintainer[]). Two model families recommended; raw outputs archived; high-severity findings adjudicated by humans before release.
5. **Pitfalls explicitly forbidden:** call implemented modules placeholders; recommend already-existing commands; treat HF card as absent; ignore the v7 track; use outdated platform requirements; treat AUC as the only metric; ignore relational leakage; ignore account/contact split overlap; over-plan LTV/leaderboard for v1 scope.
6. **Citation discipline:** repo evidence as `path:Lstart-Lend`, web evidence with title + URL + access date + exact fact supported. Mark unverified items explicitly.
7. **Required final-report TOC** (the structure v2 follows): Executive Summary → Evidence and Method → Current-State Audit (12 subsections) → Alpha Forensics → External Research → Release Spec → Gap Matrix → Roadmap → v2 Feedback Plan → Appendices.
8. **Out-of-scope for v1:** LTV labels as first-class outputs, leaderboard mini-site, second vertical, plugin architecture, web UI.

## Useful artifacts / templates / schemas

- Code-area audit table (12 areas × files to inspect × questions)
- Dataset-forensics probe taxonomy (5 categories: direct / time-window / relational / split / model-realism)
- Release-tree spec (`leadforge-v1-lead-scoring/` directory layout)
- LLM-critique JSON output schema
- Acceptance rubric (12 criteria) for the final v2 report
- Suggested opening thesis ("Leadforge is not a blank-slate idea…")

## Limitations / blind spots

- Meta document; doesn't itself produce findings, just specifies the process.

## Items unique to this source

- The relational-leakage probe class (8.3) — without this in the spec, v2 would not have caught the blocker.
- The "five-lane" framing (framework readiness vs curated dataset readiness vs platform readiness vs educator readiness vs feedback-loop readiness).
- The exact release-tree layout that the v2 author then largely adopted.

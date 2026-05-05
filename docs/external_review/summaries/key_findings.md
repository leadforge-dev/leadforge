# Key Findings — Action-Prioritized Synthesis

Distilled from the six review files. Items are ranked by what they imply
for the v1 release. This file is NOT a roadmap; it is the input list a
roadmap would consume after a process-and-recommendations sign-off pass.

## Severity legend

- **CRITICAL** — release blocker; must verify and resolve before any v1 publish
- **HIGH** — release-quality gate; should resolve before v1 publishes or ship with explicit acknowledgment
- **MEDIUM** — improves the release substantially but could be deferred to a fast v1.1
- **LOW / Defer** — accepted-with-different-approach or out-of-scope candidates

---

## CRITICAL

### 1. Public relational tables reconstruct the label
**Source:** chatgpt_report_v2.md §0, §2.7, §3.5, §6 (gap matrix), §7 (Milestone 2)
**Evidence:** Local 500-lead `student_public` smoke bundle:
- `tables/leads.parquet` contained `converted_within_90_days` and `conversion_timestamp`
- `tables/opportunities.parquet.close_outcome == "closed_won"` + `customers` + `subscriptions` reconstruct the target with 100% accuracy via joins

**Implication:** v1 cannot ship as best-in-class until either (a) public relational tables are made snapshot-safe (drop target/timestamp from `leads`, drop `close_outcome`/`closed_at` from `opportunities`, omit `customers`/`subscriptions` from public), or (b) full-horizon relational tables are moved entirely to an instructor companion.

**First action:** reproduce the finding locally on the alpha bundle and confirm severity before designing the fix.

---

## HIGH

### 2. Difficulty signal is too flat across the alpha tiers on AUC
**Source:** chatgpt_report_v2.md §2.4, §3.2
**Evidence:** Alpha LR AUC 0.886 / 0.880 / 0.870; HistGBM 0.866-0.868. AP and P@K do degrade meaningfully (intro 0.785 / 79% → advanced 0.271 / 26%), but model-family deltas don't reward sophistication enough.
**Implication:** v1 difficulty gates must include calibration, lift, P@K, AP, and model-family deltas — not just AUC. The release is a teaching dataset; "GBM beats LR realistically" is a pedagogical requirement.

### 3. No Kaggle `dataset-metadata.json` generator
**Source:** chatgpt_report_v2.md §2.10, §4.4, §7 Milestone 3 / Guid §11 / G1 / G2
**Evidence:** Repo has `release/HF_DATASET_CARD.md` and `release/README.md` but no Kaggle metadata.
**Implication:** Build `scripts/package_kaggle_release.py` that produces validated `dataset-metadata.json` (correct field names, `expectedUpdateFrequency`, title 6-50 chars, subtitle 20-80 chars, slug 3-50 chars), copies a `dataset-cover-image.png` (≥560×280, with 2:1 header and 1:1 thumbnail crops), validates against current Kaggle requirements, and supports a dry-run mode.

### 4. HF README needs hardening to be a real dataset card
**Source:** chatgpt_report_v2.md §2.10, §4.5; Crit §10; Guid §7
**Evidence:** Existing `release/HF_DATASET_CARD.md` has YAML configs but is not the final repo `README.md`.
**Implication:** Build `scripts/package_hf_release.py` that emits a final `README.md` with `pretty_name`, `tags: [tabular, lead-scoring, synthetic-data, crm, b2b, datasets, pandas]`, `configs` for all tiers with `default: true` on the main config, and a verified local `load_dataset()` smoke test.

### 5. Release validation must move beyond `leadforge validate`
**Source:** chatgpt_report_v2.md §2.9, §5.6, §7 Milestone 4; Guid §10 Milestone D; G1/G2 Phase 2/3
**Evidence:** Current validation handles structural / FK / leakage-column / realism / difficulty / drift but produces no charts, no calibration curves, no Brier/log loss, no relational-leakage probes, no public-vs-instructor diff assertion, no cross-seed bands.
**Implication:** Add `leadforge/validation/release_quality.py` + `leadforge/validation/leakage_probes.py` + `leadforge/validation/reporting.py` + `scripts/validate_release_candidate.py`. Output `release/validation/validation_report.{json,md}` and `figures/{lift_curve_*,calibration_intermediate,leakage_delta,split_shift,value_capture}.png`. Acceptance: no critical leakage findings, metrics in tier bands, charts auto-generated.

### 6. Snapshot-safe relational export design needs to land before any data goes public
**Source:** chatgpt_report_v2.md §5.2, §7 Milestone 2
**Evidence:** Direct consequence of finding #1.
**Implication:** New module `leadforge/render/relational_snapshot_safe.py` + new validator `leadforge/validation/relational_leakage.py`. Public relational tables must filter event tables to `timestamp <= lead_created_at + snapshot_day`, drop terminal-state fields, omit conversion-conditional entities. Full-horizon stays in instructor companion only.

### 7. Notebook sequence (4 notebooks) — only one exists today
**Source:** chatgpt_report_v2.md §5.5, §7 Milestone 5; Guid §10 Milestone E; G1/G2 implicit
**Evidence:** Only one release notebook (`01_baseline_lead_scoring.ipynb`) exists.
**Implication:** Add `02_relational_feature_engineering.ipynb`, `03_leakage_and_time_windows.ipynb`, `04_lift_calibration_value_ranking.ipynb`. All run top-to-bottom; outputs match validation report within tolerance.

---

## MEDIUM

### 8. Channel-conditional conversion rates as differential predictor design
**Source:** gemini_report_v2.md §3.1
**Evidence:** Industry data shows MQL→SQL ranges from <1% (email) to 51% (SEO). Current recipe uses generic motif families without explicit channel attribution as a strong predictor.
**Implication:** Either accept-with-different-approach (channel signal already partially present via motif structure) or extend the simulation to encode source-channel as a top-tier conditional probability. Risk: non-trivial DGP work right when we should be hardening for release.

### 9. Train/test split policy — temporal/cohort + account-overlap audit + group/similarity leakage
**Source:** chatgpt_report_v2.md §5.6 (account/contact overlap); gemini_report_v2.md §6 (time-based splits over random shuffle); Guid §8.4
**Evidence:** Current splits are deterministic 70/15/15 lead-level random. Real CRM use cases score future leads from accounts with prior activity — same-account-train/test may be intentional but must be audited and documented.
**Implication:** Add account/contact overlap probe; add cohort-time-shift split as an additional evaluation axis; document the choice in the data card.

### 10. v7 teaching lessons should be ported into the v1 multi-table release
**Source:** chatgpt_report_v2.md §3.4, §5.5
**Evidence:** v7 track has purely-causal trap, value-aware ranking, cohort split, GBM-vs-LR honest delta, lecture sequencing — already proven in `lead_scoring_intro/RELEASE_v7.md`.
**Implication:** Make v1 documentation (and notebooks) explicitly inherit the v7 teaching arc.

### 11. LLM-as-a-judge integration as release-quality gate
**Source:** all reviewers; concrete schema in chatgpt_report_v2.md §7 Milestone 6 + Guid §12
**Evidence:** Repo has no LLM critique today.
**Implication:** Build `leadforge/validation/llm_critique.py` with provider abstraction (env-var creds, skips cleanly without). At least 2 model families. Output schema fixed. High-severity findings require human adjudication before release.

### 12. Mode-collapse / semantic-diversity validation
**Source:** gemini_report_v2.md §6
**Evidence:** Current cohort-level diversity measured only by stage/category distribution; not by trajectory variety.
**Implication:** Add a diversity probe (likely as part of the LLM critique rubric) — sample N trajectories, ask judge whether the cohort covers the full firmographic / behavioral space.

### 13. Demographic noise injection (NLP-forcing categorical permutations)
**Source:** gemini_report_v2.md §3.3
**Evidence:** Job titles and similar categorical fields are likely standardized today.
**Implication:** Optional enrichment — adds pedagogical realism by forcing string-cleanup before modeling. Lower priority than leakage fixes.

### 14. Cover image asset
**Source:** chatgpt_report_v2.md §4.4; Guid §11
**Evidence:** No cover image in the repo.
**Implication:** Need `release/dataset-cover-image.png` ≥560×280 with documented 2:1/1:1 crops. Sourcing/design TBD.

### 15. Versioning / naming clarification
**Source:** chatgpt_report_v2.md §2.1
**Evidence:** `pyproject.toml` says `1.0.0` + Production/Stable; public dataset still alpha.
**Implication:** Name the upcoming dataset release explicitly (e.g., `leadforge-lead-scoring-v1`) so package-version vs framework-maturity vs curated-dataset-v1 don't get conflated. Cheap; do early.

### 16. Issue templates + break-me guide + v2 decision log
**Source:** chatgpt_report_v2.md §7 Milestone 7 + §8; Guid §10 Milestone G
**Evidence:** No GitHub issue templates today.
**Implication:** Add `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml`, `realism_feedback.yml`, plus `docs/release/break_me_guide.md` and `v2_decision_log.md` once feedback starts flowing.

---

## LOW / Defer / Out-of-scope candidates

### 17. CI workflow for release-candidate packaging
Useful but later — manual run of `scripts/validate_release_candidate.py` covers the use case until v1 ships.

### 18. `leadforge release ...` CLI subcommands
Convenient but not required for v1 if `scripts/{build,validate,package_kaggle,package_hf,publish_*}.py` cover the workflows. Subcommand consolidation is a good v1.1 polish target.

### 19. Macro framing as data-card narrative (CAC ratios, growth decline)
Useful pedagogical context for the dataset card but not a release blocker.

### 20. Channel-conditional rates / log-normal sales cycles / demographic noise injection
Real DGP improvements; should be staged and benchmarked against the current alpha. Risk: rebuilding part of the engine right when we should be hardening for release.

### 21. Per-vertical industry calibration (cybersecurity, fintech)
Out of scope for v1 (single vertical: B2B SaaS procurement). Note for v2 / second vertical.

### 22. LTV labels as first-class outputs / leaderboard mini-site / second vertical
Out of scope per current `.agent-plan.md` and explicitly out per Guid §3.6 / C2 §8.

---

## Counts

- 1 critical
- 6 high
- 9 medium
- 6 low/defer

## Recommended next step

Process-and-recommendations pass on every numbered item above with action codes:
- accept
- accept-with-different-approach
- reject
- out-of-scope-and-open-issue
- defer

Then sign-off, then a single coherent roadmap.

# Summary — chatgpt_report_v2.md

**Source:** `docs/external_review/chatgpt/chatgpt_report_v2.md` (781 lines, ~49 KB)
**Author:** ChatGPT (second attempt, evidence-grounded)
**Verdict in one line:** The single most actionable artifact in this corpus — a forensic, file:line-cited audit + 7-milestone roadmap that surfaces THE release blocker (relational leakage in `student_public` mode).

## Document role

The substantive ChatGPT review — evidence-first, repo-aware, release-oriented. Builds on the critique (what went wrong with v1) and the guidance (how to do v2). Followed the prescribed methodology and produced a verdict + gap matrix + roadmap.

## Top points (ranked by importance to the v1 release)

### THE release blocker — verify before anything else
1. **Public relational tables leak the label end-to-end.** In a 500-lead `student_public` smoke bundle generated locally:
   - `tables/leads.parquet` still contained `converted_within_90_days` and `conversion_timestamp`.
   - `tables/opportunities.parquet.close_outcome == "closed_won"`, plus `customers` and `subscriptions` existing only for converted leads, **reconstructs the target with 100% accuracy** via joins.
   - Acceptable only if those relational tables are documented as post-outcome world records, not if they are marketed as feature-engineering inputs for the lead-scoring task.
   - Recommended fix: **snapshot-safe relational export** for public bundles (drop target/timestamp from `leads`, drop `close_outcome`/`closed_at` from `opportunities`, omit `customers`/`subscriptions` from public). Move full-horizon tables to instructor companion only.

### Audit findings — by repo area, with evidence
2. Architecture and design docs aligned with implementation (`README.md:L1-L6,L34-L56,L74-L127`). Strength.
3. **Versioning friction:** `pyproject.toml` declares `version = "1.0.0"` and `Production/Stable`, while the public dataset is still alpha. Recommends naming the upcoming dataset release explicitly (e.g., `leadforge-lead-scoring-v1`) so package-version vs framework-maturity vs curated-dataset-v1 don't get conflated.
4. Public API exists (`generator.py:L43-L122,L124-L248`) — vertical-slice generator is real. Gap: no release-oriented APIs (`leadforge.release.build_release_candidate()` etc.).
5. CLI exists (`generate`, `inspect`, `validate`, `list-recipes`). Gaps: no `release` subcommands, no `--json` (recently shipped on `inspect`), no dry-run publishing, no credential checks.
6. Recipes + difficulty profiles are first-class. Gap: alpha baselines show LR AUC ≈0.87-0.89 across all tiers and HistGBM ≈0.866-0.868 — too flat to demonstrate "stronger modeling lifts realistically." Need difficulty gates on calibration, lift, P@K, AP, model-family deltas.
7. Hidden-graph + motif sampler implemented (5 motif families, rewiring, validation). Gap: no public-facing diversity summary across seeds.
8. Simulation engine: real 90-day discrete-time simulator with stage transitions, conversion hazards, churn, direct conversion, post-conversion entities. Gap: data card needs a "simulation simplifications" section listing what's modeled / approximate / not modeled.
9. Bundle writer + snapshot logic implemented. **Critical gap:** flat task path is much safer than the full relational path; relational rendering needs the snapshot-safe variant.
10. Exposure modes work (`student_public` vs `research_instructor`); redaction targets known columns (`current_stage`, `is_sql`). Gap: redaction does not enforce "no public join path reconstructs label" — needs `leadforge/validation/relational_leakage.py`.
11. Validation suite is real and broad (`bundle_checks`, `realism`, `difficulty`, `drift`, `lead_scoring`). Gap: not yet a single reproducible release report with charts, calibration, Brier/log loss, leakage probes, public/instructor diff assertions, cross-seed bands, LLM critique.
12. Release tooling partial: `scripts/build_public_release.py`, `release/HF_DATASET_CARD.md` (with YAML), `release/README.md` exist. Missing: Kaggle metadata, final HF README with `pretty_name`/`tabular`/`datasets`/`pandas`/`default: true`/tested configs, cover image, publisher scripts, post-upload smoke tests, more notebooks.
13. CI runs lint/mypy/pytest plus v5/v6/v7 dataset validation jobs. Missing: release-candidate workflow.

### Alpha forensics
14. Alpha LR baselines: intro 41.5% conv → AUC 0.886 / AP 0.785 / P@100 79%; intermediate 20.1% → 0.880/0.559/65%; advanced 7.9% → 0.870/0.271/26%. AP and P@K degrade meaningfully across tiers; AUC stays flat. v1 needs to show difficulty in calibration, lift, value capture, and stronger-model deltas, not only AP.
15. **v7 lessons to carry into v1:**
    - Keep student path simple and safe.
    - Keep leakage traps clearly separated from student-facing features.
    - Teach value-aware ranking (not just probability ranking).
    - Include cohort/time-shift evaluation.
    - Make tree/GBM lift over LR visible but not absurd.
    - Document limitations bluntly.
    - Provide a lecture/notebook sequence.

### External research grounding
16. **Public lead-scoring dataset census:** X Education on Kaggle (9240 rows, flat, overused, leakage-suspect status fields) → `shawhin/lead-scoring-x` on HF (5688 rows, only 7 features) → UCI Online Shoppers (12330 sessions, e-commerce not B2B). Gap is real; leadforge can plausibly be best-in-class.
17. **Industry realism citations:** HubSpot (fit + engagement + combined scoring), Salesforce, Adobe RT-CDP B2B (predictive lead/account scoring → opportunity-stage events, account-level activity aggregation, tree models). Frontiers 2025 paper (real CRM, Jan 2020 - Apr 2024, 23154 records, 67 fields, 15 classifiers, gradient boosting wins; key features: source, status, reason for status, last activity).
18. **Synthetic data evaluation:** SDMetrics quality report (column shapes, column-pair trends, multi-table cardinality + intertable trends). Datasheets for Datasets + Data Cards Playbook.
19. **Kaggle requirements (verified from official docs):** `dataset-metadata.json` adjacent to files, Data Package style, fields: `title` (6-50 chars), `subtitle` (20-80 chars), `description`, `id` (slug 3-50 chars), `licenses` (one entry), `resources` (with `schema.fields` in order if provided), `keywords`, `expectedUpdateFrequency` (never/annually/quarterly/monthly/weekly/daily/hourly), `userSpecifiedSources`, `image`. Cover image `dataset-cover-image.png/.jpg/.jpeg/.webp`, **minimum 560×280**, with 2:1 header and 1:1 thumbnail crops.
20. **HF requirements:** README.md as dataset card, YAML metadata (license, language, tags, size, configs/data_files), `load_dataset()` viewer support, configs with `default: true`, manual split/subset configuration documented.

### Recommended v1 release shape (canonical tree)
21. Public Kaggle/HF release: intro/intermediate/advanced flat lead-scoring task splits + snapshot-safe relational tables + feature dictionary with leakage flags + validation report + charts + notebooks + data card + break-me guide.
22. Separate instructor/research companion: full world graph + latent registry + mechanisms + full-horizon relational tables + leakage-trap materials + reproducibility manifest.
23. Recommended that the instructor companion live in a separate GitHub Release artifact or HF repo/config, NOT in the default Kaggle dataset.
24. Notebooks (4): `01_intro_flat_csv_baseline` → `02_relational_feature_engineering` → `03_leakage_and_time_windows` → `04_lift_calibration_value_ranking`. All must run top-to-bottom and reproduce validation metrics within tolerance.

### "v1 ready" definition
25. Fresh release candidate generates from code; passes structural, snapshot, redaction, **relational-leakage**, split-leakage, calibration, lift, top-K, value-ranking, and platform-packaging checks; renders valid Kaggle and HF packages; notebooks run top-to-bottom; no unresolved high-severity LLM/human review findings.

## Useful artifacts / templates / schemas

- Gap matrix (Area / Current evidence / Gap / Severity / Recommended fix / Acceptance criterion) — directly portable to a roadmap.
- 7-milestone roadmap: Audit → Snapshot-safe relational → Platform packaging → Validation hardening → Docs+notebooks → LLM critique → Dry-run publish + feedback.
- Release-validation metric checklist (~25 items): row counts, class balance, ROC/PR-AUC, log loss, Brier, calibration, lift@1/5/10%, P@50/100, recall@K, top-decile rate, expected ACV at K, model deltas (LR vs GBM vs source-only vs engagement-only vs leakage-probe vs ID-only vs stage-only vs post-snapshot-aggregates), account/contact overlap, near-duplicates, public/instructor diff, snapshot-window audit, relational-join leakage audit, cross-seed stability, cross-tier difficulty ordering.
- v2 feedback triage labels: critical-leakage / realism / difficulty / documentation / platform / notebook / pedagogy / v2-idea / out-of-scope-v1.

## Limitations / blind spots

- Test suite full run timed out at 53% (300s budget) — not a failure, just incomplete dynamic verification.
- Did not upload to Kaggle/HF; did not run `load_dataset()` against a real HF repo; did not run a full multi-model leakage probe beyond the smoke-bundle finding.
- Did not download every alpha Parquet file from the public dataset repo — relied on public GitHub pages + locally-regenerated artifacts.
- Notes M12 polish items (`--json` on `inspect`) as gaps without realizing they shipped in PR #60 the same day this report was prepared.

## Items unique to this source

- The relational-leakage finding (THE blocker)
- File:line-cited current-state audit
- Concrete gap matrix with severities
- Snapshot-safe relational export design
- Public-vs-companion split with explicit recommendation to keep companion off Kaggle
- v1-ready definition (the acceptance contract)
- v2 feedback triage labels
- Frontiers 2025 paper as a real B2B-realism citation (23154 records, 15 classifiers)
- "Pin the timestamp; verify byte-equal regeneration" as a release-readiness check
- Alpha LR/HistGBM gap finding (model-family deltas should be larger to reward sophistication)

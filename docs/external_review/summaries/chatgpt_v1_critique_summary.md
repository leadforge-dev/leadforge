# Summary — leadforge_report_v1_critique.md

**Source:** `docs/external_review/chatgpt/leadforge_report_v1_critique.md` (678 lines, ~31 KB)
**Author:** ChatGPT (self-critique of its own v1 report)
**Verdict in one line:** A methodology rebuke that reset the second attempt — also the place where the corrected platform facts (Kaggle 560×280 minimum cover image, `expectedUpdateFrequency` field name) live.

## Document role

A forensic critique of `chatgpt_report_v1.md` against the original task prompt. Diagnoses why v1 was inadequate, scores it on a rubric, lays out the better process the v2 author should follow, and ships an improved-roadmap sketch (Milestones A-F) that v2 then expanded.

## Top points

1. **Diagnosis:** v1's biggest failure was *methodological under-inspection* — it skimmed architecture docs and platform docs, then inferred a roadmap, instead of building an evidence matrix from the actual code, tests, and release artifacts.
2. **Scorecard verdict:** Prompt compliance C-, Repository review D, Dataset audit C-, External research C, Roadmap C-, Citation D, Strategic usefulness C.
3. **Major factual corrections:** repo is not skeletal (937 tests, ~10.4k LoC under `leadforge/`); CLI exists; HF card and release scripts exist; validation modules exist; `lead_scoring_intro/` v6/v7 track exists.
4. **Process prescription (7 phases):** evidence inventory → static audit → dynamic reproducibility audit → alpha dataset forensics → external research → release spec + acceptance gates → LLM critique loop. v2 follows this.
5. **Distinguish two products:** framework readiness vs curated dataset readiness — must run as parallel lanes with separate acceptance criteria.
6. **Concreteness gradient (weak vs strong example):**
   - Weak: "Add better validation."
   - Strong: "Add `leadforge/validation/release_quality.py` and `scripts/validate_release_candidate.py` that read each bundle's manifest, feature dictionary, task splits, and flat CSV; compute ROC-AUC, PR-AUC, Brier, calibration bins, lift@K, leakage-probe metrics, split-shift summaries, redaction checks, relational rejoin checks; write `validation/validation_report.{json,md}` and figures. Acceptance: no high-severity leakage; metrics within configured difficulty bands; intentional public/instructor diff."
7. **Corrected platform facts (durable, accurate as of 2026-05-05):**
   - Kaggle metadata file is `dataset-metadata.json`; supported fields: `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, `image`.
   - Kaggle cover image: `dataset-cover-image.png` (or `.jpg/.jpeg/.webp`), **minimum 560×280** (not 1200×400 as v1 claimed), with header and thumbnail crops specified.
   - HF YAML supports `configs` and `data_files` for splits/subsets; mark one config `default: true`.
8. **Citation discipline:** every consequential repo claim must have file path + line range; every web claim needs URL + access date; bibliography grouped by platform docs / academic / industry / repository files.

## Useful artifacts / templates / schemas

- Acceptance rubric for the v2 report (8 dimensions): evidence fidelity, current-state accuracy, research depth, platform correctness, release specificity, pedagogical value, adversarial readiness, citation quality.
- Improved-roadmap sketch (Milestones A-F) — porting v7 lessons explicitly into v1.
- LLM critique JSON output schema (severity / category / claim / evidence / reproducer / suggested_fix).

## Limitations / blind spots

- Self-referential — its job is to fix v1, not to do the substantive review itself. Subsumed by v2.

## Items unique to this source

- The 8-dimension evidence-fidelity / current-state-accuracy / etc. acceptance rubric that should be applied to any future report.
- Corrected platform facts with timestamps.
- The framework-vs-dataset lane separation argument.

# Guidance for the Second Attempt at the Leadforge v1 Release Report

**Purpose:** This document is guidance to attach to the second attempt at the original leadforge research/report task. It is not the final leadforge roadmap itself. Its job is to prevent the second attempt from repeating the first attempt’s methodological mistakes and to force an evidence-first, code-aware, release-oriented report.

**Inputs the second attempt must use:**

1. The original task prompt.
2. The attached Repomix package: `leadforge-repomix-output.xml`.
3. The critique report: `leadforge_report_v1_critique.md`.
4. Current official platform documentation and current public lead-scoring dataset landscape from the web.

**High-level instruction to the second-attempt author:**
Treat the critique report as a constraint file, not as optional context. Verify its claims against the Repomix package, then produce a final report that is forensic, current, and directly actionable for publishing a v1 educational lead-scoring dataset to Kaggle and Hugging Face.

---

## 1. Non-negotiable objective

The final second-attempt report must answer the original prompt at the requested depth:

- Review the current state of leadforge through the code, documentation, tests, release scripts, and alpha / quasi-release dataset assets.
- Review the current state of the generated datasets and release artifacts.
- Conduct a deep research expedition into what a best-in-class synthetic lead-scoring educational dataset should look like on Kaggle and Hugging Face.
- Produce a concrete roadmap to v1 release.
- Provide a project critique: positives, negatives, risks, and opportunities.

The output should be useful to an implementer who needs to make code and release changes, not merely to a reader who wants a generic strategic overview.

---

## 2. Core lesson from the failed first attempt

The first report failed mainly because it **under-inspected the actual repository**. It treated implemented components as placeholders, missed release assets that already exist, and produced generic advice instead of a repository-grounded plan.

The second attempt must therefore be built around an evidence matrix:

- What exists now?
- Where is it in the repo?
- How mature is it?
- What did dynamic checks show?
- What is missing for v1?
- What exact release artifact, command, test, or code path should be added?

Do not write the final report from memory, from architectural intent alone, or from platform docs alone.

---

## 3. Mandatory corrections to carry forward

The final report must explicitly avoid the following false or misleading claims unless a fresh audit proves otherwise.

### 3.1 Do not say the repo is mostly skeletal

The critique found evidence of a working end-to-end generation path. The second attempt must verify this.

Files to inspect carefully:

- `leadforge/api/generator.py`
- `leadforge/api/bundle.py`
- `leadforge/simulation/population.py`
- `leadforge/simulation/engine.py`
- `leadforge/structure/sampler.py`
- `leadforge/mechanisms/*`
- `leadforge/render/*`
- `leadforge/exposure/*`
- `leadforge/validation/*`

The correct posture is not “implement the engine.” The correct posture is “audit whether the existing engine is realistic, sufficiently validated, and release-grade.”

### 3.2 Do not say the CLI is absent

The critique reports that CLI commands already exist:

- `leadforge generate`
- `leadforge list-recipes`
- `leadforge inspect`
- `leadforge validate`

Verify this in:

- `leadforge/cli/main.py`
- `leadforge/cli/commands/generate.py`
- `leadforge/cli/commands/list_recipes.py`
- `leadforge/cli/commands/inspect.py`
- `leadforge/cli/commands/validate.py`

The roadmap should focus on CLI hardening and release automation, such as:

- `leadforge release build`
- `leadforge release validate`
- `leadforge release package-kaggle`
- `leadforge release package-hf`
- `leadforge release publish-kaggle`
- `leadforge release publish-hf`
- `--json` output where missing
- dry-run publishing
- credentials checks
- deterministic artifact checks

### 3.3 Do not say there is no Hugging Face packaging

The critique found existing Hugging Face-oriented material:

- `release/HF_DATASET_CARD.md`
- `release/README.md`

The correct finding is likely:

- Hugging Face packaging partially exists.
- It needs hardening into a full Hub `README.md` with current metadata, configs, default config, dataset viewer-friendly file structure, examples, and possibly `dataset_info`.
- Kaggle-specific metadata is likely missing or incomplete and must be verified.

### 3.4 Do not say built-in validation is missing

The critique found implemented validation layers, including:

- `leadforge/validation/bundle_checks.py`
- `leadforge/validation/realism.py`
- `leadforge/validation/difficulty.py`
- `leadforge/validation/drift.py`
- validation reports in `lead_scoring_intro/`

The correct finding is likely:

- Validation exists.
- It must be raised to release-grade: persisted reports, charts, leakage probes, platform checks, cross-seed checks, LLM critique, acceptance thresholds, and CI gates.

### 3.5 Do not ignore the `lead_scoring_intro` track

The original task mentioned the user’s one-CSV intro-course dataset. The attached repo contains a substantial `lead_scoring_intro/` section.

Inspect:

- `lead_scoring_intro/RELEASE_v7.md`
- `lead_scoring_intro/BACKGROUND_v7.md`
- `lead_scoring_intro/validation_v7_report.json`
- `lead_scoring_intro/lead_scoring_intro_v7.csv`
- `lead_scoring_intro/lead_scoring_intro_v7_instructor.csv`
- `scripts/build_v7_snapshot.py`
- `scripts/validate_v7_dataset.py`

The final report should extract lessons from this track, especially:

- Leakage trap design.
- Student vs instructor versions.
- Teaching sequence.
- Lift and value-aware ranking.
- Cohort split degradation.
- Calibration and leakage-trap metrics.
- How the v7 single-CSV teaching artifact should inform the richer v1 multi-table release.

### 3.6 Separate the framework from the curated dataset

The original task has two intertwined products:

1. **The leadforge framework/package** — API, CLI, recipe system, simulation, validation, release tooling.
2. **The curated v1 lead-scoring dataset release** — the best-in-class public educational dataset family generated by the framework.

The final report must maintain separate lanes:

- Framework readiness.
- Dataset readiness.
- Platform readiness.
- Documentation/readiness for educators.
- Feedback-loop readiness.

Do not collapse these into one generic “project roadmap.”

---

## 4. Required methodology for the second attempt

### Phase 0 — Evidence inventory

Extract the Repomix XML into a working directory.

Recommended extraction approach:

```python
import re
from pathlib import Path

xml = Path("leadforge-repomix-output.xml").read_text(encoding="utf-8")
out = Path("leadforge_extracted")
out.mkdir(exist_ok=True)

for m in re.finditer(r'<file path="([^"]+)">\n(.*?)</file>', xml, re.S):
    path = out / m.group(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.group(2), encoding="utf-8")
```

Then create an inventory:

- total files
- Python files
- tests
- docs
- scripts
- release artifacts
- generated CSV/JSON/Markdown files
- recipe files
- validation modules
- notebooks, if any
- CI workflows, if any

Do not trust the critique’s counts blindly. Recompute them.

Useful commands:

```bash
find leadforge_extracted -maxdepth 3 -type f | sort
find leadforge_extracted/leadforge -name "*.py" | wc -l
find leadforge_extracted/tests -name "*.py" | wc -l
grep -R "NotImplementedError\|TODO\|pass$" -n leadforge_extracted/leadforge || true
grep -R "dataset-metadata.json\|HF_DATASET_CARD\|huggingface\|kaggle" -n leadforge_extracted || true
```

### Phase 1 — Static code audit

Audit these areas in separate subsections.

| Area | Files to inspect | What to determine |
|---|---|---|
| Public API | `leadforge/api/generator.py`, `leadforge/api/bundle.py`, `leadforge/api/recipes.py` | Is generation end-to-end? Are defaults, overrides, exposure modes, and difficulty handled cleanly? |
| CLI | `leadforge/cli/*` | Which commands exist? Are `--json`, errors, dry-run, and release commands missing? |
| Core models | `leadforge/core/models.py`, enums, RNG, IDs, hashing | Are data contracts typed and reproducible? |
| Recipes | `leadforge/recipes/*` | What recipe metadata, narrative, schema, difficulty, and motif settings exist? |
| Structure | `leadforge/structure/*` | Are motif graphs sampled and validated? Is rewiring meaningful and documented? |
| Mechanisms | `leadforge/mechanisms/*` | Are hazards, static features, scores, policies, and measurement logic plausible? |
| Simulation | `leadforge/simulation/population.py`, `engine.py`, `world.py`, `state.py` | What is actually simulated? What is simplified? What events are generated? |
| Rendering | `leadforge/render/*`, `leadforge/api/bundle.py` | What tables, snapshots, tasks, dictionaries, manifests, graph exports, cards are written? |
| Exposure | `leadforge/exposure/*` | What is redacted in public mode? Can redacted truths be reconstructed? |
| Validation | `leadforge/validation/*` | What checks exist? Which release-grade checks are missing? |
| Release tooling | `scripts/*`, `release/*` | What already builds platform-ready assets? What is missing? |
| Tests | `tests/*` | How broad is the test coverage? What important release risks are not tested? |
| Intro dataset track | `lead_scoring_intro/*` | What teaching/release lessons should inform v1? |

For each area, the report should state:

- What exists.
- Evidence from files.
- Strengths.
- Gaps.
- Release implications.
- Concrete suggested changes.

### Phase 2 — Dynamic reproducibility audit

If the environment permits, install and run the project.

Suggested commands:

```bash
cd leadforge_extracted

python -m pip install -e ".[dev]" || python -m pip install -e .
python -m pytest --collect-only -q
python -m pytest -q

leadforge list-recipes

leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intermediate \
  --n-leads 500 \
  --out /tmp/leadforge_smoke

leadforge inspect /tmp/leadforge_smoke
leadforge validate /tmp/leadforge_smoke

python scripts/build_public_release.py \
  /tmp/leadforge_release \
  --generation-timestamp 2026-01-01T00:00:00+00:00
```

Record:

- Commands run.
- Runtime.
- Exit codes.
- Failures and likely causes.
- Generated file tree.
- Row counts.
- Whether checksums are deterministic.
- Whether validation passes.
- What was not run and why.

If dynamic checks cannot be run, the final report must say so plainly and must not imply they were run.

### Phase 3 — Alpha release and dataset forensic audit

The second attempt should not only read repository READMEs. It should inspect actual release bundles where accessible.

Minimum checks for each relevant bundle:

- Manifest schema and row counts.
- Train/valid/test split sizes.
- Class balance by split.
- Flat CSV columns.
- Relational table presence.
- Feature dictionary coverage.
- Public vs instructor exposure diff.
- Redacted-column enforcement.
- Snapshot/label time-window logic.
- Potential leakage via IDs, dates, stage columns, opportunity status, post-snapshot events.
- Account/contact leakage across splits.
- Duplicate leads or near-duplicate rows.
- Train/test distribution shift.
- Metrics: ROC-AUC, PR-AUC, log loss, Brier score, calibration, lift@K, precision@K, top-decile conversion.
- Stronger baselines: logistic regression, tree/GBM, simple target-encoding pipeline where safe, and intentionally “bad” leakage model for demonstration.
- Value-aware ranking: expected ACV / opportunity value if present.
- Cohort shift: time-based split or lead-source split, not only random splits.

The final report should include a forensic subsection for:

- `leadforge-datasets` alpha release family.
- `lead_scoring_intro` v7 CSV track.
- Any differences between the two.

### Phase 4 — External research expedition

The original prompt demands deep research. The second attempt should do a real census and literature/platform review.

Use current web search. Do not rely on stored knowledge for current platform requirements or current public datasets.

#### 4.1 Public lead-scoring dataset census

Search at least:

- Kaggle
- Hugging Face
- GitHub
- UCI / common educational repositories
- Data.World or other public dataset catalogs if relevant

Queries to run:

```text
lead scoring dataset Kaggle
"lead scoring" "X Education" dataset
"lead scoring" Hugging Face dataset
"predictive lead scoring" dataset GitHub
"CRM lead scoring" dataset machine learning
"lead conversion prediction" dataset
```

For each dataset found, record:

- Name and URL.
- Domain.
- Row count.
- Feature count.
- Label definition.
- Flat vs relational.
- Documentation/card quality.
- Baselines if present.
- Known leakage concerns.
- What leadforge can do better.

The final report should include a competitor/benchmark table, but keep prose outside tables. Tables should use short entries only.

#### 4.2 Lead-scoring and B2B GTM realism

Research both academic and industry sources.

Topics:

- Predictive lead scoring methods.
- CRM data leakage.
- Conversion funnel stages and stage definitions.
- MQL/SQL/opportunity/closed-won conversion dynamics.
- B2B SaaS sales-cycle durations and ACV distributions.
- Inbound/outbound/partner channel mix.
- Buying committees and multi-threaded accounts.
- Lead-source attribution limitations.
- Lift curves, top-K prioritization, and precision/recall tradeoffs.
- Calibration and score interpretability.

Prefer high-quality sources:

- Academic papers.
- Vendor docs/articles from credible GTM vendors, used critically.
- Industry benchmarks from sources such as Salesforce, HubSpot, Demandbase, Gartner, Forrester, OpenView, SaaS benchmark reports, or similar if accessible.
- Explain uncertainty when industry benchmarks conflict.

#### 4.3 Synthetic data generation and evaluation

Research:

- Synthetic tabular data quality dimensions: fidelity, utility, privacy.
- TSTR/TRTS evaluation.
- SDMetrics / SDV-style metrics.
- Relational synthetic data metrics.
- Disclosure risk, membership inference, exact-match risk.
- Constraint validation.
- Dataset-level mechanism design / scenario coverage.
- Diversity and difficulty calibration.

Use sources such as:

- SDV / SDMetrics documentation.
- Academic papers on synthetic tabular data evaluation.
- Google Data Cards / Dataset documentation standards.
- Datasheets for Datasets.
- Data Cards Playbook.
- Any current work on dataset-level mechanism design or synthetic data evaluation, if relevant.

#### 4.4 Platform requirements

Use official docs only for platform packaging details.

Current platform facts that should be verified at report time:

- Kaggle metadata requires `dataset-metadata.json` next to uploaded files, follows Data Package style, and supports fields such as `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image`. The official Kaggle API docs also describe supported licenses, data types, and image requirements.
- Kaggle cover image guidance currently says `dataset-cover-image.png` / `.jpg` / `.jpeg` / `.webp` can be placed beside `dataset-metadata.json`, with minimum 560×280 dimensions and specified header/thumbnail crops.
- Hugging Face dataset repositories render `README.md` as the dataset card, use YAML front matter for metadata, and support `configs` / `data_files` for splits and subsets.
- Hugging Face Datasets can automatically load supported formats such as CSV and Parquet when repository structure and metadata are compatible.

Primary sources to use and cite:

- Kaggle API dataset metadata documentation: `https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets_metadata.md`
- Hugging Face Hub dataset card docs: `https://huggingface.co/docs/hub/datasets-cards`
- Hugging Face Datasets repository structure docs: `https://huggingface.co/docs/datasets/repository_structure`
- Hugging Face data files configuration docs: `https://huggingface.co/docs/hub/datasets-data-files-configuration`

### Phase 5 — Build a gap matrix

The final report must include a gap matrix. Suggested columns:

- Area
- Current evidence
- Gap
- Severity
- Recommended fix
- Files/commands affected
- Acceptance criterion

Example rows the second attempt should verify:

| Area | Likely current state | Likely gap |
|---|---|---|
| Kaggle packaging | Release README and HF card exist | `dataset-metadata.json` generator likely missing |
| Hugging Face packaging | `release/HF_DATASET_CARD.md` exists | Needs full card, configs, default config, tested `load_dataset()` |
| Validation | Structural/realism/difficulty/drift validators exist | Needs release-quality report, charts, leakage probes, LLM critique |
| Release builder | `scripts/build_public_release.py` exists | Needs platform-specific package/publish commands |
| Teaching assets | v7 intro docs exist | Need polished notebooks for Kaggle/HF |
| Feedback loop | likely informal | Need issue templates, break-me guide, triage labels |
| Metrics | baselines exist | Need calibration, lift, top-K, value ranking, seed stability |

---

## 5. Required final report format

The second-attempt report should be a single comprehensive Markdown report with this structure.

```text
# Leadforge v1 Lead-Scoring Dataset Release Plan

## 0. Executive Summary
- Verdict.
- What is already strong.
- Top release blockers.
- Recommended release shape.
- Definition of “v1 ready.”

## 1. Evidence and Method
- Inputs reviewed.
- Repository extraction method.
- Commands run.
- Web sources used.
- What was not verified.
- Evidence-quality limitations.

## 2. Current-State Audit of Leadforge
### 2.1 Architecture and design docs
### 2.2 Public API
### 2.3 CLI
### 2.4 Recipe system and difficulty profiles
### 2.5 Hidden graph and motif sampler
### 2.6 Mechanisms and simulation engine
### 2.7 Relational rendering and snapshot task generation
### 2.8 Exposure/redaction modes
### 2.9 Validation suite
### 2.10 Release tooling
### 2.11 Test suite
### 2.12 Documentation

For each subsection:
- What exists.
- Evidence.
- Strengths.
- Gaps.
- Release implications.

## 3. Existing Dataset and Alpha Release Forensics
### 3.1 leadforge-datasets alpha release inventory
### 3.2 Intro/intermediate/advanced difficulty tiers
### 3.3 Public vs instructor mode
### 3.4 `lead_scoring_intro` v7 lessons
### 3.5 Baselines, lift, calibration, leakage, and drift
### 3.6 What currently makes the dataset hard/easy to break

## 4. External Research
### 4.1 Public lead-scoring dataset census
### 4.2 Lead-scoring and B2B GTM realism
### 4.3 Synthetic data generation and evaluation
### 4.4 Dataset documentation standards
### 4.5 Kaggle release requirements
### 4.6 Hugging Face release requirements
### 4.7 Lessons for leadforge

## 5. Best-in-Class v1 Release Specification
### 5.1 Dataset family shape
### 5.2 File tree
### 5.3 Public bundle contents
### 5.4 Instructor/research companion contents
### 5.5 Kaggle package
### 5.6 Hugging Face package
### 5.7 Dataset cards and documents
### 5.8 Notebooks
### 5.9 Validation report
### 5.10 Feedback and break-me process

## 6. Gap Matrix
- Repository gaps.
- Dataset gaps.
- Platform gaps.
- Documentation gaps.
- Validation gaps.
- Pedagogical gaps.

## 7. Roadmap to v1
### Milestone 1: Release audit and acceptance gates
### Milestone 2: Platform package generation
### Milestone 3: Release validation hardening
### Milestone 4: Documentation and notebooks
### Milestone 5: LLM critique integration
### Milestone 6: Dry-run publication
### Milestone 7: Public release and feedback intake

Each milestone:
- Goal.
- Work items.
- Files likely touched.
- Commands.
- Acceptance criteria.
- Risks.

## 8. Suggested v2 Feedback Plan
- Break-me guide.
- Issue templates.
- Metrics requested from users.
- Triage process.
- How feedback becomes v2 decisions.
- Explicitly keep leaderboard/LTV out of the v1 milestone.

## 9. Appendices
- Commands run.
- Candidate release tree.
- Kaggle metadata template.
- Hugging Face README/YAML template.
- Release validation JSON schema.
- LLM critique prompt schema.
- Bibliography.
```

---

## 6. Required level of concreteness

The final report should contain PR-sized work, not only broad themes.

Weak recommendation:

> Add better validation.

Strong recommendation:

> Add `leadforge/validation/release_quality.py` and `scripts/validate_release_candidate.py`. The script should read each bundle’s `manifest.json`, `feature_dictionary.csv`, task splits, and flat CSV; compute ROC-AUC, PR-AUC, Brier score, calibration bins, lift@1%, lift@5%, lift@10%, precision@50/100, leakage-probe metrics, split-shift summaries, redaction checks, and relational rejoin leakage checks; then write `validation/validation_report.json`, `validation/validation_report.md`, and figures. Acceptance: no high-severity leakage probes; metrics within configured difficulty bands; all public/instructor diffs are intentional.

Weak recommendation:

> Publish to Kaggle.

Strong recommendation:

> Add `scripts/package_kaggle_release.py` that reads bundle manifests and `feature_dictionary.csv`, generates `kaggle/dataset-metadata.json`, copies `dataset-cover-image.png`, writes resource descriptions for flat CSVs and Parquet bundle files, validates title/subtitle/id/license/image dimensions against Kaggle docs, creates a zip, and offers a dry-run command. Acceptance: the package can be uploaded with `kaggle datasets create -p <dir> --dir-mode zip` after credentials are configured.

---

## 7. Required platform-correctness guardrails

The final report must not repeat the first report’s platform inaccuracies.

### Kaggle

Use current official documentation.

Specific guidance to verify and cite:

- The upload folder should include `dataset-metadata.json` beside data files.
- `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image` are supported fields.
- `title` and `subtitle` have length constraints.
- `resources[].schema.fields` should include all fields in order when provided.
- `expectedUpdateFrequency` uses values like `never`, `annually`, `quarterly`, `monthly`, `weekly`, `daily`, `hourly`.
- Cover image minimum is 560×280 according to the current Kaggle API docs, not 1200×400.
- The recommended sibling name is `dataset-cover-image.png` or supported alternatives.

### Hugging Face

Use current official documentation.

Specific guidance to verify and cite:

- Hub dataset cards are repository `README.md` files.
- YAML metadata block controls visible metadata.
- Use `language`, `license`, `pretty_name`, `tags`, `task_categories`, `size_categories`, and `configs`.
- Include `tags: tabular`, `pandas`, and `datasets` where appropriate.
- Use `configs` and `data_files` for intro/intermediate/advanced and possibly a separate relational bundle config.
- Mark one config as `default: true` if appropriate.
- Test `load_dataset()` locally or state that this was not tested.

---

## 8. Dataset-forensics requirements

The final report should discuss not only whether a model performs well, but whether the dataset is **hard to break**.

Required probes:

### 8.1 Direct leakage probes

- Train models with all features.
- Train without known leakage traps.
- Train using only suspect temporal/stage/opportunity columns.
- Train using IDs or hashed IDs if present.
- Train using post-snapshot aggregates if present.
- Compare performance deltas.

### 8.2 Time-window leakage

Check that all public features intended to be pre-snapshot are derived from events at or before `lead_created_at + snapshot_day`. Verify that label resolution uses the full label horizon but not as a feature source, except explicitly documented teaching leakage traps.

### 8.3 Relational leakage

The public flat CSV may be safe while relational tables reveal label information. Check:

- Opportunity status after snapshot.
- Customer/subscription rows that only exist for conversions.
- Sales activities after snapshot.
- Stage tables.
- Join paths that reconstruct `is_sql`, `current_stage`, or terminal states.

### 8.4 Split leakage

Check:

- Same account appearing in train/test.
- Same contact appearing in train/test.
- Near-duplicate leads in different splits.
- Temporal split leakage if lead dates overlap in unrealistic ways.

This is especially important because real CRM use cases often score future leads from accounts with prior activity. The report should decide whether account overlap is intentional and document it.

### 8.5 Model realism

Compute or request:

- ROC-AUC.
- PR-AUC.
- Brier score.
- Calibration curves.
- Lift curves.
- Precision@K.
- Recall@K.
- Top-decile conversion.
- Expected value captured at K if ACV exists.
- Baseline comparison against naive source-only or engagement-only models.

A best-in-class dataset should not be “solved” by a single shortcut, but should still reward better modeling.

---

## 9. Research deliverables that must appear in the final report

### 9.1 Public lead-scoring dataset benchmark

The report should include a concise dataset census table and then discuss it in prose.

Suggested columns:

- Dataset
- Platform
- Domain
- Rows
- Shape
- Documentation quality
- Main weakness

Keep table entries short. Put details in prose.

### 9.2 Documentation benchmark

Compare leadforge’s intended documentation against:

- Hugging Face dataset card template.
- Data Cards Playbook.
- Kaggle dataset metadata / README conventions.
- High-quality Kaggle notebooks.

### 9.3 Synthetic data benchmark

Discuss what “best ever synthetic lead scoring dataset” should mean operationally:

- Relational, not just flat.
- Narrative-grounded.
- Deterministic and reproducible.
- Has latent truth in instructor mode.
- Uses time windows correctly.
- Has realistic class imbalance and lift behavior.
- Has multiple difficulty tiers.
- Includes validation reports and notebooks.
- Includes a break-me guide.
- Is transparent about limitations.

### 9.4 Educational design benchmark

Discuss what makes it useful for teaching:

- Intro flat CSV path.
- Advanced relational path.
- Leakage-trap lesson.
- Calibration and lift.
- Class imbalance.
- Feature engineering.
- Temporal validation.
- Instructor-only truth artifacts.
- Assignment/rubric possibilities.
- Student-friendly notebook and instructor guide.

---

## 10. Roadmap requirements

The roadmap should be written as a sequence of release work packages. Suggested work packages:

### Milestone A — Evidence-backed current-state audit

Deliverables:

- `docs/release/v1_current_state_audit.md`
- regenerated release bundles
- command log
- inventory table

Acceptance:

- Full or partial test results recorded.
- Release builder behavior verified.
- Existing HF/Kaggle packaging status classified correctly.
- v7 intro lessons captured.

### Milestone B — Release-candidate specification

Deliverables:

- `docs/release/v1_release_spec.md`
- canonical release file tree
- list of public/instructor artifacts
- v1 acceptance gates

Acceptance:

- Public and instructor artifact scopes are explicit.
- Out-of-scope items are explicit: no LTV, no leaderboard mini-site, no other GTM task.

### Milestone C — Platform package automation

Deliverables:

- `scripts/package_kaggle_release.py`
- `scripts/package_hf_release.py`
- `release/kaggle/dataset-metadata.json`
- `release/huggingface/README.md`
- `release/dataset-cover-image.png`

Acceptance:

- Kaggle metadata validates against official docs.
- HF `load_dataset()` works or known blockers are documented.
- Dry-run command does not require credentials.

### Milestone D — Release validation hardening

Deliverables:

- `leadforge/validation/release_quality.py`
- `leadforge/validation/leakage_probes.py`
- `leadforge/validation/reporting.py`
- `scripts/validate_release_candidate.py`
- `release/validation/validation_report.json`
- `release/validation/validation_report.md`
- figures

Acceptance:

- No critical leakage findings.
- Metrics are within configured bands.
- Figures and reports are generated automatically.
- Validation output is included in platform packages.

### Milestone E — Notebooks and teaching materials

Deliverables:

- `notebooks/01_intro_flat_csv_baseline.ipynb`
- `notebooks/02_relational_feature_engineering.ipynb`
- `notebooks/03_leakage_and_time_windows.ipynb`
- `notebooks/04_lift_calibration_value_ranking.ipynb`
- `docs/release/instructor_guide.md`
- `docs/release/student_quickstart.md`

Acceptance:

- Notebooks run top-to-bottom.
- Notebook metrics match validation report within tolerance.
- Student vs instructor usage is clear.

### Milestone F — LLM critique integration

Deliverables:

- `leadforge/validation/llm_critique.py`
- `docs/release/llm_critique_prompt.md`
- `release/validation/llm_critique_summary.md`
- raw model-output archive

Acceptance:

- Runs with provider credentials.
- Skips gracefully without credentials.
- Produces structured findings with severity, evidence, and suggested fix.
- No unresolved high-severity findings before release.

### Milestone G — Publish and feedback loop

Deliverables:

- `scripts/publish_kaggle.py`
- `scripts/publish_hf.py`
- GitHub issue templates
- break-me guide
- release notes
- public feedback instructions

Acceptance:

- Private/draft upload tested.
- Public download/load smoke tests pass.
- Feedback channels are linked from Kaggle, HF, GitHub, and README.

---

## 11. Suggested final release artifact tree

The second-attempt report should propose a concrete tree similar to this:

```text
leadforge-v1-lead-scoring/
  README.md
  LICENSE
  CITATION.cff
  CHANGELOG.md
  dataset-cover-image.png

  docs/
    DATASET_CARD.md
    GENERATION_METHOD.md
    VALIDATION_REPORT.md
    FEATURE_DICTIONARY.md
    BREAK_ME_GUIDE.md
    INSTRUCTOR_GUIDE.md

  data/
    intro/
      lead_scoring.csv
      train.csv
      validation.csv
      test.csv
      manifest.json
      feature_dictionary.csv
    intermediate/
      ...
    advanced/
      ...
    relational/
      intro/
      intermediate/
      advanced/

  instructor_companion/
    intermediate_instructor/
      metadata/
      graph/
      mechanism_summary.json
      latent_registry.json

  validation/
    validation_report.json
    validation_report.md
    figures/
      lift_curve_intro.png
      lift_curve_intermediate.png
      lift_curve_advanced.png
      calibration_intermediate.png
      leakage_delta.png
      split_shift.png

  notebooks/
    01_intro_flat_csv_baseline.ipynb
    02_relational_feature_engineering.ipynb
    03_leakage_and_time_windows.ipynb
    04_lift_calibration_value_ranking.ipynb

  kaggle/
    dataset-metadata.json
    README.md

  huggingface/
    README.md
```

The final report should decide whether instructor companion artifacts should be publicly downloadable, gated, omitted from Kaggle, or stored separately. It should explain the tradeoff: transparency and reproducibility versus student-exercise leakage.

---

## 12. LLM critique guidance

The original prompt explicitly requests built-in deep release validation using outside LLMs. The final report should propose this as a concrete module and workflow.

Suggested input bundle:

- `README.md`
- `DATASET_CARD.md`
- `GENERATION_METHOD.md`
- `manifest.json`
- `feature_dictionary.csv`
- `validation_report.json`
- first 100 public rows
- public/instructor diff summary
- mechanism summary if instructor mode is available

Suggested output schema:

```json
{
  "release_id": "leadforge-lead-scoring-v1",
  "model": "provider/model/version",
  "run_timestamp": "ISO-8601",
  "overall_score": 0,
  "findings": [
    {
      "severity": "critical|high|medium|low|nit",
      "category": "leakage|realism|documentation|platform|ethics|pedagogy|code",
      "claim": "...",
      "evidence": "file/path:line or artifact reference",
      "reproducer": "optional command",
      "suggested_fix": "..."
    }
  ],
  "missing_sections": [],
  "questions_for_maintainer": []
}
```

Guidelines:

- Use at least two model/provider families when possible.
- Save raw outputs and parsed findings.
- Treat LLM outputs as review inputs, not ground truth.
- Require human adjudication of high-severity findings before release.
- Include LLM critique summaries in the release validation directory.

---

## 13. Citation and evidence standards

The final report must use durable citations.

### 13.1 Repository evidence

Use extracted file paths and line ranges. Example format:

```text
`leadforge/api/generator.py:L42-L117`
```

To create line-numbered extracts:

```bash
nl -ba leadforge/api/generator.py | sed -n '1,160p'
```

When quoting or summarizing repository content, cite exact file paths. Do not cite the whole Repomix package generically for every claim.

### 13.2 Web evidence

Use official or primary sources for platform requirements.

For web citations, include:

- Title.
- URL.
- Access date.
- Exact fact supported.

Do not use hidden browser/source IDs as the only citation in a downloadable Markdown report. They are not portable.

### 13.3 Research evidence

For academic claims, prefer:

- original papers
- official documentation
- peer-reviewed or arXiv/OpenReview papers when appropriate

For industry claims, note that sources are often vendor-authored and may be biased. Use them for plausible ranges and practices, not as hard universal truths.

### 13.4 Unverified items

Mark unverified items clearly:

- “Verified by command.”
- “Verified by static inspection.”
- “Observed in alpha release.”
- “Reported by critique, not independently verified.”
- “Not verified in this pass.”

The final report should include a “What I did not verify” subsection.

---

## 14. Pitfalls to avoid

Do not:

- Call implemented modules placeholders without evidence.
- Recommend implementing commands that already exist.
- Treat Hugging Face packaging as absent without inspecting `release/HF_DATASET_CARD.md`.
- Ignore the intro v7 CSV dataset track.
- Use outdated or unofficial Kaggle/HF platform requirements.
- Rely on a single vendor blog as “industry knowledge.”
- Treat AUC as the only success metric.
- Ignore lift, precision@K, calibration, and value-aware ranking.
- Ignore relational leakage.
- Ignore split leakage through accounts/contacts.
- Over-plan LTV or leaderboard work for the v1 milestone.
- Produce a roadmap with no file names, commands, artifacts, or acceptance criteria.
- Use citations that cannot be followed outside the chat environment.
- End with vague “next steps” instead of a concrete PR/release plan.

---

## 15. Recommended stance and tone

The report should be candid and precise.

Good stance:

> Leadforge appears much further along than the first report recognized. The right v1 task is not to create the framework from scratch, but to harden an already functional generator and release pipeline into a best-in-class public dataset product.

Good stance:

> The current alpha release already has strong bones: deterministic generation, relational tables, public/instructor exposure, manifests, feature dictionaries, baseline metrics, and validation. The remaining work is release-grade packaging, deeper adversarial validation, stronger documentation, notebooks, and a public feedback loop.

Bad stance:

> The simulation engine must be implemented before v1.

Bad stance:

> No Kaggle/Hugging Face packaging exists.

Bad stance:

> Add validation.

---

## 16. Rubric for accepting the second-attempt report

Score the second-attempt report against this rubric before using it.

| Criterion | Pass condition |
|---|---|
| Evidence-first review | Every consequential repo claim has file-path evidence or command evidence. |
| Correct current-state classification | Existing, partial, missing, and out-of-scope items are clearly separated. |
| Dynamic checks | Commands are run where possible; failures and limitations are reported. |
| Alpha dataset forensics | Existing releases and v7 intro dataset are analyzed, not just mentioned. |
| External research depth | Public datasets, industry practice, synthetic-data evaluation, and platform requirements are surveyed. |
| Platform accuracy | Kaggle and HF requirements are sourced from current official docs. |
| Actionable roadmap | Milestones include files, commands, artifacts, gates, risks, and acceptance criteria. |
| Pedagogical value | Notebooks, leakage lessons, instructor mode, calibration, lift, and assignments are addressed. |
| Adversarial readiness | Break-me guide, leakage probes, LLM critique, and feedback triage are included. |
| Scope control | LTV, other tasks, and leaderboard are kept out of v1 except as future notes. |
| Citation quality | Citations are durable and human-readable. |
| Honesty | Unverified claims and failed checks are explicitly labeled. |

---

## 17. Minimal acceptable second-attempt report

If time is constrained, the second attempt must still include:

1. Corrected current-state audit of the package.
2. Corrected current-state audit of release/HF/Kaggle packaging.
3. Discussion of `lead_scoring_intro` v7.
4. External platform requirements from official docs.
5. Public lead-scoring dataset census.
6. Gap matrix.
7. Roadmap with files, commands, deliverables, and acceptance criteria.
8. Citation/evidence appendix.

A shorter but accurate and evidence-based report is better than a longer generic report.

---

## 18. Suggested opening thesis for the final report

The second-attempt report may use a thesis like this, if supported by the audit:

> Leadforge is not a blank-slate synthetic-data idea. It already appears to contain an end-to-end deterministic CRM world generator, a relational bundle writer, public/instructor exposure modes, validation modules, release scripts, a Hugging Face-style dataset card, and a mature intro-course CSV lineage. The v1 milestone should therefore be framed as a release-hardening and evidence-building project: prove the generator’s realism, make leakage and difficulty measurable, produce platform-native Kaggle and Hugging Face packages, ship polished notebooks and data cards, and create a public break-me feedback loop. The roadmap should concentrate on release quality, not on re-implementing core generation.

Only use this thesis if the actual second-attempt audit verifies it.

---

## 19. Appendix: official platform documentation checked while preparing this guidance

The second-attempt author should re-check these at report time because platform requirements change.

- Kaggle API dataset metadata documentation: `https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets_metadata.md`
  - Notes: documents `dataset-metadata.json`, supported metadata fields, licenses, data types, update frequencies, and image requirements.
- Hugging Face Datasets repository structure: `https://huggingface.co/docs/datasets/repository_structure`
  - Notes: documents repository structures, `README.md`, supported formats, `load_dataset()`, splits, and YAML `configs`.
- Hugging Face Hub dataset cards: `https://huggingface.co/docs/hub/datasets-cards`
  - Notes: documents README-based dataset cards and YAML metadata.
- Hugging Face Hub data files configuration: `https://huggingface.co/docs/hub/datasets-data-files-configuration`
  - Notes: documents automatic/manual split and subset configuration.

---

## 20. Final instruction to the second-attempt author

Do not optimize for sounding impressive. Optimize for making the leadforge v1 release shippable.

The final report should let the project owner answer:

- What exactly is already working?
- What exactly is missing?
- What exactly should be built next?
- What evidence proves the dataset is realistic, useful, and not trivially broken?
- What files and commands will produce a Kaggle/Hugging Face release?
- What artifacts will convince educators, Kaggle users, Hugging Face users, and skeptics that this is a serious synthetic CRM dataset?

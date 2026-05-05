# Critique of `chatgpt_report_v1.md` and Suggested Better Process

## Executive verdict

The generated report is useful as a **first rough planning memo**, but it does **not** satisfy the original task prompt. The prompt asked for a thorough code-and-release review, a deep research expedition, and an actionable roadmap toward a best-in-class Kaggle/Hugging Face lead-scoring dataset. The report instead gives a mostly generic roadmap, lightly references the alpha dataset repository, and misses major evidence present in the attached `leadforge-repomix-output.xml` package.

The biggest problem is not tone or formatting. It is **methodological under-inspection**. The report appears to have skimmed the architecture documents and external platform docs, then inferred a future roadmap. It did not build an evidence matrix from the repository, did not run or audit the package, did not inspect the existing release assets, and did not distinguish what already exists from what remains to build.

This caused several materially wrong or misleading statements, including:

- It says many modules are placeholders and that the simulation pipeline, CLI, validation, and release packaging need to be implemented. The attached repo already contains implemented generator, population, hidden graph, simulation, bundle writer, CLI commands, validation modules, release scripts, release README, and Hugging Face dataset card files.
- It says there is no Kaggle/HF packaging. There is no Kaggle `dataset-metadata.json`, but there **is** a Hugging Face dataset card and a release README in `release/`.
- It says built-in evaluation is missing. The repo contains bundle validation, realism checks, difficulty checks, drift/cross-seed checks, v7 dataset validation, release validation reports, and a large test suite.
- It gives platform guidance that is partly inaccurate or too loose, such as an unsupported “1200×400” Kaggle image claim and non-current field names like `updateFrequency` instead of `expectedUpdateFrequency`.

A better report should be much more forensic: it should audit the actual code, release artifacts, generated data, validation scripts, tests, and external platform requirements; then produce a gap matrix, acceptance gates, and a PR-sized roadmap.

## What I reviewed

I reviewed:

1. The task prompt in the current conversation.
2. The generated report: `chatgpt_report_v1.md`.
3. The attached core package: `leadforge-repomix-output(1).xml`.
4. The extracted repository files from the Repomix XML.
5. Current public documentation for Kaggle dataset metadata and Hugging Face dataset cards/repository structure.

I also performed lightweight static and dynamic checks on the extracted repo:

- Extracted **194 files** from the Repomix XML.
- Counted **78 Python files** under `leadforge/`.
- Counted roughly **10,398 lines** under `leadforge/` and **9,312 lines** under `tests/`.
- Found **81 classes**, **286 functions**, **0 `NotImplementedError` occurrences**, **0 TODO occurrences**, and only **2 literal `pass` statements** in `leadforge/`.
- `pytest --collect-only` found **937 tests**.
- A partial test run after editable install passed the first ~53% of tests before the execution timeout. I do **not** treat this as a failed test run; it only means I did not complete the full dynamic validation within this critique pass.

## Scorecard for the generated report

| Dimension | Grade | Why |
|---|---:|---|
| Prompt compliance | C- | Covers the requested headings, but not at the requested depth. |
| Repository review | D | Misses substantial implemented code and release assets. |
| Dataset release audit | C- | Mentions alpha bundles, but does not inspect data or artifacts deeply. |
| External research | C | Uses a few relevant sources, but too shallow and vendor-blog heavy. |
| Roadmap actionability | C- | Generic milestones; few concrete files, commands, gates, or PRs. |
| Citation quality | D | Browser-internal citations are not portable; several claims are miscited. |
| Strategic usefulness | C | Good high-level instincts, but unsafe as an execution plan. |

The report is directionally aligned with leadforge’s vision, but it would mislead an implementer about what is already done.

## Major factual and evidentiary problems

### 1. It misclassifies the repo as mostly skeletal

The report states that the repo includes skeletons and “many functions are placeholders,” and later recommends “complete the simulation pipeline.” That is not supported by the attached package.

Evidence in the attached package shows an implemented end-to-end generation path:

- `leadforge/api/generator.py` builds a `Generator` from a recipe, resolves config and narrative, samples a hidden graph, builds population, simulates the world, and returns a populated `WorldBundle`.
- `leadforge/structure/sampler.py` selects a motif family, performs stochastic rewiring, and returns a validated hidden world graph.
- `leadforge/simulation/population.py` generates accounts, contacts, leads, and latent states.
- `leadforge/simulation/engine.py` contains a detailed discrete-time 90-day simulation with stage transitions, conversion hazards, event emission, opportunity creation, customers, and subscriptions.
- `leadforge/api/bundle.py` writes relational Parquet tables, snapshot task splits, dataset card, feature dictionary, exposure metadata, and manifest.

The right critique would not be “implement the engine.” It would be:

- Audit whether the engine is realistic enough.
- Identify where mechanisms are too simple, over-tuned, or under-documented.
- Assess whether difficulty profiles are stable across seeds.
- Test whether public artifacts remain leakage-safe under relational feature engineering.
- Add release-grade validations and publishing automation where missing.

### 2. It misses existing CLI implementation

The report’s roadmap says to implement `leadforge generate`, `list-recipes`, `inspect`, and `validate`. Those commands already exist in the attached repo.

Evidence:

- `leadforge/cli/commands/generate.py` implements generation, override handling, config resolution, bundle generation, and save.
- `leadforge/cli/commands/inspect.py` reads `manifest.json` and prints recipe, seed, mode, difficulty, horizon, package version, schema version, motif family, table row counts, task rows, and metadata presence.
- `leadforge/cli/commands/validate.py` calls `validate_bundle()` and exits nonzero on failures.

A better roadmap would focus on gaps in those commands:

- Add `--json` output to `inspect` and `validate`.
- Add `release build`, `release validate`, `release package-kaggle`, `release package-hf`, and `release publish-*` commands.
- Add a dry-run mode for publishing.
- Add environment-variable checks for Kaggle/HF credentials.
- Add artifact hashing and upload manifest verification.

### 3. It says there is no Hugging Face packaging, but there is

The report says: “No Kaggle/HF packaging.” That is only half true.

The attached repo contains:

- `release/HF_DATASET_CARD.md`, with YAML front matter for `language`, `license`, `task_categories`, `tags`, `size_categories`, and `configs` for intro/intermediate/advanced splits.
- `release/README.md`, with release layout, quick-start code, dataset summary, leakage handling, research companion explanation, and provenance.
- `scripts/build_public_release.py`, which builds intro, intermediate, advanced, and intermediate instructor bundles; writes flat CSVs for public bundles; copies the license; and validates each bundle.

What is missing is more specific:

- A Kaggle `dataset-metadata.json` template/generator.
- A final HF `README.md` that satisfies the full dataset-card template, not just a concise release card.
- A release asset manifest for platform upload.
- A publishing command or CI workflow.
- A cover image asset.
- Automated post-upload smoke tests.

### 4. It underestimates existing validation

The report says built-in evaluation is lacking. That is too broad.

The repo already has multiple validation layers:

- `leadforge/validation/bundle_checks.py` validates required files, table files, task split files, hashes, FK integrity, leakage columns, and exposure redaction.
- `leadforge/validation/realism.py` checks conversion-rate guardrails, nonempty tables, feature ranges, boolean dtypes, and stage diversity where available.
- `leadforge/validation/difficulty.py` checks known difficulty profiles and difficulty ordering across bundles.
- `leadforge/validation/drift.py` checks cross-seed stability and degenerate conversion-rate patterns.
- `lead_scoring_intro/validation_v7_report.json` contains concrete v7 metrics including baseline AUC, PR-AUC, value-aware ranking uplift, leakage-trap deltas, missingness, and cohort split degradation.

The better critique is not that validation is absent. It is that v1 release validation needs to become **release-grade**, with explicit acceptance thresholds, persisted reports, charts, adversarial leakage probes, platform packaging checks, and LLM critique artifacts.

### 5. It ignores the `lead_scoring_intro` v6/v7 track

The task prompt explicitly mentioned a one-CSV lead-scoring dataset used in an Intro to ML course. The attached repo contains a substantial `lead_scoring_intro/` track:

- `lead_scoring_intro/RELEASE_v7.md` documents a v7 educational dataset, a purely causal leakage trap, snapshot definition, student/instructor files, column dictionary, baseline metrics, tree-model comparison, value-aware ranking, cohort split evaluation, known limitations, and lecture guidance.
- `lead_scoring_intro/validation_v7_report.json` stores metrics used by that release document.
- `scripts/build_v7_snapshot.py` and `scripts/validate_v7_dataset.py` support generation and validation.

The generated report almost completely misses this. That is a major omission because the v7 CSV track is likely one of the best sources of lessons for v1: leakage trap design, lecture sequencing, cohort shift, value-aware ranking, and student/instructor split design.

### 6. It does not distinguish “framework v1” from “curated dataset v1”

The task contains two related products:

1. The `leadforge` package/framework.
2. A curated, exemplary, v1 lead-scoring dataset family generated by the framework.

The report treats these as one blended thing. That makes the roadmap blurry. A better report should maintain two parallel lanes:

- **Framework readiness lane:** engine, config, CLI, validation, documentation, release automation, publishing integrations, reproducibility.
- **Dataset readiness lane:** chosen recipe/seed(s), size, splits, public/instructor variants, data cards, notebooks, validation reports, public challenge framing, feedback channels.

Each lane should have its own acceptance criteria and release gates.

### 7. It gives a generic roadmap, not an execution plan

The roadmap is too high-level. It says things like “add engineered features,” “expand motifs,” and “implement validation checks,” but does not identify:

- Which files to change.
- Which commands should exist.
- Which release artifacts should be produced.
- What acceptance thresholds define success.
- What the Kaggle/HF upload directory should look like.
- Which validation reports should be persisted.
- Which notebooks should be shipped.
- Which tests should gate CI.
- Which items are out of scope for the next milestone.

For example, “Add engineered features” should become a concrete feature plan:

- Add `engagement_velocity_7d`, `high_intent_session_ratio`, `multi_threaded_account`, `stakeholder_coverage`, `days_to_first_sales_activity`, and `source_normalized_activity_rate` only if they are causally available before the snapshot window.
- For each new feature, add a schema entry, feature dictionary description, leakage flag, snapshot test, monotonicity or range test, and at least one validation check.
- Require a clean-model/lift delta report with and without each feature family.

### 8. It weakly satisfies the “deep research expedition” requirement

The external research in the report is thin. It cites a few platform docs, one industry article, a synthetic-data vendor blog, and a Google research blog. That is not enough for the stated goal: “best ever synthetic lead scoring dataset.”

Missing research streams include:

- Public lead-scoring dataset census: Kaggle, HF, UCI, GitHub, Data.World, and common bootcamp datasets such as X Education.
- Lead-scoring literature: predictive lead scoring, sales funnel modeling, survival/hazard modeling, uplift/lift evaluation, conversion-rate calibration, CRM data leakage patterns.
- B2B GTM realism: funnel conversion benchmarks, sales cycle durations, lead source mix, enterprise buying committee dynamics, SDR/outbound/inbound attribution, opportunity creation rates.
- Synthetic tabular data evaluation: SDMetrics, TSTR/TRTS, SynthCity, statistical fidelity metrics, privacy/disclosure risk, plausibility constraints, graph/relational synthetic data evaluation.
- Dataset documentation standards: Data Cards, Dataset Nutrition Labels, Model Cards analogues, MLCommons/Croissant if applicable, Kaggle metadata, HF dataset card specs.
- Educational dataset design: notebooks, assignments, instructor keys, leakage traps, calibration exercises, lift curves, cohort shift, and rubrics.

### 9. Its citations are not publication-grade

The report’s citations are internal browser IDs such as `【176731919908143†L15-L89】`. In a downloaded Markdown file, these are not durable references. They do not identify the source title, URL, or accessed date.

There are also citation-matching problems. For example, broad architecture claims are repeatedly cited to the same line span that appears to be a dataset-card snippet rather than the actual architecture specification. The report’s cited line ranges are too broad to be useful and sometimes do not support the claim precisely.

A better report should use:

- Source title.
- URL or repository path.
- Access date for web sources.
- Exact file path and line range for repository evidence.
- A bibliography grouped by platform docs, academic research, industry evidence, and repository files.

### 10. It contains platform-specific inaccuracies

The Kaggle section should be corrected. Kaggle’s current dataset metadata docs state that the upload folder should contain `dataset-metadata.json`; supported fields include `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image`. The docs also describe a recommended `dataset-cover-image.png` sibling file and specify a **minimum** image size of **560×280**, with header and thumbnail crops. The generated report instead mentions fields such as `isPrivate`, `maintainer`, and `updateFrequency`, and says a cover image must be 1200×400. That is not precise enough for an automated publisher.

The Hugging Face section is broadly right but underspecified. HF dataset cards use `README.md` with YAML metadata. The YAML can include `configs` and `data_files` so that multiple subsets/splits load without custom code. The attached repo already has such a card, but it needs hardening: `pretty_name`, `tags: tabular`, `dataset_info`, `default: true` for the main config, and a clearer split between task splits and relational tables.

## What the report did well

The report has useful instincts:

- It recognizes that leadforge should be a world simulator, not a generic tabular sampler.
- It emphasizes narrative context, reproducibility, leakage safety, documentation, and validation.
- It correctly points toward Kaggle and HF metadata/data-card requirements.
- It recommends external LLM critique, which matches the original prompt.
- It identifies the importance of lift curves, precision@K, and teaching notebooks.

These are good high-level themes. The problem is that the report stops before doing the hard work of connecting those themes to the actual codebase and release artifacts.

## Better process for producing the report the user actually asked for

### Phase 0: Build an evidence inventory

Extract the Repomix XML into files. Produce a source inventory:

- File tree by module.
- Python module/function/class counts.
- Docs inventory.
- Release artifact inventory.
- Test inventory.
- Scripts inventory.
- Dataset artifact inventory.

Create a table of evidence claims with source paths and line ranges. Do not write the final report until this matrix exists.

### Phase 1: Static code and architecture audit

Audit each package layer:

- `api/`: public surface, config precedence, bundle lifecycle.
- `cli/`: implemented commands, missing flags, error behavior, JSON output.
- `recipes/`: recipe schema, difficulty profiles, extensibility.
- `structure/`: motif families, graph validity, rewiring semantics.
- `simulation/`: population generation, stage transitions, hazards, event emission, direct conversion, churn, post-simulation entities.
- `render/`: relational tables, snapshots, task splits, manifests, cards.
- `exposure/`: public/instructor redaction and metadata filtering.
- `validation/`: invariants, realism, difficulty, drift, leakage, artifact integrity.
- `release/` and `scripts/`: build, validate, package, publish readiness.

For each layer, classify findings as:

- Exists and seems mature.
- Exists but needs hardening.
- Missing.
- Risky/unclear.

### Phase 2: Dynamic reproducibility audit

Install the package in editable mode. Run:

```bash
python -m pytest --collect-only -q
python -m pytest -q
leadforge list-recipes
leadforge generate --recipe b2b_saas_procurement_v1 --seed 42 --mode student_public --difficulty intermediate --out /tmp/leadforge_smoke
leadforge inspect /tmp/leadforge_smoke
leadforge validate /tmp/leadforge_smoke
python scripts/build_public_release.py /tmp/leadforge_release --generation-timestamp 2026-01-01T00:00:00+00:00
```

Record:

- Whether tests pass.
- Runtime and memory.
- Bundle row counts.
- Hash determinism.
- Validation errors/warnings.
- Any mismatch between docs and artifacts.

### Phase 3: Alpha dataset forensic audit

Download or inspect the alpha public dataset repository and the generated bundles, not just README snippets.

For each tier:

- Load task splits and flat CSV.
- Verify manifest hashes.
- Check row counts and class balance.
- Compute baseline AUC, PR-AUC, log loss, calibration curves, lift@K, precision@K, expected-value ranking, and top-decile conversion rates.
- Test leakage probes: label permutation, timestamp leakage, post-snapshot event leakage, ID leakage, duplicates, account-contact leakage across splits, relational rejoin leakage.
- Compare train/valid/test distribution shift.
- Compare public vs instructor bundles.
- Check whether redacted columns can be reconstructed from remaining relational tables.
- Generate a validation report with charts and JSON.

### Phase 4: Public dataset and competitor census

Search Kaggle, HF, GitHub, and UCI for lead-scoring datasets and notebooks. For each public dataset, record:

- Topic/domain.
- Row count and feature count.
- Whether relational or flat.
- Presence/quality of card/README.
- Label definition.
- Known leakage columns.
- Baseline performance.
- Educational value.
- Weaknesses that leadforge can explicitly surpass.

This would provide evidence for “best-in-class” rather than assuming it.

### Phase 5: Current platform packaging research

Use official docs only for platform requirements.

For Kaggle, produce an exact `dataset-metadata.json` schema and a publishing command. Include current license identifiers, resource schema requirements, image requirements, and API behavior.

For Hugging Face, produce a final `README.md` dataset card with YAML metadata, configs, data files, and optional `dataset_info`. Test `load_dataset()` locally or in a temporary repo if possible.

### Phase 6: Release specification and acceptance gates

Define a v1 release candidate as a directory plus validation reports. Each release candidate should include:

```text
release_candidate/
  README.md
  LICENSE
  DATASET_CARD.md
  CHANGELOG.md
  CITATION.cff
  dataset-cover-image.png
  kaggle/
    dataset-metadata.json
    README.md
  huggingface/
    README.md
  validation/
    validation_report.json
    validation_report.md
    figures/
      lift_curve.png
      calibration_curve.png
      missingness_heatmap.png
      conversion_by_source.png
      leakage_delta.png
  notebooks/
    01_intro_baseline.ipynb
    02_feature_engineering_from_relational_tables.ipynb
    03_leakage_and_time_windows.ipynb
    04_lift_curves_and_value_ranking.ipynb
  bundles/
    intro/
    intermediate/
    advanced/
    intermediate_instructor/
```

Acceptance gates should include:

- `leadforge validate` passes all bundles.
- Release-level validation passes across tiers and seeds.
- Kaggle metadata validates locally.
- Hugging Face `load_dataset()` works for all configs.
- Dataset cards pass markdown and metadata linting.
- Baseline metrics fall in target ranges.
- No forbidden leakage columns in public artifacts.
- Public/instructor exposure diff is exactly as intended.
- External LLM critiques produce no unresolved high-severity findings.
- A human spot-check confirms the first notebook runs end-to-end.

### Phase 7: LLM critique loop

Add a release critique runner, but make it structured and auditable.

Inputs:

- `README.md` / dataset card.
- `manifest.json`.
- `feature_dictionary.csv`.
- `validation_report.json`.
- Sample rows.
- Public/instructor diff summary.
- Mechanism summary, if instructor mode.

Output schema:

```json
{
  "model": "provider/model/version",
  "release_id": "...",
  "overall_score": 0,
  "findings": [
    {
      "severity": "critical|high|medium|low|nit",
      "category": "leakage|realism|documentation|platform|ethics|pedagogy|code",
      "claim": "...",
      "evidence": "file/path:line or artifact reference",
      "reproducer": "optional command or check",
      "suggested_fix": "..."
    }
  ],
  "missing_sections": [],
  "questions_for_maintainer": []
}
```

Use at least two providers or two model families when available. Save raw model outputs, parsed JSON, and an adjudicated summary. Treat LLM findings as review inputs, not as pass/fail truth.

## Better report format to aim for

A strong version of the original report should look like this:

```text
# Leadforge v1 Lead-Scoring Dataset Release Plan

## 0. Executive Summary
- One-page verdict.
- Top 10 release blockers.
- Recommended release shape.
- Definition of “v1 ready.”

## 1. Evidence and Method
- Inputs reviewed.
- Commands run.
- Web sources used.
- What was not verified.
- Source/evidence map.

## 2. Current-State Audit
### 2.1 Package architecture
### 2.2 Generator and simulation engine
### 2.3 Recipes and difficulty profiles
### 2.4 Rendering and bundle schema
### 2.5 Exposure modes and redaction
### 2.6 Validation and tests
### 2.7 Release tooling
### 2.8 Documentation

Each subsection:
- What exists.
- Evidence.
- Strengths.
- Gaps.
- Release implications.

## 3. Alpha Dataset Forensics
- Bundle inventory.
- Schema audit.
- Baseline metrics.
- Lift curves.
- Calibration.
- Leakage probes.
- Missingness and drift.
- Public vs instructor diff.
- Student CSV v7 lessons.

## 4. External Research
### 4.1 Public lead-scoring datasets and their weaknesses
### 4.2 Lead-scoring and B2B GTM realism
### 4.3 Synthetic data generation/evaluation standards
### 4.4 Dataset documentation standards
### 4.5 Kaggle/Hugging Face release standards

## 5. Best-in-Class Release Specification
- Dataset family name and positioning.
- File tree.
- Platform-specific packaging.
- Dataset cards.
- Notebooks.
- Validation report.
- Feedback channels.

## 6. Gap Matrix
- Gap.
- Current evidence.
- Severity.
- Recommended fix.
- Target artifact/test.

## 7. Roadmap
### Milestone 1: Release audit and gates
### Milestone 2: Release builder and platform packaging
### Milestone 3: Validation hardening
### Milestone 4: Documentation and notebooks
### Milestone 5: LLM critique integration
### Milestone 6: Dry run and publication

Each milestone:
- Goal.
- PRs/files.
- Commands.
- Acceptance criteria.
- Risks.

## 8. v2 Feedback Plan
- Break-me guide.
- Issue templates.
- Metrics requested from users.
- Triage labels.
- Planned v2 decision log.

## 9. Appendices
- Exact commands.
- Validation JSON schema.
- Dataset card template.
- Kaggle metadata template.
- HF README template.
- LLM critique prompt.
- Bibliography.
```

This format would directly answer the original prompt while giving the project owner a buildable plan.

## Concrete improved roadmap for leadforge from the current attached state

### Milestone A — Audit current release candidate

Deliverables:

- `docs/release/v1_current_state_audit.md`
- `release/validation/validation_report.md`
- `release/validation/validation_report.json`
- `release/validation/figures/*.png`

Work:

- Run full tests.
- Generate fresh release bundles with fixed timestamp.
- Validate all bundles.
- Verify public/instructor diffs.
- Reproduce baselines.
- Add missing evidence from `lead_scoring_intro` v7 into the v1 design notes.

Acceptance criteria:

- Full tests pass or all failures are triaged.
- Release bundles regenerate byte-identically with pinned timestamp.
- Validation report is produced from code, not hand-written.
- Known limitations are explicit and reconciled across README, HF card, and dataset card.

### Milestone B — Platform packaging

Deliverables:

- `release/kaggle/dataset-metadata.json`
- `release/kaggle/README.md`
- `release/huggingface/README.md`
- `release/dataset-cover-image.png`
- `scripts/package_kaggle_release.py`
- `scripts/package_hf_release.py`

Work:

- Generate Kaggle metadata from manifest and feature dictionary.
- Convert the existing HF card into a full HF README with `pretty_name`, `tabular` tag, configs, and dataset information.
- Validate image dimensions and file paths.
- Produce zip/tar artifacts.

Acceptance criteria:

- `kaggle datasets create --dir-mode zip` can run in dry-run/local packaging mode.
- `load_dataset(local_path, name="intermediate")` works for HF-style structure.
- All public artifacts have stable checksums.

### Milestone C — Release validation hardening

Deliverables:

- `leadforge/validation/release_quality.py`
- `leadforge/validation/leakage_probes.py`
- `leadforge/validation/reporting.py`
- `scripts/validate_release_candidate.py`

Work:

- Add lift curves, calibration, precision@K, AP, log loss, expected-value ranking.
- Add adversarial leakage probes.
- Add cross-seed and cross-tier stability summaries.
- Add relational rejoin leakage checks.
- Add account leakage / split independence checks.
- Add data-card consistency checks: manifest vs README vs feature dictionary.

Acceptance criteria:

- No high-severity leakage findings.
- Metrics fall within configured target bands.
- Charts and JSON are generated automatically.
- Validation output is included in both Kaggle and HF releases.

### Milestone D — Documentation and notebooks

Deliverables:

- `notebooks/01_baseline_lead_scoring.ipynb`
- `notebooks/02_relational_feature_engineering.ipynb`
- `notebooks/03_leakage_and_time_windows.ipynb`
- `notebooks/04_lift_curves_and_value_ranking.ipynb`
- `docs/release/break_me_guide.md`
- `docs/release/instructor_guide.md`

Work:

- Turn existing v7 teaching guidance into notebook structure.
- Include a “try to break this dataset” guide.
- Add a short modeling baseline and a stronger tree/GBM baseline.
- Add warnings about `total_touches_all` and how to use/remove it.
- Include expected outputs and sanity checks.

Acceptance criteria:

- Notebooks run top-to-bottom.
- Notebook outputs match validation report within tolerance.
- Every public-facing artifact links to the issue tracker and feedback instructions.

### Milestone E — LLM release critique

Deliverables:

- `leadforge/validation/llm_critique.py`
- `docs/release/llm_critique_prompt.md`
- `release/validation/llm_critique_raw/*.json`
- `release/validation/llm_critique_summary.md`

Work:

- Implement provider abstraction with env-var credentials.
- Create structured critique prompts.
- Feed dataset card, manifests, feature dictionary, validation report, and samples.
- Save raw and summarized findings.

Acceptance criteria:

- At least two independent critiques run successfully when credentials are present.
- No unresolved high-severity findings before release.
- LLM critique is optional and skipped cleanly without credentials.

### Milestone F — Dry run, publish, and feedback loop

Deliverables:

- `scripts/publish_kaggle.py`
- `scripts/publish_hf.py`
- `.github/ISSUE_TEMPLATE/dataset_breakage_report.yml`
- `.github/ISSUE_TEMPLATE/realism_feedback.yml`
- `docs/release/v1_release_notes.md`

Work:

- Upload private/draft versions.
- Smoke-test downloads and HF loading.
- Publish public versions.
- Open a “break this dataset” discussion and issue templates.

Acceptance criteria:

- Kaggle page renders with files, metadata, and notebook.
- HF page renders with card, configs, and dataset viewer where supported.
- Download and load examples work from a clean environment.
- Feedback intake is documented.

## Suggested grading rubric for the final v1 report

Use this rubric before accepting a future research report:

1. **Evidence fidelity**: Every claim about the repo or datasets has a file path, line range, command output, or artifact reference.
2. **Current-state accuracy**: The report distinguishes existing, partial, missing, and future work.
3. **Research depth**: The report surveys public lead-scoring datasets, industry lead-scoring practices, synthetic-data evaluation methods, and platform docs.
4. **Platform correctness**: Kaggle/HF instructions are current and tested.
5. **Release specificity**: The roadmap names exact files, commands, artifacts, tests, and acceptance gates.
6. **Pedagogical value**: The report addresses notebooks, instructor mode, leakage teaching, lift curves, calibration, value ranking, and cohort shift.
7. **Adversarial readiness**: The report includes how users should break the dataset and how feedback becomes v2 work.
8. **Citation quality**: All sources are durable and human-readable.

## Current official documentation corrections to preserve in future work

The next report should use official platform docs directly:

- Kaggle’s current metadata docs say the upload folder should include `dataset-metadata.json`; supported fields include `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image`. The image guidance uses `dataset-cover-image.png` and a minimum image size of 560×280, with specific header and thumbnail crops. Source: Kaggle API `datasets_metadata.md`, accessed 2026-05-05.
- Hugging Face dataset cards are repository `README.md` files with YAML metadata at the top. Metadata helps display license, language, size, tags, and data-file configuration. Source: Hugging Face Hub dataset card docs, accessed 2026-05-05.
- Hugging Face repository structure docs support `configs` and `data_files` in YAML for multiple configurations and splits, which is relevant to intro/intermediate/advanced dataset tiers. Source: Hugging Face Datasets repository structure docs, accessed 2026-05-05.
- Data Cards should document provenance, motivation, dataset overview, sampling, transformations, annotations/labels, validation, sensitivity, limitations, and maintenance. Source: Google Data Cards Playbook, accessed 2026-05-05.

## Bottom line

The generated report captured some right themes but did not perform the requested level of inspection or research. It should not be used as the roadmap for leadforge v1. The next iteration should be evidence-first, code-aware, artifact-aware, and release-gated. The best report would read less like generic advice and more like a product/engineering release plan backed by concrete repository findings, generated metrics, platform-ready artifacts, and acceptance criteria.

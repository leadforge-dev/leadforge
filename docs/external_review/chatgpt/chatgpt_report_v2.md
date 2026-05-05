# Leadforge v1 Lead-Scoring Dataset Release Plan

## 0. Executive summary

**Verdict:** Leadforge is much further along than a greenfield project. The current repo contains an end-to-end deterministic generator, motif-sampled hidden graphs, population generation, a 90-day simulation engine, relational bundle writing, public/instructor exposure modes, CLI commands, validation modules, a public-release builder, a Hugging Face-style card, a baseline release notebook, and a mature `lead_scoring_intro` v7 teaching lineage. The v1 work should therefore be framed as **release hardening and adversarial validation**, not core implementation.

**The biggest release blocker I found is not absence of generation; it is public relational leakage.** In a local 500-lead `student_public` smoke bundle, `tables/leads.parquet` still contained `converted_within_90_days` and `conversion_timestamp`, and `tables/opportunities.parquet.close_outcome == "closed_won"` plus `customers` existence reconstructed the target with **100% accuracy**. This is acceptable only if those relational tables are documented as post-outcome world records, not if they are marketed as feature-engineering inputs for a lead-scoring task. For a best-in-class public Kaggle/HF dataset, the public relational path must be made **snapshot-safe** or moved to an instructor/research companion.

**Recommended v1 release shape:**

```text
Public Kaggle/HF release:
  intro / intermediate / advanced flat lead-scoring task splits
  snapshot-safe relational tables only
  feature dictionary with leakage flags
  validation report, charts, notebooks, data card, break-me guide

Separate instructor/research companion:
  full world graph
  latent registry
  mechanisms
  full-horizon relational tables
  leakage-trap materials
  reproducibility manifest
```

**Definition of v1 ready:** A fresh release candidate can be generated from code; passes structural, snapshot, redaction, relational-leakage, split-leakage, calibration, lift, top-K, value-ranking, and platform packaging checks; renders valid Kaggle and Hugging Face packages; has notebooks that run top-to-bottom; and has no unresolved high-severity LLM or human review findings.

---

## 1. Evidence and method

I treated the second-attempt guidance and critique as constraints, not optional context. Those files specifically require an evidence-first, code-aware, release-oriented audit and warn against calling implemented components skeletal, ignoring existing CLI/release/validation/HF assets, or missing the `lead_scoring_intro` v7 track.   I extracted and inspected the attached Repomix package.

**Repository inventory from extracted Repomix package:**

| Item                            | Count |
| ------------------------------- | ----: |
| Total files                     |   194 |
| Python files                    |   149 |
| Python files under `leadforge/` |    78 |
| Test files                      |    56 |
| Scripts                         |    15 |
| Markdown/RST/TXT docs           |    22 |
| Notebooks                       |     2 |
| CSV files                       |     4 |
| YAML/YML files                  |     8 |
| Release files                   |     3 |
| `lead_scoring_intro/` files     |     9 |

Line counts from the extracted package: `leadforge/` ≈10.7k lines, `tests/` ≈9.4k lines, `scripts/` ≈3.9k lines, `lead_scoring_intro/` ≈4.8k lines, `docs/` ≈5.6k lines.

**Dynamic checks run:**

| Command / check                                                                        | Result                                                                                                                                      |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `PYTHONPATH=. python -m pytest --collect-only -q`                                      | 937 tests collected, exit 0                                                                                                                 |
| Full `pytest -q`                                                                       | Timed out at 300s around 53%; no failures observed before timeout                                                                           |
| CLI help through Typer app                                                             | Worked; commands present                                                                                                                    |
| `leadforge list-recipes`                                                               | Worked; found `b2b_saas_procurement_v1`                                                                                                     |
| `leadforge generate ... --mode student_public --difficulty intermediate --n-leads 500` | Exit 0; bundle generated                                                                                                                    |
| `leadforge inspect /tmp/leadforge_smoke`                                               | Exit 0; reported 9 tables, task splits, no metadata dir                                                                                     |
| `leadforge validate /tmp/leadforge_smoke`                                              | Exit 0                                                                                                                                      |
| Same smoke generation in `research_instructor` mode                                    | Exit 0; metadata dir present                                                                                                                |
| `scripts/build_public_release.py ... --generation-timestamp ...`                       | Timed out before final summary, but generated intro/intermediate/advanced/intermediate_instructor bundles; separate validation calls passed |
| Same-seed determinism smoke check                                                      | Two saved bundles with pinned timestamp matched by tree/hash comparison                                                                     |

**What I did not verify:** I did not complete the full test suite within the timeout; I did not upload to Kaggle or Hugging Face; I did not run `load_dataset()` against a real HF repo; I did not run a full multi-model leakage-probe suite beyond the smoke-bundle relational leakage check; and I did not download every alpha Parquet file from the public dataset repo, though I did inspect the public GitHub pages and local regenerated release artifacts.

---

## 2. Current-state audit of Leadforge

### 2.1 Architecture and design docs

**Exists:** The repo has strong design intent: world-first, relational-first, narrative-grounded synthetic CRM generation, one B2B SaaS procurement vertical, exposure modes, difficulty profiles, and LTV-ready but not LTV-shipping foundations. The README frames Leadforge as a simulated commercial-world generator, not a row sampler, and documents CLI/API usage, exposure modes, difficulty profiles, output bundle shape, and deterministic/relational/simulation-driven principles. `README.md:L1-L6`, `README.md:L34-L56`, `README.md:L74-L127`.

**Strength:** The design docs and implementation are aligned enough that the project is not just speculative documentation.

**Gap:** `pyproject.toml` already declares `version = "1.0.0"` and `Development Status :: 5 - Production/Stable`, while the public dataset release state is still alpha and v1 release blockers remain. `pyproject.toml:L5-L24`. This can confuse users because package version, framework maturity, and curated dataset v1 are not the same product milestone.

**Release implication:** Rename the upcoming public dataset release explicitly, for example `leadforge-lead-scoring-v1`, and document that it is the first curated public dataset release even if the Python package version is already `1.0.0`.

---

### 2.2 Public API

**Exists:** `Generator.from_recipe()` loads a registered recipe, resolves config, applies overrides, loads narrative, and constructs a `WorldSpec`. `Generator.generate()` samples the hidden graph, loads difficulty profile parameters, builds population, simulates the world, and returns a populated `WorldBundle`. `leadforge/api/generator.py:L43-L122`, `leadforge/api/generator.py:L124-L248`.

**Strength:** This is a working vertical-slice generator, not a placeholder.

**Gap:** The API does not yet expose release-oriented workflows: build a release candidate, validate release quality, package for Kaggle/HF, or publish/dry-run.

**Release implication:** Keep `Generator` as the framework API. Add release APIs separately, for example `leadforge.release.build_release_candidate()` and CLI wrappers.

---

### 2.3 CLI

**Exists:** The Typer CLI registers `list-recipes`, `generate`, `inspect`, and `validate`. `leadforge/cli/main.py:L1-L42`. `generate` supports recipe, seed, exposure mode, output, difficulty, population counts, horizon, and override YAML. `leadforge/cli/commands/generate.py:L12-L86`. `inspect` reads `manifest.json` and prints recipe, seed, mode, difficulty, horizon, package version, schema version, motif, table counts, task rows, and metadata presence. `leadforge/cli/commands/inspect.py:L14-L75`. `validate` calls `validate_bundle()` and exits nonzero on errors. `leadforge/cli/commands/validate.py:L12-L42`.

**Strength:** The core CLI is present and usable.

**Gaps:** `inspect` and `validate` do not yet expose `--json`; no `release` subcommands exist; no dry-run publishing; no credential checks; no platform package validation.

**Release implication:** Do not “implement CLI” from scratch. Add:

```text
leadforge release build
leadforge release validate
leadforge release package-kaggle
leadforge release package-hf
leadforge release publish-kaggle --dry-run
leadforge release publish-hf --dry-run
```

---

### 2.4 Recipe system and difficulty profiles

**Exists:** Recipe objects include id, title, vertical, primary task, supported modes, difficulty profiles, population defaults, horizon, label window, and snapshot day. Config precedence is implemented across defaults, override files, and explicit args. `leadforge/api/recipes.py:L30-L45`, `leadforge/api/recipes.py:L140-L164`, `leadforge/api/recipes.py:L187-L240`.

**Strength:** Difficulty is a first-class named profile, not an accidental result.

**Gap:** Current release validation does not fully prove that difficulty profiles reward stronger modeling rather than simply changing base rate. The public alpha baselines show AUC is roughly flat across tiers while AP and P@K collapse, which is pedagogically useful, but LogReg and HistGBM are too close for “better modeling lifts realistically” in the alpha. The alpha `BASELINES.md` reports LogReg AUC ≈0.87–0.89 and HistGBM ≈0.866–0.868 across tiers. ([GitHub][1])

**Release implication:** For v1, require difficulty gates that include not only conversion rate and AUC, but also AP, P@K, lift@K, calibration, Brier score, log loss, and model-family deltas.

---

### 2.5 Hidden graph and motif sampler

**Exists:** `sample_hidden_graph()` selects a motif, applies stochastic rewiring, and returns a validated graph. `leadforge/structure/sampler.py:L26-L83`. The motif library includes fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, and buying-committee-friction structures. `leadforge/structure/motifs.py:L46-L230`. Graph validation checks acyclicity, type legality, nondegeneracy, and outcome reachability. `leadforge/structure/graph.py:L252-L315`. Rewiring can drop optional nodes, jitter edge weights, and inject latent confounders. `leadforge/structure/rewiring.py:L42-L125`.

**Strength:** This directly implements the “distribution over plausible worlds” idea.

**Gaps:** Release reports do not yet summarize graph diversity across seeds/tier releases, motif frequencies, structural edit distances, or which mechanisms drive the released v1 seed. External reviewers need those summaries without opening `world_spec.json`.

**Release implication:** Add a `validation/graph_diversity.md` and public-safe `mechanism_summary_public.json`.

---

### 2.6 Mechanisms and simulation engine

**Exists:** The simulation engine is a discrete 90-day world simulator. It creates RNG substreams, assigns mechanisms, evolves leads daily, applies churn/stage transitions/direct conversion, emits touches/sessions/sales activities, updates labels from conversion within the label window, and creates opportunities, customers, and subscriptions. `leadforge/simulation/engine.py:L166-L210`, `leadforge/simulation/engine.py:L260-L398`, `leadforge/simulation/engine.py:L416-L476`.

Population generation creates accounts, contacts, leads, latent scores, and category-latent correlations. `leadforge/simulation/population.py:L141-L211`, `leadforge/simulation/population.py:L219-L380`, `leadforge/simulation/population.py:L410-L495`.

**Strength:** Leadforge has real simulation machinery, including post-conversion entities and multiple event tables.

**Gaps:** Some realism choices need external calibration and release documentation: sales-cycle timing, partner/inbound/outbound mix, opportunity lifecycle, direct conversion rate, rep policy, and customer/subscription treatment in public bundles.

**Release implication:** The v1 data card must include a “simulation simplifications” section. It should say which CRM phenomena are modeled, which are approximate, and which are not modeled.

---

### 2.7 Relational rendering and snapshot task generation

**Exists:** The bundle writer writes relational Parquet tables, builds snapshot task splits, writes dataset cards and feature dictionaries, applies exposure metadata, and writes a manifest. Redaction is applied to both relational tables and task splits. `leadforge/api/bundle.py:L62-L140`. Relational rendering writes accounts, contacts, leads, touches, sessions, sales activities, opportunities, customers, and subscriptions. `leadforge/render/relational.py:L42-L83`. Snapshot building supports `snapshot_day`, filters touches/sessions/sales/opportunities by snapshot cutoff, computes early/recent touches, expected ACV, and applies noise/missingness/outliers. `leadforge/render/snapshots.py:L57-L131`, `leadforge/render/snapshots.py:L153-L243`, `leadforge/render/snapshots.py:L307-L404`. Task splitting is deterministic and writes train/valid/test Parquet plus a task manifest. `leadforge/render/tasks.py:L20-L77`.

**Strength:** The flat task path is much safer than the full relational path.

**Critical gap:** Public relational tables currently include post-outcome information. In my smoke bundle, `tables/leads.parquet` included `converted_within_90_days` and `conversion_timestamp`; `tables/opportunities.parquet` included `close_outcome` and `closed_at`; `customers` and `subscriptions` existed only for converted leads. Those fields reconstruct the label directly. This is not just a documentation issue if the public release invites relational feature engineering.

**Release implication:** For v1, create a **snapshot-safe relational export**:

```text
data/relational_snapshot_safe/<tier>/
  leads.parquet              # no target, no conversion timestamp
  touches.parquet            # only events <= snapshot_time
  sessions.parquet           # only events <= snapshot_time
  sales_activities.parquet   # only events <= snapshot_time
  opportunities.parquet      # only opps created <= snapshot_time, no close_outcome/closed_at
```

Move full-horizon `customers`, `subscriptions`, full opportunities, and label-bearing lead records to instructor/research companion only.

---

### 2.8 Exposure and redaction modes

**Exists:** Exposure filters distinguish `student_public` and `research_instructor`; instructor mode writes metadata, public mode does not. `leadforge/exposure/filters.py:L23-L58`. Metadata writing includes graph, GraphML, latent registry, world spec, and mechanism summary. `leadforge/exposure/metadata.py:L1-L70`. Feature specs differentiate leakage-risk documentation from redaction policy, and `current_stage` / `is_sql` are redacted in public mode. `leadforge/schema/features.py:L16-L57`, `leadforge/schema/features.py:L153-L165`, `leadforge/schema/features.py:L287-L304`.

**Strength:** The exposure-mode architecture is real and valuable.

**Gap:** Redaction currently targets known columns, but does not yet enforce a full “no public join path reconstructs label” guarantee. The alpha exposure delta says `current_stage` and `is_sql` are removed and redaction is applied uniformly, but it does not address target labels, opportunity close outcome, customer existence, or subscription existence in public relational tables. ([GitHub][2])

**Release implication:** Add `leadforge/validation/relational_leakage.py` and fail v1 if any public relational join path can reconstruct the target above a strict threshold.

---

### 2.9 Validation suite

**Exists:** `validate_bundle()` checks required files, tables, task split files, hashes, foreign keys, unexpected leakage columns, exposure redaction, realism, and difficulty. `leadforge/validation/bundle_checks.py:L26-L55`, `leadforge/validation/bundle_checks.py:L71-L260`. Realism checks cover conversion-rate guardrails, nonempty tables, ranges, booleans, and stage diversity. `leadforge/validation/realism.py:L23-L162`. Difficulty validation defines profile target ranges and ordering. `leadforge/validation/difficulty.py:L12-L102`. Drift validation checks cross-seed stability. `leadforge/validation/drift.py:L44-L104`.

The lead-scoring validation module already includes ROC-AUC, PR-AUC, precision/recall/lift@K, value-aware ranking, leakage-trap deltas, group determinism, and v7 validation flow. `leadforge/validation/lead_scoring.py:L120-L161`, `leadforge/validation/lead_scoring.py:L423-L537`, `leadforge/validation/lead_scoring.py:L733-L808`.

**Strength:** Validation is present and nontrivial.

**Gaps:** Release-grade validation is not yet a single reproducible artifact with charts, calibration, Brier/log loss, relational leakage probes, split leakage probes, public/instructor diff assertions, cross-seed bands, and LLM critique.

**Release implication:** The v1 release should not rely only on `leadforge validate`; it needs `scripts/validate_release_candidate.py` producing `validation_report.json`, `validation_report.md`, and figures.

---

### 2.10 Release tooling, HF material, and notebooks

**Exists:** `scripts/build_public_release.py` builds intro/intermediate/advanced public bundles plus an intermediate instructor bundle, writes flat CSVs for public bundles, pins generation timestamps, copies the license, and validates bundles. `scripts/build_public_release.py:L1-L21`, `scripts/build_public_release.py:L37-L87`, `scripts/build_public_release.py:L112-L170`.

`release/HF_DATASET_CARD.md` already has YAML front matter with license, task categories, tags, size category, and configs for intro/intermediate/advanced splits. `release/HF_DATASET_CARD.md:L1-L44`. `release/README.md` already describes the release layout, quick start, dataset summary, leakage handling, research companion, and provenance. `release/README.md:L1-L167`. The repo contains a baseline release notebook, and the examples notebook inspects generated worlds.

**Strength:** Hugging Face packaging is partial, not absent.

**Gaps:** Kaggle metadata is missing; HF card is not yet a final repo `README.md` with `pretty_name`, `tabular`, `datasets`, `pandas`, `default: true`, and tested configs; there is no cover image; no publisher scripts; no post-upload smoke tests; and only one release notebook is present.

**Release implication:** Add platform package generation scripts, not hand-authored upload folders.

---

### 2.11 Test suite and CI

**Exists:** CI runs Ruff, mypy, tests on Python 3.11/3.12, and v5/v6/v7 dataset validation jobs. `.github/workflows/ci.yml:L13-L52`, `.github/workflows/ci.yml:L61-L140`. The extracted test suite collected 937 tests.

**Strength:** Test coverage is broad for a small project.

**Gap:** CI does not yet gate full release-candidate packaging, Kaggle/HF metadata validation, relational leakage, notebook execution, or release report generation.

**Release implication:** Add a release-candidate CI workflow that runs on demand and uploads validation artifacts.

---

## 3. Existing dataset and alpha release forensics

### 3.1 Public alpha release inventory

The public `leadforge-datasets` repo currently has a `v0.1.0-alpha` release folder with five bundles: intro, intermediate, advanced, intermediate instructor, and tiny demo. The README reports all bundles are generated from `b2b_saas_procurement_v1`, seed 42, leadforge 1.0.0, bundle schema v4, and 5,000 leads for the three main public tiers. It also lists companion artifacts: `BASELINES.md`, `EXPOSURE_DELTA.md`, `provenance.json`, `build.sh`, `validation.log`, and `baselines.py`. ([GitHub][3])

The alpha validation log reports all five bundles passed `leadforge validate`. ([GitHub][4])

### 3.2 Alpha difficulty tiers

The alpha baselines show a useful but incomplete difficulty story:

| Tier         | Train conversion | LogReg AUC | LogReg AP | LogReg P@100 |
| ------------ | ---------------: | ---------: | --------: | -----------: |
| intro        |            41.5% |      0.886 |     0.785 |          79% |
| intermediate |            20.1% |      0.880 |     0.559 |          65% |
| advanced     |             7.9% |      0.870 |     0.271 |          26% |

The alpha interpretation is reasonable: rank-order AUC remains high, while AP/P@K degrade as positives become sparser. ([GitHub][1]) The v1 release should go further: calibration, lift curves, and value capture need to be generated as figures, and a stronger model should show some realistic improvement over a simple model.

### 3.3 Public vs instructor mode

The alpha exposure delta documents that public and instructor bundles share the same recipe/seed/difficulty, with public redacting `current_stage` and `is_sql` and omitting hidden-truth metadata. ([GitHub][2]) This is a good pattern. The missing piece is a deeper assertion: no public relational table or join path should reveal the label, terminal opportunity status, post-snapshot activity, or customer existence.

### 3.4 `lead_scoring_intro` v7 lessons

The `lead_scoring_intro` v7 track is one of the strongest assets in the repo. It defines a 1,000-row student CSV at snapshot day 20, with target conversion within 90 days, student and instructor variants, and a purely causal temporal leakage trap in the instructor file. `lead_scoring_intro/RELEASE_v7.md:L19-L38`.

The v7 release records baseline AUC ≈0.671 and PR-AUC ≈0.426, GBM improving LR by ≈0.072 AUC, value-aware ranking uplift of 13.4% at K=25 and 20.3% at K=50, a subtle leakage-trap delta of ≈0.013, and a cohort split AUC drop of ≈0.089. `lead_scoring_intro/RELEASE_v7.md:L121-L196`, `lead_scoring_intro/validation_v7_report.json`.

**Lessons to carry into v1:**

1. Keep the student path simple and safe.
2. Keep leakage traps clearly separated from student-facing features.
3. Teach value-aware ranking, not only probability ranking.
4. Include cohort/time-shift evaluation.
5. Make tree/GBM lift over LR visible but not absurd.
6. Document limitations bluntly.
7. Provide a lecture/notebook sequence. `lead_scoring_intro/RELEASE_v7.md:L206-L278`.

### 3.5 What currently makes the dataset easy or hard to break

**Harder than common public datasets:**

* Relational world with accounts, contacts, leads, touches, sessions, sales activities, opportunities, customers, and subscriptions.
* Hidden motifs and stochastic rewiring.
* Public/instructor modes.
* Windowed snapshot logic.
* Feature dictionary with leakage flags.
* Difficulty tiers.
* Value-aware ACV signal.
* Cohort split lesson in v7.

**Easy to break today:**

* Public relational tables leak the label and post-outcome state unless redesigned.
* Alpha LogReg AUC is high and close to HistGBM, so the alpha may not reward model sophistication enough.
* Release-level validation does not yet probe ID leakage, account/contact split leakage, relational join leakage, calibration, Brier/log loss, or post-snapshot event leakage.
* Public release docs and feature dictionary must make the intentional leakage trap impossible to miss for Kaggle/HF users.

---

## 4. External research

### 4.1 Public lead-scoring dataset census

| Dataset                              | Platform     |                    Domain |   Rows | Shape             | Documentation quality | Main weakness                                     |
| ------------------------------------ | ------------ | ------------------------: | -----: | ----------------- | --------------------- | ------------------------------------------------- |
| X Education Lead Scoring             | Kaggle       |          Online education |  9,240 | Flat, 37 cols     | Many notebooks        | Overused, flat, leakage-suspect status/tag fields |
| `shawhin/lead-scoring-x`             | Hugging Face |     Processed X Education |  5,688 | Flat, 7 features  | Minimal card          | Very reduced feature set                          |
| Online Shoppers Purchasing Intention | UCI          | E-commerce session intent | 12,330 | Flat, 17 features | Solid UCI metadata    | Not CRM/B2B lead scoring                          |
| GitHub/PyCaret demos                 | GitHub/blogs |       Usually X Education | Varies | Flat              | Tutorial-centric      | Repeats same source dataset                       |

The canonical public X Education dataset is a flat online-education CRM dataset. A public EDA article reports 9,240 rows and 37 columns, and a 38.54% conversion rate. ([Analytics Vidhya][5]) The Hugging Face processed variant uses only seven key features and reports 5,688 rows. ([Hugging Face][6]) The UCI Online Shoppers dataset is a useful adjacent benchmark, but it is session-level e-commerce intent, not B2B CRM lead scoring; UCI reports 12,330 sessions, 17 features, and 84.5% negative class. ([UCI Machine Learning Repository][7])

**Implication:** Leadforge can plausibly be best-in-class if it ships relational/snapshot-safe data, data cards, validation, notebooks, and break-me artifacts. The public landscape is shallow.

### 4.2 Lead-scoring and B2B GTM realism

Current product documentation and case-study literature support Leadforge’s fit + engagement + stage/process design.

HubSpot’s scoring tool distinguishes fit scores based on properties, engagement scores based on events, and combined scores using both property values and events. ([HubSpot Knowledge Base][8]) Salesforce describes lead scoring as ranking leads based on behavior, demographics, and engagement to help sellers prioritize effort. ([Salesforce][9]) Adobe Real-Time CDP B2B describes predictive lead/account scoring as learning from opportunity-stage conversion events, aggregating person activities to account level, and using tree-based random forest/gradient boosting methods. ([Experience League][10])

The 2025 Frontiers B2B lead-scoring case study is especially relevant. It used real CRM data from January 2020 to April 2024, evaluated 15 classifiers, and found Gradient Boosting superior; it also identified source and lead status as important predictive features. ([Frontiers][11]) The same paper reports 23,154 CRM records and 67 fields, including source, status, reason for status, last activity, and contact fields, and later highlights lead source, reason/status, lead classification, product, responses, account type, and interest level as important. ([Frontiers][11]) It also notes B2B processes can involve longer consultative sales cycles and overloaded sales reps, matching Leadforge’s prioritization framing. ([Frontiers][11])

**Implication for v1:** The release should emphasize lead prioritization, top-K sales capacity, lift, value capture, calibration, and process timing, not only binary classification AUC.

### 4.3 Synthetic data generation and evaluation

For pure synthetic data like Leadforge, “fidelity to real data” is not enough because there is no single real reference dataset. Still, synthetic-data evaluation literature and tooling point to useful axes: statistical quality, relational/cardinality preservation, utility, privacy/disclosure risk, and documentation.

SDMetrics’ Quality Report evaluates statistical similarity through column shapes, column pair trends, and for multi-table data, cardinality and intertable trends. ([Synthetic Data Vault][12]) Leadforge should borrow the idea of multi-axis reporting, but adapt it to **mechanism-designed synthetic worlds**: validity, leakage safety, difficulty, utility, structural diversity, narrative plausibility, and public artifact correctness.

Datasheets for Datasets argues datasets should document motivation, composition, collection/creation, recommended uses, and related information. ([arXiv][13]) Google’s Data Cards Playbook defines data cards as structured summaries of essential dataset facts for stakeholders across the lifecycle and includes themes such as authorship, dataset overview, motivation, provenance, transformations, annotations/labeling, validation, sampling, and benchmarks. ([Google Research][14])

**Implication for v1:** Leadforge should ship a generated validation report and a human-readable data card. The card should not only describe files; it should describe DGP, snapshot policy, label policy, leakage traps, limitations, intended use, out-of-scope use, and maintenance.

### 4.4 Kaggle release requirements

Kaggle’s current official API docs say a dataset upload folder must contain `dataset-metadata.json` next to the uploaded files, and the metadata follows the Data Package specification. Supported fields include `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image`. ([GitHub][15])

Important Kaggle constraints: `title` must be 6–50 characters, `subtitle` 20–80 characters, dataset slug 3–50 characters, exactly one license entry is evaluated, and `resources[].schema.fields` must include all fields in order if provided. ([GitHub][15]) Supported `expectedUpdateFrequency` values include `never`, `annually`, `quarterly`, `monthly`, `weekly`, `daily`, and `hourly`. ([GitHub][15]) Kaggle’s cover image guidance currently recommends `dataset-cover-image.png` or `.jpg/.jpeg/.webp` beside `dataset-metadata.json`, with minimum 560×280 dimensions and specified 2:1 header and 1:1 thumbnail crops. ([GitHub][15])

**Implication:** Add a generator for `release/kaggle/dataset-metadata.json`, a cover image, and validation against these constraints.

### 4.5 Hugging Face release requirements

Hugging Face dataset repos render `README.md` as the dataset card, with YAML metadata at the top for license, language, tags, size, and data-file configuration. ([Hugging Face][16]) Supported repository structures and file formats such as CSV and Parquet can be loaded automatically with `load_dataset()` and can show a Dataset Viewer. ([Hugging Face][17]) The YAML `configs` field defines splits and subsets; multiple configurations can be loaded by name, and `default: true` can set the default config. ([Hugging Face][17]) Hugging Face also documents manual split/subset configuration and notes Parquet viewer-size issues can be mitigated with smaller row groups and page indexes. ([Hugging Face][18])

**Implication:** Convert `release/HF_DATASET_CARD.md` into the final HF repo `README.md`, add `pretty_name`, `tags: [tabular, lead-scoring, synthetic-data, crm, b2b, datasets, pandas]`, `configs` for all tiers, a default config, and test local `load_dataset()`.

---

## 5. Best-in-class v1 release specification

### 5.1 Dataset family shape

Ship a family, not one CSV:

```text
leadforge-lead-scoring-v1
  intro_public
  intermediate_public
  advanced_public
  intermediate_research_companion
```

The public tiers should be snapshot-safe. The research companion should be clearly marked “not for student exercises.”

### 5.2 Canonical public release tree

```text
leadforge-lead-scoring-v1/
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
    STUDENT_QUICKSTART.md
    LIMITATIONS.md

  data/
    intro/
      train.csv
      validation.csv
      test.csv
      lead_scoring.csv
      manifest.json
      feature_dictionary.csv
    intermediate/
      ...
    advanced/
      ...

    relational_snapshot_safe/
      intro/
        accounts.parquet
        contacts.parquet
        leads.parquet
        touches.parquet
        sessions.parquet
        sales_activities.parquet
        opportunities.parquet
      intermediate/
      advanced/

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
      value_capture.png

  notebooks/
    01_intro_flat_csv_baseline.ipynb
    02_relational_feature_engineering.ipynb
    03_leakage_and_time_windows.ipynb
    04_lift_calibration_value_ranking.ipynb

  kaggle/
    dataset-metadata.json

  huggingface/
    README.md
```

### 5.3 Public bundle contents

Public bundle should include:

* Flat task splits with labels.
* Snapshot-safe relational tables with labels and post-outcome fields removed.
* Feature dictionary with `leakage_risk`, `available_at`, `derived_from`, `entity_level`, and `recommended_for_modeling`.
* Manifest with row counts, checksums, recipe, seed, package version, schema version, snapshot day, horizon, and validation report hash.
* Notebook-safe starter path that excludes leakage-trap features by default.

### 5.4 Instructor/research companion contents

Instructor companion should include:

* Full hidden graph.
* Full world spec.
* Mechanism summary.
* Latent registry.
* Full-horizon relational tables.
* Instructor leakage-trap features.
* Public/instructor diff report.
* LLM critique raw outputs and adjudication.

Recommendation: keep this out of the default Kaggle dataset. Put it in a separate GitHub Release artifact or a separate HF repo/config named clearly as instructor/research material. This preserves teaching utility while enabling external audit.

### 5.5 Notebooks

Minimum notebooks:

1. **Intro flat CSV baseline:** LR/GBM, AUC, PR-AUC, P@K, lift, calibration.
2. **Relational feature engineering:** only snapshot-safe tables; demonstrate legal joins.
3. **Leakage and time windows:** deliberately add leakage trap and post-snapshot fields; show why invalid.
4. **Lift, calibration, value ranking:** use `expected_acv`, `P(convert) × expected_acv`, calibration curves, thresholding.

Acceptance: all notebooks run top-to-bottom and reproduce validation metrics within tolerance.

### 5.6 Validation report

Minimum release validation metrics:

* Row counts, class balance, split sizes.
* ROC-AUC, PR-AUC, log loss, Brier score.
* Calibration bins and reliability curve.
* Lift@1/5/10%, precision@50/100, recall@K.
* Top-decile conversion rate.
* Expected ACV captured at K.
* LR vs GBM vs source-only vs engagement-only vs leakage-probe models.
* ID-only model.
* Stage/opportunity/customer-only suspect models.
* Post-snapshot aggregate leakage model.
* Account/contact overlap across splits.
* Near-duplicate rows across splits.
* Public/instructor diff.
* Snapshot-window audit.
* Relational join leakage audit.
* Cross-seed stability.
* Cross-tier difficulty ordering.

---

## 6. Gap matrix

| Area                     | Current evidence                                                                                        | Gap                                         | Severity | Recommended fix                                                     | Acceptance criterion                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------- | -------- | ------------------------------------------------------------------- | -------------------------------------------- |
| Core generation          | End-to-end API exists; graph → population → simulation → bundle. `leadforge/api/generator.py:L124-L248` | Not the blocker                             | Low      | Keep stable                                                         | Smoke generation passes                      |
| CLI                      | `generate`, `inspect`, `validate`, `list-recipes` exist. `leadforge/cli/main.py:L39-L42`                | No release commands / JSON                  | Medium   | Add `leadforge release ...`, `--json`                               | Machine-readable CI output                   |
| Public relational tables | Writes full leads/opps/customers/subscriptions. `leadforge/render/relational.py:L42-L83`                | Direct target/post-outcome leakage          | Critical | Add snapshot-safe relational export; move full horizon to companion | No public join path reconstructs label       |
| Flat task                | Snapshot filtering exists. `leadforge/render/snapshots.py:L57-L243`                                     | Needs full leakage probes                   | High     | Add time-window and suspect-feature probes                          | No high-severity leakage                     |
| Exposure                 | Public/instructor modes exist. `leadforge/exposure/filters.py:L23-L58`                                  | Redaction too narrow for relational leakage | Critical | Expand redaction/safe-export policy                                 | Relational leak test passes                  |
| Validation               | Bundle, realism, difficulty, drift, v7 metrics exist                                                    | No release report/charts/calibration/LLM    | High     | Add `release_quality.py`, `leakage_probes.py`, reporting            | `validation_report.{json,md}` generated      |
| HF packaging             | `release/HF_DATASET_CARD.md` exists                                                                     | Needs final README/config/default/load test | Medium   | Add `package_hf_release.py`                                         | Local `load_dataset()` works                 |
| Kaggle packaging         | No `dataset-metadata.json` found                                                                        | Missing platform package                    | High     | Add `package_kaggle_release.py`                                     | Metadata validates; dry-run package produced |
| Notebooks                | One release baseline notebook                                                                           | Missing relational/leakage/value sequence   | Medium   | Add 4 notebooks                                                     | All execute                                  |
| v7 lessons               | Strong v7 track exists                                                                                  | Not fully propagated into v1 spec           | Medium   | Port v7 teaching sequence                                           | Data card/notebooks include v7 lessons       |
| Feedback loop            | Alpha repo exists                                                                                       | No issue templates/break-me guide           | Medium   | Add GitHub templates + guide                                        | Public pages link feedback channels          |
| Scope                    | LTV-ready internals exist                                                                               | Risk of v1 scope creep                      | Medium   | State out-of-scope clearly                                          | No LTV/leaderboard work in v1                |

---

## 7. Roadmap to v1

### Milestone 1 — Release audit and acceptance gates

**Goal:** Freeze the current-state evidence and define v1 gates.

**Work items:**

* Add `docs/release/v1_current_state_audit.md`.
* Add `docs/release/v1_acceptance_gates.md`.
* Regenerate intro/intermediate/advanced/instructor bundles with pinned timestamp.
* Record command logs.
* Record public/instructor diff.
* Record known relational leakage finding.

**Files likely touched:**

```text
docs/release/v1_current_state_audit.md
docs/release/v1_acceptance_gates.md
scripts/build_public_release.py
```

**Commands:**

```bash
python -m pytest --collect-only -q
python -m pytest -q
python scripts/build_public_release.py /tmp/leadforge_v1_rc \
  --generation-timestamp 2026-01-01T00:00:00+00:00
leadforge validate /tmp/leadforge_v1_rc/intermediate
```

**Acceptance:** Full tests pass or failures triaged; release bundles regenerate; relational leakage is documented as a blocker, not ignored.

---

### Milestone 2 — Snapshot-safe public relational export

**Goal:** Remove direct and join-based label leakage from public relational data.

**Work items:**

* Add `leadforge/render/relational_snapshot_safe.py`.
* Add `leadforge/validation/relational_leakage.py`.
* Drop target and conversion timestamps from public `leads`.
* Filter event tables to `timestamp <= lead_created_at + snapshot_day`.
* Drop `close_outcome` and `closed_at` from public `opportunities`.
* Omit `customers` and `subscriptions` from public feature-engineering exports.
* Keep full-horizon tables only in instructor companion.

**Acceptance:** A leak probe using only public relational tables cannot reconstruct `converted_within_90_days` above configured tolerance; a customer/opportunity-only model fails because those fields are absent or snapshot-safe.

---

### Milestone 3 — Platform package generation

**Goal:** Build Kaggle and HF upload folders from release manifests.

**Work items:**

```text
scripts/package_kaggle_release.py
scripts/package_hf_release.py
release/kaggle/dataset-metadata.json
release/huggingface/README.md
release/dataset-cover-image.png
```

**Kaggle acceptance:**

* `dataset-metadata.json` contains valid `title`, `subtitle`, `description`, `id`, `licenses`, `resources`, `keywords`, `expectedUpdateFrequency`, `userSpecifiedSources`, and `image`.
* Title/subtitle/slug/image constraints pass.
* Resource schemas include fields in order.
* Package zip is produced without credentials.

**HF acceptance:**

* `README.md` has YAML metadata with `pretty_name`, `license`, `language`, `task_categories`, `size_categories`, `tags`, and `configs`.
* Main config is `default: true`.
* `load_dataset(local_path, "intermediate")` works or blocker is recorded.

---

### Milestone 4 — Release validation hardening

**Goal:** Turn validation into a release artifact.

**Work items:**

```text
leadforge/validation/release_quality.py
leadforge/validation/leakage_probes.py
leadforge/validation/reporting.py
scripts/validate_release_candidate.py
release/validation/validation_report.json
release/validation/validation_report.md
release/validation/figures/*.png
```

**Acceptance:**

* No critical leakage findings.
* Metrics within configured tier bands.
* Calibration and lift charts generated.
* Relational leak probes pass.
* Public/instructor diff is intentional.
* Report is included in Kaggle/HF packages.

---

### Milestone 5 — Documentation and notebooks

**Goal:** Make the release usable by educators, students, and external breakers.

**Work items:**

```text
docs/release/DATASET_CARD.md
docs/release/GENERATION_METHOD.md
docs/release/BREAK_ME_GUIDE.md
docs/release/STUDENT_QUICKSTART.md
docs/release/INSTRUCTOR_GUIDE.md
notebooks/01_intro_flat_csv_baseline.ipynb
notebooks/02_relational_feature_engineering.ipynb
notebooks/03_leakage_and_time_windows.ipynb
notebooks/04_lift_calibration_value_ranking.ipynb
```

**Acceptance:** Notebooks run top-to-bottom; notebook metrics match validation report; leakage-trap use is clearly separated from normal modeling.

---

### Milestone 6 — LLM critique integration

**Goal:** Add the external LLM review loop requested in the original milestone.

**Work items:**

```text
leadforge/validation/llm_critique.py
docs/release/llm_critique_prompt.md
release/validation/llm_critique_raw/*.json
release/validation/llm_critique_summary.md
```

**Input bundle:**

* README / dataset card.
* Generation method.
* Manifest.
* Feature dictionary.
* Validation report.
* First 100 public rows.
* Public/instructor diff.
* Public-safe mechanism summary.

**Output schema:**

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

**Acceptance:** Runs with credentials, skips cleanly without credentials, produces structured findings, and no unresolved high-severity findings remain.

---

### Milestone 7 — Dry-run publication and public feedback loop

**Goal:** Publish safely and invite breakage reports.

**Work items:**

```text
scripts/publish_kaggle.py
scripts/publish_hf.py
.github/ISSUE_TEMPLATE/dataset_breakage_report.yml
.github/ISSUE_TEMPLATE/realism_feedback.yml
docs/release/v1_release_notes.md
```

**Acceptance:**

* Kaggle private/draft upload tested.
* HF private repo upload tested.
* Download/load smoke tests pass.
* Public pages link to break-me guide and issue templates.
* LTV, leaderboard, and other task families remain out of v1 scope.

---

## 8. Suggested v2 feedback plan

The v1 public framing should explicitly ask users to break the dataset in these ways:

1. Find direct leakage.
2. Reconstruct labels through relational joins.
3. Beat baseline lift with legitimate features.
4. Show unrealistic marginal or joint distributions.
5. Show unrealistic sales-cycle or funnel dynamics.
6. Identify documentation ambiguity.
7. Find platform loading/viewer problems.
8. Propose better industry calibration sources.

Feedback should be triaged into:

```text
critical-leakage
realism
difficulty
documentation
platform
notebook
pedagogy
v2-idea
out-of-scope-v1
```

Keep a `docs/release/v2_decision_log.md` that records accepted/rejected feedback and why. Do not add LTV, leaderboard, or other GTM tasks to v1.

---

## 9. Bottom line

Leadforge’s current state is strong. The right next move is not to build the generator; it is to make the curated public release impossible to dismiss.

The required release-hardening work is concrete:

1. Fix public relational leakage.
2. Generate platform-native Kaggle/HF packages.
3. Produce a release validation report with charts and adversarial probes.
4. Port v7’s strongest teaching lessons into the multi-table v1 release.
5. Add notebooks, break-me guide, issue templates, and LLM critique.
6. Publish public data and separate instructor/research truth cleanly.

Until the relational leakage issue is fixed, the v1 dataset should not be released as a best-in-class public lead-scoring dataset. Once fixed, Leadforge has enough implemented machinery to plausibly exceed the current public lead-scoring dataset landscape.

[1]: https://github.com/leadforge-dev/leadforge-datasets/blob/main/releases/v0.1.0-alpha/BASELINES.md "leadforge-datasets/releases/v0.1.0-alpha/BASELINES.md at main · leadforge-dev/leadforge-datasets · GitHub"
[2]: https://github.com/leadforge-dev/leadforge-datasets/blob/main/releases/v0.1.0-alpha/EXPOSURE_DELTA.md "leadforge-datasets/releases/v0.1.0-alpha/EXPOSURE_DELTA.md at main · leadforge-dev/leadforge-datasets · GitHub"
[3]: https://github.com/leadforge-dev/leadforge-datasets "GitHub - leadforge-dev/leadforge-datasets · GitHub"
[4]: https://github.com/leadforge-dev/leadforge-datasets/blob/main/releases/v0.1.0-alpha/validation.log "leadforge-datasets/releases/v0.1.0-alpha/validation.log at main · leadforge-dev/leadforge-datasets · GitHub"
[5]: https://www.analyticsvidhya.com/blog/2022/09/exploratory-data-analysis-eda-on-lead-scoring-dataset/ "Exploratory Data Analysis (EDA) on Lead Scoring Dataset -"
[6]: https://huggingface.co/datasets/shawhin/lead-scoring-x "shawhin/lead-scoring-x · Datasets at Hugging Face"
[7]: https://archive.ics.uci.edu/ml/datasets/Online%2BShoppers%2BPurchasing%2BIntention%2BDataset "UCI Machine Learning Repository"
[8]: https://knowledge.hubspot.com/scoring/build-lead-scores?utm_source=chatgpt.com "Build lead scores to qualify contacts, companies, and deals"
[9]: https://www.salesforce.com/blog/lead-scoring/?utm_source=chatgpt.com "Lead Scoring: How to Find the Best Prospects in 4 Steps"
[10]: https://experienceleague.adobe.com/en/docs/experience-platform/rtcdp/b2b-cdp-ai-ml/predictive-lead-and-account-scoring-intro/predictive-lead-and-account-scoring?utm_source=chatgpt.com "Predictive lead and account scoring in Real-Time CDP B2B"
[11]: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1554325/full "Frontiers | The relevance of lead prioritization: a B2B lead scoring model based on machine learning"
[12]: https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report "Quality Report | SDMetrics"
[13]: https://arxiv.org/abs/1803.09010?utm_source=chatgpt.com "Datasheets for Datasets"
[14]: https://sites.research.google/datacardsplaybook/ "The Data Cards Playbook - Data Cards Playbook"
[15]: https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets_metadata.md "kaggle-cli/docs/datasets_metadata.md at main · Kaggle/kaggle-cli · GitHub"
[16]: https://huggingface.co/docs/hub/datasets-cards "Dataset Cards · Hugging Face"
[17]: https://huggingface.co/docs/datasets/repository_structure "Structure your repository · Hugging Face"
[18]: https://huggingface.co/docs/hub/datasets-data-files-configuration "Data files Configuration · Hugging Face"

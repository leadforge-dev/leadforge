
# leadforge — Detailed Roadmap / Implementation Plan

**Status:** v1 implementation roadmap  
**Project type:** opinionated open-source Python framework + CLI  
**Repository / package / CLI name:** `leadforge`  
**License:** MIT  
**Primary target:** synthetic lead-scoring dataset generation, with LTV-ready foundations  
**Vertical for v1:** mid-market procurement / AP automation SaaS  
**Primary v1 task:** `converted_within_90_days`

---

## 1. Purpose of this document

This document converts the locked product/design decisions and the architecture/spec contract into an execution plan for `leadforge`.

It defines:
- the recommended implementation sequence,
- milestone boundaries,
- PR-sized work breakdowns,
- dependencies between workstreams,
- acceptance criteria,
- release gates,
- and what should explicitly be deferred until after v1.

This is the **operational** companion to the design and architecture documents.

It assumes the following are already locked:
- `leadforge` is a world-first, relational-first synthetic CRM/funnel generator.
- v1 supports one vertical only: mid-market procurement / AP automation SaaS.
- v1 ships a library, CLI, sample datasets, and example notebooks.
- v1 uses a hybrid discrete-time simulator.
- v1 exposes both `student_public` and `research_instructor` modes.
- v1 includes post-conversion customer/subscription entities internally, but does **not** yet ship LTV labels as first-class outputs.
- v1 graph variability comes from named motif/template families with stochastic rewiring.
- the primary supervised task is `converted_within_90_days`.

---

## 2. Delivery philosophy

## 2.1 Ship narrow, polished, and extensible

The project should not try to "earn" flexibility by building a huge generic core before it can generate a good dataset. The correct path is:

1. Build the narrowest end-to-end system that already reflects the final worldview.
2. Keep interfaces clean enough that later growth is additive, not a rewrite.
3. Prefer one excellent vertical and one excellent task over early breadth.

## 2.2 Build the vertical and framework together

`leadforge` is not the kind of system where it is wise to build a fully abstract engine before any recipe exists. The first vertical should drive the architecture.

That means:
- recipe work starts early,
- schema and simulation are validated against the v1 vertical continuously,
- and sample datasets are generated throughout the build rather than at the end.

## 2.3 Favor vertical slices over horizontal speculation

Each milestone should produce something runnable:
- a command,
- a world object,
- a bundle,
- a sample dataset,
- a notebook,
- a validation report.

Avoid long periods of “core work” that cannot generate artifacts.

## 2.4 Optimize for future contributors

Because this is intended as polished OSS, the early roadmap must include:
- repo hygiene,
- docs,
- stable manifests,
- tests,
- example outputs,
- and contributor-visible architecture.

---

## 3. Workstreams

The project naturally decomposes into nine workstreams:

1. **Foundation / repo scaffolding**
2. **Recipe + narrative system**
3. **Schema + entity/event model**
4. **Structure layer (motifs, templates, rewiring, graph validation)**
5. **Mechanism + simulation engine**
6. **Rendering + artifact bundle**
7. **Exposure modes + redaction**
8. **CLI + examples + notebooks**
9. **Validation + release engineering**

These should not be developed in perfect isolation, but they are useful planning buckets.

---

## 4. Release strategy

Recommended tagged releases:

- **v0.1.0** — repository and generation skeleton
- **v0.2.0** — first end-to-end narrow world generator
- **v0.3.0** — motif/template variability + exposure modes
- **v0.4.0** — polished relational outputs + flat task export + metadata
- **v0.5.0** — CLI-complete, docs-complete, sample-data-complete release candidate
- **v1.0.0** — polished OSS release

This does **not** mean six public releases are mandatory. It means the implementation plan should behave as though each stage could be released cleanly.

---

## 5. Milestone overview

## Milestone 0 — Project foundation and OSS scaffolding

### Goal
Create a professional-quality repository and package skeleton that can support the rest of the work without later cleanup becoming painful.

### Why this milestone exists
The quality bar is “polished OSS from the beginning.” That means the project cannot start as an unstructured prototype.

### Scope
- Initialize repository layout.
- Create Python package skeleton.
- Add versioning strategy.
- Add base tooling:
  - Ruff / formatting
  - pytest
  - mypy or pyright
  - pre-commit
  - GitHub Actions CI
- Add initial README.
- Add MIT license.
- Add contribution guidance.
- Add minimal docs scaffold.
- Add package install and CLI entrypoint skeleton.

### Deliverables
- installable `leadforge` package
- runnable `leadforge --help`
- passing CI
- documented dev environment

### Acceptance criteria
- `pip install -e .` works
- `leadforge --help` works
- tests run in CI
- lint/type checks run in CI
- repo has stable top-level structure matching the architecture doc closely enough to grow into

### Suggested PR breakdown
1. Repo bootstrap + package skeleton
2. Tooling + CI + pre-commit
3. CLI entrypoint + versioning + README baseline

### Out of scope
- real generation logic
- recipe system
- schema or simulation internals

---

## Milestone 1 — Canonical config, recipe, and model objects

### Goal
Establish the typed configuration and recipe system that everything else will depend on.

### Why this milestone exists
Without a stable recipe/config contract, the rest of the implementation will drift.

### Scope
- Implement typed models for:
  - `GenerationConfig`
  - `Recipe`
  - `WorldSpec` placeholder
  - `WorldBundle` placeholder
  - `ExposureMode`
- Implement recipe loading and validation.
- Add v1 recipe directory and baseline files for:
  - `b2b_saas_procurement_v1`
  - narrative defaults
  - difficulty profiles
- Implement RNG root and deterministic substream utilities.

### Deliverables
- `load_recipe("b2b_saas_procurement_v1")`
- `Generator.from_recipe(...)` skeleton
- parsed recipe object with validated defaults
- deterministic config hashing / manifest identity primitives

### Acceptance criteria
- recipe can be loaded from disk and validated
- config precedence rules are implemented
- seeds produce deterministic config-resolved objects
- exposure modes are represented centrally, not ad hoc

### Suggested PR breakdown
1. Core models + enums + RNG utilities
2. Recipe registry and loading
3. Initial recipe files + validation tests

### Out of scope
- graph sampling
- simulation
- rendering

---

## Milestone 2 — Narrative layer and dataset-card foundations

### Goal
Make the vertical concrete and human-readable early, rather than leaving narrative polish to the end.

### Why this milestone exists
Teaching realism is the top product priority. Narrative is not decoration.

### Scope
- Implement narrative spec types:
  - company
  - product
  - market
  - buyer roles
  - GTM motion
  - funnel stage vocabulary
- Add dataset-card rendering skeleton
- Add feature-dictionary scaffolding
- Add recipe-bound narrative templates

### Deliverables
- rendered draft narrative from recipe + config
- dataset card skeleton object
- feature dictionary schema definition

### Acceptance criteria
- a generation run can already render a meaningful narrative summary even if no real data are generated yet
- narrative fields are versioned and serializable
- recipe-specific defaults are reflected in output consistently

### Suggested PR breakdown
1. Narrative models and recipe binding
2. Dataset card renderer
3. Feature dictionary schema scaffold

### Out of scope
- relational tables
- target labels

---

## Milestone 3 — Schema and relational model backbone

### Goal
Lock the internal relational world model and table contracts.

### Why this milestone exists
The project’s entire future depends on relational-first internals, including LTV-readiness.

### Scope
- Implement canonical entity/table schemas for:
  - accounts
  - contacts
  - leads
  - touches
  - sessions
  - sales activities
  - opportunities
  - customers
  - subscriptions
- Implement ID generation and foreign-key consistency helpers
- Define row-building utilities and typed intermediate records
- Define canonical Parquet writing helpers and schema validation

### Deliverables
- schema module with typed row contracts
- validators for required columns and FK integrity
- empty or toy tables renderable via a bundle

### Acceptance criteria
- all required v1 tables can be instantiated as empty/placeholder tables and serialized
- table schemas are documented in code and tests
- IDs are deterministic and namespace-safe
- foreign-key validation passes on synthetic toy data

### Suggested PR breakdown
1. Entity and relationship models
2. ID generation + FK validation
3. Parquet serialization + schema tests

### Out of scope
- real data generation
- world dynamics

---

## Milestone 4 — Hidden structure layer: motif families, graph model, and constraints

### Goal
Implement the hidden-world variability mechanism.

### Why this milestone exists
This is where `leadforge` becomes more than a fixed hidden-world simulator.

### Scope
- Implement typed graph representation
- Implement node categories and edge metadata
- Implement v1 motif/template families:
  1. fit-dominant
  2. intent-dominant
  3. sales-execution-sensitive
  4. demo/trial-mediated
  5. buying-committee-friction
- Implement stochastic rewiring under constraints
- Implement graph validation:
  - acyclicity
  - reachability
  - node-type legality
  - nondegeneracy

### Deliverables
- `sample_hidden_graph(recipe, seed, difficulty, ...)`
- graph exports to JSON and GraphML
- deterministic motif selection with rewiring
- unit tests for graph validity

### Acceptance criteria
- different seeds yield meaningfully different graphs
- all sampled graphs pass structural validation
- graph export files are stable and readable
- each named motif family is represented in tests and examples

### Suggested PR breakdown
1. Graph representation + node types + validators
2. Motif family implementations
3. Rewiring rules + graph exporters
4. Graph sampling tests and debug visualizations

### Out of scope
- full simulation over time
- rendering public/research metadata differences

---

## Milestone 5 — Population generation and latent state initialization

### Goal
Generate the base commercial world population before dynamic events begin.

### Why this milestone exists
Dynamic behavior must start from a plausible population of accounts, contacts, leads, and latent traits.

### Scope
- Implement account population generation
- Implement contact generation conditional on account properties
- Implement lead creation generation
- Sample core latent traits:
  - account fit
  - budget readiness
  - process maturity
  - problem awareness
  - authority
  - responsiveness
  - engagement propensity
  - sales friction
- Bind latent sampling to the selected hidden graph and motif family

### Deliverables
- initial accounts/contacts/leads tables with latent backing state
- population builder module
- deterministic population counts with recipe + config controls

### Acceptance criteria
- population generation works without time evolution yet
- population output is coherent with vertical narrative
- latent state generation respects graph dependencies
- counts and distributions are sane under all bundled difficulty profiles

### Suggested PR breakdown
1. Account generation
2. Contact generation
3. Lead generation
4. Latent-state initialization tied to graph/mechanisms

### Out of scope
- event simulation
- target derivation

---

## Milestone 6 — Mechanism layer v1

### Goal
Implement the node and transition mechanisms that turn world structure into behavior.

### Why this milestone exists
The graph alone is not enough; the system needs mechanism families that express commercial dynamics.

### Scope
- Implement mechanism base classes
- Implement v1 mechanism families:
  - static categorical/bounded draws
  - additive/logistic influence
  - threshold and saturating transforms
  - interaction terms
  - count/event intensity mechanisms
  - discrete-time transition/hazard mechanisms
  - measurement/noise mechanisms
- Implement mechanism summary serialization
- Bind mechanisms to graph nodes/edges

### Deliverables
- functioning mechanism library
- mechanism assignment pass from graph to `WorldSpec`
- mechanism summary object

### Acceptance criteria
- every generated world has an explicit mechanism assignment
- mechanisms are deterministic given seed and config
- mechanism summaries can be serialized
- mechanism family usage is covered by tests

### Suggested PR breakdown
1. Base mechanism interfaces
2. Static and influence mechanisms
3. Transition/count mechanisms
4. Measurement/noise mechanisms
5. Mechanism summary serialization

### Out of scope
- full event engine
- public bundle publication

---

## Milestone 7 — Hybrid discrete-time simulation engine

### Goal
Produce the first real evolving world and derive conversion outcomes from events, not direct labels.

### Why this milestone exists
This is the heart of the v1 product.

### Scope
- Implement world state container
- Implement discrete-time scheduler (daily steps)
- Implement event emission for:
  - touches
  - sessions
  - sales activities
  - stage changes
  - opportunity creation
  - conversion events
- Implement state update logic driven by mechanisms
- Implement post-conversion customer/subscription creation for converted leads
- Implement `converted_within_90_days` as event-derived outcome

### Deliverables
- `simulate_world(world_spec)` end-to-end
- populated relational tables
- valid conversion timestamps and outcomes
- post-conversion customer/subscription rows for converted cases

### Acceptance criteria
- a full world can be simulated end to end from recipe + config + seed
- `converted_within_90_days` is derived from simulated trajectory
- event timestamps are valid and ordered
- tables are mutually consistent
- post-conversion entities exist but no LTV labels are emitted yet

### Suggested PR breakdown
1. World state + scheduler
2. Touch/session emission
3. Sales-activity and stage progression
4. Opportunity + conversion creation
5. Customer/subscription creation
6. End-to-end simulation tests

### Out of scope
- flat ML snapshots
- exposure modes
- polished CLI

---

## Milestone 8 — Observation model and CRM messiness

### Goal
Make the output feel like business data rather than exposed simulation truth.

### Why this milestone exists
Operational realism is a core differentiator.

### Scope
- Implement measurement layer for:
  - missing enrichment
  - source noise
  - stage-label lag
  - imperfect attribution
  - partial session visibility
  - proxy scores
  - duplicate/merge artifacts (low-rate, optional)
- Ensure these affect observed tables and derived features, not hidden truth

### Deliverables
- observation pass over simulated world
- observed relational tables distinct from hidden internal truth
- configurable intensity via difficulty profiles

### Acceptance criteria
- public-facing observed tables differ meaningfully from hidden internal state
- difficulty profiles affect operational messiness in expected direction
- artifact validation still passes after noise application

### Suggested PR breakdown
1. Measurement model primitives
2. Attribution/source noise
3. Stage/update lag and proxy fields
4. Optional duplicate/merge artifacts

### Out of scope
- truth exposure filtering
- CLI inspection tools

---

## Milestone 9 — Rendering layer: bundle, manifests, dataset card, graph export

### Goal
Turn the simulated world into a polished artifact bundle.

### Why this milestone exists
A generator without a clean artifact contract is not useful as OSS or teaching infrastructure.

### Scope
- Implement `WorldBundle`
- Implement canonical bundle save layout
- Render:
  - relational tables
  - manifest
  - public summary
  - dataset card
  - feature dictionary
  - graph exports
  - world spec and mechanism summaries (internally first)
- Add file hashing and schema versioning

### Deliverables
- `bundle.save(path)` implementation
- canonical directory structure on disk
- stable `manifest.json`
- readable `dataset_card.md`

### Acceptance criteria
- bundle save is deterministic and complete
- all expected files are written for a research/instructor bundle
- dataset card is readable and useful
- manifest contents match actual written files

### Suggested PR breakdown
1. Manifest and bundle objects
2. Dataset card rendering
3. Graph and metadata exporters
4. Save/load helpers and file hashing

### Out of scope
- task splits
- public/research filtering rules

---

## Milestone 10 — Flat supervised task export and split logic

### Goal
Produce the actual ML-ready task that v1 is centered around.

### Why this milestone exists
The project is not complete until a student/instructor can use the generated data directly for modeling.

### Scope
- Implement flat snapshot feature export for `converted_within_90_days`
- Implement snapshot anchor rule
- Implement leakage-safe feature derivation
- Implement train/valid/test splits
- Implement task manifest
- Implement feature provenance mapping

### Deliverables
- `tasks/converted_within_90_days/train.parquet`
- `valid.parquet`
- `test.parquet`
- `task_manifest.json`
- feature dictionary with `derived_from`

### Acceptance criteria
- flat export can be trained on immediately
- no post-anchor leakage is present in features
- split strategy is documented and reproducible
- target semantics are consistent with relational events

### Suggested PR breakdown
1. Snapshot feature aggregation
2. Leakage checks and target derivation
3. Split generation
4. Task manifest and provenance

### Out of scope
- notebooks
- final CLI polish

---

## Milestone 11 — Exposure modes and publication filtering

### Goal
Implement `student_public` and `research_instructor` mode differences cleanly.

### Why this milestone exists
Truth exposure is a locked product decision and central to the framework’s usefulness.

### Scope
- Implement exposure-mode filter layer
- Define which files/fields are published in each mode
- Implement metadata redaction
- Ensure publication filtering happens after world generation, not before

### Deliverables
- exposure filter module
- public bundle publication
- instructor/research bundle publication
- comparison tests showing differential outputs from same world

### Acceptance criteria
- same hidden world can publish to both modes
- public mode excludes full hidden truth by default
- instructor mode includes rich hidden truth
- manifests clearly indicate mode and file inventory

### Suggested PR breakdown
1. Exposure-mode models and rules
2. Metadata filtering/redaction
3. Mode-differential tests and examples

### Out of scope
- advanced third/fourth exposure modes

---

## Milestone 12 — CLI completion and user workflows

### Goal
Make the project usable through a polished command-line interface.

### Why this milestone exists
The CLI is part of the required v1 deliverable and a major usability surface.

### Scope
- Implement required commands:
  - `list-recipes`
  - `generate`
  - `inspect`
  - `validate`
- Add human-readable output
- Add optional JSON output for machine use
- Add friendly error handling

### Deliverables
- fully functional CLI matching the architecture doc
- command help text
- command-level tests

### Acceptance criteria
- a user can generate a valid bundle entirely via CLI
- inspect and validate commands work on saved bundles
- error messages are clean and actionable

### Suggested PR breakdown
1. `list-recipes` and `generate`
2. `inspect`
3. `validate`
4. CLI output polish and tests

### Out of scope
- shell completion
- advanced interactive UI

---

## Milestone 13 — Validation harness and release gates

### Goal
Implement strong checks so that `leadforge` does not silently produce broken or degenerate worlds.

### Why this milestone exists
The framework’s credibility depends on generated artifacts being structurally and operationally sound.

### Scope
- Implement validation categories:
  - structural validation
  - simulation validation
  - artifact validation
  - realism/difficulty validation
- Define release gates for sample bundles
- Add automated validation to CI for committed sample bundles

### Deliverables
- `leadforge validate` full implementation
- validation test corpus
- release-gate checklist

### Acceptance criteria
- all generated sample bundles pass validation in CI
- key degenerate failure modes are caught
- validation output is understandable to users

### Suggested PR breakdown
1. Structural + simulation validation
2. Artifact validation
3. Realism/difficulty validation
4. CI integration for sample bundles

### Out of scope
- benchmark-oriented causal-identification scoring

---

## Milestone 14 — Sample datasets and example notebooks

### Goal
Make the project immediately useful and inspectable.

### Why this milestone exists
A polished OSS release needs examples, not just code.

### Scope
- Generate and commit:
  - one `student_public` sample bundle
  - one `research_instructor` sample bundle
- Create notebooks:
  1. Inspecting a generated world
  2. Lead-scoring baseline workflow
  3. Public vs instructor mode comparison
  4. optional recipe customization walkthrough
- Ensure notebooks run end to end

### Deliverables
- committed sample data
- committed notebooks
- notebook execution instructions

### Acceptance criteria
- notebooks work against committed sample bundles
- sample bundles are deterministic and versioned
- users can understand project value from examples quickly

### Suggested PR breakdown
1. Sample bundle generation + commit policy
2. Notebook 1 + 2
3. Notebook 3 (+ optional 4)
4. notebook smoke tests / CI if feasible

### Out of scope
- giant sample datasets
- polished marketing site

---

## Milestone 15 — Documentation, polish, and v1.0 release candidate

### Goal
Prepare a genuinely polished OSS release.

### Why this milestone exists
The project’s quality bar explicitly requires polish, not just functionality.

### Scope
- Improve README and package docs
- Add quickstart guides
- Add architecture diagram(s)
- Add “how to add a recipe later” notes
- Audit API/doc consistency
- Add changelog and release notes
- Review naming and manifest clarity
- Run final sample generation and validation passes

### Deliverables
- release candidate tag
- final docs set
- updated sample bundles if needed
- v1.0 checklist

### Acceptance criteria
- new user can install and generate a bundle via docs only
- docs match actual CLI/API
- all release gates pass
- repo presents as polished OSS

### Suggested PR breakdown
1. Docs and quickstart pass
2. API/CLI cleanup pass
3. release candidate artifacts and changelog

### Out of scope
- second vertical
- LTV labels
- plugin system

---

## 6. Cross-milestone dependency graph

A simplified dependency ordering:

- **M0** → everything
- **M1** depends on M0
- **M2** depends on M1
- **M3** depends on M1
- **M4** depends on M1 and loosely on M2/M3
- **M5** depends on M3 and M4
- **M6** depends on M4 and M5
- **M7** depends on M5 and M6
- **M8** depends on M7
- **M9** depends on M3, M7, M8
- **M10** depends on M7, M8, M9
- **M11** depends on M9 and M10
- **M12** depends on M1 and M9–M11
- **M13** depends on M4, M7, M9, M10, M11
- **M14** depends on M10–M13
- **M15** depends on all prior milestones

If parallelization is needed:
- M2 and M3 can proceed partly in parallel after M1.
- M4 can begin before M3 is fully feature-complete.
- M12 docs/CLI work can begin once bundle/save behavior stabilizes.

---

## 7. PR strategy

The implementation should prefer **small-to-medium PRs** with clear artifact outcomes.

Recommended PR characteristics:
- one logical capability per PR
- tests included
- docs touched if user-visible behavior changes
- no giant “foundation” PRs that change many layers at once unless unavoidable

Ideal PR size:
- roughly 300–900 lines of meaningful diff for normal PRs
- occasionally larger for schema or CLI integration, but avoid chronic mega-PRs

Every milestone above intentionally includes suggested PR groupings. Those should be treated as guidance, not law.

---

## 8. Testing strategy by layer

## 8.1 Unit tests
For:
- config parsing
- recipe loading
- RNG determinism
- ID generation
- graph validation
- mechanism behavior
- serialization helpers
- CLI argument parsing

## 8.2 Property/invariant tests
For:
- graph acyclicity
- foreign-key integrity
- nonnegative counts/durations where required
- deterministic output under same seed
- exposure filtering monotonicity

## 8.3 Integration tests
For:
- end-to-end generation via API
- end-to-end generation via CLI
- bundle save/load
- snapshot export + split generation

## 8.4 Golden artifact tests
For:
- sample bundle file inventory
- manifest shape
- selected metadata outputs
- maybe selected row-count expectations under committed samples

## 8.5 Notebook smoke tests
If feasible in CI:
- import and execute notebooks in lightweight mode against sample bundles

---

## 9. Release gates

A version should not be tagged as release candidate or v1.0 unless all gates below pass.

### Gate A — Generation gate
- Can generate both `student_public` and `research_instructor` bundles through API and CLI.

### Gate B — Artifact gate
- Required files exist and validate.
- Manifests are complete and accurate.
- Dataset card is rendered.

### Gate C — Task gate
- `converted_within_90_days` export is present and leakage-safe.
- Train/valid/test splits exist.

### Gate D — Structural gate
- Different seeds can yield different hidden worlds under the same recipe.
- Graph exports are valid.

### Gate E — Quality gate
- CI passes.
- Docs are coherent.
- Example notebooks run.
- Sample bundles validate.

### Gate F — OSS gate
- Install instructions work.
- Quickstart works.
- License and contributing docs present.
- Release notes written.

---

## 10. Explicit deferrals to post-v1

The following are intentionally deferred until after v1 unless a compelling reason appears:

1. **Second vertical**
2. **First-class LTV labels**
3. **Continuous-time or richer event engine**
4. **Plugin architecture for third-party recipe packs**
5. **Semisynthetic calibration against real data**
6. **External-API enrichment features**
7. **Learned generative backends**
8. **Large benchmark suite / paper-grade evaluation framework**
9. **Web UI or dashboard**
10. **Advanced intervention/uplift task surfaces beyond small internal hooks**

This list matters. It protects v1 from collapsing under ambition.

---

## 11. What the very first working end-to-end prototype should look like

Before striving for all v1 polish, the first meaningful prototype should achieve this minimal vertical slice:

- load recipe
- build deterministic config
- sample one motif family
- generate accounts, contacts, leads
- simulate touches/sessions/sales actions over 90 days
- derive `converted_within_90_days`
- render the relational tables
- render one flat task export
- save a minimal manifest and dataset card

If a PR sequence loses sight of this, the project is drifting.

---

## 12. Recommended implementation order in plain language

If I were starting the repo from zero, I would proceed in this exact practical order:

1. scaffold repo/package/CLI
2. lock config + recipe loading
3. lock schema + IDs
4. lock graph + motif families
5. generate population + latent state
6. implement mechanism library
7. implement simulation engine
8. implement observation/noise
9. render bundle + metadata
10. render flat task export
11. add exposure modes
12. complete CLI
13. add validation harness
14. commit sample data + notebooks
15. do final polish and release

That sequence minimizes rework while still keeping artifacts visible early.

---

## 13. Suggested ownership / focus by milestone type

If multiple assistants or contributors are involved, the cleanest division is:

### Contributor A — Core framework / config / packaging
M0, M1, parts of M9, M12, M15

### Contributor B — Schema / relational model / rendering
M3, parts of M9, M10

### Contributor C — Graph / motif / mechanism work
M4, M6

### Contributor D — Simulation engine
M5, M7, M8

### Contributor E — Validation / docs / examples
M13, M14, parts of M15

Even if one person writes most of it, this mental split helps.

---

## 14. Final roadmap statement

The correct implementation plan for `leadforge` is:

> **build a narrow but fully real end-to-end world simulator for one B2B SaaS revenue problem, then harden it into polished OSS through structured milestones that preserve narrative realism, relational internals, controlled hidden-world variability, and dual truth-exposure modes.**

That is the roadmap this document commits to.

Implementation should now proceed milestone by milestone, with each milestone yielding runnable artifacts and reducing ambiguity for the next.

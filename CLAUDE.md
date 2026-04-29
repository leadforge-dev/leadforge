# CLAUDE.md — leadforge

## Branch & PR Workflow (mandatory)

**Never push directly to `main`.** Every piece of work — feature, bugfix, doc update, plan update — follows this sequence:

1. `git checkout main && git pull` — ensure main is up to date.
2. `git checkout -b <descriptive-branch-name>` — branch from latest main.
3. Do the work; commit to the branch.
4. Update `.agent-plan.md` to reflect project state *after* the PR merges; commit that update to the same branch (same PR).
5. Open a PR against `main` on GitHub with a detailed description.
6. Apply the appropriate **labels** to the PR (create new ones if none fit — see label taxonomy below).
7. Assign the PR to the appropriate **milestone** (create a new one on GitHub if none fits).

Never use `git push origin main`, `git push --force origin main`, or any variant that targets `main` directly.

> **Team enforcement:** The above is reinforced by GitHub branch protection on `main`. The local `.git/hooks/pre-push` hook installed in this repo is a personal convenience only — it is not versioned and will not be present for other contributors.

### Label taxonomy

**Type** (one required):
`type: feature` · `type: bugfix` · `type: docs` · `type: test` · `type: refactor` · `type: ci` · `type: chore`

**Layer** (one or more, when touching package code):
`layer: core` · `layer: narrative` · `layer: schema` · `layer: structure` · `layer: mechanisms` · `layer: simulation` · `layer: render` · `layer: exposure` · `layer: validation` · `layer: cli` · `layer: api` · `layer: recipes`

**Status** (optional):
`status: in progress` · `status: needs review` · `status: blocked`

Existing labels that predate this taxonomy: `bug` · `documentation` · `enhancement` · `good first issue` · `help wanted` · `foundation` — use when appropriate.

### Milestone map

| Milestone | Covers | Roadmap |
|---|---|---|
| v0.1.0 — Repo & CLI skeleton | M0 | Foundation, CI, package scaffold |
| v0.2.0 — First end-to-end world | M1–M3 | Config/recipe, narrative, schema |
| v0.3.0 — Motif variability + exposure modes | M4–M6 | Structure, mechanisms, exposure |
| v0.4.0 — Polished relational output + task export | M7–M10 | Simulation, observation, render, task |
| v0.5.0 — CLI-complete release candidate | M11–M13 | CLI, validation harness |
| v1.0.0 — Polished OSS release | M14–M15 | Sample data, notebooks, docs polish |

If work spans multiple milestones, assign to the earliest one it unblocks.

---

## Project Identity
- Package / repo / CLI: `leadforge`
- License: MIT
- Purpose: opinionated Python framework + CLI for generating synthetic CRM/funnel datasets from simulated commercial worlds
- v1 vertical: mid-market procurement / AP automation SaaS
- Primary v1 task: `converted_within_90_days`

---

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Linting / formatting | Ruff |
| Type checking | mypy or pyright |
| Testing | pytest |
| Pre-commit | pre-commit hooks |
| CI | GitHub Actions |
| Tabular data | pandas + pyarrow / Parquet |
| Graph internals | networkx.DiGraph |
| Config models | dataclasses or Pydantic |
| CLI | (choose at M0 — typer or click) |
| File format (tables) | Parquet (canonical); CSV optional later |
| File format (metadata) | JSON |
| File format (narrative) | Markdown |
| File format (graph) | GraphML + JSON |

---

## CLI Commands

```bash
leadforge list-recipes
leadforge generate --recipe b2b_saas_procurement_v1 --seed 42 --mode student_public --difficulty intermediate --n-leads 5000 --out ./out/demo_bundle
leadforge inspect ./out/demo_bundle
leadforge validate ./out/demo_bundle
```

---

## Dev Commands

```bash
pip install -e ".[dev]"          # editable install with dev deps
pytest                            # run all tests
ruff check .                      # lint
ruff format .                     # format
mypy leadforge/                   # type check
pre-commit run --all-files        # pre-commit suite
```

---

## Architectural Invariants

### Generation
- All generation is **deterministic given (recipe, config, seed, version)**.
- All stochastic components derive from a single seeded RNG root; substreams must be derived deterministically.
- External API calls are **never required** — always optional behind extras.

### Data model
- Internal world is **relational-first**. Flat ML exports are derived products.
- Use **typed dataclasses/models** for all config, recipe, world-spec, manifest, and task-manifest objects. No ad hoc dicts at boundaries.
- Entity IDs are stable, opaque strings (e.g., `acct_000001`, `lead_000001`), unique within namespace and deterministic per run.

### Schema
- Tables: accounts, contacts, leads, touches, sessions, sales_activities, opportunities, customers, subscriptions.
- Primary task table rows = one lead snapshot, anchored at snapshot time.
- **No flat feature may use events occurring after the snapshot anchor** (leakage rule, non-negotiable).

### Hidden world
- World structure varies via **named motif/template families + stochastic rewiring** — never a single fixed DGP or unconstrained random graph.
- Required v1 motif families: fit-dominant, intent-dominant, sales-execution-sensitive, demo/trial-mediated, buying-committee-friction.
- Graph must be a **DAG** (acyclic). Validate on construction.

### Truth exposure
- Filtering happens **during rendering/publication, not during simulation**.
- `student_public` mode: excludes latent registry, full world spec, mechanism summary, rich hidden graph.
- `research_instructor` mode: full truth — hidden graph, world spec, latent registry, mechanism summary, provenance.
- `ExposureMode` enum is central, not ad hoc strings scattered through rendering code.

### Output bundle
```
bundle_root/
  manifest.json          # required in all modes
  dataset_card.md        # required in all modes
  feature_dictionary.csv # required in all modes
  tables/                # relational Parquet tables
  tasks/converted_within_90_days/{train,valid,test}.parquet + task_manifest.json
  metadata/              # exposure-mode filtered
```
- `manifest.json` must include: package version, recipe id, seed, generation timestamp, exposure mode, difficulty profile, table inventory with row counts, file hashes, `bundle_schema_version`.

### LTV
- Customer/subscription entities **exist in v1 internals** and may appear in relational outputs.
- LTV labels are **not** first-class task outputs in v1.

### Simulation
- v1 uses **hybrid discrete-time simulator** (daily steps, 90-day horizon for primary task).
- `converted_within_90_days` is **event-derived** (not a directly sampled Bernoulli).

---

## Package Layout (canonical)

```
leadforge/
  api/            generator.py, recipes.py, bundle.py
  cli/            main.py, commands/{generate,list_recipes,inspect,validate}.py
  core/           rng.py, ids.py, time.py, enums.py, models.py, exceptions.py, ...
  narrative/      spec.py, company.py, product.py, personas.py, market.py, funnel.py, dataset_card.py
  schema/         entities.py, relationships.py, events.py, features.py, tasks.py, dictionaries.py
  structure/      node_types.py, graph.py, motifs.py, templates.py, rewiring.py, sampler.py, constraints.py
  mechanisms/     base.py, static.py, transitions.py, counts.py, categorical.py, scores.py, hazards.py, measurement.py, policies.py
  simulation/     world.py, state.py, population.py, scheduler.py, engine.py, interventions.py
  render/         relational.py, snapshots.py, metadata.py, manifests.py, graph_export.py, notebooks.py
  exposure/       modes.py, filters.py, redaction.py
  validation/     invariants.py, artifact_checks.py, realism.py, difficulty.py, drift.py
  recipes/        registry.py, b2b_saas_procurement_v1/{recipe,narrative,schema,motifs,difficulty_profiles}.yaml
  examples/       notebooks/, configs/
  sample_data/    public/, instructor/
```

---

## Public API Contract (high-level)

```python
from leadforge.api import Generator, list_recipes

gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42, exposure_mode="student_public")
bundle = gen.generate(n_accounts=1500, n_contacts=4200, n_leads=5000, difficulty="intermediate")
bundle.save("./out/procurement_world_001")
```

Key abstractions: `Recipe`, `GenerationConfig`, `WorldSpec`, `WorldBundle`, `ExposureMode`.

---

## Config Precedence (highest → lowest)
1. Explicit function args / CLI flags
2. User override YAML/JSON file (`--override`)
3. Recipe defaults
4. Package defaults

---

## Commit and PR Conventions
- Small-to-medium PRs: ~300–900 lines of meaningful diff.
- One logical capability per PR; tests included.
- PR title describes capability, not file list.
- Tests required for: config parsing, recipe loading, RNG determinism, graph validation, mechanism behavior, serialization, CLI arg parsing.
- Property tests required for: graph acyclicity, FK integrity, deterministic output under same seed, exposure filtering monotonicity.

---

## Hard Constraints — Do Not Violate
- Never use a single fixed hidden world (DGP must vary by motif family + rewiring).
- Never leak post-snapshot-anchor data into flat task features.
- Never require external APIs for core generation.
- Never publish hidden truth in `student_public` mode.
- Never derive `converted_within_90_days` as a directly sampled label; it must emerge from simulated events.
- Never skip schema versioning in `manifest.json`.
- Do not add LTV labels as first-class task outputs in v1.

---

## Repository Map

```
leadforge/                    # Python package root
├── api/                      # Public API: Generator, Recipe, Bundle
│   ├── generator.py          # Generator.from_recipe() → .generate() → WorldBundle
│   ├── recipes.py            # Recipe loading, config resolution
│   └── bundle.py             # write_bundle() orchestrator
├── cli/                      # Click CLI
│   ├── main.py               # CLI entry point
│   └── commands/             # generate, inspect, validate, list_recipes
├── core/                     # Foundational utilities
│   ├── rng.py                # RNGRoot with named substreams
│   ├── ids.py                # Deterministic ID generation (acct_000001, etc.)
│   ├── models.py             # GenerationConfig, WorldSpec, WorldBundle
│   ├── enums.py              # ExposureMode, DifficultyProfile
│   └── exceptions.py         # Custom exception hierarchy
├── narrative/                # Vertical narrative (company, market, personas)
│   ├── spec.py               # NarrativeSpec and sub-spec dataclasses
│   └── dataset_card.py       # Markdown dataset card renderer
├── schema/                   # Relational data model
│   ├── entities.py           # 9 entity row dataclasses (AccountRow, LeadRow, etc.)
│   ├── features.py           # LEAD_SNAPSHOT_FEATURES — canonical feature spec
│   ├── relationships.py      # FK constraints (ALL_CONSTRAINTS)
│   ├── tasks.py              # SplitSpec, TaskManifest, CONVERTED_WITHIN_90_DAYS
│   └── dictionaries.py       # Feature dictionary CSV writer
├── structure/                # Hidden world graph
│   ├── graph.py              # WorldGraph (DAG wrapper)
│   ├── motifs.py             # 5 motif families
│   ├── rewiring.py           # Stochastic graph perturbation
│   └── sampler.py            # sample_hidden_graph()
├── mechanisms/               # Node/edge behavior
│   ├── policies.py           # assign_mechanisms() — motif → MechanismAssignment
│   ├── hazards.py            # ConversionHazard
│   ├── transitions.py        # StageSequence, HazardTransition
│   ├── counts.py             # PoissonIntensity, RecencyDecayIntensity
│   ├── categorical.py        # CategoricalInfluence, CHANNEL_QUALITY_SCORES
│   └── scores.py             # LatentScore
├── simulation/               # World evolution
│   ├── engine.py             # simulate_world() — 90-day daily loop
│   ├── state.py              # LeadSimState (per-lead mutable state)
│   └── population.py         # build_population() — accounts, contacts, leads
├── render/                   # Bundle output
│   ├── snapshots.py          # build_snapshot() — ML-ready lead table
│   ├── relational.py         # to_dataframes() — 9-table dict
│   ├── tasks.py              # write_task_splits() — train/valid/test Parquet
│   └── manifests.py          # build_manifest(), write_manifest()
├── exposure/                 # Truth filtering
│   ├── modes.py              # apply_exposure() dispatch
│   ├── metadata.py           # write_metadata_dir() for instructor mode
│   └── filters.py            # BundleFilter, FILTERS dict
├── validation/               # Bundle quality checks
│   ├── bundle_checks.py      # validate_bundle() orchestrator
│   ├── invariants.py         # Determinism + exposure monotonicity
│   ├── realism.py            # Conversion rates, feature ranges, stage diversity
│   ├── difficulty.py         # Known difficulty profile validation
│   └── drift.py              # Cross-seed stability
└── recipes/                  # Recipe definitions
    └── b2b_saas_procurement_v1/
        ├── recipe.yaml       # Recipe metadata + defaults
        ├── narrative.yaml    # Company, product, market, personas, funnel
        └── difficulty_profiles.yaml  # intro/intermediate/advanced
```

### Related repos

- **leadforge-datasets-private** — generated dataset archive
  - `b2b_saas_procurement_v1__intro__seed42/` — full relational bundle
  - `lead_scoring_intro/` — simplified single-CSV versions (v1–v4)
  - `scripts/` — build and validation scripts for simplified CSVs

---

## Generation Workflow

### Generate a full bundle

```bash
leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intro \
  --n-leads 5000 \
  --out ./out/bundle
```

### Build a simplified CSV (v4 example)

```bash
# In leadforge-datasets-private repo:
python scripts/build_v4_snapshot.py /path/to/bundle lead_scoring_intro/lead_scoring_intro_v4.csv
```

### Validate a simplified CSV

```bash
python scripts/validate_v4_dataset.py lead_scoring_intro/lead_scoring_intro_v4.csv
```

### Validate a full bundle

```bash
leadforge validate ./out/bundle
```

---

## student_public Mode Invariants

These are non-negotiable for any dataset published in `student_public` mode:

1. **No post-snapshot features** — all features computed from events ≤ snapshot day only.
2. **No outcome-stage columns** — `current_stage`, `funnel_stage` with `closed_won`/`closed_lost` are banned.
3. **No deterministic single-feature mapping** — for any feature value with n≥50, conversion rate must be in [2%, 98%].
4. **No hidden truth** — latent scores, mechanism parameters, world graph not included.
5. **No direct outcome columns** — `conversion_timestamp`, `close_outcome` are banned.
6. **No zero-variance features** — every included feature must have ≥2 distinct values.

Exception: deliberately included leakage traps (e.g., `total_touches_all` in v4) must be clearly documented in release notes and feature dictionary.

---

## How to Add New Features to the Snapshot

1. Add a `FeatureSpec` entry to `LEAD_SNAPSHOT_FEATURES` in `leadforge/schema/features.py`.
2. Compute the feature value in `build_snapshot()` in `leadforge/render/snapshots.py`.
3. If the feature needs new event data, add it to the simulation loop in `leadforge/simulation/engine.py`.
4. Update `leadforge/schema/dictionaries.py` if the feature dictionary format changes.
5. Run `pytest` and `leadforge validate` on a generated bundle.
6. Update the feature dictionary CSV description.

---

## v4 Dataset Plan

The current focus is producing a v4 lead scoring intro dataset. See `docs/v4/` for:
- `design.md` — requirements, contract, engine changes, implementation plan (single source of truth)
- `validation_spec.md` — automated validation checks
- `planning_pr_review.md` — self-review of the planning PR and treatment plan

---

## Reference Docs
- Design decisions: `docs/leadforge_design_doc.md`
- Architecture/spec: `docs/leadforge_architecture_spec.md`
- Implementation roadmap: `docs/leadforge_implementation_plan.md`
- v4 dataset plan: `docs/v4/design.md`

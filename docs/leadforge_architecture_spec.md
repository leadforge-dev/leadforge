# leadforge — Architecture & Specification Document

**Status:** v1 architecture and specification document  
**Project type:** opinionated open-source Python framework + CLI  
**Repository / package / CLI name:** `leadforge`  
**License:** MIT  
**Primary target:** synthetic lead-scoring dataset generation, with LTV-ready foundations  
**Vertical for v1:** mid-market procurement / AP automation SaaS  
**Primary v1 task:** `converted_within_90_days`

---

## 1. Purpose of this document

This document translates the product decisions from the design document into a concrete technical contract for `leadforge`.

It defines:
- the package architecture,
- the canonical internal abstractions,
- the v1 schema and artifact model,
- the public Python and CLI surfaces,
- the generation pipeline,
- truth-exposure behavior,
- data and metadata contracts,
- validation rules,
- and extension points.

This is the document that implementation work should follow.

It is **not** the milestone plan. Sequencing and work breakdown belong in the roadmap / implementation plan.

---

## 2. Scope and architectural stance

`leadforge` v1 is a **world simulator with dataset renderers**, not a generic tabular generator.

The architecture is therefore organized around seven layers:

1. **Narrative layer** — the concrete company/product/market story.
2. **Schema layer** — entities, relationships, events, and exported views.
3. **Structure layer** — hidden world motifs, typed graph family, and stochastic rewiring.
4. **Mechanism layer** — conditional generators, transition rules, measurement logic.
5. **Simulation layer** — hybrid discrete-time world evolution.
6. **Rendering layer** — relational tables, snapshot tasks, dataset cards, metadata.
7. **Validation layer** — invariants, realism checks, and artifact integrity.

The v1 design center is:
- **one vertical**,
- **one primary task**,
- **hybrid discrete-time simulation**,
- **relational internals**,
- **flat supervised export**,
- **dual truth exposure**,
- and **LTV-ready foundations without LTV labels yet**.

---

## 3. Architectural principles

## 3.1 Deterministic generation

Every generated world must be reproducible from:
- recipe identifier,
- configuration,
- package version,
- seed,
- and truth exposure mode.

All stochastic components must use a single seeded RNG root, with substreams derived deterministically.

## 3.2 Strongly typed internal model

Internals should distinguish explicitly between:
- account-level objects,
- contact-level objects,
- lead-level objects,
- events,
- latent variables,
- observed variables,
- targets,
- and metadata artifacts.

This should be enforced via typed dataclasses / models rather than ad hoc dicts.

## 3.3 Relational-first generation

The internal world should generate normalized tables and events first. Flat ML datasets are derived products.

## 3.4 Narrative-anchored semantics

Every feature, event, stage, and label must map to an interpretable business concept in the selected vertical.

## 3.5 Structured variability over unrestricted randomness

World structure should vary through a controlled family of motif templates with stochastic rewiring, not through unconstrained random graph generation.

## 3.6 Exposure-mode separation

The code should separate:
- **truth creation**,
- **truth storage**,
- and **truth publication**.

A generated world may include rich hidden truth internally even if the published artifact omits it.

## 3.7 v2 extensibility

Nothing in v1 should assume that all dynamics can be reduced forever to a simple discrete-time lead funnel. The architecture must support a later richer event engine, LTV projections, and additional tasks without conceptual rework.

---

## 4. Canonical package layout

```text
leadforge/
  __init__.py
  version.py

  api/
    __init__.py
    generator.py
    recipes.py
    bundle.py

  cli/
    __init__.py
    main.py
    commands/
      generate.py
      list_recipes.py
      inspect.py
      validate.py

  core/
    rng.py
    ids.py
    time.py
    enums.py
    paths.py
    hashing.py
    serialization.py
    models.py
    exceptions.py

  narrative/
    __init__.py
    spec.py
    company.py
    product.py
    personas.py
    market.py
    funnel.py
    dataset_card.py

  schema/
    __init__.py
    entities.py
    relationships.py
    events.py
    features.py
    tasks.py
    dictionaries.py

  structure/
    __init__.py
    node_types.py
    graph.py
    motifs.py
    templates.py
    rewiring.py
    sampler.py
    constraints.py

  mechanisms/
    __init__.py
    base.py
    static.py
    transitions.py
    counts.py
    categorical.py
    scores.py
    hazards.py
    measurement.py
    policies.py

  simulation/
    __init__.py
    world.py
    state.py
    population.py
    scheduler.py
    engine.py
    interventions.py

  render/
    __init__.py
    relational.py
    snapshots.py
    metadata.py
    manifests.py
    graph_export.py
    notebooks.py

  exposure/
    __init__.py
    modes.py
    filters.py
    redaction.py

  validation/
    __init__.py
    invariants.py
    artifact_checks.py
    realism.py
    difficulty.py
    drift.py

  recipes/
    __init__.py
    registry.py
    b2b_saas_procurement_v1/
      recipe.yaml
      narrative.yaml
      schema.yaml
      motifs.yaml
      difficulty_profiles.yaml

  examples/
    notebooks/
    configs/

  sample_data/
    public/
    instructor/
```

This layout is intentionally opinionated. It optimizes for clarity of responsibilities rather than ultra-minimal package depth.

---

## 5. Public abstractions

## 5.1 `Recipe`

A recipe is the canonical user-facing generation preset.

It binds:
- vertical,
- default narrative,
- schema family,
- motif family set,
- difficulty defaults,
- available tasks,
- and supported exposure modes.

Example canonical identifier:

```text
b2b_saas_procurement_v1
```

## 5.2 `GenerationConfig`

The full resolved config for a world generation run.

It should include:
- recipe id,
- seed,
- output path,
- exposure mode,
- difficulty profile,
- row/population counts,
- horizon settings,
- enabled exports,
- optional overrides,
- package version.

## 5.3 `WorldSpec`

The fully instantiated hidden world specification after sampling but before simulation.

It contains:
- resolved narrative,
- resolved schema,
- chosen motif family,
- concrete hidden graph,
- concrete mechanisms,
- policy parameters,
- time horizon,
- population plan.

## 5.4 `WorldBundle`

The in-memory result of one generation run.

It should provide access to:
- relational tables,
- snapshot tables,
- metadata,
- manifest,
- dataset card,
- graph exports,
- optional hidden truth objects.

## 5.5 `ExposureMode`

An enum-like type with at least:
- `student_public`
- `research_instructor`

Future additional modes may exist, but these two are normative in v1.

---

## 6. Public Python API

The library API should be concise and explicit.

### 6.1 High-level API

```python
from leadforge.api import Generator, list_recipes

recipes = list_recipes()

gen = Generator.from_recipe(
    "b2b_saas_procurement_v1",
    seed=42,
    exposure_mode="student_public",
)

bundle = gen.generate(
    n_accounts=1500,
    n_contacts=4200,
    n_leads=5000,
    difficulty="intermediate",
)

bundle.save("./out/procurement_world_001")
```

### 6.2 Lower-level API

```python
from leadforge.api import load_recipe
from leadforge.core.models import GenerationConfig
from leadforge.simulation.world import build_world_spec, simulate_world
from leadforge.render.metadata import render_bundle

recipe = load_recipe("b2b_saas_procurement_v1")
config = GenerationConfig(...)
world_spec = build_world_spec(recipe, config)
world = simulate_world(world_spec)
bundle = render_bundle(world, config)
```

### 6.3 API rules

- The high-level API must cover common use.
- Lower-level imports may exist, but they are secondary.
- v1 should avoid exposing too many unstable internal classes as part of the supported public API.

---

## 7. CLI specification

The CLI command is:

```bash
leadforge
```

## 7.1 Required commands

### `leadforge list-recipes`
Lists available recipes.

Example:

```bash
leadforge list-recipes
```

Expected output fields:
- recipe id
- title
- primary task
- vertical
- supported modes

### `leadforge generate`
Generates a dataset bundle.

Example:

```bash
leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intermediate \
  --n-leads 5000 \
  --out ./out/demo_bundle
```

Required options:
- `--recipe`
- `--seed`
- `--mode`
- `--out`

Recommended options:
- `--difficulty`
- `--n-accounts`
- `--n-contacts`
- `--n-leads`
- `--task`
- `--horizon-days`
- `--override path/to/config.yaml`

### `leadforge inspect`
Inspects an existing bundle.

Example:

```bash
leadforge inspect ./out/demo_bundle
```

Outputs summary information:
- recipe
- version
- seed
- mode
- files present
- table sizes
- task summary
- exposure summary

### `leadforge validate`
Runs artifact and schema validation on a generated bundle.

Example:

```bash
leadforge validate ./out/demo_bundle
```

## 7.2 CLI behavior rules

- Non-zero exit codes on generation failure or invalid configuration.
- Human-readable output by default.
- Optional `--json` for machine-readable command summaries.
- CLI must not require external APIs.

---

## 8. Canonical v1 recipe

## 8.1 Recipe id

```text
b2b_saas_procurement_v1
```

## 8.2 Narrative defaults

- Product: procurement and AP workflow automation software
- Geography: US and UK
- Target company size: 200–2,000 employees
- GTM motion: inbound + SDR-assisted + partner-driven
- Buyer roles: finance director, procurement manager, operations lead, IT/security reviewer
- Deal model: subscription SaaS with moderate ACV and moderate cycle length

## 8.3 Primary task

`converted_within_90_days`

## 8.4 Supported exposure modes

- `student_public`
- `research_instructor`

## 8.5 Supported difficulty profiles in v1

- `intro`
- `intermediate`
- `advanced`

These should map to bundled presets rather than free-form tuning only.

---

## 9. Internal world model

## 9.1 Core entity types

The v1 internal world must support these canonical entity types.

### Account
Represents a company or buying organization.

Examples of conceptual fields:
- `account_id`
- `company_name`
- `industry`
- `region`
- `employee_band`
- `estimated_revenue_band`
- `process_maturity`
- `tech_stack_profile`
- `latent_account_fit`
- `latent_budget_readiness`

### Contact
Represents an individual person associated with an account.

Examples:
- `contact_id`
- `account_id`
- `job_title`
- `role_function`
- `seniority`
- `buyer_role`
- `email_domain_type`
- `latent_problem_awareness`
- `latent_contact_authority`

### Lead
Represents a lead object for pre-conversion tracking.

Examples:
- `lead_id`
- `contact_id`
- `account_id`
- `created_at`
- `lead_source`
- `first_touch_channel`
- `current_stage`
- `owner_rep_id`
- `is_mql`
- `is_sql`
- `converted_within_90_days`

### Touch
Represents a marketing or sales touch.

Examples:
- `touch_id`
- `lead_id`
- `touch_type`
- `touch_channel`
- `touch_timestamp`
- `campaign_id`
- `response_signal`

### Session
Represents web or trial behavior.

Examples:
- `session_id`
- `lead_id`
- `timestamp`
- `session_type`
- `page_views`
- `pricing_page_views`
- `demo_page_views`
- `trial_activity_score`

### SalesActivity
Represents SDR/AE action.

Examples:
- `activity_id`
- `lead_id`
- `rep_id`
- `activity_type`
- `timestamp`
- `outcome`

### Opportunity
Represents a post-qualification opportunity object.

Examples:
- `opportunity_id`
- `lead_id`
- `created_at`
- `stage`
- `estimated_acv`
- `close_outcome`

### Customer
Represents a converted customer. Included in v1 internals for future-proofing.

### Subscription
Represents post-conversion subscription state. Included in v1 internals for future-proofing.

## 9.2 Required relationships

- `Account 1---N Contact`
- `Account 1---N Lead`
- `Contact 1---N Lead` (usually 1 but not required)
- `Lead 1---N Touch`
- `Lead 1---N Session`
- `Lead 1---N SalesActivity`
- `Lead 0..N Opportunity`
- `Opportunity 0..1 Customer`
- `Customer 1---N Subscription` (future-ready)

## 9.3 ID rules

All entity IDs should be stable, opaque string identifiers.

Examples:
- `acct_000001`
- `cnt_000001`
- `lead_000001`
- `touch_000001`

IDs must be unique within their namespaces and deterministic for a given run.

---

## 10. Simulation model

## 10.1 Simulation stance

v1 uses a **hybrid discrete-time simulator**.

This means:
- static and slowly varying hidden traits are sampled up front,
- observed states evolve over discrete time steps,
- touches, sessions, and sales actions are emitted as events,
- conversion outcome emerges from the simulated world trajectory,
- and flat features are derived later from the event history.

## 10.2 Time model

- Canonical v1 time unit: **day**
- Canonical horizon for main task: **90 days from lead creation**
- Simulation window per lead: from `lead_created_at` through `lead_created_at + 90d` for primary task generation
- Some entities may predate lead creation conceptually, but v1 exported lead-scoring task is anchored to the lead creation time

## 10.3 World state decomposition

World state should include:
- market context state,
- account latent state,
- contact latent state,
- lead funnel state,
- rep/policy state,
- and event log state.

## 10.4 Outcome generation

The label `converted_within_90_days` should be a projection over simulated events, not a directly sampled independent Bernoulli.

Minimum acceptable derivation:
- a lead reaches qualifying conditions,
- opportunity creation and/or conversion event happens,
- event timestamp occurs within the 90-day horizon.

## 10.5 Post-conversion inclusion

If a lead converts in the simulation, post-conversion customer/subscription records may be created in v1 internals and outputs, but v1 snapshot tasks do not yet expose LTV labels.

---

## 11. Hidden structure model

## 11.1 Node type system

The hidden graph should be a typed graph using at least the following node categories:

- `global_context`
- `account_latent`
- `contact_latent`
- `lead_state`
- `engagement_state`
- `sales_process_state`
- `observable_feature_source`
- `outcome`
- `post_conversion_state`

## 11.2 v1 motif/template families

The recipe must support a small family of named hidden-world motif templates.

Required families:

### 1. Fit-dominant
Conversion is primarily driven by account/contact fit, with engagement partly downstream of fit.

### 2. Intent-dominant
Behavioral engagement and urgency dominate conversion, even among mixed-fit leads.

### 3. Sales-execution-sensitive
Follow-up timing, rep quality, and process friction materially affect outcomes.

### 4. Demo/trial-mediated
Product demonstration or trial progression acts as a major mediator.

### 5. Buying-committee-friction
Multiple stakeholders and approval friction materially influence progression.

## 11.3 Rewiring rules

Each motif family should allow stochastic rewiring under constraints.

Permitted variability includes:
- optional mediator nodes,
- optional additional latent confounders,
- changed edge strengths,
- alternate measurement proxies,
- alternate stage bottlenecks,
- alternate source-channel influence structure.

Forbidden variability includes:
- chronologically impossible edges,
- nonsensical entity cross-level dependencies,
- orphaned outcome nodes,
- degenerate fully deterministic worlds with no meaningful noise.

## 11.4 Graph representation

Internally, v1 should use a directed graph abstraction that supports:
- node typing,
- edge metadata,
- topological ordering,
- export to GraphML / JSON,
- validation checks.

`networkx.DiGraph` is acceptable in v1.

---

## 12. Mechanism layer specification

## 12.1 Mechanism families

v1 should implement a limited but meaningful set of mechanism types.

Required mechanism families:

### Static latent mechanisms
Used for account/contact trait generation.
- categorical draws
- ordinal draws
- bounded numeric draws
- mixture distributions

### Influence mechanisms
Used for latent-state propagation.
- additive weighted combinations
- logistic transforms
- saturating transforms
- threshold transforms
- interaction terms

### Transition mechanisms
Used for stage or state movement over time.
- discrete-time hazard/probability transitions
- dwell-time sensitivity
- event-triggered state jumps

### Count/event mechanisms
Used for touches/sessions/actions.
- count intensity functions
- recency-sensitive event probabilities
- channel-conditioned event selection

### Measurement mechanisms
Used to turn hidden truth into observed imperfect CRM fields.
- missingness
- delayed updates
- noisy categorization
- proxy compression
- deduplication / duplication artifacts

## 12.2 Mechanism contract

All mechanism classes should conceptually expose:

```python
class Mechanism:
    def sample(self, context, rng):
        ...
```

Where `context` includes the relevant parent states and time information.

## 12.3 Required modeling patterns

The mechanism system must support these v1 patterns:
- fit affecting engagement but not perfectly determining it
- engagement affecting progression but not perfectly determining it
- source/channel influencing both volume and quality
- rep quality affecting follow-up effectiveness
- buying-committee friction reducing or delaying conversion
- latent variables observed through imperfect proxies rather than directly

---

## 13. Observation and measurement model

This is a first-class architectural layer.

## 13.1 Purpose

To transform hidden world state into realistic CRM-like observations.

## 13.2 Required v1 artifacts of messiness

The v1 observation model should support:
- incomplete enrichment on some accounts/contacts,
- imperfect first-touch attribution,
- delayed stage updates,
- noisy lead source normalization,
- partial session visibility,
- sparse rep notes / activity logs,
- derived scores that are not identical to hidden truth,
- and optional duplicate/merge artifacts at low rate.

## 13.3 Exposure-mode interaction

The hidden truth is generated before observation filtering. Exposure mode controls what metadata are rendered, not what hidden truth exists.

---

## 14. Rendering layer

## 14.1 Output bundle structure

A saved bundle should have this canonical shape:

```text
bundle_root/
  manifest.json
  dataset_card.md
  feature_dictionary.csv
  tables/
    accounts.parquet
    contacts.parquet
    leads.parquet
    touches.parquet
    sessions.parquet
    sales_activities.parquet
    opportunities.parquet
    customers.parquet
    subscriptions.parquet
  tasks/
    converted_within_90_days/
      train.parquet
      valid.parquet
      test.parquet
      task_manifest.json
  metadata/
    public_summary.json
    graph.graphml
    graph.json
    world_spec.json
    latent_registry.json
    mechanism_summary.json
    provenance.json
```

Not all metadata files are published in all exposure modes.

## 14.2 File format policy

Default file format should be **Parquet**.

Reasons:
- efficient,
- typed,
- good Python interoperability,
- suitable for larger generated datasets.

Optional CSV export may be supported later, but Parquet should be the canonical contract.

## 14.3 Dataset card

`dataset_card.md` should include:
- recipe id,
- package version,
- seed,
- narrative summary,
- task summary,
- exposure mode,
- table inventory,
- feature categories,
- label definition,
- suggested use cases,
- and caveats.

## 14.4 Feature dictionary

The bundle must contain a machine-readable feature dictionary for flat task exports.

Minimum columns:
- `feature_name`
- `feature_group`
- `dtype`
- `entity_level`
- `description`
- `derived_from`
- `available_in_public_mode`

---

## 15. Flat supervised export specification

## 15.1 Task directory

The primary task export lives at:

```text
tasks/converted_within_90_days/
```

## 15.2 Required splits

v1 should emit:
- `train.parquet`
- `valid.parquet`
- `test.parquet`

## 15.3 Row unit

Each row in the primary task export represents **one lead snapshot**.

## 15.4 Snapshot anchor

The snapshot should be taken at a clearly defined anchor time. For v1 default:
- anchor = lead creation date plus allowable observed pre-window behavior according to recipe settings

The exact default pre-window should be defined in recipe config; for example, a short lookback and same-day initialization. The critical requirement is that feature computation must not leak information from after the snapshot anchor.

## 15.5 Required columns

Every task table must include:
- `row_id`
- `lead_id`
- `account_id`
- `contact_id`
- `snapshot_time`
- features...
- target column: `converted_within_90_days`
- split indicator optional but not required if already split by file

## 15.6 Feature groups

The flat export should draw from these feature groups:
- firmographic
- contact / buyer profile
- source / attribution
- behavioral aggregates
- sales-process aggregates
- stage/state aggregates
- derived engagement / fit proxies

## 15.7 Leakage rules

No flat feature may use events occurring after the snapshot anchor.

`task_manifest.json` must document:
- snapshot rule,
- horizon,
- target semantics,
- leakage policy,
- split policy.

---

## 16. Relational table specifications

Below are the minimum required v1 schemas. Additional columns are allowed if documented.

## 16.1 `accounts`

Required columns:
- `account_id` (string, pk)
- `company_name` (string)
- `industry` (categorical)
- `region` (categorical)
- `employee_band` (categorical)
- `estimated_revenue_band` (categorical)
- `process_maturity_band` (categorical)
- `created_at` (datetime)

Optional/internal:
- hidden latent/account-fit fields in research mode metadata, not necessarily in public relational tables

## 16.2 `contacts`

Required:
- `contact_id`
- `account_id`
- `job_title`
- `role_function`
- `seniority`
- `buyer_role`
- `email_domain_type`
- `created_at`

## 16.3 `leads`

Required:
- `lead_id`
- `contact_id`
- `account_id`
- `lead_created_at`
- `lead_source`
- `first_touch_channel`
- `current_stage`
- `owner_rep_id`
- `is_mql`
- `is_sql`
- `converted_within_90_days`
- `conversion_timestamp` (nullable)

## 16.4 `touches`

Required:
- `touch_id`
- `lead_id`
- `touch_timestamp`
- `touch_type`
- `touch_channel`
- `campaign_id` (nullable)
- `touch_direction` (marketing/sales/partner)

## 16.5 `sessions`

Required:
- `session_id`
- `lead_id`
- `session_timestamp`
- `session_type`
- `page_views`
- `pricing_page_views`
- `demo_page_views`
- `session_duration_seconds`

## 16.6 `sales_activities`

Required:
- `activity_id`
- `lead_id`
- `rep_id`
- `activity_timestamp`
- `activity_type`
- `activity_outcome`

## 16.7 `opportunities`

Required:
- `opportunity_id`
- `lead_id`
- `created_at`
- `stage`
- `estimated_acv`
- `close_outcome`
- `closed_at` (nullable)

## 16.8 `customers`

Required in v1 if conversions occur:
- `customer_id`
- `opportunity_id`
- `account_id`
- `customer_start_at`

## 16.9 `subscriptions`

Required in v1 only at minimal future-ready level if customers are emitted:
- `subscription_id`
- `customer_id`
- `plan_name`
- `subscription_start_at`
- `subscription_status`

---

## 17. Metadata specification

## 17.1 `manifest.json`

The canonical artifact manifest. Must include:
- package version
- recipe id
- seed
- generation timestamp
- exposure mode
- difficulty profile
- task list
- table inventory with row counts
- file hashes
- bundle schema version

## 17.2 `public_summary.json`

A lightweight summary safe for public bundles.

## 17.3 `world_spec.json`

A rich, serializable representation of the sampled world. Research/instructor mode only.

## 17.4 `graph.json` and `graph.graphml`

Two representations of the hidden graph.

`graph.json` should be canonical for machine use. `graph.graphml` is for tooling interoperability.

## 17.5 `latent_registry.json`

Lists latent variables, their meanings, and exposure policies. Research/instructor mode.

## 17.6 `mechanism_summary.json`

Human-readable and machine-readable summary of mechanism assignments. Research/instructor mode.

## 17.7 `provenance.json`

Describes how flat features derive from relational tables and event windows.

---

## 18. Exposure mode filtering

## 18.1 `student_public`

Must include:
- dataset card
- manifest
- feature dictionary
- relational tables
- task tables
- public summary

Must exclude by default:
- latent registry
- full world spec
- mechanism summary
- rich hidden graph with sensitive causal annotations

May include:
- a redacted or simplified graph summary if explicitly desired by recipe design, but not full hidden truth

## 18.2 `research_instructor`

Includes everything in public mode plus:
- full hidden graph
- world spec
- latent registry
- mechanism summary
- detailed provenance
- any additional diagnostics

## 18.3 Filtering implementation rule

Filtering must happen during rendering/publication, not during simulation.

---

## 19. Difficulty profile specification

## 19.1 Difficulty model

Difficulty in v1 should be represented as named profiles, not only raw parameter tuning.

## 19.2 Required bundled profiles

### `intro`
- stronger primary signals
- lower noise
- simpler missingness
- weaker confounding
- lower feature count

### `intermediate`
- balanced realism
- moderate noise
- moderate confounding
- moderate class imbalance
- some noisy measurement artifacts

### `advanced`
- stronger proxy distortion
- stronger confounding
- more irrelevant or weakly relevant features
- more stage and attribution noise
- more distribution shift between splits

## 19.3 Implementation rule

Profiles should compile to concrete mechanism and rendering parameters through recipe-defined defaults.

---

## 20. Split specification

## 20.1 Default split strategy

v1 should support a default split strategy suitable for teaching realism.

Recommended default:
- time-aware split when possible,
- otherwise deterministic random split stratified on the target.

## 20.2 Required task split metadata

Each task manifest must state:
- split strategy type,
- split ratios,
- any temporal cutoff,
- whether target stratification was used,
- whether account-level leakage prevention was enforced.

## 20.3 Leakage-prevention option

The system should support account-aware splitting to reduce the chance of the same account appearing across train and test when the recipe or user requests it.

---

## 21. Validation layer

## 21.1 Validation categories

v1 validation should be divided into four categories.

### Structural validation
- graph is acyclic
- graph satisfies node-type constraints
- outcome node reachable from meaningful parents
- no empty motif instantiation

### Simulation validation
- required entities created
- event timestamps valid and ordered
- transitions legal
- conversion semantics consistent

### Artifact validation
- required files present
- schemas valid
- IDs unique where required
- foreign keys resolvable where required
- manifests internally consistent

### Realism/difficulty validation
- target base rate within recipe bounds
- event counts within sane ranges
- no obviously degenerate feature columns
- difficulty profile constraints approximately respected

## 21.2 Validation execution

Validation should run:
- automatically during generation,
- and separately via `leadforge validate`.

---

## 22. Sample dataset specification

The repository should ship sample generated bundles for at least:
- one `student_public` example
- one `research_instructor` example

These should be modest in size and generated deterministically.

Each sample should have:
- committed manifest,
- committed dataset card,
- example notebook using it.

---

## 23. Example notebooks specification

v1 must ship at least these notebooks:

1. **Inspecting a generated world**
   - loads bundle
   - explores tables
   - reads dataset card

2. **Lead-scoring baseline workflow**
   - trains baseline models on `converted_within_90_days`
   - demonstrates evaluation and calibration

3. **Public vs instructor mode comparison**
   - shows what changes in metadata exposure

Optional fourth notebook:
4. **Recipe customization walkthrough**

---

## 24. Configuration format

## 24.1 Config sources

Generation config may come from:
- CLI flags,
- Python keyword args,
- recipe defaults,
- optional user override YAML/JSON file.

## 24.2 Precedence

Highest to lowest:
1. explicit function args / CLI flags
2. override config file
3. recipe defaults
4. package defaults

## 24.3 Canonical override file shape

Example:

```yaml
recipe: b2b_saas_procurement_v1
seed: 42
mode: student_public
out: ./out/demo_bundle
population:
  n_accounts: 1500
  n_contacts: 4200
  n_leads: 5000
difficulty: intermediate
task:
  name: converted_within_90_days
  horizon_days: 90
exports:
  relational: true
  snapshots: true
  metadata: true
```

---

## 25. Serialization rules

## 25.1 Canonical serialization targets

- tabular data: Parquet
- manifests and metadata: JSON
- human-facing narrative card: Markdown
- graph interchange: GraphML + JSON

## 25.2 Schema versioning

Every bundle must include a `bundle_schema_version` in `manifest.json`.

## 25.3 Version pinning

Every bundle must include the exact `leadforge` package version used to generate it.

---

## 26. Error model and exceptions

The package should define structured exceptions, including at minimum:
- invalid recipe
- invalid config
- graph construction error
- simulation error
- render error
- validation error

CLI should catch and report these cleanly.

---

## 27. Internal implementation guidance

## 27.1 Data modeling

Use dataclasses or Pydantic-like models for:
- config objects,
- recipe metadata,
- world spec,
- manifest objects,
- task manifests.

## 27.2 Table construction

Use pandas and/or pyarrow-compatible structures for v1 rendering.

## 27.3 Graph tooling

Use `networkx` in v1 unless a clearly better lightweight alternative appears.

## 27.4 Optional dependencies

Any LLM or external API integration must be behind optional extras and must not affect core functionality.

---

## 28. Non-goals at the architecture/spec level

This spec intentionally does not require in v1:
- a full plugin ecosystem,
- multiple verticals,
- continuous-time simulation,
- causal-identification proof machinery,
- privacy-preserving training on real data,
- learned generator backends,
- or production-scale distributed generation.

These may come later, but the architecture should not assume them today.

---

## 29. Acceptance criteria for the architecture

The architecture/spec should be considered successfully implemented for v1 when all of the following are true:

1. A user can generate a full bundle via library or CLI.
2. The bundle includes relational tables, a flat task export, a dataset card, and manifests.
3. Two exposure modes publish different metadata surfaces from the same underlying world model.
4. Different seeds can yield different hidden worlds within the same recipe.
5. The primary task `converted_within_90_days` is correctly defined, rendered, and documented.
6. Customer/subscription entities exist in the internal world and can appear in outputs for converted leads.
7. Validation passes for generated bundles.
8. Example notebooks run against shipped sample data.

---

## 30. Final architectural statement

`leadforge` v1 should be implemented as:

> **a deterministic, relational-first, hybrid discrete-time world simulator for a concrete B2B SaaS lead-scoring vertical, with controlled hidden-world variability, configurable truth exposure, polished artifact rendering, and a narrow but extensible public API.**

That is the architecture this spec commits to. The roadmap document should now sequence its implementation.

# Foundation Document: A Python Framework for Generating Synthetic Lead-Scoring and LTV Datasets

**Working title:** `leadgen`  
**Purpose of this document:** a foundation memo for review by other AI assistants and future human design work.  
**Status:** exploratory design document, not a final architecture spec.  
**Date:** 2026-04-17

---

## 1. Executive summary

The proposed package should **not** be “yet another synthetic tabular data generator” that learns a single table distribution from seed data and resamples from it. The target is more ambitious:

1. A **domain simulator** for revenue problems, beginning with **lead scoring** and later extending naturally into **LTV / CLV**.
2. A **narrative-first** system: every generated dataset should come from a concrete company, product, buyer universe, funnel, and go-to-market motion.
3. A **structure-sampling** system: the data-generating process (DGP) should itself be variable. Users should be able to sample not only parameters of one fixed graph, but also the **graph family, graph topology, latent variables, mechanisms, event logic, and observation process**.
4. A **ground-truth-bearing** system: generated datasets should optionally ship with the hidden graph, latent variables, interventions, mechanism metadata, and evaluation hooks.
5. A **relational and temporal** system under the hood, even if the first exported task is a flat lead-scoring table.

The research strongly suggests that this project should be designed as a **hybrid simulator** rather than a pure tabular generator:

- Pure tabular generators are useful, but current literature emphasizes that **structural fidelity** remains a major weakness in synthetic tabular data generation and evaluation.[^R1][^R2]
- Recent work in SCM-based generation and benchmarking shows the value of **randomly sampled causal graphs**, flexible causal mechanisms, and downstream causal evaluation tasks rather than only marginal/statistical similarity.[^R1][^R3][^R4]
- Real lead scoring in industry is not just “one label from one row”; vendors and practice organize the problem around **fit + engagement + stage transitions + sales interactions + account context**.[^R5][^R6][^R7]
- CLV/LTV modeling literature strongly favors behavioral and temporal views of customer value, especially transaction processes, churn/dropout, and persistence models.[^R8][^R9]
- The best analogous synthetic-data systems in other domains are **modular simulators** (for example Synthea’s Generic Module Framework) rather than monolithic flat-table generators.[^R10]

### Recommended core direction

Build a **graph-and-simulation framework** with five layers:

1. **Narrative layer** – company, product, market, GTM model, personas, funnel stages, constraints.
2. **Schema layer** – entities/tables/events/features/labels/aggregation views.
3. **Structural layer** – sampled latent-variable graph / SCM / event dependencies / observation graph.
4. **Mechanism layer** – node-wise conditional mechanisms, policy rules, funnel transitions, time processes.
5. **Rendering layer** – relational event logs, CRM-like tables, feature snapshots, supervised ML tasks, graph metadata.

For the initial vertical, the best choice is a **mid-market B2B SaaS company** with a **multi-touch marketing and SDR-assisted sales funnel**. That vertical is the cleanest bridge between lead scoring now and LTV later.

---

## 2. Why this project is worth doing

### 2.1 Public lead-scoring data is weak

Public lead-scoring datasets are shallow, few, and usually either classroom-grade or narrow case studies. By contrast, recent B2B lead-scoring work still treats real company data as a valuable case study, not a mature public benchmark ecosystem.[^R11]

### 2.2 Existing synthetic tabular tooling is not enough for this goal

General-purpose synthetic data libraries such as SDV are useful for learning and sampling tabular patterns, including conditional sampling, but they are not by themselves a solution to a rich, graph-variable, narrative-grounded funnel simulator.[^R12]

The key problem is not only realism in marginal distributions. It is:

- realism of **structural dependencies**,
- realism of **funnel dynamics over time**,
- realism of **measurement and CRM artifacts**,
- realism of **business interventions**, and
- realism of **task difficulty**.

Recent benchmark work argues that existing evaluation often misses exactly these structural dimensions.[^R1][^R2]

### 2.3 The strongest adjacent precedents are modular simulators and SCM-based benchmark generators

Three especially relevant precedents emerge from the literature:

- **Synthea**: a modular, domain-grounded synthetic generator with disease modules and state transitions, showing the value of a narrative/module approach instead of a single learned distribution.[^R10]
- **causalAssembly**: a semisynthetic generator that uses a causal graph plus estimated node conditionals to guarantee Markov adherence to a ground-truth structure, showing a strong pattern for graph-first generation.[^R3]
- **CausalPlayground / CauTabBench / TabPFN prior work**: strong evidence that synthetic datasets sampled from families of SCMs and graph priors are useful both scientifically and practically.[^R1][^R4][^R13][^R14]

This suggests that the right abstraction is **not** “train a generator and sample rows,” but rather “specify or sample a world, simulate it, then render multiple dataset views from it.”

---

## 3. Research synthesis

### 3.1 What industry lead scoring actually looks like

Vendor documentation and industry practice consistently frame lead scoring around combinations of:

- **fit**: firmographic / demographic properties,
- **engagement**: behavioral signals,
- **combined scores**: fit + engagement,
- **deal/account context**, and
- **stage transitions / sales interactions**.[^R5][^R6][^R15]

HubSpot’s current scoring model explicitly separates **fit scores** from **engagement scores**, with combined scores allowing both property groups and event groups.[^R5] Salesforce describes lead scoring as ranking leads based on **behavior, demographics, and engagement**.[^R6]

**Implication for the framework:** lead scoring should not be modeled as a single direct Bernoulli on a flat row. The latent structure should usually include:

- lead/company fit,
- current intent / urgency,
- awareness and product understanding,
- channel mix and touch history,
- sales contactability / responsiveness,
- account-level readiness / budget / authority,
- pipeline friction, and
- conversion policy / sales playbook.

### 3.2 What academic lead-scoring work adds

The recent B2B lead-scoring case study in *Frontiers in Artificial Intelligence* is helpful because it is very concrete: a B2B software company, real CRM data from 2020–2024, multiple classification algorithms, and explicit emphasis on prioritization, not merely classification.[^R11] Feature-importance findings such as the influence of **source** and **lead status** are also instructive.[^R11]

The systematic review by Wu, Andreev, and Benyoucef (2023) indicates that the literature is still fragmented, with traditional vs predictive models and a relatively small number of studies overall.[^R16]

**Implication:** the package should generate not just a target label, but also the artifacts that make real lead scoring hard and useful:

- source/channel,
- lifecycle stage,
- handoff timing,
- qualification status,
- lead decay,
- sales effort ordering,
- time-to-conversion,
- lead quality under noisy CRM operations.

### 3.3 CLV / LTV research points toward temporal, behavioral models

Classic CLV literature emphasizes customer value as a longitudinal process connected to acquisition, retention, and cross-sell allocation.[^R9] Fader and Hardie’s work on customer-base analysis, Pareto/NBD, and BG/NBD makes the key point that customer value often emerges from **transaction frequency, recency, and dropout / inactivity processes** rather than one-shot labels.[^R8]

Microsoft’s current product framing also reflects that CLV lives alongside propensity and journey signals, not in isolation.[^R7]

**Implication:** even though v1 targets lead scoring, the internal world model should already support:

- accounts/contacts that can become customers,
- repeated interactions or purchases,
- churn / survival / inactivity processes,
- subscription or repeat-purchase revenue,
- interventions that affect both short-term conversion and long-term value.

### 3.4 Synthetic tabular generation research: realism is not enough

Recent synthetic-tabular research makes several points relevant here:

1. **Structural fidelity matters.** CauTabBench argues that tabular synthesis still struggles with complex dependencies and that benchmark datasets should be generated from **randomly sampled causal graphs** and evaluated using structural and downstream causal tasks.[^R1]
2. **Evaluation is incomplete without structure.** TabStruct pushes structural fidelity as a core evaluation dimension and argues that conventional evaluations alone are insufficient.[^R2]
3. **SCM-based synthetic priors are powerful.** The TabPFN prior explicitly samples high-level dataset hyperparameters, then constructs a DAG / SCM and propagates noise through a computational graph to generate datasets.[^R14]
4. **Relational synthesis is underexplored but crucial.** Work on synthetic relational tabular data via SCMs explicitly notes that real-world data often spans multiple linked tables and that current methods inadequately address this.[^R4]

**Implication:** your framework should treat the graph as a first-class output artifact, and should support **random graph sampling**, **mechanism sampling**, **relational generation**, and **task rendering**.

### 3.5 Simulation precedents are more relevant than generic GANs

RetailSynth is especially instructive. It is not just a table sampler; it is a **multi-stage simulation environment** calibrated to a retail domain and designed to capture heterogeneity such as price sensitivity and past experiences.[^R17]

Synthea goes even further: it simulates people over time through modular state machines backed by domain knowledge and reference statistics.[^R10]

**Implication:** for lead scoring and later LTV, a world simulator with event/state progression is likely more valuable than a one-shot generative model. A learned tabular generator may still be useful later as one backend, but it should not define the project’s conceptual core.

---

## 4. Product vision

### 4.1 What the package should produce

The package should be able to generate some or all of the following from one coherent synthetic world:

- **raw relational CRM-like data**
  - accounts
  - contacts
  - leads
  - sessions
  - marketing touches
  - SDR activities
  - opportunities
  - product trials
  - subscription / billing records (later)
- **snapshot feature tables** for supervised ML
- **labels / targets** for one or more tasks
- **ground-truth metadata**
  - DAG / SCM
  - latent variables
  - interventions
  - measurement process
  - leakage map
  - feature provenance
- **evaluation bundles**
  - train/validation/test splits
  - temporal splits
  - distribution-shift splits
  - counterfactual challenge sets
  - stress-test datasets

### 4.2 What “good” looks like

A generated dataset should support at least four use modes:

1. **Teaching / portfolio** – realistic enough to feel concrete.
2. **Benchmarking** – enough hidden truth to compare modeling strategies rigorously.
3. **Method development** – supports ablations, interventions, and known structure.
4. **Tooling demos** – renders clean CRM-style datasets, dashboard slices, and score explanations.

---

## 5. Recommended initial vertical

## Mid-market B2B SaaS with a mixed PLG + marketing + SDR motion

### Narrative anchor

A company sells a cloud software product to operations / finance / procurement teams at 200–2,000 employee firms. The product may be, for example:

- procurement workflow automation,
- spend analytics / vendor management,
- AP automation,
- revenue ops workflow tooling,
- compliance workflow software.

### Why this is the best starting vertical

It gives you, naturally:

- firmographic fit variables,
- account/contact/lead structure,
- inbound and outbound acquisition,
- marketing touches and web/product behavior,
- demo / trial conversions,
- sales-assist dynamics,
- long sales cycles,
- opportunity stages,
- post-conversion subscription and expansion paths.

That means the same simulated world can later power:

- **lead scoring**,
- **MQL/SQL prediction**,
- **opportunity win prediction**,
- **time-to-conversion prediction**,
- **ACV prediction**,
- **LTV / CLV prediction**,
- **uplift / intervention evaluation**.

### Concrete entities for v1

- **Account**: company-level object
- **Contact**: person-level buyer / user / evaluator
- **Lead**: contact/account seen in pre-opportunity funnel
- **Touch**: marketing interaction (email, webinar, paid, organic, partner, SDR)
- **Session**: website / product visit or trial activity
- **Sales activity**: call, email, meeting, sequence step
- **Outcome**: opportunity created, qualified, converted to closed-won, or no conversion within horizon

---

## 6. Core design principle: world-first, dataset-second

The framework should internally simulate a **world**, not directly a supervised dataset.

### 6.1 Canonical hidden layers

A sensible hidden structure for the first vertical might include:

- `market_segment`
- `competitive_pressure`
- `economic_cycle`
- `account_need_intensity`
- `account_budget_readiness`
- `account_process_maturity`
- `contact_seniority`
- `contact_role_fit`
- `contact_problem_awareness`
- `contact_solution_awareness`
- `inbound_intent`
- `outbound_receptiveness`
- `trial_friction`
- `sales_capacity`
- `sales_execution_quality`
- `pricing_fit`
- `product_fit`
- `latent_ltv_potential`

Some should be account-level, some contact-level, some time-varying, some global.

### 6.2 Observable layers

Observed variables then arise through measurement processes, not direct copying of latent truth:

- form fill fields,
- CRM enrichment,
- web analytics,
- campaign attributions,
- SDR notes/statuses,
- stage labels,
- timestamps,
- counts and recencies,
- product events,
- eventually billing outcomes.

This distinction is critical. Real lead scoring is difficult partly because **the business truth is latent and the data are noisy operational traces**.

---

## 7. Proposed architecture

## 7.1 Layer A — Narrative specification

A narrative spec defines the concrete world.

Example objects:

```yaml
vertical: b2b_saas_procurement
company:
  name_style: "synthetic"
  pricing_model: "subscription_seat_plus_platform"
  acv_range: [12000, 85000]
  sales_motion: [inbound, outbound, partner, trial_assisted]
market:
  regions: [US, UK, DACH]
  target_company_size: [200, 2000]
  industries: [software, manufacturing, healthcare, retail]
buyers:
  personas:
    - finance_director
    - procurement_manager
    - ops_lead
    - it_security_reviewer
funnel:
  stages: [new, engaged, mql, sql, demo, opportunity, closed_won, closed_lost]
```

This should be concrete enough that generated data feel like they belong to a real company.

## 7.2 Layer B — Schema specification

A schema compiler takes the narrative and builds:

- entities,
- relationships,
- event types,
- snapshot views,
- task definitions.

For v1, the engine should probably generate relational data first and derive a flat ML table second.

## 7.3 Layer C — Structure prior / graph sampler

This is the heart of the project.

Instead of a single fixed DAG, define a **family of valid graphs** constrained by narrative semantics.

### Recommended approach

Use a **typed graph grammar** or **constraint-based DAG sampler**.

- Nodes have types: `global`, `account`, `contact`, `touch`, `session`, `sales_process`, `outcome`, `billing`.
- Edges are allowed or forbidden by type constraints.
- Some substructures are mandatory (for example fit must influence conversion somehow).
- Some are optional or variable (for example webinar attendance may or may not mediate intent).
- Some motifs can be sampled with varying probability.

### Why this matters

If all datasets are generated from one fixed hidden graph with parameter perturbations, models and benchmarking will overfit to that worldview. The graph family must be broad enough that datasets have **different causal stories**, not just different coefficients.

### Recommended structure modes

Support at least three families:

1. **SCM-first DAG mode**  
   Clean acyclic graph over latent and observed variables.
2. **State-machine / event-process mode**  
   Time-based transitions, suitable for funnel progression.
3. **Hybrid mode**  
   Static DAG for background traits + event simulator for touches, sessions, and stage transitions.

The hybrid mode is likely the default winner.

## 7.4 Layer D — Mechanism library

Each node or transition needs a conditional mechanism.

Mechanism families should include:

- linear-Gaussian / GLM-style,
- tree/rule-based,
- monotonic splines,
- mixture models,
- small neural mechanisms,
- survival / hazard mechanisms,
- count processes (Poisson / NB / zero-inflated),
- dropout / inactivity models,
- bounded / categorical transforms,
- thresholded business-rule policies.

A strong design pattern is what TabPFN describes: sample a graph, propagate noise through varied computational mappings, and post-process to realistic feature spaces.[^R14]

## 7.5 Layer E — Event simulator

For lead scoring, important phenomena are sequential:

- campaign touch order,
- session recency/frequency,
- SDR response timing,
- trial activation,
- lead decay,
- stage advancement,
- meeting/no-show,
- deal creation.

Therefore a lightweight event simulator is highly recommended. The simulator can run in discrete time (daily or weekly ticks) or continuous time with sampled event times.

### Recommendation

For v1, prefer **discrete-time simulation**. It is simpler, easier to debug, and easier to align with common CRM/reporting views.

## 7.6 Layer F — Observation / measurement model

This layer should create realism via imperfections:

- missing enrichment fields,
- delayed CRM updates,
- imperfect source attribution,
- noisy status codes,
- proxy features instead of true intent,
- duplicated leads,
- merged contacts/accounts,
- sales-rep policy heterogeneity,
- partial observability.

This layer is not a nice-to-have. It is where much of the real-world difficulty comes from.

## 7.7 Layer G — Task renderer

A single synthetic world should render multiple tasks.

### Lead scoring v1 tasks

- `lead_converts_within_90d` (binary)
- `lead_stage_reaches_sql_within_30d` (binary)
- `days_to_opportunity` (regression / survival)
- `expected_pipeline_value_90d` (regression)
- `priority_band` (ordinal classification)

### Future LTV tasks

- `customer_ltv_12m`
- `net_revenue_12m`
- `first_renewal_probability`
- `expansion_revenue_12m`
- `gross_margin_ltv`

---

## 8. Open questions and design alternatives

## 8.1 What is the primary abstraction?

### Option A — DAG / SCM first

**Pros**
- principled
- clean hidden truth
- supports interventions/counterfactuals
- aligned with current tabular synthesis research[^R1][^R14]

**Cons**
- awkward for long sequential journeys
- can become artificial if too static

### Option B — Event simulator first

**Pros**
- better for realistic funnel dynamics
- natural for sequences, delays, recency, and sales operations
- better bridge to LTV

**Cons**
- easier to build an ad hoc simulator with weak causal semantics
- harder to expose clean ground truth

### Option C — Hybrid (recommended)

Static / slowly varying latent DAG + event/state transition layer + measurement model.

---

## 8.2 Should graph structures be fully random?

### Option A — Fully sampled from broad priors

**Pros**: maximal diversity  
**Cons**: risk of incoherent or implausible worlds

### Option B — Library of templates + stochastic rewiring

**Pros**: safer, more interpretable, easier debugging  
**Cons**: less expressive

### Recommendation

Start with **template families + typed stochastic rewiring**. Avoid both brittle fixed graphs and unconstrained randomness.

---

## 8.3 Should the first release be flat-table or relational?

### Flat first

**Pros**: simpler user adoption  
**Cons**: weak foundation for realism and later LTV

### Relational first

**Pros**: better architecture, better realism, future-proof[^R4]  
**Cons**: heavier implementation and testing burden

### Recommendation

Internally: **relational first**.  
Externally: provide a **flat-table supervised export** in v1.

---

## 8.4 How should mechanisms be authored?

### Option A — Pure Python code

Flexible, powerful, but hard to inspect and share.

### Option B — DSL / config only

Easy to serialize and compare, but can become too restrictive.

### Recommendation

Use a **code-first core with a serializable declarative layer**. In other words:

- underlying mechanisms implemented in Python classes,
- worlds/graphs/mechanisms serializable to YAML/JSON,
- reproducible seeds and manifests.

---

## 8.5 How much hidden truth should be exposed?

### Minimal truth exposed

Better for realistic benchmark use, but weaker for scientific analysis.

### Full truth exposed

Great for method development and education, but less realistic.

### Recommendation

Support **truth exposure levels**:

- `none`
- `partial` (graph only)
- `research` (graph + mechanisms + latent vars + interventions)

---

## 8.6 Should generation be purely synthetic or semisynthetic?

### Pure synthetic

Pros: no licensing/privacy dependency; cleaner control.

### Semisynthetic

Pros: stronger realism; can calibrate marginals or conditional mechanisms to seed data, similar in spirit to causalAssembly.[^R3]

### Recommendation

Design pure synthetic as the main mode, but leave a clean extension point for **calibration against seed distributions** later.

---

## 8.7 Should interventions be first-class in v1?

Interventions matter because lead scoring is often affected by policies:

- faster follow-up,
- webinar invitations,
- SDR sequencing,
- routing changes,
- pricing offers,
- trial assistance.

### Recommendation

Yes, but narrowly. Support a small intervention API from the start:

- change touch policy,
- change follow-up latency,
- change scoring threshold routing,
- change campaign allocation.

This is strategically important for later uplift / policy tasks.

---

## 8.8 What kinds of realism should be optimized?

This project has at least six realism axes:

1. **Narrative realism** – plausible company/world.
2. **Schema realism** – plausible CRM/log tables.
3. **Statistical realism** – sensible marginals/correlations.
4. **Structural realism** – causal and high-order dependencies.[^R1][^R2]
5. **Operational realism** – missingness, delay, duplicates, pipeline messiness.
6. **Task realism** – models should face meaningful signal, confounding, and nontrivial tradeoffs.

A key design question is which axes are optimized first and which are left “good enough.”

---

## 9. Recommended design outline

## 9.1 Package shape

```text
leadgen/
  core/
    ids.py
    random.py
    schema.py
    manifests.py
  narrative/
    verticals/
      b2b_saas_procurement.py
    personas.py
    channels.py
    funnel.py
  graph/
    node_types.py
    graph_grammar.py
    dag_sampler.py
    motif_library.py
    validation.py
  mechanisms/
    base.py
    static.py
    counts.py
    categorical.py
    survival.py
    transitions.py
    measurement.py
  simulation/
    world.py
    event_engine.py
    policies.py
    interventions.py
  render/
    relational.py
    snapshots.py
    ml_tasks.py
    graph_metadata.py
  eval/
    fidelity.py
    structure.py
    task_difficulty.py
    drift.py
  datasets/
    recipes/
      lead_scoring_v1.yaml
  cli/
    main.py
```

## 9.2 Main abstractions

### `NarrativeSpec`
Concrete business world.

### `SchemaSpec`
Entities, relationships, event types, exported views.

### `StructurePrior`
Family of allowed graphs and motif priors.

### `WorldModel`
Sampled world instance with graph + mechanisms + policies.

### `Simulator`
Runs the event/state process over time.

### `Renderer`
Exports tables, ML tasks, metadata.

### `EvaluationBundle`
Produces diagnostics and challenge splits.

## 9.3 Example generation flow

1. Load `NarrativeSpec` for `b2b_saas_procurement`.
2. Sample company-level parameters.
3. Compile schema.
4. Sample a typed graph from allowed motif priors.
5. Sample node mechanisms.
6. Generate base account/contact populations.
7. Simulate touches, sessions, SDR activities, stage changes.
8. Apply observation model / CRM noise.
9. Render relational tables.
10. Render feature snapshot at selected time horizon.
11. Render labels.
12. Save metadata bundle.

---

## 10. Recommended outputs for v1

## 10.1 Required outputs

- `accounts.csv`
- `contacts.csv`
- `leads.csv`
- `touches.csv`
- `sessions.csv`
- `sales_activities.csv`
- `lead_snapshot_train.csv`
- `lead_snapshot_valid.csv`
- `lead_snapshot_test.csv`
- `dataset_manifest.json`
- `ground_truth_graph.graphml`
- `feature_dictionary.md`

## 10.2 Strongly recommended metadata

- latent variable registry
- node mechanism summary
- intervention hooks
- leakage notes
- feature provenance map
- task definition file
- calibration summary

---

## 11. Evaluation framework for the generator itself

Do **not** evaluate the framework only by “does the table look plausible?”

### 11.1 Statistical checks

- marginal distributions
- pairwise and selected higher-order dependencies
- sparsity / cardinality / missingness patterns
- event count and recency distributions

### 11.2 Structural checks

Inspired by CauTabBench and TabStruct:[^R1][^R2]

- graph recovery on exposed structure,
- conditional independence alignment,
- motif preservation,
- intervention-response plausibility,
- counterfactual challenge sets.

### 11.3 Task checks

Generated tasks should show a healthy spectrum of difficulty:

- simple baselines should not dominate,
- signal should exist but be incomplete,
- calibration should matter,
- temporal splitting should hurt somewhat,
- there should be heterogeneity across datasets.

### 11.4 Narrative checks

Human sanity review:

- do channels/sources match the company motion?
- do job titles and company sizes match the product sold?
- do stage transitions make business sense?
- do generated reps and policies behave plausibly?

---

## 12. Suggested first implementation milestone

## Milestone 1 — verticalized hybrid simulator for lead scoring only

### Scope

- one vertical: `b2b_saas_procurement`
- one primary task: `lead_converts_within_90d`
- relational internal world
- flat snapshot ML export
- typed graph grammar with several motifs
- discrete-time event engine
- observation/noise model
- metadata bundle with exposed graph

### Graph motifs to support initially

1. **Fit-dominant funnel**  
   Company fit strongly drives both engagement propensity and conversion.
2. **Intent-dominant funnel**  
   Behavioral signals dominate; fit matters less.
3. **Sales-execution-sensitive funnel**  
   Follow-up speed and rep quality matter materially.
4. **Trial-mediated funnel**  
   Product activation is a key mediator.
5. **Account-consensus funnel**  
   Multi-contact buying committee matters.

The first milestone does not need infinite graph flexibility. It needs a clean system in which these motifs can be combined, rewired, and perturbed.

---

## 13. Design recommendations stated plainly

1. **Do not start with GANs or diffusion.** Start with a world simulator.
2. **Do not start with a flat table.** Start with relational event generation and export flat views.
3. **Do not hard-code one graph.** Use a typed graph family with stochastic motifs.
4. **Do not expose only labels.** Expose metadata and truth levels.
5. **Do not treat lead scoring as isolated from later LTV.** Use a vertical where account/contact/opportunity/customer continuity is natural.
6. **Do not optimize only statistical fidelity.** Optimize structural, operational, and task realism too.
7. **Do not make the first vertical generic.** Make it concrete and vivid.

---

## 14. Key unresolved design questions for reviewer assistants

The following are the main questions I would want Gemini / Claude / future assistants to respond to.

### Strategy questions

1. Is the recommended first vertical the right one, or is there a stronger bridge between lead scoring and LTV?
2. Should the package aim first at **benchmark research**, **teaching realism**, or **general synthetic CRM generation**?
3. What minimum graph diversity is needed before the project stops being “one fixed hidden worldview”?

### Modeling questions

4. What is the best hybrid formalism: SCM + HMM/state machine? SCM + event simulation? SCM + agent-based simulation?
5. Which latent variables should be mandatory across most lead-scoring worlds?
6. Which mechanism families are essential in v1, and which are overengineering?
7. How should multi-contact / buying-committee dynamics be represented without exploding complexity?

### Product / API questions

8. Should the public API be recipe-driven (`generate_dataset("b2b_saas_procurement")`) or world-builder-driven?
9. How should graph and mechanism metadata be serialized for reproducibility and review?
10. What should the artifact contract be: CSVs + JSON + GraphML? Parquet + DuckDB? something else?

### Evaluation questions

11. What evaluation harness is most informative for a generator like this?
12. What difficulty targets should generated lead-scoring tasks satisfy?
13. What anti-shortcut checks should be built in to avoid trivial leakage?

### Roadmap questions

14. What is the cleanest path from lead scoring to CLV/LTV without architectural rework?
15. When should semisynthetic calibration against real marginals be introduced?
16. When, if ever, should a learned tabular generator backend be plugged in?

---

## 15. A rough roadmap

## Phase 0 — foundation

- finalise vertical
- define narrative schema
- define graph grammar and motif library
- define event vocabulary
- define output artifact contract

## Phase 1 — lead scoring v1

- hybrid world simulator
- primary conversion label
- CRM noise model
- flat snapshot export
- metadata bundle
- baseline evaluation notebooks

## Phase 2 — richer funnel tasks

- SQL / opportunity / ACV tasks
- rep and policy heterogeneity
- intervention API
- challenge splits and stress tests

## Phase 3 — LTV bridge

- customer/subscription layer
- renewal/churn/expansion events
- 12-month value labels
- retention interventions

## Phase 4 — calibration / semisynthetic mode

- fit selected marginals or conditionals to seed data
- narrative adaptation tools
- recipe search / automatic difficulty control

---

## 16. Final recommendation

The project should be framed as:

> **A modular synthetic revenue-world simulator for CRM and GTM ML tasks, beginning with lead scoring in one concrete B2B SaaS vertical, built around sampled graph structures, event dynamics, and realistic observation noise.**

That framing is stronger than “synthetic lead-scoring dataset generator” because it:

- makes the architectural direction clearer,
- protects the project from collapsing into a one-table toy,
- creates a natural path to LTV,
- aligns with the most relevant recent research on SCM-based synthetic generation and structural fidelity,[^R1][^R2][^R14]
- and mirrors the strongest real-domain precedent: modular, domain-grounded simulation.[^R10]

---

## References

[^R1]: R. Tu et al., **“Causality for Tabular Data Synthesis: A High-Order Structure Causal Benchmark Framework”** (2024). OpenReview / arXiv. Key idea: benchmark datasets generated from randomly sampled causal graphs; evaluation includes structural and causal tasks. Source: https://arxiv.org/html/2406.08311v1 and https://openreview.net/pdf?id=BqV8qbLDqR

[^R2]: X. Jiang, N. Simidjievski, M. Jamnik, **“TabStruct: Measuring Structural Fidelity of Tabular Data”** (2025). OpenReview. Key idea: structural fidelity should be a core dimension for evaluating tabular generators. Source: https://openreview.net/forum?id=XOPH34Extq and https://openreview.net/forum?id=QccV7Wi3sN

[^R3]: K. Göbler et al., **“causalAssembly: Generating Realistic Production Data for Benchmarking Causal Discovery”** (AISTATS/causal learning context, 2024). Key idea: combine a ground-truth causal graph with estimated conditional mechanisms to generate semisynthetic data that adhere to the graph. Source: https://proceedings.mlr.press/v236/gobler24a.html and https://proceedings.mlr.press/v236/gobler24a/gobler24a.pdf

[^R4]: F. Hoppe et al., **“Generating Synthetic Relational Tabular Data via Structural Causal Models”** (2025). Key idea: relational tabular data generation with causal relations across tables is underexplored but important. Source: https://aclanthology.org/2025.trl-1.2.pdf

[^R5]: HubSpot Knowledge Base, **“Build lead scores to qualify contacts, companies, and deals”** and related scoring docs (2025–2026). Key idea: explicit split between fit scores, engagement scores, and combined scores. Source: https://knowledge.hubspot.com/scoring/build-lead-scores and https://knowledge.hubspot.com/scoring/understand-the-lead-scoring-tool

[^R6]: Salesforce, **“Lead Scoring: How to Find the Best Prospects in 4 Steps”** (2025). Key idea: lead scoring ranks leads based on behavior, demographics, and engagement. Source: https://www.salesforce.com/blog/lead-scoring/

[^R7]: Microsoft Learn / Dynamics 365 Customer Insights docs (2025–2026). Key idea: customer insights stack exposes CLV, propensity, activity history, and lead/contact usage together. Source: https://learn.microsoft.com/en-us/dynamics365/customer-insights/data/whats-new-customer-insights and related Customer Insights docs

[^R8]: P. S. Fader and B. G. S. Hardie, **“Probability Models for Customer-Base Analysis”** (2009), plus BG/NBD notes and antecedents. Key idea: CLV emerges from behavioral processes such as purchase frequency and dropout/inactivity. Source: https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_09.pdf and https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf

[^R9]: S. Gupta et al., **“Modeling Customer Lifetime Value”** (2006). Key idea: implementable CLV models for segmentation and resource allocation across acquisition, retention, and cross-selling. Source: https://www.anderson.ucla.edu/sites/default/files/documents/areas/fac/marketing/JSR2006%280%29.pdf

[^R10]: J. Walonoski et al. / Synthea project, **Synthea synthetic patient generator** and Generic Module Framework. Key idea: modular domain simulation over lifespans/states is a powerful precedent for synthetic-data systems. Source: https://synthetichealth.github.io/synthea/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC7651916/

[^R11]: L. González-Flores, J. Rubiano-Moreno, G. Sosa-Gómez, **“The relevance of lead prioritization: a B2B lead scoring model based on machine learning”** (2025). Key idea: concrete case study with real CRM data from a B2B software company; prioritization is the real business objective. Source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1554325/full

[^R12]: SDV documentation. Key idea: conditional sampling and general-purpose synthetic tabular generation are useful, but the abstraction is still table/distribution oriented. Source: https://docs.sdv.dev/sdv/single-table-data/sampling/conditional-sampling and https://docs.sdv.dev/sdv/multi-table-data/sampling/conditional-sampling

[^R13]: A. W. M. Sauter, E. Acar, A. Plaat, **“CausalPlayground: Addressing Data-Generation Requirements in Cutting-Edge Causality Research”** (2024). Key idea: standardized platform for generating, sampling, and sharing SCMs with fine-grained control and interactive environments. Source: https://arxiv.org/abs/2405.13092 and https://github.com/sa-and/CausalPlayground

[^R14]: N. Hollmann et al., **“Accurate predictions on small data with a tabular foundation model”** (Nature, 2025). Key idea: synthetic training data generated from SCMs by sampling dataset hyperparameters, building a DAG, propagating noise through diverse computational mappings, and post-processing. Source: https://www.nature.com/articles/s41586-024-08328-6

[^R15]: Salesforce Account Engagement / Einstein lead scoring docs. Key idea: behavior scoring uses engagement pattern data and weighs recency/frequency; account engagement supports AI-driven lead scoring and grading. Source: Salesforce Help pages, including Einstein behavior scoring and lead scoring docs.

[^R16]: M. Wu, P. Andreev, M. Benyoucef, **“The state of lead scoring models and their impact on sales performance”** (2023). Key idea: systematic review showing a still-limited and fragmented lead-scoring literature. Source: discoverable via PMC/ResearchGate and journal pages.

[^R17]: Y. Xia et al., **“RetailSynth: Synthetic Data Generation for Retail AI Systems Evaluation”** (2023). Key idea: multi-stage behavioral simulation calibrated to public data can create realistic domain-specific synthetic transactional environments. Source: https://arxiv.org/html/2312.14095v1

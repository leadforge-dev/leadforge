# leadforge — Design Document

**Status:** v1 design document  
**Project type:** opinionated open-source Python framework + CLI  
**Repository / package / CLI name:** `leadforge`  
**License:** MIT  
**Primary target:** synthetic lead-scoring dataset generation, with LTV-ready foundations  
**Vertical for v1:** mid-market procurement / AP automation SaaS

---

## 1. Purpose of this document

This document defines the **product and design-level decisions** for `leadforge`.

It is intended to lock:
- what `leadforge` is,
- who it is for,
- what it must ship in v1,
- what it must explicitly not attempt in v1,
- what design principles must guide all implementation work,
- and what success looks like.

This is **not** the low-level architecture spec. It deliberately avoids overcommitting to internal interfaces, schema minutiae, or module contracts that belong in the later architecture/spec document.

---

## 2. Executive summary

`leadforge` is an opinionated Python framework for generating **narrative-grounded synthetic revenue datasets**, starting with **lead scoring** and designed from the outset to support **later LTV / CLV extensions**.

The core idea is that `leadforge` should not behave like a generic synthetic tabular resampler. It should instead generate data from a **simulated commercial world**:
- a specific company,
- selling a specific product,
- to a specific kind of buyer,
- through a specific go-to-market motion,
- within a plausible funnel,
- under a variable hidden data-generating process.

In v1, `leadforge` will support one concrete vertical:

> a mid-market B2B SaaS company selling procurement and AP workflow automation software to 200–2,000 employee firms in the US and UK, through a mixed inbound marketing, SDR-assisted, and partner-driven motion.

The first primary supervised task is:

> **predict whether a lead converts within 90 days** (`converted_within_90_days`).

Internally, `leadforge` should simulate a **relational CRM-like world** with account, contact, lead, touch, session, sales-activity, opportunity, and post-conversion customer/subscription concepts. Externally, it should also export **flat ML-ready snapshots**, rich metadata, and optional ground truth depending on the selected exposure mode.

The framework should prioritize:
1. **teaching realism**,
2. **reusable open-source framework quality**,
3. **benchmark / research usefulness**,
4. **demo / portfolio value**.

---

## 3. Problem statement

Public datasets for lead scoring and adjacent revenue problems are usually unsatisfying for serious teaching and project work:
- too small,
- too overused,
- too shallow,
- too obviously solvable,
- too disconnected from real CRM operations,
- or too generic to sustain repeated instructional use.

Meanwhile, generic synthetic data tooling typically focuses on distributional resemblance rather than:
- meaningful commercial narratives,
- realistic funnel dynamics,
- causal or structural variety,
- measurement artifacts,
- CRM messiness,
- and pedagogically useful hidden truth.

This creates a gap:

> we want datasets that feel like they came from a real commercial setting, are reusable across classes and projects, support multiple ML skills, and do not collapse into one fixed hidden worldview.

`leadforge` exists to fill that gap.

---

## 4. Product vision

### 4.1 One-sentence vision

`leadforge` generates **high-quality synthetic CRM and GTM datasets from simulated commercial worlds**, beginning with lead scoring and expanding naturally toward customer-value problems.

### 4.2 Longer vision

A user should be able to ask `leadforge` to generate a concrete world such as:
- a procurement automation SaaS vendor,
- with a particular market mix,
- a particular funnel shape,
- a particular dominant buying motion,
- a particular difficulty level,
- and a particular ground-truth exposure policy,

and receive a bundle containing:
- realistic CRM-like tables,
- ML snapshot tables,
- target labels,
- a dataset card / narrative,
- and optional hidden graph/mechanism metadata.

The resulting artifact should be useful for:
- courses,
- assignments,
- capstone projects,
- tutorials,
- open-source demonstrations,
- and, secondarily, methodological benchmarking.

---

## 5. Design priorities

These priorities are ordered and should break ties.

### 5.1 First priority: teaching realism

If forced to choose, `leadforge` should prefer:
- more believable business context,
- more coherent feature naming,
- more realistic operational artifacts,
- more useful student tasks,
- and more interpretable generated worlds,

over abstract generality or benchmark purity.

The framework should generate datasets that let students practice:
- problem framing,
- feature understanding,
- train/validation/test design,
- leakage awareness,
- class imbalance handling,
- calibration,
- thresholding,
- error analysis,
- and business-aware interpretation.

### 5.2 Second priority: reusable OSS framework

`leadforge` should be polished, composable enough to extend, and pleasant to use as an open-source package. It should not be a one-off internal script dump.

This implies:
- careful package structure,
- a strong CLI,
- good docs,
- example notebooks,
- sample datasets,
- reproducibility,
- deterministic seeded generation,
- and a clean artifact contract.

### 5.3 Third priority: benchmark / research usefulness

The framework should preserve enough hidden truth and structural rigor to support more scientific usage, but that should not distort the product toward a research-only benchmark generator.

### 5.4 Fourth priority: demo / portfolio polish

A polished generated dataset and narrative are valuable, but visual polish should not dominate core realism or framework integrity.

---

## 6. Product stance

`leadforge` is explicitly:
- an **opinionated framework**, not a minimal toolkit,
- **offline-capable by default**,
- **deterministic given a seed**,
- **open-source first**,
- **narrative-first**,
- and **world-first rather than table-first**.

Optional API-based enrichments may exist later, but the core product must not depend on them.

---

## 7. Naming and identity

The canonical project identity is:
- repository: `leadforge`
- Python package: `leadforge`
- CLI command: `leadforge`

The name should be treated as product-facing, not internal only. The docs, dataset cards, example notebooks, and CLI help should all use it directly.

---

## 8. Core design principles

## 8.1 World-first, dataset-second

`leadforge` should generate a **simulated commercial world** and then render one or more datasets from it.

This means the underlying abstraction is not “a row with a label,” but rather:
- companies/accounts,
- people/contacts,
- leads,
- touches and sessions,
- sales actions,
- opportunities,
- and later customers/subscriptions.

## 8.2 Narrative-grounded always

Every generated dataset should belong to a concrete story.

No anonymous `x1`, `x2`, `x3` worlds.
Features should have names that a business student can reason about.
The generated narrative should explain:
- who the seller is,
- what is being sold,
- to whom,
- through what funnel,
- and what the target means.

## 8.3 Structural variability is mandatory

The hidden DGP must vary meaningfully across generated worlds.

`leadforge` should not rely on one fixed latent graph with only parameter perturbations. Instead, it should sample from a small but meaningful family of motif/template worlds with stochastic rewiring.

The objective is not maximal randomness. The objective is **multiple sensible hidden worlds**.

## 8.4 Relational internals, flat exports

The internal representation should be relational and event-aware, even if the initial course-facing export is a flat table.

This keeps the v1 architecture aligned with later goals such as:
- opportunity prediction,
- retention / churn,
- subscription evolution,
- and LTV.

## 8.5 Truth exposure must be configurable

Different users need different visibility.

`leadforge` should support at least two official modes:
- **student/public mode**: limited truth exposure
- **research/instructor mode**: rich truth exposure

The default public generated dataset should not need to reveal the hidden world. But the framework must support fully exposing it when requested.

## 8.6 Realism includes operational messiness

Realism is not only about statistics. It includes:
- delayed CRM updates,
- imperfect attribution,
- missing enrichment,
- duplicates,
- proxy measurements,
- noisy stage labels,
- variable rep quality,
- and nontrivial but plausible shortcuts.

## 8.7 v1 should not block v2

The v1 implementation should support a **hybrid discrete-time simulator** and explicitly leave room for a richer event engine later, without rewriting the conceptual foundations.

---

## 9. Primary users

### 9.1 Course instructor / teaching designer

Wants:
- reusable realistic datasets,
- adjustable difficulty,
- student-safe outputs,
- optional hidden truth for grading or review,
- and examples that map to ML course topics.

### 9.2 Student / portfolio builder

Wants:
- a believable dataset,
- understandable business context,
- meaningful feature names,
- multiple artifact types,
- and a challenge that is neither trivial nor impossible.

### 9.3 OSS user / practitioner

Wants:
- a clean API,
- strong docs,
- sample recipes,
- reproducible generation,
- and extensibility to related verticals.

### 9.4 Research / methods user

Wants:
- controllable graph families,
- hidden truth,
- metadata bundles,
- and reproducible worlds for experimentation.

This user matters, but is not the top priority.

---

## 10. v1 vertical definition

## 10.1 Selected vertical

The v1 vertical is:

> **mid-market procurement / AP automation SaaS**

### 10.2 Company narrative

A B2B SaaS company sells procurement and accounts-payable workflow automation software to firms with roughly 200–2,000 employees in the US and UK.

Typical customer pain points include:
- manual invoice workflows,
- fragmented approval chains,
- poor procurement visibility,
- vendor onboarding inefficiency,
- weak spend controls,
- and compliance / audit pressure.

### 10.3 Buying roles

Primary buyers and evaluators may include:
- finance directors,
- procurement managers,
- operations leads,
- controllership stakeholders,
- and secondary IT/security reviewers.

### 10.4 Go-to-market motion

The default v1 narrative assumes a mixed motion:
- inbound marketing,
- SDR-assisted follow-up,
- partner referrals,
- and optionally product/demo-led touchpoints.

### 10.5 Why this vertical

This vertical is the best v1 choice because it naturally supports:
- rich firmographic fit,
- multi-stakeholder buying,
- meaningful behavioral engagement,
- moderate-length sales cycles,
- pipeline stage transitions,
- subscription post-conversion structure,
- and natural extension into later LTV problems.

It also fits well with an economics/management teaching context.

---

## 11. v1 product scope

## 11.1 What v1 must ship

v1 must ship:
- a Python library,
- a CLI,
- sample generated datasets,
- and example notebooks.

## 11.2 What v1 must generate

At minimum, v1 should generate:
- relational CRM-style outputs,
- flat supervised ML exports,
- a narrative / dataset card,
- and a graph / metadata bundle with exposure controlled by mode.

## 11.3 Primary v1 supervised task

The primary task is:

> `converted_within_90_days`

This should represent whether a lead converts within a 90-day horizon according to the world’s funnel and policy logic.

This task should be central in docs, notebooks, and default recipes.

## 11.4 Included but not yet surfaced as primary outputs

The internal world in v1 should also model enough structure to later support:
- customer/subscription entities,
- post-conversion evolution,
- renewal / churn pathways,
- and revenue-related extensions.

However, v1 should **not** expose LTV labels as first-class tasks yet.

---

## 12. What v1 is not

To protect clarity and execution quality, v1 is **not**:
- a multi-vertical framework out of the gate,
- a general-purpose synthetic tabular package,
- a full dynamic event-stream simulator for every possible CRM action,
- an LTV task package yet,
- a privacy-preserving synthetic data engine,
- or a research benchmark suite first and foremost.

---

## 13. Product experience goals

## 13.1 Library experience

A user should be able to do something conceptually like:
- choose a recipe / vertical,
- choose a mode,
- choose a seed,
- choose a difficulty profile,
- generate a world,
- and save the outputs.

The experience should feel intentional and “batteries included,” not like assembling dozens of independent components.

## 13.2 CLI experience

The CLI should be a first-class product surface.

A user should be able to:
- list available recipes,
- generate a dataset bundle,
- choose output directory,
- choose truth exposure mode,
- choose seed,
- and later inspect or summarize generated artifacts.

## 13.3 Notebook experience

Example notebooks should make `leadforge` immediately useful for:
- teaching demos,
- exploratory analysis,
- baseline modeling,
- and dataset inspection.

## 13.4 Artifact experience

The output bundle should feel polished enough that a user could hand it to students or publish it as a sample project artifact with little or no cleanup.

---

## 14. Reality model for v1

The underlying world in v1 should include several conceptual layers.

### 14.1 Market / context layer

Examples:
- region,
- macro conditions,
- industry mix,
- market pressure,
- partner ecosystem effects.

### 14.2 Account / company layer

Examples:
- company size,
- industry,
- process maturity,
- growth stage,
- likely pain intensity,
- purchasing complexity,
- budget readiness.

### 14.3 Contact / buyer layer

Examples:
- role,
- seniority,
- problem awareness,
- decision authority,
- responsiveness,
- product fit from their perspective.

### 14.4 Engagement / behavioral layer

Examples:
- sessions,
- content downloads,
- demo intent,
- recency/frequency signals,
- campaign-origin effects,
- touch responsiveness.

### 14.5 Sales-process layer

Examples:
- follow-up timing,
- rep quality,
- stage transitions,
- qualification logic,
- meeting success,
- friction and delays.

### 14.6 Outcome layer

Examples:
- conversion within horizon,
- time to opportunity,
- stage progression,
- and later customer/subscription outcomes.

These layers should be reflected in the later architecture, but this design doc intentionally does not lock their exact implementation.

---

## 15. Truth exposure modes

## 15.1 Student / public mode

Purpose:
- produce realistic datasets that behave like externally received business data.

Characteristics:
- limited or no hidden graph exposure,
- no easy leakage from metadata,
- polished business-facing artifacts,
- enough documentation to use the dataset,
- but no need to reveal all hidden mechanisms.

## 15.2 Research / instructor mode

Purpose:
- enable review, grading, benchmarking, debugging, and deeper methodological work.

Characteristics:
- graph exposure,
- richer provenance metadata,
- latent-variable registries where appropriate,
- generation manifests,
- and mechanism summaries.

## 15.3 User sovereignty

Users generating their own datasets may choose whichever exposure policy they want. The framework must support “generate rich internal truth but only publish the flat public artifacts.”

---

## 16. Difficulty philosophy

Difficulty should be treated as a product feature, not an accidental side effect.

The framework should eventually let users influence things like:
- class imbalance,
- noise level,
- strength of fit vs behavior signal,
- degree of confounding,
- number of irrelevant features,
- missingness and measurement quality,
- policy heterogeneity,
- and train/test shift.

For v1, difficulty should be strong enough that:
- simple baselines are clearly beatable,
- the task is nontrivial,
- and different modeling choices meaningfully matter,

but not so difficult that the output feels arbitrary or unteachable.

---

## 17. Graph variability strategy

The chosen v1 strategy is:

> a few named motif/template families with stochastic rewiring

This is the correct tradeoff for v1 because it avoids two bad extremes:
- a single fixed worldview, and
- unconstrained random graph generation.

The named families should represent distinct but plausible commercial stories. For example:
- fit-dominant worlds,
- intent-dominant worlds,
- sales-execution-sensitive worlds,
- trial-mediated worlds,
- buying-committee-friction worlds.

The later spec can define these concretely. At the design level, the important point is that `leadforge` should sample from **a small family of believable hidden worlds**, not from one invariant one.

---

## 18. Simulation philosophy

The chosen v1 direction is:

> a hybrid discrete-time simulator, explicitly leaving room for a richer dynamic event engine in v2

This means the conceptual foundation should support:
- world state,
- repeated touches,
- recency/frequency effects,
- stage progression,
- and horizon-based outcomes,

without requiring a fully general continuous-time simulator in the first release.

This is the right balance between realism and execution quality.

---

## 19. Output philosophy

v1 should emit a polished bundle, not just a file.

The bundle should conceptually include:
- relational data,
- flat ML-ready data,
- narrative and feature descriptions,
- metadata / manifests,
- and optional hidden truth.

The exact filenames and schemas belong in the spec doc, but the product-level requirement is that a generated world should feel like a **complete dataset package**.

---

## 20. Quality bar

The required quality bar is:

> **very polished OSS from the beginning**

This affects all downstream decisions.

`leadforge` should aim for:
- clear packaging,
- strong typing and testing,
- clean docs,
- thoughtful examples,
- stable command behavior,
- deterministic outputs,
- and professional artifact presentation.

This does not require v1 to be huge. It requires v1 to be coherent and well-finished.

---

## 21. External dependencies and APIs

The framework may support optional external-API enrichments later, but:
- all such use must remain optional,
- there must always be a non-API path,
- and the core framework must remain fully useful without them.

This keeps the project reproducible, OSS-friendly, and portable.

---

## 22. Licensing decision

The package license is:

> **MIT**

This reflects the desire for broad reuse, low adoption friction, and a permissive OSS posture.

---

## 23. Success criteria

The design-level success criteria for v1 are:

### 23.1 Product success

A user can generate a coherent dataset bundle with:
- one command or a short library workflow,
- a believable narrative,
- realistic CRM-style tables,
- a usable flat lead-scoring export,
- and strong documentation.

### 23.2 Teaching success

An instructor can use a generated dataset for a realistic lead-scoring assignment or project without major additional fabrication.

### 23.3 OSS success

An outside developer can understand the project structure, run generation, inspect outputs, and imagine how to extend the framework.

### 23.4 Structural success

Different seeds can yield genuinely different hidden commercial worlds while staying within the same vertical.

### 23.5 Future-readiness success

The v1 world already contains enough conceptual structure that later LTV work extends naturally rather than requiring a conceptual reboot.

---

## 24. Key risks

### 24.1 Risk: too much ambition in v1

Mitigation:
- keep one vertical,
- one primary task,
- one simulation style,
- and motif families instead of a giant graph language.

### 24.2 Risk: realism becomes hand-authored and brittle

Mitigation:
- rely on template families plus controlled variability,
- not one giant manually curated universe.

### 24.3 Risk: benchmark value undermines teaching clarity

Mitigation:
- let benchmark usefulness emerge from good design,
- not drive the project from benchmark ambitions first.

### 24.4 Risk: flat-table convenience distorts the architecture

Mitigation:
- keep relational internals non-negotiable.

### 24.5 Risk: hidden truth leaks too much in public datasets

Mitigation:
- keep truth exposure modes explicit and separate.

---

## 25. Non-goals for the later docs

This design document intentionally does **not** settle:
- exact module boundaries,
- exact schema definitions,
- exact CLI commands and flags,
- exact metadata contracts,
- exact motif implementations,
- or detailed milestone sequencing.

Those belong in:
1. the architecture/spec document,
2. and then the roadmap / implementation plan.

---

## 26. Decision log

The following decisions are now locked.

- Project name: `leadforge`
- Package / repo / CLI identity unified under `leadforge`
- License: MIT
- Primary priority: teaching realism
- Secondary priority: reusable OSS framework
- Third priority: benchmark / research usefulness
- Fourth priority: demo / portfolio value
- v1 vertical: mid-market procurement / AP automation SaaS
- v1 deliverables: library + CLI + sample datasets + example notebooks
- v1 outputs: relational CRM-style tables + flat export + graph/metadata bundle + narrative/dataset card
- Primary v1 task: `converted_within_90_days`
- v1 simulation stance: hybrid discrete-time simulator
- v2 compatibility requirement: richer event engine should be possible later
- Graph variability stance: motif/template families with stochastic rewiring
- Truth exposure stance: dual mode (`student/public`, `research/instructor`)
- LTV stance: include post-conversion customer/subscription entities now, but do not ship LTV labels as first-class outputs yet
- External APIs: always optional
- Product stance: opinionated framework
- Quality bar: polished OSS from the start

---

## 27. Final product statement

`leadforge` should be built as:

> **a polished, opinionated, narrative-first open-source framework for generating synthetic CRM and funnel datasets from simulated commercial worlds, beginning with lead scoring in a concrete procurement / AP automation SaaS vertical, using relational internals, configurable truth exposure, and variable hidden world structures.**

That is the design center. The later architecture/spec and roadmap documents should preserve it.


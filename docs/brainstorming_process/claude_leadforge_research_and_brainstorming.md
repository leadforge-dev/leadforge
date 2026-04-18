# SynthLeadLTV — Foundation Document

**A Framework for Deep, Narrative-Anchored, Structure-Variable Synthetic Dataset Generation for Lead Scoring and LTV Prediction**

*Version 0.1 — foundation / discussion draft, intended for critical review by collaborating AI assistants (Gemini, other Claude instances, etc.) and by the human project owner.*

---

## 0. How to read this document

This is a **discussion document**, not a spec. It has three jobs:

1. **Ground** the project in what already exists — in the academic literature, in industry practice, and in the open-source tooling — so we don't reinvent things badly and we know which shoulders to stand on.
2. **Surface** the open questions and genuine design forks where we still need to decide, with enough context that a thoughtful reviewer can push back meaningfully.
3. **Propose** a rough, opinionated architecture that is concrete enough to attack, but loose enough to rewrite.

**If you are an AI assistant reviewing this document, please:**

- Challenge framing and terminology early if something feels wrong. Several terms here (e.g., "DGP structure" vs "DGP parameters", "vertical", "narrative skeleton") are load-bearing and need to be solid.
- Where we present design alternatives, **take sides** and say why. A review that adds a sixth alternative without ranking the existing five is less useful than one that says "option B is clearly wrong because X, and the right move between A and C depends on Y".
- Flag anything that looks like a false assumption about how real lead/LTV data is actually generated.
- Point out pedagogical pitfalls — this is ultimately a teaching tool for undergrads in economics/management, not a research benchmark.
- Don't be afraid to say "scope this much smaller".

---

## 1. Motivation and problem framing

### 1.1 The gap this project exists to fill

Customer lifetime value (CLV/LTV) prediction and lead scoring are two of the most economically important predictive tasks in modern business, and they are natural case studies for an undergraduate ML course in economics and management. Unfortunately, the available datasets are poor for teaching:

- **Small and over-trodden on Kaggle.** The canonical lead-scoring dataset (an Indian edtech CRM export with ~9,000 rows and a `Converted` label) is the basis of hundreds of nearly-identical notebooks. A student searching any feature name will find a worked solution immediately. The IBM Watson Auto Insurance CLV dataset has the same problem.
- **Synthetic but shallow.** The few purpose-built synthetic lead/LTV datasets are generated from a small number of hand-written rules, produce unrealistic joint distributions, and usually have an obvious "tell" (a single feature that drives the label).
- **Real but narrow.** The UCI Online Retail dataset is widely used as a CLV case study via the `lifetimes` package, but it is a single-company, single-year, B2C non-contractual setting. It doesn't let a student explore what changes when the business is B2B, subscription, or high-touch.

The contractual/non-contractual distinction matters a lot pedagogically and is under-served by available data: [Fader, Hardie & Lee's](https://www.brucehardie.com/papers/018/) BG/NBD formulation and the older [Schmittlein, Morrison & Colombo](https://www.jstor.org/stable/2631608) Pareto/NBD assume a non-contractual continuous setting, where "churn" is latent — but most undergrad exercises are done on contractual subscription data where churn is observed. A good generator should be able to produce both, with the student seeing why the modeling choice has to change.

### 1.2 What we want to build

A Python framework that **generates synthetic datasets for lead-scoring and LTV-prediction tasks**, with the following design commitments:

1. **Narrative-anchored.** Every generated dataset is tied to a concrete, human-readable business context: industry vertical, product/service, company size, go-to-market motion, representative leads, representative sales stages. No "X1, X2, X3" feature names. Features have meaningful names, sensible units, and the target has a clear business interpretation.
2. **Generalisable engine, single vertical to start.** The abstractions are designed so that adding a new vertical is a matter of writing a vertical module, not rewriting the engine. We ship with exactly one vertical at v1.
3. **Deep, adaptable DGP.** The data-generating process is not a fixed graph with random parameters. The **structure** of the DGP — which variables exist, which cause which, how many latent factors, how the target is formed — is itself sampled from a space of plausible structures.
4. **Lead-scoring first, LTV-capable foundation.** The v1 artifact is a lead-scoring dataset generator. But the core abstractions must make the extension to LTV-style time-to-event / monetary-value targets an extension rather than a rewrite.
5. **Suitable for teaching, not just benchmarking.** Outputs must be human-inspectable, narratively coherent, and rich enough to support feature engineering, imbalance handling, probability calibration, error analysis, and (later) survival analysis and LTV regression.

### 1.3 What we are **not** building

- We are **not** building a privacy-preserving generator. There is no real data to protect; all of our "realism" constraints come from domain knowledge, not from fitting to a real corpus. This rules out most of the SDV/CTGAN/TVAE family as the **backbone** of the generator (though we may borrow pieces — see §6).
- We are **not** building a production-grade simulator of a specific real company. The goal is pedagogically rich realism, not operational fidelity.
- We are **not** building a system that requires an LLM at training or inference time. LLMs may optionally be used for narrative flavor (lead company names, industry descriptions) but the statistical DGP itself should be self-contained, deterministic given a seed, and auditable.
- We are **not** trying to beat TabPFN's prior at its own game. TabPFN samples **abstract** SCMs with arbitrary node semantics to pre-train a foundation model. We sample **semantically grounded** SCMs anchored to a business narrative. The two priors live in different spaces.

---

## 2. Background: what the literature and industry already give us

This section is the "what you need to know" reference for anyone designing the framework. Reviewers should flag omissions and disagreements.

### 2.1 The modeling landscape for lead scoring

Lead scoring, academically, is a fairly standard binary classification problem on a lead-level feature vector, where the label is "this lead eventually converted into a paying customer within horizon H." The interesting questions are almost entirely about **what the features should be** and **how imbalanced the label is**.

The B2B SaaS literature organises features into four canonical buckets, which we should mirror (see e.g. [Scalarly's B2B scoring guide](https://scalarly.com/blog/b2b-lead-scoring-model/) and [Saber's ML scoring overview](https://www.saber.app/glossary/machine-learning-scoring)):

- **Demographic / identity**: job title, seniority, role function, decision-making authority, email domain type.
- **Firmographic**: company size (employees, revenue), industry, geography, funding stage, tech stack age.
- **Technographic**: tools the prospect's company already uses, compatibility signals, tech maturity.
- **Behavioral / engagement**: pages visited, content downloaded, webinar attendance, email open rates, time-on-pricing-page, response latency to outbound.

The dominant modelling choice in industry is gradient-boosted trees (XGBoost, LightGBM, Adobe's B2B CDP uses a tree-based ensemble), with logistic regression as the interpretable baseline — exactly the pedagogical arc Shay's syllabus already follows.

A recurring pattern in the academic and applied literature (e.g. the [Frontiers in AI 2025 case study of a B2B software company](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1554325/full)) is that two features consistently dominate importance: **lead source** (which marketing channel the lead came through) and **lead status / stage** (how far they've moved down the pipeline). Any generator that doesn't make these features both prominent and noisy will produce something that "looks wrong" to a domain expert.

### 2.2 The modeling landscape for LTV

LTV is harder and more heterogeneous than lead scoring. The landscape splits along two axes:

**Axis 1 — Contractual vs. non-contractual.**
- *Contractual* (subscription, telco, insurance): churn is observed. Standard tools are survival analysis (Kaplan-Meier, Cox proportional hazards, Aalen additive), combined with a monetary model for revenue per active period. This is the setting most closely parallel to the course's churn case study.
- *Non-contractual* (e-commerce, episodic B2B purchases): churn is latent. The canonical family is the "Buy-Till-You-Die" models:
  - **Pareto/NBD** ([Schmittlein, Morrison & Colombo 1987](https://www.jstor.org/stable/2631608)) — Poisson purchases while alive, exponential dropout; latent per-customer λ ~ Gamma and μ ~ Gamma.
  - **BG/NBD** ([Fader, Hardie & Lee 2005](https://www.brucehardie.com/papers/018/)) — easier-to-fit variant where dropout happens immediately after a purchase with probability p, and heterogeneity in p is Beta.
  - **MBG/NBD** — fixes BG/NBD's assumption that never-repeat customers haven't churned.
  - **Gamma-Gamma** ([Fader & Hardie](http://www.brucehardie.com/notes/025/)) — the standard partner model for the monetary side; assumes monetary value per transaction is independent of transaction frequency.

**Axis 2 — Probabilistic vs. ML vs. hybrid.**
- Probabilistic models (BG/NBD + Gamma-Gamma) are statistically principled and interpretable but rely on assumptions that break for many real businesses.
- Pure ML regression on engineered RFM features (random forest, GBM) handles heterogeneity better but discards the time dimension.
- Hybrid models — most importantly [Wang, Liu & Miao's 2019 Zero-Inflated LogNormal DNN approach](https://research.google/pubs/a-deep-probabilistic-model-for-customer-lifetime-value-prediction/) — explicitly model P(LTV > 0) and log(LTV | LTV > 0) as two heads, capturing the one-time-buyer zero-inflation that destroys naive MSE regression.

**What this means for the generator.** The generator must at minimum be able to produce:
- A time-to-churn event (censored or observed) per customer.
- A purchase-count and purchase-value history per customer, with realistic heterogeneity.
- A scalar LTV target computable as the discounted sum of predicted future net revenue over a horizon.

The zero-inflated, heavy-tailed shape of LTV is **the** pedagogically interesting feature. If the generator produces Gaussian-ish LTVs, we've failed.

### 2.3 Synthetic data generation: what's out there and why none of it is quite right

Three broad traditions are relevant:

**Tradition A — Learn from real data and resample (SDV / CTGAN / TVAE / TabDDPM).** These are the dominant tools for privacy-preserving synthetic tabular data. They are the wrong tool for us: we have no real data to fit, and they famously struggle to preserve joint structure, causal relationships, and downstream-ML utility when the reference dataset is small or the prior is weak. The [SDV vs SynthCity comparison](https://arxiv.org/abs/2506.17847) makes the weaknesses concrete.

**Tradition B — Structural Causal Model (SCM) priors.** This is the most relevant tradition. The clearest exemplar is [TabPFN's prior](https://www.nature.com/articles/s41586-024-08328-6) ([Hollmann et al. 2025](https://pubmed.ncbi.nlm.nih.gov/39780007/)): pre-training generates millions of synthetic tabular datasets by (1) sampling high-level hyperparameters, (2) sampling a DAG over nodes, (3) assigning random functional mappings (small MLPs, decision-tree-like rules, sigmoids, modular arithmetic) to each edge with Gaussian noise, (4) sampling which nodes are "features" vs "targets". Recent work extends this to [relational / multi-table settings](https://arxiv.org/abs/2507.03528) and to [causal-structure-aware autoregressive generation](https://arxiv.org/abs/2603.10254).

This is architecturally close to what we want. The critical difference: TabPFN's prior is **deliberately semantically empty** — the goal is a foundation model that generalizes across all tabular problems, so nodes have no inherent meaning. We want the opposite: **every node has meaning**, drawn from a domain ontology, and the space of DAGs we sample from is constrained to business-plausible ones.

**Tradition C — Agent-based / discrete-event simulation.** The marketing-science community (and tools like Simfluence) have long generated synthetic customer journeys via agent-based models: each simulated lead has internal state (interest level, budget, buying stage, decision-committee members), evolves over time under stochastic rules, and emits observable touchpoints. This is the right tradition for generating **realistic sequences of events**, but it's overkill for a first pass at lead scoring (which collapses the sequence to a feature vector) and under-powered for capturing clean causal structure on its own.

**The synthesis we want: SCM backbone, agent-based flavor, narrative scaffolding.** The statistical engine should be an SCM family in the TabPFN style, but with node semantics constrained by a domain ontology. Time-series and sequential features should be emitted by small, tightly-scoped agent-based subroutines that live *inside* selected nodes of the SCM (e.g., a node "behavioral_score" is internally a compressed summary of a simulated 30-day touchpoint sequence).

### 2.4 Multi-touch attribution, Markov chains, and why they matter for LTV

For the LTV extension, a realistic generator must assign revenue attribution across touchpoints. The industry-standard approach is [Markov-chain multi-touch attribution using "removal effects"](https://www.databricks.com/blog/2021/08/23/solution-accelerator-multi-touch-attribution.html). For our purposes, this is a **consistency target**, not a modeling choice: if our generator emits realistic touchpoint sequences and a realistic conversion/LTV outcome, students should be able to apply standard Markov-chain attribution and get sensible, non-degenerate results. This is a useful sanity check on the generator itself.

### 2.5 Survival analysis as the time-to-event substrate

For both lead scoring (time-to-conversion) and LTV (time-to-churn) the right latent substrate is a hazard function with time-varying covariates. Cox proportional hazards ([Cox 1972](https://www.jstor.org/stable/2985181)) is the interpretability-friendly default; Aalen additive and parametric (Weibull) hazards are viable alternatives. The generator should internally carry a hazard function per customer, use it to draw event times, and allow the exercise-facing target to be a thresholded binary ("converted within H days?") or a censored time ("days-to-conversion, with right-censoring at H").

---

## 3. Key design commitments and where they come from

These are the commitments that fall out of the research review. They are the "unless you can convince me otherwise" list for reviewers.

### 3.1 The DGP is a distribution over DGPs, not a single DGP

This is the user's explicit requirement and it is correct. The risk we're designing against is the "solved puzzle" problem: if every dataset from the generator has the same causal graph with different parameters, a student can reverse-engineer the structure once and then has a perfect prior on every future dataset.

The TabPFN prior addresses this by sampling fresh SCMs each time. We will do the same, but over a **constrained family of domain-plausible SCMs**.

**Two-level sampling:**
- **Level 1 — Structure sampling.** Sample a DAG from the family of plausible structures for this vertical. This determines which latent factors exist, which features are driven by which factors, whether the target is a direct function of observable features or is mediated by a latent "propensity," and so on.
- **Level 2 — Parameter sampling.** Given the structure, sample concrete parameters: coefficient magnitudes, noise scales, feature distributions, base rates, interaction strengths.

A student working with dataset *k* of this generator should not be able to infer the structure of dataset *k+1* with high confidence.

### 3.2 The vertical is a first-class citizen

A "vertical" is a bundle of:
- A human-readable narrative (who the company is, what they sell, who the leads are).
- A **domain ontology**: named variables with types, units, plausible ranges, and semantic tags (e.g., `company_size:firmographic:numeric:log-normal:[10, 1e6]`).
- A **structural prior**: a distribution over DAGs on the ontology, expressed as soft rules (e.g., "firmographic features are root causes," "engagement is mediated by a latent interest variable," "the target is non-trivially driven by at least one firmographic and at least one behavioral feature").
- A **functional-form library**: which kinds of edge mechanisms (linear, logistic, threshold, saturating, interaction, tree-like) are appropriate for this vertical.
- **Lead / customer archetypes**: named personas that skew the sampled distribution (e.g., "bargain-hunter small business," "enterprise evaluator," "tire-kicker student").

At v1 we ship one vertical. My strong recommendation is **a mid-market B2B SaaS product** (see §5 for why), but this is open for debate.

### 3.3 The generator must be deterministic given a seed and fully auditable

Every generated dataset must come bundled with:
- The sampled DAG (as a machine-readable object and a rendered image).
- The sampled parameters.
- The "ground-truth" feature importance ranking (computable directly from the DGP).
- The ground-truth feature-target causal path (which features are causes vs. confounds vs. colliders vs. noise).

This is non-negotiable and has two purposes: it enables **instructor introspection** (Shay can verify that the dataset is solvable and appropriately difficult), and it enables **automated evaluation** of student models against ground truth (not just against held-out data).

### 3.4 Difficulty is a tunable knob, not a consequence

The generator must expose an explicit difficulty axis. At minimum:
- **Signal-to-noise ratio** of the target.
- **Class imbalance** (for lead scoring) / zero-inflation rate (for LTV).
- **Number of spurious / irrelevant features.**
- **Strength of non-linear interactions.**
- **Presence of distribution shift** between train and test.
- **Missingness pattern** (MCAR / MAR / MNAR).

Shay should be able to ask for "week 3 difficulty" and get a dataset where logistic regression with light feature engineering is competitive, and "week 11 difficulty" where it is not.

### 3.5 LTV-readiness from v1

Even though v1 only ships the lead-scoring target, the internal data model must already represent the full customer journey as a sequence of events with timestamps, and the lead-scoring label must be derived from it as a late projection ("did this lead convert within H days?"). Adding LTV is then primarily a matter of: (a) continuing the simulation past conversion, (b) adding transaction/revenue generation, and (c) defining LTV targets as different projections of the same underlying trajectory.

---

## 4. Open design questions

These are the real forks. Reviewers: please pick sides.

### 4.1 How structured should the structural prior be?

**Option A — Thin ontology, thick structural prior.** The ontology lists ~20 canonical variables for the vertical. The structural prior is a small set of rigid rules: "roots must be firmographic/demographic; target must be a leaf; at most 3 latent factors; mediators are drawn from a named shortlist." Every sampled DAG is essentially a perturbation of a canonical business theory.
- *Pros:* Easy to implement. Generated datasets are always recognisable. Low risk of nonsense.
- *Cons:* Students who see 3–5 datasets may learn the canonical theory implicitly, defeating the purpose.

**Option B — Thick ontology, thin structural prior.** The ontology lists ~50–100 variables. The structural prior is expressed as probability distributions over edge counts, node-type-to-edge-type transition probabilities, and functional-form weights. Generated DAGs can look quite different from each other.
- *Pros:* Much higher between-dataset variance. Harder for students to "solve" the family.
- *Cons:* Risk of generating implausible or contradictory DAGs. Requires extensive sanity-check machinery.

**Option C — Hybrid with a "business theory" layer.** The prior samples at two levels: first, pick one of N named "business theories" (e.g., "demand-pull," "relationship-driven," "product-led growth"), each of which is a loose constraint over DAGs; then sample a DAG consistent with that theory.
- *Pros:* Pedagogically rich — the theory name is itself a learnable concept. Preserves between-dataset variance while keeping each dataset coherent.
- *Cons:* More implementation complexity. Requires us to actually write out ~5 plausible business theories.

**My tentative recommendation: Option C.** But I want this challenged.

### 4.2 Where do latent variables come from?

Latent variables (unobserved drivers) are the key to making the target non-trivial. The question is whether they should be:

- **Always present as a named "interest / propensity" node** that mediates between observable features and the target. Simple and pedagogically clean — the student is learning to predict an observable proxy for a latent interest. This matches how propensity modeling is actually taught.
- **Sampled as part of the structural prior** — sometimes the graph has zero mediators, sometimes three. Richer but harder to reason about.
- **A single always-present "consideration" latent, plus 0–2 additional context-dependent latents.** Compromise position.

### 4.3 How do we sample the functional forms on edges?

For each edge parent → child, we need a mechanism `child = f(parents) + noise`. Options drawn from TabPFN's edge library and the causal-TGAN literature:

- **Linear / affine:** simplest, appropriate for firmographic → firmographic propagation.
- **Generalised linear (logistic, Poisson):** appropriate for observable → target edges.
- **Small MLP with 1 hidden layer:** flexible, used by TabPFN.
- **Decision-tree-like rules:** appropriate for threshold effects ("company size crosses 500 employees → buying committee forms").
- **Saturating / sigmoid:** appropriate for engagement metrics.
- **Interaction terms** (product of two parents): essential — this is what makes tree ensembles beat linear models, and it must be present in our DGP or we're testing the wrong thing.

A reasonable default is a weighted mixture sampled per-edge, with weights conditioned on parent-type and child-type.

### 4.4 How do we make the generator feel "narratively real" without hand-authoring everything?

Real lead data has names, company domains, industries, sales reps, timestamps with weekday structure, free-text notes. We have three plausible approaches:

1. **Faker + rule tables.** Standard Python `faker` library for names, emails, company names; curated lookup tables for industries, titles, tech stacks, etc. Fast, deterministic, no LLM dependency.
2. **Optional LLM enrichment layer.** Generator emits structurally-valid rows; an optional post-processor uses an LLM to add realistic company names, paraphrase free-text fields, etc. Off by default.
3. **Pure-statistical + light faker.** Accept that a synthetic dataset will look synthetic in its string fields, and lean into it.

**My tentative recommendation: (1) with an optional (2) hook.** The generator must work offline with zero API calls. LLM enrichment is a nice-to-have, not a dependency.

### 4.5 What's the minimum viable interface?

A reasonable v1 surface:

```python
from synthlead import Generator, Vertical

gen = Generator(
    vertical=Vertical.load("b2b_saas_v1"),
    target="lead_conversion",   # or "ltv_12mo" in v2
    difficulty="intermediate",  # or a dict of explicit knobs
    seed=42,
)

dataset = gen.sample(n_rows=10_000)
dataset.features       # pandas DataFrame
dataset.target         # pandas Series
dataset.metadata       # sampled DAG, ground-truth importances, narrative
dataset.save("./out/")
```

The question: **how much should the user see of the DGP at generation time?** Zero (black box) / metadata only (current proposal) / full DAG exposed for inspection? For instructor use the answer is "full," for student use it's "zero," which argues for two modes.

### 4.6 How do we validate that generated datasets are "good"?

This is the most important question in the whole project and the one I have the least developed answer for.

Candidate validation axes:
- **Plausibility** — a domain expert looking at a sampled dataset thinks it looks like a real lead-scoring dataset. Hard to automate; requires human review.
- **Learnability** — standard models (logistic regression, random forest, XGBoost) achieve reasonable but not perfect performance. We can automate this.
- **Non-trivialness** — the majority-class baseline should be clearly beatable; but a single-feature classifier should not solve it. Automated.
- **Feature-importance consistency** — the ground-truth feature importances computable from the DAG should match, in broad ordering, the importances a trained XGBoost recovers. Automated. This is our primary internal sanity check.
- **Structural diversity** — across many samples from the generator, we should see high variance in the sampled DAGs (measured via structural Hamming distance or similar). Automated.
- **Narrative coherence** — a text description auto-generated from the DAG should read plausibly. Semi-automated; can use an LLM as a judge.

We should build these into the generator's CI from day one.

### 4.7 Open statistical subquestions (hard ones)

- How do we handle the fact that lead scoring and LTV operate on overlapping but distinct populations (all leads vs. converted customers)? The generator must produce a superset of leads and then let the user project to whichever subset the target requires.
- How do we simulate realistic selection bias (sales rep cherry-picking which leads to pursue) without making it a dominant signal?
- How do we handle time explicitly? The "timestamp" on a row could be conversion time, touch time, or snapshot time, and these are not interchangeable.
- Do we need a concept of a "market" that all leads share (so aggregate conversion rate changes over time), or is each lead i.i.d.? For LTV, the aggregate signal is essential. For lead scoring at v1, we might get away without it.

---

## 5. Proposed v1 vertical: mid-market B2B SaaS

### 5.1 Why this vertical

- It's the most-written-about lead scoring case in the industry literature, so reviewers and domain experts can sanity-check our output.
- It has all the pedagogically interesting features: firmographic + behavioral + technographic features, long sales cycles, buying committees, clear channel-source variation, meaningful imbalance, plausibility of both logistic regression and tree ensembles.
- Extending from lead scoring to LTV is natural because SaaS has subscription revenue, observable churn, and expansion revenue — the three textbook LTV components.
- It fits the economics/management framing of the course.

### 5.2 Narrative skeleton

*Company:* "Helix Analytics" — a mid-market B2B SaaS vendor selling a business-intelligence platform to companies with 50–5,000 employees. Founded 2017, 180 employees, ~$30M ARR, selling through inbound marketing, outbound SDRs, and a partner channel. Deal sizes $5k–$200k ACV. Sales cycles 30–180 days.

*Lead population:* 60% inbound (web visitors, content downloaders, free-trial signups), 25% outbound-sourced, 15% partner-referred. Company size log-normal; industry mix weighted toward tech, finance, retail, healthcare. Titles ranging from analyst (low authority, high engagement) to VP/C-level (high authority, low engagement).

*Target:* `converted_within_90_days ∈ {0, 1}`, base rate ~10–20% (a lever).

*Underlying "truth":* a latent `fit` (how well the product matches the lead's need) combines with a latent `intent` (how urgently they're buying) to drive a conversion hazard. Features are noisy observations of these two latents, with realistic confounds (e.g., enterprise leads engage less but convert at higher rates; inbound leads engage more but have lower fit).

### 5.3 Candidate variable ontology (draft)

| Category | Variables (v1 draft) |
|---|---|
| Firmographic | `company_size_employees`, `company_revenue_usd`, `industry`, `country`, `funding_stage`, `years_since_founded` |
| Demographic / identity | `lead_title_seniority`, `lead_function` (e.g., eng/sales/ops), `lead_tenure_months`, `email_domain_type` (corp/personal) |
| Technographic | `uses_competitor`, `uses_complement_stack`, `has_data_warehouse`, `tech_maturity_score` |
| Channel / source | `lead_source` (inbound/outbound/partner), `first_touch_channel`, `utm_campaign_category` |
| Behavioral (aggregate) | `page_views_30d`, `content_downloads_30d`, `pricing_page_visits`, `demo_requested`, `trial_started`, `days_to_first_touch`, `session_duration_avg` |
| Sales interaction | `discovery_calls_held`, `emails_exchanged`, `days_since_first_contact`, `assigned_rep_experience_years` |
| Latents (unobserved) | `fit`, `intent`, `budget_readiness` |
| Target | `converted_within_90_days`, (v2+: `ltv_12mo`, `time_to_churn_days`, `time_to_convert_days`) |

This list is explicitly a starting point. Reviewers: what am I missing?

---

## 6. Proposed architecture (rough)

```
┌─────────────────────────────────────────────────────────────┐
│                        VERTICAL MODULE                      │
│  • narrative skeleton (markdown template)                   │
│  • ontology (typed variable list)                           │
│  • structural prior (distribution over DAGs on ontology)    │
│  • functional-form library (edge mechanisms)                │
│  • archetype library (customer personas)                    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     STRUCTURE SAMPLER                       │
│  • sample business theory (Option C)                        │
│  • sample DAG consistent with theory + ontology             │
│  • validate: acyclic, connected, target is reachable,       │
│    no degenerate structures                                 │
│  Output: concrete DAG G                                     │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     PARAMETER SAMPLER                       │
│  for each node in G:                                        │
│      sample a functional form consistent w/ parent+child    │
│      types, from the vertical's function library            │
│      sample its parameters                                  │
│  sample noise scales                                        │
│  sample base rates / offsets                                │
│  Output: concrete SCM M = (G, f, ε)                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  TRAJECTORY SIMULATOR                       │
│  for each lead (row):                                       │
│      sample archetype (optional skew)                       │
│      propagate noise through SCM to draw latents + feats    │
│      draw event times from hazard function (survival core)  │
│      emit event log                                         │
│  Output: per-lead event sequences + feature snapshot        │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   TARGET PROJECTOR                          │
│  project event sequence to target:                          │
│      lead_scoring → threshold on conversion event           │
│      ltv_12mo    → discounted sum of revenue in horizon     │
│      time_to_churn → first churn event time + censor flag   │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  DIFFICULTY MODULATOR                       │
│  apply configured difficulty transforms:                    │
│    • inject noise / MCAR-MAR-MNAR missingness               │
│    • add spurious / redundant features                      │
│    • introduce train/test distribution shift                │
│    • adjust class imbalance via rejection sampling          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  SERIALIZATION + METADATA                   │
│  • DataFrame out                                            │
│  • metadata.json: DAG, parameters, ground-truth importance  │
│  • narrative.md: auto-rendered human-readable story         │
│  • optional: Faker enrichment (names, domains, timestamps)  │
└─────────────────────────────────────────────────────────────┘
```

Each of these boxes is a module with a narrow interface, which means:
- Adding a vertical = writing a new vertical module.
- Adding a new target family (LTV) = extending the simulator and projector.
- Swapping the structure sampler for something smarter later doesn't change the rest.

### 6.1 Key implementation choices to make early

- **DAG representation.** `networkx.DiGraph` is fine for v1; performance is not a concern at ≤100 nodes.
- **SCM implementation.** Write our own; don't depend on a causal-inference library as a hard dependency. We want full control over edge mechanisms.
- **Random state.** Single seeded `numpy.random.Generator` threaded through every sampler. No wall-clock, no global state, no hash-based randomness.
- **Output format.** Pandas DataFrame for v1; the metadata is a separate JSON + an image of the DAG (matplotlib/graphviz). Optionally a nicely formatted markdown "dataset card" similar to HuggingFace datasheets.

### 6.2 Things to borrow from existing libraries

- `lifetimes` — for validation only, and for the LTV extension. We will fit BG/NBD + Gamma-Gamma on our generated non-contractual data and confirm the fits are reasonable.
- `lifelines` — for generating time-to-event targets with a real hazard model, and for validation.
- `networkx` — DAG representation and topological sort for SCM propagation.
- `faker` — string fields.
- `sdmetrics` (optional) — some of their univariate / joint distribution diagnostics are useful for internal validation.

We will explicitly **not** depend on SDV, CTGAN, or any learn-from-data library as part of the core generator.

---

## 7. A v1 milestone plan (rough)

This is a suggestion, not a commitment. Calibrate against reviewer feedback.

- **M0 — Scaffolding.** Package skeleton, config system, seeded RNG plumbing, CI with lint/type-check/test. Cost: small.
- **M1 — Ontology + fixed DAG.** Write the B2B SaaS ontology, one hand-crafted canonical DAG, hand-written parameters. Produce a single reproducible lead-scoring dataset. Validate that a logistic regression + random forest behave sensibly on it.
- **M2 — Structure sampler.** Replace the hand-crafted DAG with a sampler over the structural prior (Option C from §4.1). Produce datasets whose DAGs differ across seeds.
- **M3 — Parameter sampler + functional library.** Make edge mechanisms varied and sampled, not hand-picked.
- **M4 — Difficulty modulator.** Add the noise / missingness / imbalance / shift controls.
- **M5 — Trajectory simulator + time-to-event core.** Move from "snapshot features" to "event sequence + late projection." This is the piece that unblocks LTV.
- **M6 — LTV extension.** Add monetary-value generation and LTV target projections.
- **M7 — Second vertical.** Prove the engine generalizes by adding a second vertical (candidates: B2C subscription media, B2B services / consulting, small-business lending).

M1 is probably 2–4 days of focused work. M2–M5 are the substantive R&D steps. M6–M7 are extensions.

---

## 8. Explicit open questions for reviewers

I want opinions on these before I touch any code.

1. **Is the proposed vertical right?** Is mid-market B2B SaaS the best choice for v1, or should we start with something closer to the economics students' intuition — consumer lending, insurance, or a B2C retail setting?
2. **Option A/B/C from §4.1 — which structural prior strategy?** I lean C. Argue me out of it or reinforce it.
3. **Should the generator expose time explicitly from v1, or fake it at v1 and add it in v2?** The clean answer is "v1 already has a trajectory simulator." The pragmatic answer is "snapshot features are plenty for a first lead-scoring dataset, don't over-engineer."
4. **What is the right granularity of the variable ontology?** 20 variables or 80? More feels more realistic but risks generating nonsense graphs.
5. **How do we avoid the "solved" problem at scale?** Even with structure sampling, a motivated student can run the generator 100 times, fit a meta-model, and learn the prior. Is that acceptable (it's a genuine ML skill) or should we actively obfuscate?
6. **Should LTV support be a goal or a non-goal for v1?** The user's brief says "LTV is a roadmap target, but foundations should take it into account." I've interpreted that as "design for LTV, ship lead scoring." Is that the right read?
7. **What's missing from §2?** I've surveyed probabilistic LTV models, lead-scoring feature taxonomies, SCM-based tabular synthesis, agent-based customer journey simulation, Markov attribution, and survival analysis. What important body of work am I not citing?
8. **Are the validation axes in §4.6 sufficient, or do we need formal tests?** In particular: should we validate against a real published dataset's joint moments, even though we're not trying to match any specific dataset?
9. **What's the right license?** MIT / Apache-2 for the generator code; what about the generated datasets themselves? (They're synthetic, so no privacy concerns, but there may be pedagogical reasons to mark them clearly.)
10. **What's the right name?** Working title is `SynthLeadLTV`. Bad. Suggestions welcome.

---

## 9. Selected references and further reading

Numbered for ease of cross-reference in review comments.

**LTV modeling (probabilistic):**
1. Schmittlein, Morrison, & Colombo (1987). "Counting Your Customers: Who Are They and What Will They Do Next?" *Management Science*. — Pareto/NBD.
2. Fader, Hardie, & Lee (2005). "'Counting Your Customers' the Easy Way: An Alternative to the Pareto/NBD Model." *Marketing Science*. — BG/NBD.
3. Fader & Hardie (2013). Technical notes on Gamma-Gamma. <http://www.brucehardie.com/notes/025/>
4. Gupta, Hanssens, Hardie, Kahn, Kumar, Lin, Ravishanker, & Sriram (2006). "Modeling Customer Lifetime Value." *Journal of Service Research*.

**LTV modeling (ML / hybrid):**
5. Wang, Liu, & Miao (2019). "A Deep Probabilistic Model for Customer Lifetime Value Prediction." Google Research. — ZILN loss.
6. Vanderveld, Pandey, Han, & Parekh (2016). "An Engagement-Based Customer Lifetime Value System for E-commerce." KDD.
7. Chen, Guitart, del Rio, & Perianez (2018). "Customer Lifetime Value in Video Games Using Deep Learning and Parametric Models." IEEE Big Data.

**Lead scoring:**
8. González-Flores, Rubiano-Moreno, & Sosa-Gómez (2025). "The Relevance of Lead Prioritization: A B2B Lead Scoring Model Based on Machine Learning." *Frontiers in Artificial Intelligence*. <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1554325/full>
9. D'Haen, Van den Poel, & Thorleuchter (2013). "Predicting Customer Profitability During Acquisition." *Decision Support Systems*.

**Synthetic tabular data — SCM / prior-based:**
10. Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Schirrmeister, & Hutter (2025). "Accurate Predictions on Small Data with a Tabular Foundation Model." *Nature*. — TabPFN v2, the cleanest statement of the SCM prior approach.
11. Hoppe, Franz, Kleinemeier, & Göbel (2025). "Generating Synthetic Relational Tabular Data via Structural Causal Models." TRL Workshop, ACL. — Extension to relational tables.
12. Tugnoli et al. (2026). "Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure." arXiv:2603.10254.
13. Causal-TGAN (2021). arXiv:2104.10680.

**Synthetic tabular data — learn-from-data:**
14. Patki, Wedge, & Veeramachaneni (2016). "The Synthetic Data Vault." IEEE DSAA. — SDV original.
15. Xu, Skoularidou, Cuesta-Infante, & Veeramachaneni (2019). "Modeling Tabular Data Using Conditional GAN." NeurIPS. — CTGAN.
16. Borisov, Seßler, Leemann, Pawelczyk, & Kasneci (2023). "Language Models are Realistic Tabular Data Generators." ICLR. — GReaT.

**Survival / time-to-event:**
17. Cox (1972). "Regression Models and Life-Tables." *JRSS-B*.
18. Aalen (1989). "A Linear Regression Model for the Analysis of Life Times." *Statistics in Medicine*.

**Attribution and customer journey:**
19. Anderl, Becker, von Wangenheim, & Schumann (2016). "Mapping the Customer Journey: Lessons Learned from Graph-Based Online Attribution Modeling." *International Journal of Research in Marketing*.
20. Archak, Mirrokni, & Muthukrishnan (2010). "Mining Advertiser-specific User Behavior Using Adfactors." WWW. — Early ML attribution.

**Industry / practical guides:**
21. Adobe Real-Time CDP B2B. "Predictive Lead and Account Scoring." Experience League docs.
22. Saber (2026). "Machine Learning Scoring: Definition, Examples & Use Cases." <https://www.saber.app/glossary/machine-learning-scoring>
23. Scalarly (2026). "How to Build a B2B Lead Scoring Model That Actually Works." <https://scalarly.com/blog/b2b-lead-scoring-model/>
24. PyMC Labs. "Bayesian Pareto NBD CLV Modeling with PyMC-Marketing." <https://www.pymc-labs.com/blog-posts/pareto-nbd>

---

*End of v0.1. Please tear this apart.*

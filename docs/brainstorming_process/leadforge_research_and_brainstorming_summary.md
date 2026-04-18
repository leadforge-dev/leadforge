# Foundation Document: A Framework for Dynamic, Structural Synthetic Dataset Generation for CRM Analytics

**Project Domain:** Synthetic Data Generation for Lead Scoring and Customer Lifetime Value (LTV) Prediction
**Target Audience:** Instructors, researchers, and developers in Machine Learning, Economics, and Management.
**Date:** Friday, April 17, 2026

---

## 1. Executive Summary & Problem Framing

Customer lifetime value (LTV) prediction and lead scoring are economically critical predictive tasks, making them ideal case studies for undergraduate and graduate machine learning curricula. However, the pedagogical landscape is constrained by a severe lack of suitable datasets:
* **Public datasets are shallow and overused:** Canonical datasets (e.g., Kaggle CRM exports, UCI Online Retail) are static, solved puzzles lacking complex latent dynamics. 
* **Existing synthetic tools fail on structural fidelity:** Traditional synthetic data libraries (e.g., SDV, CTGAN, Gretel) focus on privacy-preserving cloning. They learn the joint probability distribution of a flat table but remain entirely ignorant of the underlying *causal mechanics*. 
* **Static Data-Generating Processes (DGPs) are pedagogical dead-ends:** If a framework uses a single fixed causal graph and only perturbs noise parameters, students will reverse-engineer the invariant structure, turning causal discovery into a trivial parameter-estimation exercise.

**The Vision:** We propose a modular, generalizable Python framework that generates narrative-anchored synthetic datasets. Instead of sampling rows from a fixed distribution, this engine will dynamically sample the **causal graph structure itself** from a massive family of sensible business realities, simulate agent/event trajectories through that graph, and render the output into both relational CRM logs and supervised ML snapshots.

---

## 2. The Domain Anchor: B2B SaaS

To prevent the engine from generating mathematical abstractions devoid of meaning, generation must be anchored to a concrete, human-readable narrative and domain ontology. 

We establish **Mid-Market B2B SaaS (Software-as-a-Service)** as the foundational vertical. It naturally contains all pedagogically interesting features required for both lead scoring and LTV:
* **Firmographics (Macro-fit):** Annual revenue, employee count, industry vertical. (Often root exogenous causes).
* **Demographics (Micro-fit):** Job title, seniority, departmental affiliation.
* **Technographics:** Existing tech stack, integrations, maturity.
* **Behavioral / Engagement (Implicit intent):** Pricing page visits, demo requests, content downloads, webinar attendance.
* **Sales Interactions:** Discovery calls, email cadences, SDR effort.
* **LTV Mechanics:** Subscription revenue, observable churn, expansion revenue (upsell/cross-sell).

Real lead scoring is not a single direct Bernoulli outcome on a flat row; it is a combination of *fit* and *engagement* over time.

---

## 3. Mathematical & Theoretical Foundations

### 3.1 Structural Causal Models (SCMs)
The generation engine must be constructed upon SCMs to guarantee a verifiable ground-truth causal structure. An SCM provides deterministic and probabilistic mechanisms for simulating how variables are generated. It is defined by $M = \langle U, V, F, P(U) \rangle$, where:
* $V$: Endogenous (observed) variables (e.g., company size, demo requests).
* $U$: Exogenous (unobserved) noise or latent variables (e.g., buyer urgency, market shifts).
* $F$: Structural deterministic functions $v_i = f_i(pa_i, u_i)$, where $pa_i$ are the causal parents.
* $P(U)$: Joint probability distribution over the noise.

### 3.2 Dynamic Structural Sampling
To ensure structural variance across datasets, the framework must sample the Directed Acyclic Graph (DAG) before generating data. Generating a random DAG is easy; generating a *sensible* business DAG is hard. Two dominant mathematical approaches to enforce this:
1.  **Probabilistic Graph Grammars:** Iterative production rules that expand non-terminal symbols (e.g., `Marketing_Funnel`) into topologically valid substructures (e.g., `Inbound_Content_Sequence`).
2.  **Constraint-Based MCMC / Topological Tiering:** Assigning variables to chronological tiers (Context $\rightarrow$ Attributes $\rightarrow$ Latents $\rightarrow$ Interventions $\rightarrow$ Behaviors $\rightarrow$ Outcomes). Edges can only flow forward through tiers, heavily restricting the search space to logical causal flows.

### 3.3 Pedagogical Motifs
The graph sampler must intentionally inject classic causal traps to test students:
* **Confounders (Forks):** An unobserved latent (e.g., *Company Budget*) drives both an intervention (*Premium Support*) and the outcome (*LTV*), creating spurious correlations.
* **Colliders (Immoralities):** *High Intent* and *Existing Customer Status* both cause *Fast Sales Response*. Filtering by fast response spuriously correlates intent and status.

### 3.4 Modeling Time and LTV
LTV modeling splits between contractual (observed churn) and non-contractual (latent churn via Pareto/NBD or BG/NBD models). B2B SaaS allows for survival analysis (Cox proportional hazards, Aalen additive) for time-to-conversion and time-to-churn, functioning as the time-to-event substrate of the simulation.

---

## 4. Synthesized Architecture Outline

The framework should be structured as a 6-layer hybrid simulator:

1.  **Narrative & Schema Layer (DSL):** A configuration object defining the vertical, entity relationships (Accounts, Contacts, Leads), and variable vocabularies. 
2.  **Structure Sampling Engine:** Samples a specific DAG from the structural prior/grammar, ensuring acyclic properties, tier adherence, and the injection of causal motifs.
3.  **Mechanism Assigner:** Iterates through the topologically sorted nodes and assigns functional equations (linear, non-linear, thresholds) and noise distributions to each edge.
4.  **Trajectory & Event Simulator (Agent-based flavor):** Simulates the actual rows. It generates base populations (Accounts/Contacts) and steps through discrete time, emitting events (website sessions, sales calls, pipeline stage changes) based on the SCM and hazard functions.
5.  **Observation & Measurement Model:** Applies "CRM Noise." This layer introduces realistic operational imperfections: missing fields, delayed updates, imperfect attribution, and selection bias.
6.  **Task Renderer & Exporter:** Projects the relational world into specific formats:
    * Relational tables (Accounts, Touches, Opportunities).
    * Flat ML snapshot tables (e.g., `lead_scoring_dataset.csv`).
    * Ground-truth metadata (DAG JSON, latent variables, feature importances).

---

## 5. Open Questions & Design Alternatives

The following critical design forks must be resolved during the implementation phase. They represent a synthesis of differing architectural philosophies:

### 5.1 How Structured Should the Structural Prior Be?
* **Option A: Thick Ontology, Thin Prior (MCMC/Random):** Massive pool of variables. High variance, but high risk of implausible graphs. Requires heavy validation machinery.
* **Option B: Graph Grammars / Business Theories (Hybrid):** Sample from named "business theories" (e.g., Product-Led Growth vs. SDR-Heavy), which dictate specific graph templates and stochastic rewiring rules. 
    * *Synthesis Lean:* Option B provides the best balance of pedagogical coherence and dataset variance.

### 5.2 Functional Forms: Linear vs. Non-Linear vs. Rules
* **Option A: Generalized Linear Models (GLMs):** Clean, interpretable, easy to calculate exact causal effects. Lacks real-world realism.
* **Option B: Small MLPs / Decision Trees (TabPFN style):** Highly realistic, captures interaction effects naturally, but obscures the analytical ground truth from the instructor.
    * *Synthesis Lean:* A hybrid mechanism library. Linear forms for demographic propagation, saturating/sigmoid forms for engagement, and tree-like threshold rules for business logic (e.g., "MQL threshold met").

### 5.3 The Primary Abstraction: DAG vs. Event Simulator
* **Option A: Static DAG (SCM First):** Clean causal truth, easy to generate flat tables. Awkward for long, sequential customer journeys.
* **Option B: Event Simulator First:** Incredible realism for funnel dynamics, recency, and frequency. Harder to expose clean causal math.
    * *Synthesis Lean:* A **Hybrid Simulator**. A static SCM DAG defines latent traits and baseline propensities, while a discrete-time event simulator acts on those propensities to generate sequential behavioral logs.

### 5.4 Software Ecosystem: Custom vs. PPL Integration
* **Option A: Custom Python (NetworkX + NumPy):** Complete architectural control, lightweight, easier handling of unique graph grammars.
* **Option B: Probabilistic Programming Languages (PyMC, DoWhy):** Offloads mathematical heavy lifting and handles interventions automatically, but introduces heavy dependencies.
    * *Synthesis Lean:* Custom engine for DAG generation and sampling (NetworkX), with potential export compatibility to DoWhy formats for advanced causal analysis.

### 5.5 Realism Generation: Faker vs. LLMs vs. Seed Calibration
* **Option A: Pure Statistical + Faker:** Standard Python `faker` library for names/domains. Fast, deterministic, offline.
* **Option B: LLM Enrichment:** Post-processing via LLM for ultra-realistic text fields. 
* **Option C: Semi-synthetic Calibration:** Fitting marginals to a real seed dataset (like `causalAssembly`).
    * *Synthesis Lean:* Option A for the core, ensuring offline determinism. Option B and C as optional future plugins.

---

## 6. Evaluation Framework

To validate the framework itself, we must evaluate generated datasets across multiple axes beyond simple "plausibility":
1.  **Statistical Fidelity:** Marginal distributions, cardinality, and missingness match CRM expectations.
2.  **Structural Fidelity:** Can advanced causal discovery algorithms recover the basic exposed structure? Do interventions yield mathematically sound counterfactuals?
3.  **Task Difficulty (Learnability):** A simple majority-class or single-feature baseline must be beatable, but standard models (Logistic Regression, XGBoost) should achieve reasonable, non-perfect performance.
4.  **Feature Importance Consistency:** The ground-truth structural path weights must roughly align with the SHAP values recovered by an ensemble model.

---

## 7. Consolidated References & Inspirations

* **SCMs & Synthetic Data:**
    * Hollmann et al. (2025). "Accurate Predictions on Small Data with a Tabular Foundation Model" (TabPFN prior methodology).
    * Göbler et al. (2024). "causalAssembly: Generating Realistic Production Data for Benchmarking Causal Discovery."
    * Hoppe et al. (2025). "Generating Synthetic Relational Tabular Data via Structural Causal Models."
* **LTV & Customer Behavior:**
    * Fader, Hardie, & Lee (2005). "Counting Your Customers the Easy Way" (BG/NBD).
    * Wang, Liu, & Miao (2019). "A Deep Probabilistic Model for Customer Lifetime Value Prediction" (ZILN loss for neural networks).
* **Simulation Precedents:**
    * Walonoski et al. / Synthea Project. "Synthea synthetic patient generator" (Modular state-machine simulation).
* **Industry Standards:**
    * González-Flores et al. (2025). "The relevance of lead prioritization: a B2B lead scoring model based on machine learning."
    * HubSpot & Salesforce Documentation on predictive scoring models (Fit + Engagement separations).

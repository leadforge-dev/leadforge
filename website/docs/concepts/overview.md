---
sidebar_position: 1
title: Overview
---

# How leadforge works

`leadforge` generates datasets by **simulating a commercial world**, not by sampling rows from a distribution. This distinction matters:

- A distribution-sampler can reproduce the statistical shape of a CRM dataset.
- A world-simulator produces rows that have *reasons* — leads convert because they fit the ICP, have high urgency, and were engaged by a persistent SDR; leads don't convert because they stalled in technical review, or because the champion left the company.

That structure is what makes the data useful for teaching: there is something real to find, and it can be found with the right feature engineering and model choices.

## The generation pipeline

Generation runs in five sequential layers, each deterministic given the same seed:

```
1. Hidden world structure   ← sample motif family, rewire DAG
         ↓
2. Mechanism layer          ← assign mechanisms to every node
         ↓
3. Population layer         ← create accounts, contacts, leads with latent traits
         ↓
4. Simulation               ← run 90-day daily event loop
         ↓
5. Rendering                ← snapshot-safe feature extraction + relational export
```

### 1. Hidden world structure

A directed acyclic graph (DAG) of latent traits, pipeline states, and the conversion outcome is sampled from one of five **motif families** and then stochastically rewired. The motif families are:

| Family | What drives conversion |
|---|---|
| `fit_dominant` | Account/ICP fit is the primary signal |
| `intent_dominant` | Buying intent signals (sessions, demo requests) dominate |
| `sales_execution_sensitive` | SDR and AE behaviour is the strongest lever |
| `demo_trial_mediated` | Conversion is gated on a demo or trial event |
| `buying_committee_friction` | Multi-stakeholder dynamics create the main noise |

### 2. Mechanism layer

Every node in the sampled graph gets a concrete mechanism — a logistic latent score, Poisson intensity, recency-decayed engagement intensity, categorical channel influence, stage transition hazard, or conversion hazard. Parameters are calibrated per difficulty tier.

### 3. Population layer

Accounts (1,500), contacts (4,200), and leads (5,000) are instantiated with deterministic IDs (`acct_000001`, `lead_000001`) and latent trait vectors drawn from the world graph.

### 4. Simulation

A hybrid discrete-time simulator runs a 90-day daily loop. Each day, each active lead may:

- receive a touch (email, call, demo, etc.)
- generate a session
- receive a sales activity
- advance or stall in the pipeline stage sequence
- convert (via a calibrated hazard function)

Everything is event-derived — the `converted_within_90_days` label emerges from simulated events, not from a directly sampled Bernoulli.

### 5. Rendering

The simulation state is projected into:

- 9 relational tables — snapshot-filtered to ≤ anchor day for public bundles
- A flat ML-ready task table (the train/valid/test splits)
- Metadata files (manifest, feature dictionary, dataset card)

The **exposure mode** controls what gets written.

## Reproducibility

All generation is deterministic given `(recipe, config, seed, package version)`. The seed is recorded in `manifest.json` along with the package version, so any bundle can be exactly reproduced.

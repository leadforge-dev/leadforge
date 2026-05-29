---
sidebar_position: 3
title: Exposure modes
---

# Exposure modes

The **exposure mode** controls which parts of the hidden world are written into the output bundle. This is a first-class design decision: truth is filtered *during rendering*, not by omitting simulation.

| Mode | CLI flag | Who it's for |
|---|---|---|
| `student_public` | `--mode student_public` | Teaching, portfolio projects, Kaggle-style competitions |
| `research_instructor` | `--mode research_instructor` | Instructors, researchers, break-me analysis |

## `student_public`

The default. Includes:

- All 9 relational tables — but with post-conversion entities omitted and timestamps snapshot-capped
- Task splits (`train.parquet`, `valid.parquet`, `test.parquet`)
- `feature_dictionary.csv` and `dataset_card.md`
- `manifest.json` (provenance without hidden-world details)

**Excludes:**

- Latent registry (trait values)
- World spec (DGP parameters, recipe instantiation)
- Hidden causal graph (`graph.graphml`, `graph.json`)
- Mechanism summary
- `customers` and `subscriptions` tables (their existence reconstructs the label)

## `research_instructor`

Everything in `student_public`, plus:

- `metadata/world_spec.json` — full DGP and recipe instantiation
- `metadata/graph.{graphml,json}` — the hidden causal DAG
- `metadata/latent_registry.json` — per-entity latent trait values
- `metadata/mechanism_summary.json` — per-edge mechanism parameters
- Full-horizon relational tables including `customers` and `subscriptions`

The instructor companion dataset on HuggingFace (`leadforge/leadforge-lead-scoring-v1-instructor`) was built in this mode.

## Monotonicity guarantee

The exposure modes satisfy a monotonicity invariant: every artifact in `student_public` is also present in `research_instructor`. You can always load the student view from an instructor bundle; you cannot load the instructor view from a student bundle.

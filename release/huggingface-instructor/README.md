---
pretty_name: 'LeadForge: Synthetic B2B Lead Scoring (v1) — Instructor companion'
license: mit
language:
  - en
task_categories:
  - tabular-classification
size_categories:
  - 1K<n<10K
tags:
  - b2b
  - crm
  - datasets
  - lead-scoring
  - pandas
  - synthetic-data
  - tabular
configs:
  - config_name: intermediate
    default: true
    data_files:
      - split: train
        path: intermediate/tasks/converted_within_90_days/train.parquet
      - split: validation
        path: intermediate/tasks/converted_within_90_days/valid.parquet
      - split: test
        path: intermediate/tasks/converted_within_90_days/test.parquet
---

# LeadForge: Synthetic B2B Lead Scoring (v1) — Instructor companion

This is the **research / instructor companion** to the public
[`leadforge/leadforge-lead-scoring-v1`](https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1)
dataset.  It exposes the **full-horizon** view of a single difficulty
tier (`intermediate`) plus the **hidden causal structure** that the
public dataset deliberately redacts: the world graph (DAG), latent
trait registry, mechanism summary, and full-horizon relational tables
including `customers` and `subscriptions`.

It exists for instructors who want to walk students through how the
public dataset was generated, and for researchers who want to verify
that the public redactions actually remove the leakage paths the
dataset advertises.  **It is not a replacement for the public dataset
in any teaching or modelling context** — students should still train
on the public bundle.

## What this companion contains

```
.
├── intermediate/                     # research_instructor companion: full-horizon
│   ├── manifest.json                 # provenance + file hashes
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── tables/*.parquet              # full-horizon tables (incl. customers, subscriptions)
│   ├── tasks/converted_within_90_days/{train,valid,test}.parquet
│   └── metadata/                     # world_spec, graph.{graphml,json}, latent_registry, etc.
├── README.md                         # this file (HF dataset card)
├── dataset-cover-image.png           # dataset thumbnail
└── LICENSE
```

The single ``intermediate`` config exposes the same train/valid/test
parquet splits as the public dataset's ``intermediate`` config — same
seeds, same row counts (3,500 / 750 / 750), same target.  The
difference lives in the relational tables and metadata:

| File | Public `intermediate` | Instructor companion |
|---|---|---|
| `tables/leads.parquet` | redacted (label dropped) | full (label retained) |
| `tables/opportunities.parquet` | snapshot-filtered + redacted | full-horizon, full columns |
| `tables/customers.parquet` | omitted (would leak label) | included |
| `tables/subscriptions.parquet` | omitted (would leak label) | included |
| `tables/touches.parquet` etc. | filtered to ≤ snapshot day | full 90-day horizon |
| `metadata/world_spec.json` | absent | included (DGP + recipe) |
| `metadata/graph.{graphml,json}` | absent | included (hidden DAG) |
| `metadata/latent_registry.json` | absent | included (latent traits) |
| `metadata/mechanism_summary.json` | absent | included (per-edge mechanisms) |

The redaction contract is single-sourced in
[`leadforge/validation/leakage_probes.py`](https://github.com/leadforge-dev/leadforge/blob/main/leadforge/validation/leakage_probes.py)
and re-applied by
[`leadforge/render/relational_snapshot_safe.py`](https://github.com/leadforge-dev/leadforge/blob/main/leadforge/render/relational_snapshot_safe.py)
when the public bundle is built; this companion is the unfiltered
source view, so the two are always consistent by construction.

## Quick start

```python
from datasets import load_dataset

# Loads the same train/valid/test splits as the public 'intermediate'
# config; differs only in what `tables/` and `metadata/` provide.
ds = load_dataset(
    "leadforge/leadforge-lead-scoring-v1-instructor",
    name="intermediate",
)
train = ds["train"].to_pandas()

# Full-horizon relational tables — includes customers and subscriptions
# (omitted from the public dataset because their existence reconstructs
# the conversion label).
import pandas as pd
customers = pd.read_parquet(
    "hf://datasets/leadforge/leadforge-lead-scoring-v1-instructor/intermediate/tables/customers.parquet"
)
```

## Intended uses

- Teaching the **public-vs-instructor split** itself: load both
  datasets side-by-side, show students which columns and tables were
  redacted, and walk through why each was a leakage path.
- **Verifying the redaction contract:** train a model on the
  full-horizon tables, train another on the snapshot-safe public
  tables, compare AUC.  The gap is the redaction's effect.
- Teaching **causal structure and DGP transparency** using
  `metadata/world_spec.json` + `metadata/graph.json`.
- Reproducing the public dataset from the instructor view via
  [`leadforge`](https://github.com/leadforge-dev/leadforge/blob/main) source code.

## Out-of-scope uses

- **Production lead scoring.**  Same as the public dataset; the
  company, product, and customers are fictional.
- **Modelling with the unredacted view as a baseline.**  Models
  trained against the full-horizon tables look strong because they're
  directly seeing post-conversion events.  That number is not a
  baseline; it's the ceiling.
- **Demographic / fairness research.**  v1 does not model protected
  attributes.

## Composition

- **Entities.**  9 relational tables (accounts, contacts, leads,
  touches, sessions, sales_activities, opportunities, customers,
  subscriptions); per-row counts in `manifest.json`.
- **Splits.**  Identical to the public `intermediate` config: 70/15/15
  train/valid/test, deterministic given seed 42, recorded in
  `tasks/converted_within_90_days/task_manifest.json`.
- **Provenance.**  Recipe `b2b_saas_procurement_v1`, seed 42, package
  version stamped in `manifest.json` along with SHA-256 hashes for
  every parquet file.
- **Bundle schema version.**  5 (matches the public dataset).

## Maintenance, license

We *want* the dataset to be broken.  See the
[public dataset card](https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1)
for the adversarial-framing pointers, the issue templates, and the
break-me guide.  File issues at
[leadforge-dev/leadforge](https://github.com/leadforge-dev/leadforge);
PRs welcome.

| Field | Value |
|---|---|
| Generator | leadforge `1.0.0+` |
| Recipe | `b2b_saas_procurement_v1` |
| Canonical seed | 42 |
| Bundle schema version | 5 |
| Format | Parquet (canonical) |
| License | MIT — see [LICENSE](LICENSE) |
| Public dataset | [link](https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1) |

Verify integrity with `leadforge validate <bundle_dir>`; every file is
hashed in `manifest.json`.

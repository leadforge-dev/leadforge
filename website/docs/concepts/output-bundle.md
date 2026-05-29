---
sidebar_position: 4
title: Output bundle
---

# Output bundle structure

```
bundle_root/
├── manifest.json                   ← provenance, row counts, SHA-256 hashes
├── dataset_card.md                 ← human-readable documentation
├── feature_dictionary.csv          ← authoritative column spec
├── tables/                         ← 9 relational Parquet tables
│   ├── accounts.parquet
│   ├── contacts.parquet
│   ├── leads.parquet
│   ├── touches.parquet
│   ├── sessions.parquet
│   ├── sales_activities.parquet
│   └── opportunities.parquet
│   (customers.parquet, subscriptions.parquet — instructor mode only)
├── tasks/
│   └── converted_within_90_days/
│       ├── train.parquet           ← 70% split
│       ├── valid.parquet           ← 15% split
│       ├── test.parquet            ← 15% split
│       └── task_manifest.json
└── metadata/                       ← instructor mode only
    ├── world_spec.json
    ├── graph.graphml
    ├── graph.json
    ├── latent_registry.json
    └── mechanism_summary.json
```

## `manifest.json`

Records everything needed to reproduce or verify the bundle:

```json
{
  "bundle_schema_version": 5,
  "package_version": "1.0.0",
  "recipe_id": "b2b_saas_procurement_v1",
  "seed": 42,
  "generation_timestamp": "...",
  "exposure_mode": "student_public",
  "difficulty_profile": "intermediate",
  "table_inventory": { "leads": 5000, "accounts": 1500, ... },
  "file_hashes": { "tables/leads.parquet": "sha256:..." }
}
```

## Task splits

Splits are stratified by `converted_within_90_days` and fixed by seed:

| Split | Share | Rows (default 5,000 leads) |
|---|---|---|
| `train` | 70% | 3,500 |
| `valid` | 15% | 750 |
| `test` | 15% | 750 |

The split spec is recorded in `tasks/converted_within_90_days/task_manifest.json`.

## Validating a bundle

```bash
leadforge validate ./out/bundle
```

This checks:
- SHA-256 hashes in `manifest.json` match all files
- FK integrity across all relational tables
- No post-snapshot-anchor timestamps in public tables
- Conversion rate within declared tier bands
- No zero-variance features in task splits

---
sidebar_position: 3
title: Bundle schema
---

# Bundle schema reference

The bundle schema version is stamped in `manifest.json` as `bundle_schema_version`. The current version is **5**.

## `manifest.json` schema

```json
{
  "bundle_schema_version": 5,
  "package_version": "1.0.0",
  "recipe_id": "b2b_saas_procurement_v1",
  "seed": 42,
  "generation_timestamp": "2026-05-27T00:00:00Z",
  "exposure_mode": "student_public",
  "difficulty_profile": "intermediate",
  "table_inventory": {
    "accounts": 1500,
    "contacts": 4200,
    "leads": 5000,
    "touches": 38421,
    "sessions": 19847,
    "sales_activities": 12034,
    "opportunities": 1421
  },
  "file_hashes": {
    "tables/leads.parquet": "sha256:abc123..."
  }
}
```

## `task_manifest.json` schema

```json
{
  "task_id": "converted_within_90_days",
  "label_column": "converted_within_90_days",
  "label_window_days": 90,
  "primary_table": "leads",
  "split": { "train": 0.7, "valid": 0.15, "test": 0.15 },
  "description": "..."
}
```

## Entity ID format

All entity IDs are deterministic, zero-padded strings:

| Entity | Format | Example |
|---|---|---|
| Account | `acct_NNNNNN` | `acct_000001` |
| Contact | `cont_NNNNNN` | `cont_000042` |
| Lead | `lead_NNNNNN` | `lead_002501` |
| Touch | `touch_NNNNNN` | `touch_019844` |
| Session | `sess_NNNNNN` | `sess_005002` |
| Sales activity | `sact_NNNNNN` | `sact_007311` |
| Opportunity | `oppt_NNNNNN` | `oppt_000893` |

IDs are stable: the same `(recipe, seed, entity index)` always produces the same ID.

## Parquet conventions

- All tables use `snappy` compression.
- Timestamps are stored as `datetime64[us, UTC]`.
- Nullable integers use `Int64` (pandas nullable dtype), not `float64`.
- Boolean columns use `bool`, not `int`.

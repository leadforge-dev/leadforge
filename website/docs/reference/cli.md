---
sidebar_position: 1
title: CLI reference
---

# CLI reference

## `leadforge list-recipes`

```
leadforge list-recipes [--verbose]
```

Lists all available recipe IDs. Pass `--verbose` to see default parameters.

---

## `leadforge generate`

```
leadforge generate [OPTIONS]
```

Generate a full output bundle.

| Option | Default | Description |
|---|---|---|
| `--recipe ID` | required | Recipe to use (e.g. `b2b_saas_procurement_v1`) |
| `--seed INT` | `42` | Random seed — determines all stochasticity |
| `--mode MODE` | `student_public` | Exposure mode: `student_public` or `research_instructor` |
| `--difficulty TIER` | `intermediate` | Difficulty profile: `intro`, `intermediate`, `advanced` |
| `--n-leads INT` | `5000` | Number of leads to generate |
| `--out PATH` | required | Output directory (created if it doesn't exist) |
| `--override PATH` | — | YAML/JSON override file (overrides recipe defaults) |

### Example

```bash
leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intermediate \
  --n-leads 5000 \
  --out ./out/bundle
```

---

## `leadforge inspect`

```
leadforge inspect BUNDLE_DIR [--json]
```

Print a summary of `manifest.json` from a generated bundle.

```bash
leadforge inspect ./out/bundle
leadforge inspect ./out/bundle --json | jq .table_inventory
```

---

## `leadforge validate`

```
leadforge validate BUNDLE_DIR [--strict]
```

Run the full validation suite against a bundle:

- SHA-256 integrity check (every file vs. `manifest.json`)
- FK integrity across all relational tables
- Snapshot safety (no post-anchor timestamps in public mode)
- Conversion rate within declared tier bands
- No zero-variance features

Exits `0` on pass, non-zero on failure. Pass `--strict` to treat warnings as errors.

---
sidebar_position: 2
title: Quickstart
---

# Quickstart

## CLI

```bash
# List available recipes
leadforge list-recipes

# Generate a 5,000-lead intermediate bundle
leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intermediate \
  --n-leads 5000 \
  --out ./out/demo_bundle

# Inspect bundle metadata
leadforge inspect ./out/demo_bundle

# Validate bundle integrity
leadforge validate ./out/demo_bundle
```

Output at `./out/demo_bundle/`:

```
demo_bundle/
  manifest.json            ← provenance, row counts, file hashes
  dataset_card.md          ← human-readable documentation
  feature_dictionary.csv   ← feature names, types, descriptions
  tables/                  ← 9 relational Parquet tables
  tasks/
    converted_within_90_days/
      train.parquet
      valid.parquet
      test.parquet
      task_manifest.json
```

## Python API

```python
from leadforge.api import Generator

gen = Generator.from_recipe(
    "b2b_saas_procurement_v1",
    seed=42,
    exposure_mode="student_public",
)
bundle = gen.generate(n_leads=5000, difficulty="intermediate")
bundle.save("./out/demo_bundle")
```

## Load the task split

```python
import pandas as pd

train = pd.read_parquet(
    "./out/demo_bundle/tasks/converted_within_90_days/train.parquet"
)
print(train.shape)          # (3500, ~70)
print(train["converted_within_90_days"].mean())  # ≈ 0.18 on intermediate
```

## Use the pre-built dataset

If you just want to experiment with the v1 dataset without running the generator:

```python
from datasets import load_dataset

ds = load_dataset(
    "leadforge/leadforge-lead-scoring-v1",
    name="intermediate",   # or "intro" / "advanced"
)
train = ds["train"].to_pandas()
```

Or download from [Kaggle](https://www.kaggle.com/datasets/leadforge/leadforge-lead-scoring-v1).

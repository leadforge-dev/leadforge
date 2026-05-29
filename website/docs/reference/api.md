---
sidebar_position: 2
title: Python API
---

# Python API

## `Generator`

```python
from leadforge.api import Generator
```

### `Generator.from_recipe(recipe_id, *, seed, exposure_mode, **kwargs)`

Create a generator bound to a recipe.

```python
gen = Generator.from_recipe(
    "b2b_saas_procurement_v1",
    seed=42,
    exposure_mode="student_public",   # or "research_instructor"
)
```

| Parameter | Type | Default |
|---|---|---|
| `recipe_id` | `str` | required |
| `seed` | `int` | `42` |
| `exposure_mode` | `str` | `"student_public"` |

### `gen.generate(*, n_leads, difficulty, n_accounts, n_contacts)`

Run the full generation pipeline and return a `WorldBundle`.

```python
bundle = gen.generate(
    n_leads=5000,
    difficulty="intermediate",   # "intro" / "intermediate" / "advanced"
    n_accounts=1500,             # optional — overrides recipe default
    n_contacts=4200,             # optional — overrides recipe default
)
```

### `gen.world_spec`

Access the `WorldSpec` (narrative, config, hidden structure) after construction.

```python
gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
print(gen.world_spec.narrative.company.name)  # "Veridian Technologies"
```

---

## `WorldBundle`

Returned by `gen.generate()`.

### `bundle.save(path)`

Write the full bundle to disk at `path`.

```python
bundle.save("./out/bundle")
```

### `bundle.to_dataframes()`

Return all 9 relational tables as a `dict[str, pd.DataFrame]`.

```python
tables = bundle.to_dataframes()
leads = tables["leads"]
```

### `bundle.task_splits`

Access the train/valid/test DataFrames directly:

```python
train = bundle.task_splits["converted_within_90_days"]["train"]
```

---

## `list_recipes()`

```python
from leadforge.api import list_recipes

recipes = list_recipes()
print(recipes)  # ["b2b_saas_procurement_v1", ...]
```

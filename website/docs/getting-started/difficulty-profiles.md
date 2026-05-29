---
sidebar_position: 4
title: Difficulty profiles
---

# Difficulty profiles

Each recipe ships with difficulty profiles that modulate how hard the prediction task is. The profiles adjust noise, missingness, and signal strength — **not** the underlying causal structure.

| Profile | AUC (approx.) | Conversion rate | Who it's for |
|---|---|---|---|
| `intro` | ≈ 0.89 | ~28% | First-time learners, pipeline sanity-checks |
| `intermediate` | ≈ 0.79 | ~18% | Standard benchmarks, course projects |
| `advanced` | ≈ 0.68 | ~8% | Experienced practitioners, calibration and rare-event work |

## Set via CLI

```bash
leadforge generate --difficulty intro    # or intermediate / advanced
```

## Set via API

```python
bundle = gen.generate(n_leads=5000, difficulty="advanced")
```

## What changes between tiers

- **Noise level**: measurement noise on engagement features, channel attribution, and timing
- **Missingness**: the `advanced` tier has higher rates of missing values in several feature categories
- **Signal strength**: the coefficient magnitudes on causal paths are attenuated in harder tiers
- **Conversion rate**: directly calibrated so each tier hits its declared band

What does **not** change: the motif family, the graph structure, the entity IDs, the recipe narrative.

## Profile YAML

Difficulty profiles are defined in the recipe:

```
leadforge/recipes/b2b_saas_procurement_v1/difficulty_profiles.yaml
```

You can inspect the full calibration parameters by reading this file or by running:

```bash
leadforge list-recipes --verbose
```

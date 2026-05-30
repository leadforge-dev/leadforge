# leadforge

[![CI](https://github.com/leadforge-dev/leadforge/actions/workflows/ci.yml/badge.svg)](https://github.com/leadforge-dev/leadforge/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-0f766e)](https://leadforge-dev.github.io/leadforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Opinionated framework for generating synthetic CRM and GTM datasets from simulated commercial worlds.**

Created by [Shay Palachy Affek](http://www.shaypalachy.com/).

`leadforge` generates narrative-grounded synthetic revenue datasets — starting with lead scoring — designed for teaching, portfolio projects, and research. Rather than sampling rows from a distribution, it simulates a commercial world: a specific company, selling a specific product, to a specific kind of buyer, and renders realistic CRM-style outputs from that world.

**Docs:** [leadforge-dev.github.io/leadforge](https://leadforge-dev.github.io/leadforge/) · **Dataset:** [HuggingFace](https://huggingface.co/datasets/leadforge/leadforge-lead-scoring-v1) · [Kaggle](https://www.kaggle.com/datasets/derelictpanda/leadforge-lead-scoring-v1)

---

## What Makes LeadForge Different

- **World-first generation:** datasets are rendered from simulated companies, products, buyers, activities, opportunities, and outcomes.
- **Relational CRM shape:** output includes normalized tables plus task-ready train/validation/test splits for lead scoring.
- **Pedagogical realism:** snapshot discipline, redaction modes, leakage traps, calibration issues, and difficulty tiers are deliberate teaching material.

---

## Installation

Requires **Python 3.11+**.

```bash
pip install leadforge
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/leadforge-dev/leadforge.git
```

For development:

```bash
git clone https://github.com/leadforge-dev/leadforge.git
cd leadforge
pip install -e ".[dev]"
pre-commit install
```

---

## Quickstart

### CLI

```bash
# List available recipes
leadforge list-recipes

# Generate a dataset bundle
leadforge generate \
  --recipe b2b_saas_procurement_v1 \
  --seed 42 \
  --mode student_public \
  --difficulty intermediate \
  --n-leads 5000 \
  --out ./out/demo_bundle

# Inspect bundle metadata
leadforge inspect ./out/demo_bundle

# Or pipe the manifest into jq
leadforge inspect ./out/demo_bundle --json | jq .snapshot_day

# Validate bundle integrity
leadforge validate ./out/demo_bundle
```

### Python API

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

---

## Generated Data Preview

A generated bundle looks like CRM and GTM data, not a generic tabular benchmark. This compact slice comes from the intermediate lead-scoring bundle:

| split | industry | region | employee_band | lead_source | touch_count | session_count | opportunity_created | expected_acv | converted_within_90_days |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| train | logistics | UK | 200-499 | inbound_marketing | 0 | 0 | False | 66,699 | False |
| train | logistics | UK | 500-999 | inbound_marketing | 5 | 2 | False | 58,372 | False |
| train | logistics | US | 200-499 | partner_referral | 9 | 3 | True | 15,462 | False |
| train | healthcare_non_clinical | US | 200-499 | inbound_marketing | 5 | 1 | True | 30,490 | False |
| train | manufacturing | US | 1000-1999 | sdr_outbound | missing | 1 | True | 42,999 | False |

The full bundle also includes accounts, contacts, leads, touches, sessions, sales activities, opportunities, feature dictionaries, manifests, and model-ready Parquet task splits.

---

## Exposure Modes

Control what truth is visible in the output bundle:

| Mode | Purpose | Includes |
|------|---------|----------|
| `student_public` | Teaching / portfolio use | Tables, features, task splits, dataset card |
| `research_instructor` | Full truth for instructors / researchers | All of the above + hidden graph, world spec, latent registry, mechanism summary |

Set via `--mode` on the CLI or `exposure_mode=` in the Python API.

---

## Difficulty Profiles

Each recipe ships with difficulty profiles that control signal-to-noise ratio:

| Profile | Description |
|---------|-------------|
| `intro` | Strong signal, low noise — good for first-time learners |
| `intermediate` | Moderate signal, realistic noise |
| `advanced` | Weak signal, high noise — challenges experienced practitioners |

Set via `--difficulty` on the CLI or `difficulty=` in `generate()`.

---

## Output Bundle

```
bundle_root/
  manifest.json            # provenance, row counts, file hashes
  dataset_card.md          # human-readable dataset documentation
  feature_dictionary.csv   # feature names, types, descriptions
  tables/                  # 9 relational Parquet tables
  tasks/
    converted_within_90_days/
      train.parquet
      valid.parquet
      test.parquet
      task_manifest.json
  metadata/                # (research_instructor only) hidden graph, world spec, latents
```

---

## Key Design Principles

- **Deterministic**: same (recipe, seed, version) → identical output.
- **Relational-first**: 9 normalized tables; flat ML exports are derived.
- **No external APIs**: core generation never requires network access.
- **Simulation-driven labels**: `converted_within_90_days` emerges from simulated events, not sampled directly.
- **Leakage-safe**: no feature uses events after the snapshot anchor.

---

## Documentation

- [Documentation site](https://leadforge-dev.github.io/leadforge/)
- [Quickstart](https://leadforge-dev.github.io/leadforge/docs/getting-started/quickstart)
- [Output bundle reference](https://leadforge-dev.github.io/leadforge/docs/reference/output-bundle)
- [Generation method](https://leadforge-dev.github.io/leadforge/docs/dataset/generation-method)
- [Break-me guide](https://leadforge-dev.github.io/leadforge/docs/dataset/break-me)
- [Changelog](CHANGELOG.md)

---

## Development

```bash
pip install -e ".[dev]"
pytest                        # run all tests (~800)
ruff check .                  # lint
ruff format .                 # format
mypy leadforge/               # type check
pre-commit run --all-files    # full pre-commit suite
```

---

## License

MIT. See [LICENSE](LICENSE).

---

## Credits

Created by [Shay Palachy Affek ](http://www.shaypalachy.com/) [[GitHub](https://github.com/shaypal5)]

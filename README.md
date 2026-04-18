# leadforge

**Opinionated framework for generating synthetic CRM and GTM datasets from simulated commercial worlds.**

`leadforge` generates narrative-grounded synthetic revenue datasets starting with lead scoring, designed to support teaching, portfolio projects, and research. Rather than sampling rows from a distribution, it simulates a commercial world — a specific company, selling a specific product, to a specific kind of buyer — and renders realistic CRM-style outputs from that world.

---

## Installation

```bash
pip install leadforge
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

```bash
# List available recipes
leadforge list-recipes

# Coming in v0.2.0: generate a dataset bundle
# leadforge generate \
#   --recipe b2b_saas_procurement_v1 \
#   --seed 42 \
#   --mode student_public \
#   --difficulty intermediate \
#   --n-leads 5000 \
#   --out ./out/demo_bundle

# Coming in v0.4.0: inspect a generated bundle
# leadforge inspect ./out/demo_bundle

# Coming in v0.5.0: validate a generated bundle
# leadforge validate ./out/demo_bundle
```

**Python API** (coming in v0.2.0):

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

## Documentation

- [Design document](docs/leadforge_design_doc.md)
- [Architecture spec](docs/leadforge_architecture_spec.md)
- [Implementation plan](docs/leadforge_implementation_plan.md)

---

## License

MIT. See [LICENSE](LICENSE).

---
sidebar_position: 1
title: Installation
---

# Installation

**Requires Python 3.11+.**

## From PyPI

```bash
pip install leadforge
```

## From GitHub (latest development)

```bash
pip install git+https://github.com/leadforge-dev/leadforge.git
```

## Development install

```bash
git clone https://github.com/leadforge-dev/leadforge.git
cd leadforge
pip install -e ".[dev]"
pre-commit install
```

## Optional extras

| Extra | What it adds |
|---|---|
| `pip install leadforge[dev]` | Ruff, mypy, pytest, pre-commit |
| `pip install leadforge[publish]` | `huggingface_hub`, `datasets` — needed by the publish scripts |

## Verify

```bash
leadforge --version
leadforge list-recipes
```

You should see `b2b_saas_procurement_v1` in the recipe list.

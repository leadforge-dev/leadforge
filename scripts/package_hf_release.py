#!/usr/bin/env python3
"""Package the ``leadforge-lead-scoring-v1`` family for Hugging Face.

PR 5.2 — second of two PRs in Phase 5 (Platform packaging) of the v1
release roadmap.  This script:

1. Reads each public tier's ``manifest.json`` under ``release/`` and
   assembles a Hugging Face dataset card
   (``release/huggingface/README.md``) whose YAML frontmatter satisfies
   G12.1 of ``docs/release/v1_acceptance_gates.md``: ``pretty_name``,
   ``license: mit``, ``language: en``, ``task_categories:
   [tabular-classification]``, ``size_categories``, ``tags``, and one
   ``configs`` block per public tier with ``data_files`` pointing at the
   parquet task splits in the upload tree.  Exactly one config carries
   ``default: true`` (G12.2).
2. Reuses ``release/dataset-cover-image.png`` as the dataset thumbnail
   (HF datasets accept any reasonable PNG; the Kaggle floor of
   560×280 is the binding constraint and the validator lives in
   ``scripts/_release_common.py`` so both packagers share it).
3. Optionally assembles a HF-loadable upload directory under
   ``release/huggingface/`` as real-file copies of the per-tier
   bundles plus the rewritten README.  Same lesson as PR 5.1: don't
   symlink heavy bundle dirs; HF's ``datasets`` library walks the
   upload tree and silently skips broken symlinks in some versions.
4. Supports ``--variant=instructor`` to package the
   ``leadforge-lead-scoring-v1-instructor`` companion repo (G12.4)
   from ``release/intermediate_instructor/`` into a separate
   ``release/huggingface-instructor/`` tree with one config
   (``intermediate``).  The instructor variant ships its OWN dataset
   card body (``INSTRUCTOR_BODY``) — inlining the public README would
   leak three-tier prose into a single-tier dataset.

The actual ``huggingface_hub.HfApi().upload_folder()`` call lives in
PR 7.2; this script is intentionally publish-free.  ``--dry-run``
writes the README only and skips upload-tree assembly, useful for
README iteration.

Validation discipline: ``run_packager`` runs every check first and
returns ``errors`` WITHOUT writing any artifact when validation fails.
The CLI converts ``errors`` into rc=1.  Pre-flight problems (missing
release dir, unsafe ``--huggingface-dir``) raise and exit with rc=2
without touching disk.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

# Make ``scripts/`` importable regardless of how this file is loaded.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _release_common import (  # noqa: E402,F401 — must follow sys.path insert
    AGENT_REVIEWABLE_DOCS_DIR,
    AGENT_REVIEWABLE_ROOT_FILES,
    GITHUB_BLOB_BASE,
    SOURCE_TREE_BLOCK,
    ValidationError,
    load_manifest,
    replace_dir,
    replace_file,
    resolve_cover_image_path,
    rewrite_release_links,
    validate_cover_image,
    validate_readme_substitution,
    validate_upload_dir_safe,
)

# ---------------------------------------------------------------------------
# Hugging Face dataset-card YAML schema (chatgpt v2 §20, verified from
# https://huggingface.co/docs/hub/datasets-cards)
# ---------------------------------------------------------------------------

#: Allowed ``size_categories`` token (HF taxonomy).  Each public tier
#: ships task splits of 3500 / 750 / 750 rows; HF computes the size
#: bucket from ``data_files``, so the largest split (3.5K) lands the
#: dataset in the ``1K<n<10K`` bucket.  Pinned manually here rather
#: than derived from the manifest because (a) the bucket is stable
#: across the v1 family and (b) deriving from manifest row counts adds
#: I/O and a bucket-mapping helper for no real benefit at v1 scale.
HF_SIZE_BUCKET_5K: Final[str] = "1K<n<10K"

#: Public-tier configs land under ``intro``/``intermediate``/``advanced``;
#: ``intro`` is the recommended entry point — students loading the dataset
#: with no config argument land in the highest-prevalence tier, which is
#: the most forgiving teaching context. Pass ``default_config="intermediate"``
#: for graded assignments; ``default_config="advanced"`` for calibration
#: and noise-handling exercises.
DEFAULT_DEFAULT_CONFIG: Final[str] = "intro"

#: Allowed HF dataset-card ``task_categories`` token for tabular
#: binary classification.
HF_TASK_CATEGORY: Final[str] = "tabular-classification"

# ---------------------------------------------------------------------------
# Release-specific defaults
# ---------------------------------------------------------------------------

DEFAULT_PRETTY_NAME: Final[str] = "LeadForge: Synthetic B2B Lead Scoring (v1)"
DEFAULT_INSTRUCTOR_PRETTY_NAME: Final[str] = (
    "LeadForge: Synthetic B2B Lead Scoring (v1) — Instructor companion"
)

#: Tag list per the v1 release design (chatgpt v2 §20) — kept sorted at
#: render time to make determinism obvious in ``git diff`` output.
DEFAULT_TAGS: Final[tuple[str, ...]] = (
    "b2b",
    "crm",
    "datasets",
    "lead-scoring",
    "pandas",
    "synthetic-data",
    "tabular",
)

DEFAULT_LICENSE: Final[str] = "mit"
DEFAULT_LANGUAGE: Final[str] = "en"

DEFAULT_PUBLIC_TIERS: Final[tuple[str, ...]] = ("intro", "intermediate", "advanced")

#: HF expects the validation split to be called ``validation``.  The
#: committed parquet is ``valid.parquet``; the ``configs[*].data_files``
#: mapping bridges the two.
DEFAULT_TASK: Final[str] = "converted_within_90_days"
HF_SPLIT_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("train", "train"),
    ("validation", "valid"),
    ("test", "test"),
)

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_HUGGINGFACE_DIR: Final[Path] = Path("release/huggingface")
DEFAULT_HUGGINGFACE_INSTRUCTOR_DIR: Final[Path] = Path("release/huggingface-instructor")
DEFAULT_COVER_IMAGE: Final[Path] = Path("release/dataset-cover-image.png")

# ---------------------------------------------------------------------------
# README rewriting — HF-specific tree blocks
# ---------------------------------------------------------------------------

HF_UPLOAD_TREE_BLOCK: Final[str] = """```
.
├── intro/ intermediate/ advanced/    # student_public bundles, one per difficulty tier
│   ├── manifest.json                 # provenance + file hashes
│   ├── metrics.json                  # per-tier headline metrics (medians + spreads)
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── lead_scoring.csv              # flat convenience CSV (all splits)
│   ├── tables/*.parquet              # 7 snapshot-safe relational tables
│   └── tasks/converted_within_90_days/{train,valid,test}.parquet
├── docs/                             # vendored DGP / leakage / break-me docs (agent-readable)
├── metrics.json                      # top-level cross-tier metrics summary
├── claims_register.{md,json}         # claims → backing-artifact map (agent-readable)
├── README.md                         # this file (HF dataset card)
├── dataset-cover-image.png           # dataset thumbnail
└── LICENSE
```"""

HF_INSTRUCTOR_UPLOAD_TREE_BLOCK: Final[str] = """```
.
├── intermediate/                     # research_instructor companion: full-horizon
│   ├── manifest.json                 # provenance + file hashes
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── tables/*.parquet              # full-horizon tables (incl. customers, subscriptions)
│   ├── tasks/converted_within_90_days/{train,valid,test}.parquet
│   └── metadata/                     # world_spec, graph.{graphml,json}, latent_registry, etc.
├── docs/                             # vendored DGP / leakage / break-me docs (agent-readable)
├── claims_register.{md,json}         # claims → backing-artifact map (agent-readable)
├── README.md                         # this file (HF dataset card)
├── dataset-cover-image.png           # dataset thumbnail
└── LICENSE
```"""


def _hf_public_readme_text(readme: str) -> str:
    """Rewrite the public release README into the HF public dataset card.

    1. Source-repo tree diagram → public upload-tree diagram.
    2. ``](../foo)`` and ``](validation/...)`` → GitHub blob URLs
       (delegated to ``rewrite_release_links``).
    """

    text = readme.replace(SOURCE_TREE_BLOCK, HF_UPLOAD_TREE_BLOCK)
    return rewrite_release_links(text)


# ---------------------------------------------------------------------------
# Instructor-companion body
#
# Inlining the public README into the instructor card published 3-tier
# prose for a 1-tier dataset (PR 5.2 self-review #2).  The instructor
# package serves a different audience — researchers who already read
# the public dataset card — so it ships a focused, self-contained body
# that links to the public dataset for shared framing.
#
# The constant is the WHOLE markdown body (no YAML; that comes from the
# generated frontmatter).  It is byte-stable; the audit-artifact-sync
# test catches any drift against the committed instructor README.
# ---------------------------------------------------------------------------

INSTRUCTOR_BODY: Final[str] = f"""\
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

{HF_INSTRUCTOR_UPLOAD_TREE_BLOCK}

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
| `metadata/graph.{{graphml,json}}` | absent | included (hidden DAG) |
| `metadata/latent_registry.json` | absent | included (latent traits) |
| `metadata/mechanism_summary.json` | absent | included (per-edge mechanisms) |

The redaction contract is single-sourced in
[`leadforge/validation/leakage_probes.py`]({GITHUB_BLOB_BASE}/leadforge/validation/leakage_probes.py)
and re-applied by
[`leadforge/render/relational_snapshot_safe.py`]({GITHUB_BLOB_BASE}/leadforge/render/relational_snapshot_safe.py)
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
  [`leadforge`]({GITHUB_BLOB_BASE}) source code.

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

## Agent-reviewable artifacts

The companion ships the same self-contained review surface as the public
bundle so an AI reviewer (or a researcher without GitHub access) can
verify claims locally:

- ``docs/`` — vendored copies of the generation method, leakage probes
  contract, acceptance bands, break-me guide, v2 decision log, and the
  per-relational-table column descriptions (`relational_table_schemas.csv`).
- ``claims_register.{{md,json}}`` — every numerical / structural claim
  in this card paired with the artifact and path that backs it.
- ``intermediate/manifest.json`` and ``intermediate/feature_dictionary.csv``
  — SHA-256-hashed provenance and the authoritative column spec.

The instructor companion intentionally omits the top-level
``metrics.json`` (cross-tier medians would be misleading for a single
tier).  Use the public dataset's ``metrics.json`` when comparing tier
behaviour.

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
"""


# ---------------------------------------------------------------------------
# Dataclasses — one per top-level YAML block
#
# These are typed records, not invariants. Construction is unchecked;
# callers MUST run :func:`validate_card` before relying on the metadata
# being well-formed (mirrors the discipline in PR 5.1).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataFileEntry:
    """One ``configs[*].data_files[]`` entry."""

    split: str
    path: str


@dataclass(frozen=True)
class ConfigEntry:
    """One ``configs[]`` entry inside the YAML frontmatter."""

    config_name: str
    data_files: tuple[DataFileEntry, ...]
    default: bool = False


@dataclass(frozen=True)
class HuggingFaceCard:
    """Top-level HF dataset-card payload.

    ``body`` is the markdown that follows the YAML frontmatter — the
    rewritten public README for the public variant, the
    ``INSTRUCTOR_BODY`` constant for the instructor variant.
    """

    pretty_name: str
    license: str
    language: tuple[str, ...]
    task_categories: tuple[str, ...]
    size_categories: tuple[str, ...]
    tags: tuple[str, ...]
    configs: tuple[ConfigEntry, ...]
    body: str
    authors: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_card(card: HuggingFaceCard) -> list[ValidationError]:
    """Run every HF-side check against a built ``HuggingFaceCard``.

    Mirrors ``validate_metadata`` in the Kaggle packager: collects all
    field-level errors at once rather than fail-fasting.  G12.1 +
    G12.2 are surfaced here; G12.3 (load_dataset round-trip) is a
    separate test that hits the actual HF library.
    """

    errors: list[ValidationError] = []

    if not card.pretty_name.strip():
        errors.append(ValidationError(field="pretty_name", message="must be non-empty"))

    if card.license != DEFAULT_LICENSE:
        errors.append(
            ValidationError(
                field="license",
                message=f"expected {DEFAULT_LICENSE!r}, got {card.license!r}",
            )
        )

    if not card.language:
        errors.append(ValidationError(field="language", message="must contain at least one entry"))

    if HF_TASK_CATEGORY not in card.task_categories:
        errors.append(
            ValidationError(
                field="task_categories",
                message=f"must contain {HF_TASK_CATEGORY!r}",
            )
        )

    if not card.size_categories:
        errors.append(
            ValidationError(field="size_categories", message="must contain at least one entry")
        )

    if not card.tags:
        errors.append(ValidationError(field="tags", message="must contain at least one entry"))

    if not card.configs:
        errors.append(ValidationError(field="configs", message="must contain at least one config"))

    # G12.2 — exactly one config has default: true.
    defaults = [c for c in card.configs if c.default]
    if len(defaults) != 1:
        errors.append(
            ValidationError(
                field="configs",
                message=(
                    f"exactly one config must have default: true, got {len(defaults)} "
                    f"({[c.config_name for c in defaults]!r})"
                ),
            )
        )

    # Per-config integrity: non-empty data_files, every entry has a
    # split + a path.
    for i, config in enumerate(card.configs):
        if not config.config_name.strip():
            errors.append(
                ValidationError(field=f"configs[{i}].config_name", message="must be non-empty")
            )
        if not config.data_files:
            errors.append(
                ValidationError(
                    field=f"configs[{i}].data_files",
                    message="must contain at least one entry",
                )
            )
            continue
        for j, df in enumerate(config.data_files):
            if not df.split or not df.path:
                errors.append(
                    ValidationError(
                        field=f"configs[{i}].data_files[{j}]",
                        message="each entry must declare both split and path",
                    )
                )

    return errors


# ---------------------------------------------------------------------------
# Card construction
# ---------------------------------------------------------------------------


def _config_for_tier(
    *,
    config_name: str,
    task: str = DEFAULT_TASK,
    default: bool = False,
) -> ConfigEntry:
    """Pure constructor — no I/O.

    Builds a ``ConfigEntry`` from the canonical task-split layout.
    The earlier draft also opened the manifest to check the task was
    declared, but that's I/O for a structural check that
    :func:`assemble_upload_dir` already enforces (the parquet copies
    fail loud if the source paths are missing).  Keeping it pure makes
    card construction work without a fully-populated release dir,
    which is what tests want.
    """

    data_files = tuple(
        DataFileEntry(
            split=hf_split,
            path=f"{config_name}/tasks/{task}/{file_split}.parquet",
        )
        for hf_split, file_split in HF_SPLIT_NAMES
    )
    return ConfigEntry(config_name=config_name, data_files=data_files, default=default)


def _assert_tier_dir_exists(release_dir: Path, tier_dir: str) -> None:
    """Cheap preflight: tier dir must contain a ``manifest.json``.

    Catches user typos at ``--dry-run`` time rather than waiting for
    ``shutil.copytree`` to fail during assembly.  No JSON parse — we
    only care that the tier exists; the schema is the assembler's
    concern.
    """

    manifest = release_dir / tier_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.json missing for tier {tier_dir!r}: {manifest}")


def build_card(
    release_dir: Path,
    *,
    variant: str = "public",
    pretty_name: str | None = None,
    license_id: str = DEFAULT_LICENSE,
    language: Sequence[str] = (DEFAULT_LANGUAGE,),
    task_categories: Sequence[str] = (HF_TASK_CATEGORY,),
    size_categories: Sequence[str] = (HF_SIZE_BUCKET_5K,),
    tags: Sequence[str] = DEFAULT_TAGS,
    default_config: str = DEFAULT_DEFAULT_CONFIG,
    task: str = DEFAULT_TASK,
    body: str | None = None,
    authors: Sequence[str] = ("shaypal5",),
) -> HuggingFaceCard:
    """Assemble a ``HuggingFaceCard`` from the release tree.

    ``variant="public"`` builds the three-tier card pointing at
    ``release/{intro,intermediate,advanced}/`` and uses the rewritten
    public README as the body.

    ``variant="instructor"`` builds a single-config card pointing at
    ``release/intermediate_instructor/`` (flattened to ``intermediate/``
    in the upload tree) and uses :data:`INSTRUCTOR_BODY` as the body.
    The instructor variant gets a focused body rather than the public
    README to avoid publishing 3-tier prose for a 1-tier dataset.
    """

    if variant == "public":
        for tier in DEFAULT_PUBLIC_TIERS:
            _assert_tier_dir_exists(release_dir, tier)
        configs = tuple(
            _config_for_tier(
                config_name=tier,
                task=task,
                default=(tier == default_config),
            )
            for tier in DEFAULT_PUBLIC_TIERS
        )
        if pretty_name is None:
            pretty_name = DEFAULT_PRETTY_NAME
        if body is None:
            body = _hf_public_readme_text(
                (release_dir / "README.md").read_text(encoding="utf-8"),
            )
    elif variant == "instructor":
        _assert_tier_dir_exists(release_dir, "intermediate_instructor")
        configs = (_config_for_tier(config_name="intermediate", task=task, default=True),)
        if pretty_name is None:
            pretty_name = DEFAULT_INSTRUCTOR_PRETTY_NAME
        if body is None:
            body = INSTRUCTOR_BODY
    else:
        raise ValueError(f"unknown variant: {variant!r} (expected 'public' or 'instructor')")

    return HuggingFaceCard(
        pretty_name=pretty_name,
        license=license_id,
        language=tuple(language),
        task_categories=tuple(task_categories),
        size_categories=tuple(size_categories),
        tags=tuple(tags),
        configs=configs,
        body=body,
        authors=tuple(authors),
    )


# ---------------------------------------------------------------------------
# Rendering
#
# YAML serialization uses ``yaml.safe_dump`` with a custom dumper that
# forces indent-2 on top-level sequences (HF dataset-card examples in
# the wild use this style; PyYAML's default puts top-level list items
# at indent 0).  The earlier draft hand-rolled the renderer with a
# brittle quoting heuristic; PR 5.2's self-review caught the hack and
# replaced it with PyYAML.
# ---------------------------------------------------------------------------


class _IndentedDumper(yaml.SafeDumper):
    """Force indent-2 on top-level sequences.

    PyYAML's default ``increase_indent`` returns ``indentless=True``
    when emitting a top-level block sequence, putting ``- item`` at
    column 0.  HF dataset cards conventionally indent these by 2 (per
    the examples in the HF docs), so we override the flag to ``False``.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:  # noqa: ARG002
        return super().increase_indent(flow, False)


def _frontmatter_dict(card: HuggingFaceCard) -> dict[str, Any]:
    """Build the dict serialised into YAML.

    Tags are sorted at render time (mirrors the keyword sort in the
    Kaggle packager) so dataclass order doesn't leak into the rendered
    file.  The dict is built in field order so PyYAML preserves it
    (with ``sort_keys=False``).
    """

    return {
        "pretty_name": card.pretty_name,
        "license": card.license,
        **({"authors": list(card.authors)} if card.authors else {}),
        "language": list(card.language),
        "task_categories": list(card.task_categories),
        "size_categories": list(card.size_categories),
        "tags": sorted(card.tags),
        "configs": [
            {
                "config_name": config.config_name,
                **({"default": True} if config.default else {}),
                "data_files": [{"split": df.split, "path": df.path} for df in config.data_files],
            }
            for config in card.configs
        ],
    }


def render_yaml_frontmatter(card: HuggingFaceCard) -> str:
    """Render the YAML frontmatter as a deterministic string.

    The leading and trailing ``---`` markers are included so callers
    can concatenate this directly to the body.
    """

    payload = yaml.dump(
        _frontmatter_dict(card),
        Dumper=_IndentedDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=200,
    )
    return f"---\n{payload}---\n"


def render_card(card: HuggingFaceCard) -> str:
    """Render the full README (YAML frontmatter + markdown body)."""

    body = card.body
    if not body.endswith("\n"):
        body += "\n"
    return render_yaml_frontmatter(card) + "\n" + body


# ---------------------------------------------------------------------------
# Upload-directory assembly
# ---------------------------------------------------------------------------


def assemble_upload_dir(
    release_dir: Path,
    upload_dir: Path,
    *,
    variant: str = "public",
    rendered_readme: str,
    cover_image: Path = DEFAULT_COVER_IMAGE,
) -> None:
    """Assemble ``upload_dir`` for ``huggingface_hub.upload_folder()``.

    Writes the README, copies the cover image and LICENSE, and copies
    each tier bundle as a real-file copy (no symlinks; ``datasets``
    walks the tree and silently skips broken symlinks in some
    versions).  ``rendered_readme`` is the already-validated and
    rendered card text — passing it in (rather than reading it back
    from ``run_packager``) means this function produces a complete
    upload tree on its own.

    For ``variant="instructor"``, the source directory
    ``intermediate_instructor/`` is flattened to ``intermediate/`` in
    the upload tree so ``load_dataset(..., name="intermediate")``
    works against the companion repo without naming friction.
    """

    kind = "huggingface" if variant == "public" else "huggingface-instructor"
    validate_upload_dir_safe(upload_dir, release_dir, kind=kind)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # README — written from the validated card text.
    readme_path = upload_dir / "README.md"
    if readme_path.is_symlink() or readme_path.is_file():
        readme_path.unlink()
    readme_path.write_text(rendered_readme, encoding="utf-8")

    # Cover image — copy the resolved path as-is.  Path resolution
    # happens once in ``run_packager`` via ``resolve_cover_image_path``
    # so the validator and the assembler agree on which file to use.
    if cover_image.exists():
        replace_file(cover_image, upload_dir / cover_image.name)

    # LICENSE — straight copy, no rewriting.
    license_src = release_dir / "LICENSE"
    if license_src.exists():
        replace_file(license_src, upload_dir / "LICENSE")

    # Agent-reviewable root files (metrics.json, claims_register.*).
    # The public variant ships the cross-tier ``metrics.json``; the
    # instructor companion intentionally omits it (single-tier dataset
    # — cross-tier numbers would mislead).  Both variants ship the
    # claims register and the vendored docs subtree so an AI reviewer
    # never has to follow github.com/blob/main/... links to verify
    # whatever's on the README.
    public_root_files = {
        "metrics.json",
        "claims_register.md",
        "claims_register.json",
        "claims_register_source.yaml",
    }
    instructor_root_files = {
        "claims_register.md",
        "claims_register.json",
        "claims_register_source.yaml",
    }
    allow_for_variant = public_root_files if variant == "public" else instructor_root_files
    for rel, _required in AGENT_REVIEWABLE_ROOT_FILES:
        if rel not in allow_for_variant:
            continue
        src = release_dir / rel
        if src.is_file():
            replace_file(src, upload_dir / rel)

    # Vendored docs subtree.
    docs_src = release_dir / AGENT_REVIEWABLE_DOCS_DIR
    if docs_src.is_dir():
        replace_dir(docs_src, upload_dir / AGENT_REVIEWABLE_DOCS_DIR)

    # Per-tier bundles — full directory copies.  The instructor variant
    # flattens its source dir name.
    if variant == "public":
        for tier in DEFAULT_PUBLIC_TIERS:
            replace_dir(release_dir / tier, upload_dir / tier)
    elif variant == "instructor":
        replace_dir(
            release_dir / "intermediate_instructor",
            upload_dir / "intermediate",
        )
    else:
        raise ValueError(f"unknown variant: {variant!r} (expected 'public' or 'instructor')")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackagerOutcome:
    """Return value from :func:`run_packager` — used by tests + CLI."""

    card: HuggingFaceCard
    readme_path: Path
    errors: tuple[ValidationError, ...]
    assembled: bool


def run_packager(
    release_dir: Path,
    *,
    huggingface_dir: Path = DEFAULT_HUGGINGFACE_DIR,
    variant: str = "public",
    pretty_name: str | None = None,
    default_config: str = DEFAULT_DEFAULT_CONFIG,
    task: str = DEFAULT_TASK,
    cover_image: Path = DEFAULT_COVER_IMAGE,
    dry_run: bool = False,
) -> PackagerOutcome:
    """Build, validate, and write the HF dataset card.

    Validation discipline: build → validate → ONLY THEN write.  If
    validation fails the function returns an outcome with populated
    ``errors`` and **no artifact on disk**.  The CLI converts errors
    to rc=1.

    With ``dry_run=False`` (the default) the packager additionally
    assembles the HF-loadable upload directory under
    ``huggingface_dir`` (real-file copies of the per-tier bundles +
    cover image + LICENSE).  ``dry_run=True`` skips the assembly step
    entirely — useful for fast README iteration.
    """

    if not release_dir.exists():
        raise FileNotFoundError(f"release directory not found: {release_dir}")

    # Hoist the upload-dir safety check BEFORE any mkdir or write —
    # including the dry-run path (Copilot review on PR #72).  The
    # earlier draft only checked inside ``assemble_upload_dir``, so a
    # dry run with ``--huggingface-dir .`` would happily write a
    # README into ``cwd`` before the safety guard ever ran.
    # ``assemble_upload_dir`` retains its own call as defence-in-depth
    # for callers that bypass ``run_packager``.
    kind = "huggingface" if variant == "public" else "huggingface-instructor"
    validate_upload_dir_safe(huggingface_dir, release_dir, kind=kind)

    # Resolve cover image once — the same path is used for validation
    # and assembly so the two cannot disagree (Copilot review on
    # PR #72).
    resolved_cover = resolve_cover_image_path(cover_image, release_dir)

    card = build_card(
        release_dir,
        variant=variant,
        pretty_name=pretty_name,
        default_config=default_config,
        task=task,
    )

    errors: list[ValidationError] = []
    errors.extend(validate_card(card))
    errors.extend(validate_cover_image(resolved_cover))
    if variant == "public":
        # The instructor body doesn't substitute SOURCE_TREE_BLOCK —
        # it's a self-contained markdown — so the substitution guard
        # only applies to the public variant.
        errors.extend(validate_readme_substitution(release_dir, packager_name="HF"))

    readme_path = huggingface_dir / "README.md"

    # Validation gate: don't leave broken artifacts on disk.
    if errors:
        return PackagerOutcome(
            card=card,
            readme_path=readme_path,
            errors=tuple(errors),
            assembled=False,
        )

    rendered = render_card(card)
    huggingface_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # README-only — fast iteration path.  Write directly, don't
        # invoke the assembler.
        readme_path.write_text(rendered, encoding="utf-8")
    else:
        assemble_upload_dir(
            release_dir,
            huggingface_dir,
            variant=variant,
            rendered_readme=rendered,
            cover_image=resolved_cover,
        )

    return PackagerOutcome(
        card=card,
        readme_path=readme_path,
        errors=(),
        assembled=not dry_run,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate Hugging Face README.md for leadforge-lead-scoring-v1.",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release bundle root containing one subdirectory per tier (default: %(default)s)",
    )
    parser.add_argument(
        "--huggingface-dir",
        type=Path,
        default=None,
        help=(
            "output directory for README.md + assembled upload tree "
            "(default: release/huggingface for variant=public, "
            "release/huggingface-instructor for variant=instructor)"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=("public", "instructor"),
        default="public",
        help="public (3-tier) or instructor (companion repo); default: %(default)s",
    )
    parser.add_argument(
        "--default-config",
        default=DEFAULT_DEFAULT_CONFIG,
        help=(
            "config that carries default: true (G12.2; "
            "default: %(default)s; rejected for --variant=instructor "
            "which has only one config)"
        ),
    )
    parser.add_argument(
        "--cover-image",
        type=Path,
        default=DEFAULT_COVER_IMAGE,
        help="path to the dataset cover image (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate + write README only; skip assembling the upload directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    variant: str = args.variant
    huggingface_dir: Path = args.huggingface_dir or (
        DEFAULT_HUGGINGFACE_DIR if variant == "public" else DEFAULT_HUGGINGFACE_INSTRUCTOR_DIR
    )

    # Reject silent misconfiguration — instructor variant has one
    # config and ``--default-config`` is meaningless for it.
    if variant == "instructor" and args.default_config != DEFAULT_DEFAULT_CONFIG:
        print(
            f"error: --default-config={args.default_config!r} is not valid with "
            f"--variant=instructor (instructor has a single 'intermediate' config)",
            file=sys.stderr,
        )
        return 2

    try:
        outcome = run_packager(
            args.release_dir,
            huggingface_dir=huggingface_dir,
            variant=variant,
            default_config=args.default_config,
            cover_image=args.cover_image,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if outcome.errors:
        print("validation failed:", file=sys.stderr)
        for err in outcome.errors:
            print(f"  - {err.field}: {err.message}", file=sys.stderr)
        return 1

    print(f"wrote {outcome.readme_path}", file=sys.stderr)
    if outcome.assembled:
        print(f"assembled upload tree under {huggingface_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

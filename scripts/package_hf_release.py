#!/usr/bin/env python3
"""Package the ``leadforge-lead-scoring-v1`` family for Hugging Face.

PR 5.2 — second of two PRs in Phase 5 (Platform packaging) of the v1
release roadmap.  This script:

1. Reads each public tier's ``manifest.json`` + flat CSV header under
   ``release/`` and assembles a Hugging Face dataset card
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
   (``intermediate``).  Same code path; different defaults.

The actual ``huggingface_hub.HfApi().upload_folder()`` call lives in
PR 7.2; this script is intentionally publish-free.  ``--dry-run``
writes the README only and skips upload-tree assembly, useful for
README iteration.

Failed validation exits with rc=1; pre-flight errors (missing release
dir, missing tier, unsafe ``--huggingface-dir``) exit with rc=2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Make ``scripts/`` importable regardless of how this file is loaded.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ``GITHUB_BLOB_BASE`` is re-exported for tests and downstream callers
# (mirrors the symbol surface of ``scripts/package_kaggle_release.py``).
from _release_common import (  # noqa: E402,F401 — must follow sys.path insert
    GITHUB_BLOB_BASE,
    ValidationError,
    rewrite_release_links,
    validate_cover_image,
    validate_upload_dir_safe,
)

# ---------------------------------------------------------------------------
# Hugging Face dataset-card YAML schema (chatgpt v2 §20, verified from
# https://huggingface.co/docs/hub/datasets-cards)
# ---------------------------------------------------------------------------

#: Allowed ``size_categories`` tokens (HF taxonomy).  Each public tier
#: has 5,000 leads (3500 train / 750 valid / 750 test) so ``1K<n<10K``
#: is the right bucket; constants are kept here so ``--variant=instructor``
#: can pick the right bucket too if the tier sizes ever change.
HF_SIZE_BUCKET_5K: Final[str] = "1K<n<10K"

#: Public-tier configs land under ``intro``/``intermediate``/``advanced``;
#: ``intermediate`` is the recommended entry point per G12.2 (the
#: difficulty band sits between the two extremes).  Picking
#: ``intermediate`` as default mirrors ``release/HF_DATASET_CARD.md``
#: (the legacy stub PR 5.2 supersedes).
DEFAULT_DEFAULT_CONFIG: Final[str] = "intermediate"

#: Allowed HF dataset-card ``task_categories`` token for tabular
#: binary classification.
HF_TASK_CATEGORY: Final[str] = "tabular-classification"

# ---------------------------------------------------------------------------
# Release-specific defaults
# ---------------------------------------------------------------------------

DEFAULT_OWNER: Final[str] = "leadforge"
DEFAULT_DATASET_SLUG: Final[str] = "leadforge-lead-scoring-v1"
DEFAULT_INSTRUCTOR_DATASET_SLUG: Final[str] = "leadforge-lead-scoring-v1-instructor"

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
DEFAULT_INSTRUCTOR_TIERS: Final[tuple[str, ...]] = ("intermediate_instructor",)

#: HF expects the validation split to be called ``validation`` (HF
#: convention; ``valid`` would still load but the viewer labels it
#: differently).  The committed parquet is ``valid.parquet``; the
#: ``configs[*].data_files`` mapping bridges the two.
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

#: HF dataset cards do not require a cover image, but the field is
#: harmless and keeps the published thumbnail aligned with Kaggle's.
COVER_IMAGE_FILENAME: Final[str] = "dataset-cover-image.png"

# ---------------------------------------------------------------------------
# README rewriting — HF-specific tree block
# ---------------------------------------------------------------------------

#: Source-repo tree diagram from ``release/README.md`` — must match the
#: PR 4.1 README byte-for-byte; the silent-failure guard
#: (:func:`_validate_readme_substitution`) fires when the two drift
#: apart (mirrors PR 5.1's ``KAGGLE_TREE_BLOCK`` validator).
HF_TREE_BLOCK_SOURCE: Final[str] = """```
release/
├── intro/ intermediate/ advanced/    # student_public bundles, one per difficulty tier
│   ├── manifest.json                 # provenance + file hashes
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── lead_scoring.csv              # flat convenience CSV (all splits)
│   ├── tables/*.parquet              # 7 snapshot-safe relational tables
│   └── tasks/converted_within_90_days/{train,valid,test}.parquet
├── intermediate_instructor/          # research companion: full-horizon tables + metadata/
├── notebooks/01_baseline_lead_scoring.ipynb
└── validation/                       # validation_report.{json,md} + figures
```"""

HF_UPLOAD_TREE_BLOCK: Final[str] = """```
.
├── intro/ intermediate/ advanced/    # student_public bundles, one per difficulty tier
│   ├── manifest.json                 # provenance + file hashes
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── lead_scoring.csv              # flat convenience CSV (all splits)
│   ├── tables/*.parquet              # 7 snapshot-safe relational tables
│   └── tasks/converted_within_90_days/{train,valid,test}.parquet
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
│   ├── tables/*.parquet              # 9 full-horizon tables (incl. customers, subscriptions)
│   ├── tasks/converted_within_90_days/{train,valid,test}.parquet
│   └── metadata/                     # world_spec, graph.{graphml,json}, latent_registry, etc.
├── README.md                         # this file (HF dataset card)
├── dataset-cover-image.png           # dataset thumbnail
└── LICENSE
```"""


def _hf_readme_text(readme: str, *, tree_block: str = HF_UPLOAD_TREE_BLOCK) -> str:
    """Apply the HF-specific rewrites to a copy of the release README.

    Rewrites:

    1. Source-repo tree diagram → upload-tree diagram (the published
       README should describe what the *user* sees on HF, not the
       source repo layout).
    2. ``](../foo)`` → ``](GITHUB_BLOB_BASE/foo)`` and
       ``](validation/...)`` → blob URL — both via
       ``rewrite_release_links`` from ``_release_common``.

    The instructor variant calls this with
    ``tree_block=HF_INSTRUCTOR_UPLOAD_TREE_BLOCK``; otherwise the
    public upload tree is used.
    """

    text = readme.replace(HF_TREE_BLOCK_SOURCE, tree_block)
    text = rewrite_release_links(text)
    return text


def _validate_readme_substitution(release_dir: Path) -> list[ValidationError]:
    """Guard against silent drift between the README's tree diagram and
    ``HF_TREE_BLOCK_SOURCE`` (mirrors PR 5.1's Kaggle-side guard).

    Plain string ``replace()`` silently no-ops when the substring is
    not found, which would publish the source-repo tree diagram on HF.
    """

    readme = release_dir / "README.md"
    if not readme.exists():
        return []
    if HF_TREE_BLOCK_SOURCE not in readme.read_text(encoding="utf-8"):
        return [
            ValidationError(
                field="release/README.md",
                message=(
                    "HF_TREE_BLOCK_SOURCE not found verbatim in release/README.md; "
                    "the source-repo tree diagram in the README has drifted from "
                    "the constant in scripts/package_hf_release.py — the "
                    "HF README rewrite will silently no-op until the README and "
                    "HF_TREE_BLOCK_SOURCE are reconciled."
                ),
            )
        ]
    return []


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

    ``body`` is the markdown that follows the YAML frontmatter — by
    default the rewritten ``release/README.md`` (G12.1 expects the
    README to BE the dataset card; the YAML lives at the top).
    """

    pretty_name: str
    license: str
    language: tuple[str, ...]
    task_categories: tuple[str, ...]
    size_categories: tuple[str, ...]
    tags: tuple[str, ...]
    configs: tuple[ConfigEntry, ...]
    body: str


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
# Bundle reading + config building
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest.json at {path} is not a JSON object")
    return payload


def build_config_for_tier(
    release_dir: Path,
    *,
    tier_dir: str,
    config_name: str,
    task: str = DEFAULT_TASK,
    default: bool = False,
) -> ConfigEntry:
    """Build a single ``configs[]`` entry for one tier.

    ``tier_dir`` is the directory name under the upload tree
    (``intro`` / ``intermediate`` / ``advanced`` for the public
    variant; ``intermediate`` for the instructor companion which
    flattens ``intermediate_instructor/`` → ``intermediate/`` in the
    upload tree).  ``config_name`` is what users pass to
    ``load_dataset(..., name=...)``.

    The function reads ``manifest.json`` to confirm the tier exists
    and the task splits are declared; it does NOT fail when files are
    missing on disk because the upload tree is reassembled by
    :func:`assemble_upload_dir` and the manifest is the single source
    of truth for what should be there.
    """

    manifest_path = release_dir / tier_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing for tier {tier_dir!r}: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    if task not in manifest.get("tasks", {}):
        raise ValueError(
            f"task {task!r} not declared in manifest at {manifest_path}; "
            f"got {sorted(manifest.get('tasks', {}))!r}"
        )

    data_files = tuple(
        DataFileEntry(
            split=hf_split,
            path=f"{config_name}/tasks/{task}/{file_split}.parquet",
        )
        for hf_split, file_split in HF_SPLIT_NAMES
    )
    return ConfigEntry(config_name=config_name, data_files=data_files, default=default)


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
) -> HuggingFaceCard:
    """Assemble a ``HuggingFaceCard`` from the release tree.

    ``variant="public"`` builds the three-tier card pointing at
    ``release/{intro,intermediate,advanced}/``;
    ``variant="instructor"`` builds a single-config card pointing at
    ``release/intermediate_instructor/`` (flattened to
    ``intermediate/`` in the upload tree).

    When ``body`` is ``None`` (the default) we lift the contents of
    ``release/README.md`` and apply the HF-specific rewrites — HF
    renders the body as the dataset card so a full README there is
    more useful than a curated blurb (mirrors the Kaggle packager's
    description handling).
    """

    if variant == "public":
        public_tiers = (
            ("intro", "intro"),
            ("intermediate", "intermediate"),
            ("advanced", "advanced"),
        )
        configs = tuple(
            build_config_for_tier(
                release_dir,
                tier_dir=tier_dir,
                config_name=config_name,
                task=task,
                default=(config_name == default_config),
            )
            for tier_dir, config_name in public_tiers
        )
        if pretty_name is None:
            pretty_name = DEFAULT_PRETTY_NAME
        if body is None:
            body = _hf_readme_text(
                (release_dir / "README.md").read_text(encoding="utf-8"),
                tree_block=HF_UPLOAD_TREE_BLOCK,
            )
    elif variant == "instructor":
        # Companion repo flattens ``intermediate_instructor`` →
        # ``intermediate`` in the upload tree so the HF dataset slug
        # ``leadforge-lead-scoring-v1-instructor`` exposes the
        # familiar config name.
        configs = (
            build_config_for_tier(
                release_dir,
                tier_dir="intermediate_instructor",
                config_name="intermediate",
                task=task,
                default=True,
            ),
        )
        if pretty_name is None:
            pretty_name = DEFAULT_INSTRUCTOR_PRETTY_NAME
        if body is None:
            body = _hf_readme_text(
                (release_dir / "README.md").read_text(encoding="utf-8"),
                tree_block=HF_INSTRUCTOR_UPLOAD_TREE_BLOCK,
            )
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
    )


# ---------------------------------------------------------------------------
# Rendering
#
# YAML is hand-rolled rather than dumped by PyYAML because the HF
# dataset-card YAML has a precise key order and indentation style that
# the viewer is fussy about (and PyYAML's default flow style collapses
# the configs list into a single line).  Hand-rolling also makes the
# determinism contract obvious in the source.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _yaml_string(value: str) -> str:
    """Render a YAML string scalar.

    HF parses the frontmatter with PyYAML.  We use double-quoted
    strings only when the value contains characters that would change
    YAML semantics (``:``, leading ``- ``, leading whitespace).  Plain
    scalars are used otherwise so the YAML stays readable.
    """

    if value == "":
        return '""'
    needs_quoting = any(c in value for c in ":#&*!|>'\"%@`")
    needs_quoting = needs_quoting or value.startswith(("- ", " ", "?", "[", "{", "*", "&"))
    needs_quoting = needs_quoting or value.endswith(" ")
    if not needs_quoting:
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_list(values: Sequence[str], *, indent: int) -> list[str]:
    """Render a YAML block-style list."""

    pad = " " * indent
    return [f"{pad}- {_yaml_string(v)}" for v in values]


def render_yaml_frontmatter(card: HuggingFaceCard) -> str:
    """Render the YAML frontmatter as a deterministic string.

    Order: ``pretty_name``, ``license``, ``language``,
    ``task_categories``, ``size_categories``, ``tags``, ``configs``.
    Tags are sorted at render time (mirrors the keyword sort in the
    Kaggle packager) so order on the dataclass doesn't leak into the
    rendered file.
    """

    lines: list[str] = ["---"]
    lines.append(f"pretty_name: {_yaml_string(card.pretty_name)}")
    lines.append(f"license: {_yaml_string(card.license)}")
    lines.append("language:")
    lines.extend(_yaml_list(card.language, indent=2))
    lines.append("task_categories:")
    lines.extend(_yaml_list(card.task_categories, indent=2))
    lines.append("size_categories:")
    lines.extend(_yaml_list(card.size_categories, indent=2))
    lines.append("tags:")
    lines.extend(_yaml_list(sorted(card.tags), indent=2))
    lines.append("configs:")
    for config in card.configs:
        lines.append(f"  - config_name: {_yaml_string(config.config_name)}")
        if config.default:
            lines.append("    default: true")
        lines.append("    data_files:")
        for df in config.data_files:
            lines.append(f"      - split: {_yaml_string(df.split)}")
            lines.append(f"        path: {_yaml_string(df.path)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def render_card(card: HuggingFaceCard) -> str:
    """Render the full README (YAML frontmatter + markdown body)."""

    body = card.body
    if not body.endswith("\n"):
        body += "\n"
    return render_yaml_frontmatter(card) + "\n" + body


# ---------------------------------------------------------------------------
# Upload-directory assembly
# ---------------------------------------------------------------------------


def _replace_file(src: Path, dst: Path) -> None:
    """Copy ``src`` → ``dst``, replacing any existing entry at ``dst``."""

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _replace_dir(src: Path, dst: Path) -> None:
    """Copy directory ``src`` → ``dst``, replacing any existing entry."""

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def assemble_upload_dir(
    release_dir: Path,
    upload_dir: Path,
    *,
    variant: str = "public",
    cover_image: Path = DEFAULT_COVER_IMAGE,
) -> None:
    """Assemble ``upload_dir`` for ``huggingface_hub.upload_folder()``.

    Mirrors PR 5.1's Kaggle assembler: real-file copies of the per-
    tier bundles + cover image + LICENSE + the rewritten README.  No
    symlinks (the ``datasets`` library walks the upload tree and silent
    skips broken symlinks in some versions).

    For ``variant="instructor"``, the source directory
    ``intermediate_instructor/`` is flattened to ``intermediate/`` in
    the upload tree so ``load_dataset(..., name="intermediate")``
    works against the companion repo without naming friction.
    """

    kind = "huggingface" if variant == "public" else "huggingface-instructor"
    validate_upload_dir_safe(upload_dir, release_dir, kind=kind)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Cover image — reuse the one ``release/`` already ships.
    cover_src = release_dir / cover_image.name
    if not cover_src.exists():
        cover_src = cover_image
    if cover_src.exists():
        _replace_file(cover_src, upload_dir / COVER_IMAGE_FILENAME)

    # LICENSE — straight copy, no rewriting.
    license_src = release_dir / "LICENSE"
    if license_src.exists():
        _replace_file(license_src, upload_dir / "LICENSE")

    # Per-tier bundles — full directory copies.  The instructor variant
    # flattens its source dir name.
    if variant == "public":
        for tier in DEFAULT_PUBLIC_TIERS:
            _replace_dir(release_dir / tier, upload_dir / tier)
    elif variant == "instructor":
        _replace_dir(
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

    With ``dry_run=False`` (the default) the packager additionally
    assembles the HF-loadable upload directory under
    ``huggingface_dir`` (real-file copies of the per-tier bundles +
    cover image + LICENSE).  ``dry_run=True`` skips the assembly step
    entirely — useful for fast README iteration.
    """

    if not release_dir.exists():
        raise FileNotFoundError(f"release directory not found: {release_dir}")

    card = build_card(
        release_dir,
        variant=variant,
        pretty_name=pretty_name,
        default_config=default_config,
        task=task,
    )

    errors: list[ValidationError] = []
    errors.extend(validate_card(card))
    errors.extend(validate_cover_image(cover_image))
    errors.extend(_validate_readme_substitution(release_dir))

    readme_path = huggingface_dir / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(render_card(card), encoding="utf-8")

    if not dry_run:
        assemble_upload_dir(
            release_dir,
            huggingface_dir,
            variant=variant,
            cover_image=cover_image,
        )

    return PackagerOutcome(
        card=card,
        readme_path=readme_path,
        errors=tuple(errors),
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
            "default: %(default)s for public; ignored for instructor)"
        ),
    )
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help=(
            "HF dataset owner — currently informational; PR 7.2 will "
            "consume it via ``huggingface_hub.HfApi().upload_folder``. "
            "default: %(default)s"
        ),
    )
    parser.add_argument(
        "--dataset-slug",
        default=None,
        help=(
            "HF dataset slug — currently informational; PR 7.2 will "
            "consume it. defaults: leadforge-lead-scoring-v1 (public) / "
            "leadforge-lead-scoring-v1-instructor (instructor)"
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

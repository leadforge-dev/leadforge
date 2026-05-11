"""Shared primitives for the Phase 5 release packagers.

Both ``scripts/package_kaggle_release.py`` (PR 5.1) and
``scripts/package_hf_release.py`` (PR 5.2) need to:

* rewrite the ``](../foo)`` markdown links in ``release/README.md`` to
  GitHub blob URLs — the README lives one level above each upload tree,
  so the relative links break on Kaggle / HF and have to point at the
  source repo;
* rewrite the ``](validation/validation_report.md)`` link to a GitHub
  blob URL — the validation report does not ship with either upload
  tree;
* refuse to assemble into ``cwd`` / the release dir / its parent / a
  descendant of the release dir / the filesystem anchor (the assembler
  rmtrees children of the upload dir);
* validate the dataset cover image's dimensions against Kaggle's floor
  (HF reuses the same PNG for its thumbnail);
* perform idempotent file/dir replacement during upload-tree assembly;
* read tier ``manifest.json`` payloads;
* guard the source-side tree diagram in ``release/README.md`` against
  drifting away from the platform packagers' ``replace()`` substring
  (silent-failure trap).

``FieldDescriptor`` / ``ResourceSchema`` / dtype maps are deliberately
NOT extracted: HF infers schema from parquet via ``load_dataset`` and
does not need a Frictionless-shaped declaration.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from PIL import Image

# ---------------------------------------------------------------------------
# README link rewriting
# ---------------------------------------------------------------------------

GITHUB_BLOB_BASE: Final[str] = "https://github.com/leadforge-dev/leadforge/blob/main"

#: Inline relative link ``](../foo)`` → ``](GITHUB_BLOB_BASE/foo)``
#: for any markdown link that escapes the bundle root.
PARENT_RELATIVE_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\]\(\.\./([^)]+)\)")

#: The README points at ``validation/validation_report.md`` (a path
#: that lives under ``release/`` but never under either upload tree).
#: Rewrite to a GitHub blob URL so the link works on both platforms.
VALIDATION_REPORT_LINK: Final[str] = "](validation/validation_report.md)"
VALIDATION_REPORT_URL: Final[str] = f"]({GITHUB_BLOB_BASE}/release/validation/validation_report.md)"


def rewrite_release_links(text: str) -> str:
    """Apply the platform-agnostic README rewrites.

    Both Kaggle and HF use these. Tree-block substitution is
    platform-specific (each upload tree looks different) and lives in
    each packager's own module.
    """

    text = PARENT_RELATIVE_LINK_RE.sub(rf"]({GITHUB_BLOB_BASE}/\1)", text)
    text = text.replace(VALIDATION_REPORT_LINK, VALIDATION_REPORT_URL)
    return text


# ---------------------------------------------------------------------------
# Validation error type — declared before any validator that returns it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """One field-level validation failure."""

    field: str
    message: str


# ---------------------------------------------------------------------------
# Source-side tree-diagram guard
#
# ``release/README.md`` (PR 4.1) contains a "What's inside" tree
# diagram describing the SOURCE-REPO layout. Each packager substitutes
# this block for its own upload-tree diagram via ``str.replace``.
# Keeping the source-side block as a single shared constant means the
# README's tree diagram drifting away from the constant is caught
# loudly by the same guard regardless of which packager runs first.
# ---------------------------------------------------------------------------

SOURCE_TREE_BLOCK: Final[str] = """```
release/
├── intro/ intermediate/ advanced/    # student_public bundles, one per difficulty tier
│   ├── manifest.json                 # provenance + file hashes
│   ├── metrics.json                  # per-tier headline metrics (medians + spreads)
│   ├── dataset_card.md               # auto-rendered per-bundle card
│   ├── feature_dictionary.csv        # authoritative column spec
│   ├── lead_scoring.csv              # flat convenience CSV (all splits)
│   ├── tables/*.parquet              # 7 snapshot-safe relational tables
│   └── tasks/converted_within_90_days/{train,valid,test}.parquet
├── intermediate_instructor/          # research companion: full-horizon tables + metadata/
├── docs/                             # vendored DGP / leakage / break-me docs (agent-readable)
├── notebooks/                        # 01 baseline · 02 relational · 03 leakage · 04 calibration
├── metrics.json                      # top-level cross-tier metrics summary
├── claims_register.{md,json}         # claims → backing-artifact map (agent-readable)
└── validation/                       # validation_report.{json,md} + figures
```"""


def validate_readme_substitution(release_dir: Path, *, packager_name: str) -> list[ValidationError]:
    """Guard against silent drift between the README's tree diagram
    and ``SOURCE_TREE_BLOCK``.

    Plain string ``replace()`` silently no-ops when the substring is
    not found, which would publish the source-repo tree diagram on the
    target platform. ``packager_name`` is folded into the error message
    so the trace says which packager noticed the drift first.
    """

    readme = release_dir / "README.md"
    if not readme.exists():
        return []  # No README is itself a release-day issue, but not this validator's concern.
    if SOURCE_TREE_BLOCK not in readme.read_text(encoding="utf-8"):
        return [
            ValidationError(
                field="release/README.md",
                message=(
                    f"SOURCE_TREE_BLOCK not found verbatim in release/README.md; "
                    f"the source-repo tree diagram in the README has drifted from "
                    f"the constant in scripts/_release_common.py — the "
                    f"{packager_name} README rewrite will silently no-op until "
                    f"the README and SOURCE_TREE_BLOCK are reconciled."
                ),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Cover-image validation (G11.2; HF reuses the same asset)
# ---------------------------------------------------------------------------

#: Cover-image minimum dimensions per G11.2: 560 × 280 minimum, with
#: 2:1 header / 1:1 thumbnail crops in mind. HF accepts any reasonable
#: dimension for thumbnails so the Kaggle floor is the binding one.
COVER_IMAGE_MIN_WIDTH: Final[int] = 560
COVER_IMAGE_MIN_HEIGHT: Final[int] = 280


def resolve_cover_image_path(cover_image: Path, release_dir: Path) -> Path:
    """Resolve the effective cover-image path used by the packagers.

    The packagers run a validator against the cover image and a copy
    step against the same image; the earlier draft did the path
    resolution in the assembler only, which meant validate + assemble
    could disagree about WHICH file to use (Copilot review on PR #72,
    addressed in self-review pass #2).

    Resolution rule — explicit-wins, with a release-dir fallback:

    1. If ``cover_image`` exists at the path given, return it as-is.
    2. Otherwise, if ``release_dir / cover_image.name`` exists, return
       that — lets users pass a bare basename when the file lives
       under ``release/``.
    3. Otherwise return ``cover_image`` unchanged so the cover-image
       validator surfaces a clean ``not found at <path>`` error.

    Two paths sharing a basename DO NOT cause the release-dir copy to
    silently shadow the explicit path — the explicit path's existence
    is the first check and wins.
    """

    if cover_image.exists():
        return cover_image
    fallback = release_dir / cover_image.name
    if fallback.exists():
        return fallback
    return cover_image


def validate_cover_image(path: Path) -> list[ValidationError]:
    """Validate that ``path`` exists and meets the dimension floor."""

    errors: list[ValidationError] = []
    if not path.exists():
        errors.append(
            ValidationError(
                field="cover_image",
                message=f"cover image not found at {path}",
            )
        )
        return errors
    with Image.open(path) as img:
        width, height = img.size
    if width < COVER_IMAGE_MIN_WIDTH or height < COVER_IMAGE_MIN_HEIGHT:
        errors.append(
            ValidationError(
                field="cover_image",
                message=(
                    f"cover image {width}x{height} below minimum "
                    f"{COVER_IMAGE_MIN_WIDTH}x{COVER_IMAGE_MIN_HEIGHT}"
                ),
            )
        )
    return errors


# ---------------------------------------------------------------------------
# Upload-dir safety
# ---------------------------------------------------------------------------


def validate_upload_dir_safe(upload_dir: Path, release_dir: Path, *, kind: str) -> None:
    """Refuse to assemble into a path that aliases something dangerous.

    The packagers replace children of ``upload_dir`` (rmtree + recopy)
    so pointing it at any of the following would clobber unrelated
    content:

    * ``cwd`` or its anchor (``/`` on POSIX);
    * ``release_dir`` itself;
    * a parent of ``release_dir``;
    * a descendant of ``release_dir`` other than the canonical platform
      output dirs (e.g. ``release/intro`` would rmtree the intro
      bundle).

    The safe location for an upload tree is a sibling of ``release/``
    or a fresh subdirectory of ``release/`` that the packager owns
    (e.g. ``release/kaggle/``, ``release/huggingface/``).  We allow
    those by checking the resolved path against the explicit blocklist
    and against descendant containment in ``release_dir``.

    ``kind`` is folded into the error message so the trace says which
    packager refused (``kaggle`` / ``huggingface`` /
    ``huggingface-instructor``).
    """

    resolved = upload_dir.resolve()
    release_resolved = release_dir.resolve()
    blocked = {
        Path(resolved.anchor),
        Path.cwd().resolve(),
        release_resolved,
        release_resolved.parent,
    }
    if resolved in blocked:
        raise ValueError(f"refusing to assemble into unsafe --{kind}-dir: {upload_dir}")
    # Disallow descendants of release_dir other than direct children
    # owned by a packager. ``resolved.parent == release_resolved`` is
    # the canonical ok-case (release/kaggle, release/huggingface,
    # release/huggingface-instructor); deeper nesting would alias a
    # tier bundle dir.
    if resolved.is_relative_to(release_resolved) and resolved.parent != release_resolved:
        raise ValueError(
            f"refusing to assemble into unsafe --{kind}-dir: {upload_dir} "
            f"(would alias contents inside {release_dir})"
        )


# ---------------------------------------------------------------------------
# Filesystem helpers — idempotent file / dir replacement
# ---------------------------------------------------------------------------


def replace_file(src: Path, dst: Path) -> None:
    """Copy ``src`` → ``dst``, replacing any existing entry at ``dst``."""

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def replace_dir(src: Path, dst: Path) -> None:
    """Copy directory ``src`` → ``dst``, replacing any existing entry."""

    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists() and dst.is_dir():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    """Read a tier's ``manifest.json`` and return the parsed dict.

    Raises ``ValueError`` when the JSON parses to anything other than a
    top-level object (the packagers index into it expecting a dict).
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest.json at {path} is not a JSON object")
    return payload


# ---------------------------------------------------------------------------
# Per-table column descriptions (vendored under release/docs/)
# ---------------------------------------------------------------------------

#: Path within the release tree of the per-table column descriptions
#: hand-authored CSV.  Keyed by ``(table, column)``; consumed by the
#: Kaggle packager so ``resources[].schema.fields[].description`` is
#: populated for parquet tables (the preview's ``col__desc`` column
#: was previously empty for relational tables — a thin spot for AI
#: reviewers who can't open the parquet directly).
RELATIONAL_TABLE_SCHEMAS_REL: Final[Path] = Path("docs/relational_table_schemas.csv")


def load_relational_column_descriptions(release_dir: Path) -> dict[tuple[str, str], str]:
    """Load per-table column descriptions keyed by ``(table, column)``.

    Returns an empty dict if the CSV is missing — callers should treat
    the description as optional (matches the pre-PR behaviour where
    parquet schemas shipped without column docs).
    """

    import csv

    path = release_dir / RELATIONAL_TABLE_SCHEMAS_REL
    if not path.is_file():
        return {}
    descriptions: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table = row.get("table", "").strip()
            column = row.get("column", "").strip()
            description = (row.get("description") or "").strip()
            if table and column and description:
                descriptions[(table, column)] = description
    return descriptions


# ---------------------------------------------------------------------------
# Agent-reviewable artifact set
# ---------------------------------------------------------------------------

#: Files at the release root that should ship in every platform's upload
#: tree to make the bundle self-contained for agent / human review
#: without needing GitHub access.  Path tuples are ``(source_rel,
#: optional_required)``: ``required=True`` causes the packager to
#: surface a ValidationError if the file is missing at packaging time
#: (these are committed artifacts; their absence indicates the release
#: was incomplete).
AGENT_REVIEWABLE_ROOT_FILES: Final[tuple[tuple[str, bool], ...]] = (
    ("metrics.json", True),
    ("claims_register.md", True),
    ("claims_register.json", True),
    ("claims_register_source.yaml", False),
)

#: Sub-directory under the release root containing vendored docs
#: (DGP description, leakage / acceptance bands, break-me guide, etc.).
#: Copied wholesale into the upload tree when present.
AGENT_REVIEWABLE_DOCS_DIR: Final[str] = "docs"

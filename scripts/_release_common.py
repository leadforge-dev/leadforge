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
* refuse to assemble into ``cwd`` / the release dir / its parent / the
  filesystem anchor (the assembler rmtrees children of the upload dir);
* validate the dataset cover image's dimensions against Kaggle's floor
  (HF reuses the same PNG for its thumbnail).

Lifting the four to one module keeps the rules in one place — if Kaggle
ever extends the dimension floor, both packagers pick it up.
``FieldDescriptor`` / ``ResourceSchema`` / dtype maps are deliberately
NOT extracted: HF infers schema from parquet via ``load_dataset`` and
does not need a Frictionless-shaped declaration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
# Cover-image validation (G11.2; HF reuses the same asset)
# ---------------------------------------------------------------------------

#: Cover-image minimum dimensions per G11.2: 560 × 280 minimum, with
#: 2:1 header / 1:1 thumbnail crops in mind. HF accepts any reasonable
#: dimension for thumbnails so the Kaggle floor is the binding one.
COVER_IMAGE_MIN_WIDTH: Final[int] = 560
COVER_IMAGE_MIN_HEIGHT: Final[int] = 280


@dataclass(frozen=True)
class ValidationError:
    """One field-level validation failure."""

    field: str
    message: str


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
    so pointing it at ``cwd`` / ``release_dir`` / their parents / the
    filesystem anchor would clobber unrelated content. ``kind`` is
    folded into the error message so the trace says which packager
    refused (``kaggle`` / ``huggingface`` / ``huggingface-instructor``).
    """

    resolved = upload_dir.resolve()
    blocked = {
        Path(resolved.anchor),
        Path.cwd().resolve(),
        release_dir.resolve(),
        release_dir.resolve().parent,
    }
    if resolved in blocked:
        raise ValueError(f"refusing to assemble into unsafe --{kind}-dir: {upload_dir}")

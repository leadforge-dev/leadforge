"""Tests for ``scripts/_release_common.py`` shared release primitives.

The link-rewriter, source-tree-block guard, cover-image validator,
and upload-dir safety check have packager-side coverage in
``test_package_kaggle_release.py`` and ``test_package_hf_release.py``.
This file covers shared helpers that don't have a natural packager-
side home — currently the cover-image path resolver introduced in
response to the Copilot review on PR #72.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "_release_common.py"
_spec = importlib.util.spec_from_file_location("_release_common", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
common = importlib.util.module_from_spec(_spec)
sys.modules["_release_common"] = common
_spec.loader.exec_module(common)


def _make_cover(path: Path) -> None:
    """Write a minimum-Kaggle-acceptable cover image at ``path``."""

    Image.new("RGB", (1280, 640), (0, 0, 0)).save(path)


# ---------------------------------------------------------------------------
# resolve_cover_image_path — Copilot review on PR #72, item #2
# ---------------------------------------------------------------------------


def test_resolve_cover_image_path_returns_explicit_when_it_exists(tmp_path: Path) -> None:
    """An explicit path that exists is returned unchanged."""

    release = tmp_path / "release"
    release.mkdir()
    explicit = tmp_path / "explicit.png"
    _make_cover(explicit)

    resolved = common.resolve_cover_image_path(explicit, release)
    assert resolved == explicit


def test_resolve_cover_image_path_falls_back_to_release_dir(tmp_path: Path) -> None:
    """A bare basename that doesn't exist resolves to ``release_dir/<name>``."""

    release = tmp_path / "release"
    release.mkdir()
    cover_in_release = release / "dataset-cover-image.png"
    _make_cover(cover_in_release)

    resolved = common.resolve_cover_image_path(Path("dataset-cover-image.png"), release)
    assert resolved == cover_in_release


def test_resolve_cover_image_path_prefers_explicit_over_release_sibling(
    tmp_path: Path,
) -> None:
    """When BOTH the explicit path and a release-dir sibling exist, the
    explicit path wins.  Locks down the contract that two paths sharing
    a basename do not silently shadow each other (Copilot review #2).
    """

    release = tmp_path / "release"
    release.mkdir()
    explicit = tmp_path / "explicit.png"
    _make_cover(explicit)
    decoy = release / "explicit.png"
    _make_cover(decoy)

    resolved = common.resolve_cover_image_path(explicit, release)
    assert resolved == explicit
    assert resolved != decoy


def test_resolve_cover_image_path_returns_input_when_neither_exists(
    tmp_path: Path,
) -> None:
    """When neither location resolves, return the input unchanged so
    the cover-image validator can surface a clean ``not found at <path>``
    error (rather than this helper raising)."""

    release = tmp_path / "release"
    release.mkdir()
    missing = tmp_path / "nope.png"

    resolved = common.resolve_cover_image_path(missing, release)
    assert resolved == missing
    # Sanity: the validator's not-found path actually fires for this
    # input.
    errors = common.validate_cover_image(resolved)
    assert errors
    assert "not found" in errors[0].message

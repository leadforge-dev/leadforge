"""Tests for ``scripts/generate_cover_image.py``.

Locks the two acceptance properties for the Kaggle cover image:

1. it satisfies G11.2 — at least 560 × 280 pixels in the right modes;
2. the output is byte-deterministic across runs and matches the
   committed PNG (audit-artifact-sync pattern from PR 4.1).

If the simulator's headline metrics drift the cover image's pinned
literals out of date, both the determinism check here and the metrics
in ``release/validation/validation_report.md`` will need a coordinated
update.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_cover_image.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("generate_cover_image", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
generator = importlib.util.module_from_spec(_spec)
sys.modules["generate_cover_image"] = generator
_spec.loader.exec_module(generator)


_COMMITTED_COVER = _REPO_ROOT / "release" / "dataset-cover-image.png"
_COMMITTED_PRESENT = _COMMITTED_COVER.exists()


# ---------------------------------------------------------------------------
# Dimension floor + mode (G11.2)
# ---------------------------------------------------------------------------


def test_render_cover_dimensions_above_kaggle_minimum() -> None:
    """G11.2: cover image must be at least 560 × 280; we ship 1280 × 640."""

    image = generator.render_cover()
    assert image.size == (generator.CANVAS_WIDTH, generator.CANVAS_HEIGHT)
    assert image.size[0] >= 560
    assert image.size[1] >= 280
    # Ratio check — we deliberately render at 2:1 so the Kaggle header
    # crop matches the source aspect ratio.
    assert image.size[0] == 2 * image.size[1]
    assert image.mode == "RGB"


def test_write_cover_writes_png_at_target_size(tmp_path: Path) -> None:
    """``write_cover`` round-trips through Pillow at the declared dimensions."""

    out = tmp_path / "cover.png"
    generator.write_cover(out)

    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size == (generator.CANVAS_WIDTH, generator.CANVAS_HEIGHT)


# ---------------------------------------------------------------------------
# Determinism + sync with committed asset
# ---------------------------------------------------------------------------


def test_render_cover_is_byte_deterministic(tmp_path: Path) -> None:
    """Two back-to-back ``write_cover`` calls produce byte-identical PNGs.

    Pillow's PNG writer is deterministic given the same encoder
    settings; pinning those in :func:`write_cover` is what makes the
    audit-artifact-sync pattern viable for this asset.
    """

    a = tmp_path / "cover_a.png"
    b = tmp_path / "cover_b.png"
    generator.write_cover(a)
    generator.write_cover(b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.skipif(not _COMMITTED_PRESENT, reason="committed cover image not present")
def test_committed_cover_matches_fresh_regeneration(tmp_path: Path) -> None:
    """A fresh render must match the committed
    ``release/dataset-cover-image.png`` byte-for-byte.

    If this fails, the cover image drifted without a re-run of
    ``scripts/generate_cover_image.py``.  Regenerate via that script
    from the repo root and commit the new PNG alongside any code
    change that altered the rendered output.
    """

    fresh = tmp_path / "cover.png"
    generator.write_cover(fresh)
    assert fresh.read_bytes() == _COMMITTED_COVER.read_bytes()

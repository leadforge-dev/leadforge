#!/usr/bin/env python3
"""Generate the deterministic Kaggle cover image for ``leadforge-lead-scoring-v1``.

The cover image is rendered programmatically rather than hand-designed
or licensed so that:

* the asset is reproducible — re-running this script produces a
  byte-identical PNG, guarded by a determinism test in
  ``tests/scripts/test_generate_cover_image.py`` (matches the
  audit-artifact-sync pattern from PR 4.1);
* the source-of-truth for what the image *says* sits in version
  control, not in a designer's file or a stock-photo licence;
* there is no licensing question.

Output: ``release/dataset-cover-image.png`` at 1280 × 640 px (2:1
aspect, well above Kaggle's 560 × 280 minimum, with a 1:1 thumbnail
crop centred on the headline). Pillow ships with matplotlib (already a
dev / scripts extra), so this script does not require any new
dependency.

Headline metrics — conversion rates and LR AUC values — are pinned
literals sourced from the cross-seed medians (seeds 42-46) reported in
``release/validation/validation_report.md``. They are not recomputed
at render time: the cover image is intentionally a documentation-grade
artefact that lags by one validation cycle, not a live metric panel.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Layout constants (pixels)
# ---------------------------------------------------------------------------

CANVAS_WIDTH: Final[int] = 1280
CANVAS_HEIGHT: Final[int] = 640
LEFT_MARGIN: Final[int] = 80

#: Background — deep navy.
BACKGROUND: Final[tuple[int, int, int]] = (13, 27, 42)
#: Card background — slightly lighter navy.
CARD_BACKGROUND: Final[tuple[int, int, int]] = (27, 38, 59)
#: Primary text colour — pure white.
TEXT_PRIMARY: Final[tuple[int, int, int]] = (255, 255, 255)
#: Secondary text colour — pale steel.
TEXT_SECONDARY: Final[tuple[int, int, int]] = (200, 220, 240)

DEFAULT_OUT_PATH: Final[Path] = Path("release/dataset-cover-image.png")


@dataclass(frozen=True)
class TierBadge:
    """Per-tier headline shown on the cover."""

    name: str
    conversion_rate_pct: str
    lr_auc: float
    accent: tuple[int, int, int]


#: Cross-seed medians (seeds 42-46) from
#: ``release/validation/validation_report.md`` — pinned literals so the
#: cover image is reproducible without reading the report at render
#: time.
TIER_BADGES: Final[tuple[TierBadge, ...]] = (
    TierBadge("Intro", "42.7%", 0.879, (76, 175, 80)),
    TierBadge("Intermediate", "21.6%", 0.886, (255, 152, 0)),
    TierBadge("Advanced", "8.4%", 0.886, (244, 67, 54)),
)


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------


def _find_font(family: str, *, weight: str = "normal") -> Path:
    """Locate a font file via matplotlib's font manager.

    matplotlib bundles DejaVu Sans, so this resolves to a stable file
    path in any environment where matplotlib is installed (the
    ``[scripts]`` and ``[dev]`` extras both pull it in).  The same
    byte content of the font file → identical glyph rasters →
    byte-identical PNG output.
    """

    prop = fm.FontProperties(family=family, weight=weight)
    return Path(fm.findfont(prop, fallback_to_default=False))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draw_title_block(draw: ImageDraw.ImageDraw, font_paths: dict[str, Path]) -> None:
    """Render the title, tagline, and subtitle text block."""

    title_font = ImageFont.truetype(str(font_paths["bold"]), 96)
    draw.text((LEFT_MARGIN, 88), "LeadForge", font=title_font, fill=TEXT_PRIMARY)

    tagline_font = ImageFont.truetype(str(font_paths["regular"]), 40)
    draw.text(
        (LEFT_MARGIN, 208),
        "Synthetic B2B Lead Scoring · v1",
        font=tagline_font,
        fill=TEXT_SECONDARY,
    )

    subtitle_font = ImageFont.truetype(str(font_paths["regular"]), 24)
    draw.text(
        (LEFT_MARGIN, 280),
        "5,000 leads · 3 difficulty tiers · 90-day conversion · MIT",
        font=subtitle_font,
        fill=TEXT_SECONDARY,
    )


def _draw_tier_card(
    draw: ImageDraw.ImageDraw,
    *,
    badge: TierBadge,
    box: tuple[int, int, int, int],
    font_paths: dict[str, Path],
) -> None:
    """Render one tier card inside ``box`` (left, top, right, bottom)."""

    left, top, right, bottom = box
    draw.rectangle((left, top, right, bottom), fill=CARD_BACKGROUND)
    # Coloured accent stripe down the left edge.
    draw.rectangle((left, top, left + 8, bottom), fill=badge.accent)

    name_font = ImageFont.truetype(str(font_paths["bold"]), 36)
    draw.text((left + 32, top + 24), badge.name, font=name_font, fill=TEXT_PRIMARY)

    body_font = ImageFont.truetype(str(font_paths["regular"]), 22)
    draw.text(
        (left + 32, top + 80),
        f"Conversion: {badge.conversion_rate_pct}",
        font=body_font,
        fill=TEXT_SECONDARY,
    )
    draw.text(
        (left + 32, top + 116),
        f"LR AUC: {badge.lr_auc:.3f}",
        font=body_font,
        fill=TEXT_SECONDARY,
    )


def render_cover(badges: Sequence[TierBadge] = TIER_BADGES) -> Image.Image:
    """Render the cover image as a fresh ``PIL.Image`` instance."""

    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    font_paths = {
        "regular": _find_font("DejaVu Sans", weight="normal"),
        "bold": _find_font("DejaVu Sans", weight="bold"),
    }

    _draw_title_block(draw, font_paths)

    # Three equal-width cards spanning the bottom half of the canvas.
    card_top = 400
    card_bottom = 580
    card_count = len(badges)
    gap = 40
    available = CANVAS_WIDTH - 2 * LEFT_MARGIN
    card_width = (available - gap * (card_count - 1)) // card_count
    for i, badge in enumerate(badges):
        left = LEFT_MARGIN + i * (card_width + gap)
        right = left + card_width
        _draw_tier_card(
            draw,
            badge=badge,
            box=(left, card_top, right, card_bottom),
            font_paths=font_paths,
        )

    return image


def write_cover(path: Path, image: Image.Image | None = None) -> Path:
    """Render and write the cover image to ``path`` deterministically.

    Pillow's PNG writer is byte-deterministic given the same input
    image and the same encoder settings — pinning ``optimize=False``
    and a fixed ``compress_level`` removes the only sources of
    run-to-run variance.
    """

    if image is None:
        image = render_cover()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=6)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic Kaggle cover image for leadforge-lead-scoring-v1.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="output PNG path (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path: Path = args.out
    write_cover(out_path)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

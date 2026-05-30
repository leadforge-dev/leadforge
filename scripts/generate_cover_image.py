#!/usr/bin/env python3
"""Generate the Kaggle cover image for ``leadforge-lead-scoring-v1``.

The cover image is rendered programmatically rather than hand-designed
or licensed so that:

* re-running this script on the same machine produces byte-identical
  output, guarded by ``test_render_cover_is_byte_deterministic`` —
  enough for local regression detection;
* the source-of-truth for what the image *says* sits in version
  control, not in a designer's file or a stock-photo licence;
* there is no licensing question.

**Cross-platform byte equality is NOT guaranteed.** The committed
``release/dataset-cover-image.png`` was rendered on whichever machine
last ran this script; Pillow + FreeType produce slightly different
glyph rasterisation between macOS and Linux (different FreeType
versions, different font-hinting tables).  The committed PNG is
therefore one valid render — checked into git so a fresh clone has a
usable cover image without first running this script — not a
hash-locked artefact.  Tests assert dimensions and per-machine
determinism, not committed-vs-fresh byte equality.

Output: ``release/dataset-cover-image.png`` at 1280 × 640 px (2:1
aspect, well above Kaggle's 560 × 280 minimum, with a 1:1 thumbnail
crop centred on the headline). Pillow ships with matplotlib (already a
dev / scripts extra), so this script does not require any new
dependency.

Headline metrics — conversion rates and LR AUC values — are pinned
literals sourced from the cross-seed medians (seeds 42–46) reported in
``release/validation/validation_report.md``. They are not recomputed
at render time: the cover image is intentionally a documentation-grade
artefact that lags by one validation cycle, not a live metric panel.
"""

from __future__ import annotations

import argparse
import random
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
LEFT_MARGIN: Final[int] = 62

#: Background — deep navy.
BACKGROUND: Final[tuple[int, int, int]] = (13, 17, 23)
#: Card background — slightly lighter navy.
CARD_BACKGROUND: Final[tuple[int, int, int]] = (22, 27, 34)
#: Primary text colour — pure white.
TEXT_PRIMARY: Final[tuple[int, int, int]] = (255, 255, 255)
#: Secondary text colour — pale steel.
TEXT_SECONDARY: Final[tuple[int, int, int]] = (148, 163, 184)
#: Dimmer text for feature line.
TEXT_DIM: Final[tuple[int, int, int]] = (74, 85, 104)
#: Teal accent.
ACCENT_TEAL: Final[tuple[int, int, int]] = (0, 212, 170)

DEFAULT_OUT_PATH: Final[Path] = Path("release/dataset-cover-image.png")

#: x-centre of the funnel visualization (right panel).
FUNNEL_CX: Final[int] = 976
#: y-coordinate of the top edge of the funnel (top-down).
FUNNEL_TOP: Final[int] = 90


@dataclass(frozen=True)
class TierBadge:
    """Per-tier headline shown on the cover."""

    name: str
    conversion_rate_pct: str
    lr_auc: float
    accent: tuple[int, int, int]
    funnel_top_width: int
    funnel_bot_width: int


#: Cross-seed medians (seeds 42-46) from
#: ``release/validation/validation_report.md`` — pinned literals so the
#: cover image is reproducible without reading the report at render
#: time.
TIER_BADGES: Final[tuple[TierBadge, ...]] = (
    TierBadge("Intro", "42.7%", 0.879, (0, 212, 170), 320, 238),
    TierBadge("Intermediate", "21.6%", 0.886, (245, 158, 11), 238, 156),
    TierBadge("Advanced", "8.4%", 0.886, (239, 68, 68), 156, 60),
)

FUNNEL_SECTION_HEIGHT: Final[int] = 140


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
# Drawing helpers
# ---------------------------------------------------------------------------


def _draw_bokeh_background(img: Image.Image, *, seed: int = 42) -> Image.Image:
    """Overlay faint translucent circles giving a soft depth effect."""

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    rng = random.Random(seed)  # noqa: S311
    palette = [(0, 212, 170), (74, 158, 234), (245, 158, 11)]
    for _ in range(160):
        cx = rng.randint(0, CANVAS_WIDTH)
        cy = rng.randint(0, CANVAS_HEIGHT)
        r = rng.randint(20, 90)
        alpha = rng.randint(3, 12)
        col = rng.choice(palette)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*col, alpha))
    # Corner glow: top-left teal, bottom-right amber
    for gx, gy, gc, ga in [
        (110, 100, (0, 212, 170), 18),
        (1160, 540, (245, 158, 11), 14),
    ]:
        gr = 280
        d.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=(*gc, ga))
    base = img.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def _draw_vertical_accent(draw: ImageDraw.ImageDraw) -> None:
    """Teal accent bar left of the title block."""

    draw.rectangle((LEFT_MARGIN, 140, LEFT_MARGIN + 5, 490), fill=ACCENT_TEAL)


def _draw_title_block(draw: ImageDraw.ImageDraw, font_paths: dict[str, Path]) -> None:
    """Render the title, tagline, and subtitle text block."""

    tx = LEFT_MARGIN + 18

    title_font = ImageFont.truetype(str(font_paths["bold"]), 96)
    draw.text((tx, 160), "LeadForge", font=title_font, fill=TEXT_PRIMARY)

    tagline_font = ImageFont.truetype(str(font_paths["regular"]), 38)
    draw.text(
        (tx, 282), "Synthetic B2B Lead Scoring  ·  v1", font=tagline_font, fill=TEXT_SECONDARY
    )

    subtitle_font = ImageFont.truetype(str(font_paths["regular"]), 22)
    draw.text(
        (tx, 340),
        "5,000 leads  ·  3 difficulty tiers  ·  90-day conversion  ·  MIT",
        font=subtitle_font,
        fill=TEXT_DIM,
    )

    # Separator line
    draw.line((LEFT_MARGIN, 398, 640, 398), fill=(30, 45, 61), width=1)


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
    # Card border
    draw.rectangle((left, top, right, bottom), outline=badge.accent, width=1)
    # Coloured accent stripe down the left edge.
    draw.rectangle((left, top, left + 5, bottom), fill=badge.accent)

    name_font = ImageFont.truetype(str(font_paths["bold"]), 22)
    draw.text((left + 16, top + 14), badge.name, font=name_font, fill=TEXT_PRIMARY)

    body_font = ImageFont.truetype(str(font_paths["regular"]), 17)
    draw.text(
        (left + 16, top + 50),
        f"Conv: {badge.conversion_rate_pct}",
        font=body_font,
        fill=TEXT_SECONDARY,
    )
    draw.text(
        (left + 16, top + 76), f"LR AUC: {badge.lr_auc:.3f}", font=body_font, fill=TEXT_SECONDARY
    )


def _draw_funnel(draw: ImageDraw.ImageDraw, font_paths: dict[str, Path]) -> None:
    """Render the three-tier conversion funnel (right panel)."""

    label_font = ImageFont.truetype(str(font_paths["regular"]), 18)
    tier_font = ImageFont.truetype(str(font_paths["bold"]), 22)
    pct_font = ImageFont.truetype(str(font_paths["bold"]), 18)

    # "Conversion by Tier" heading above funnel
    heading = "Conversion by Tier"
    bbox = draw.textbbox((0, 0), heading, font=label_font)
    hw = bbox[2] - bbox[0]
    draw.text((FUNNEL_CX - hw // 2, FUNNEL_TOP - 28), heading, font=label_font, fill=TEXT_DIM)

    y = FUNNEL_TOP
    for badge in TIER_BADGES:
        tw = badge.funnel_top_width
        bw = badge.funnel_bot_width
        h = FUNNEL_SECTION_HEIGHT
        col = badge.accent
        bot_y = y + h

        # Trapezoid
        pts = [
            (FUNNEL_CX - tw // 2, y),
            (FUNNEL_CX + tw // 2, y),
            (FUNNEL_CX + bw // 2, bot_y),
            (FUNNEL_CX - bw // 2, bot_y),
        ]
        draw.polygon(pts, fill=(*col, 210))

        # Subtle highlight stripe at the top edge of each section
        hl_pts = [
            (FUNNEL_CX - tw // 2, y),
            (FUNNEL_CX + tw // 2, y),
            (FUNNEL_CX + tw // 2, y + 14),
            (FUNNEL_CX - tw // 2, y + 14),
        ]
        draw.polygon(hl_pts, fill=(255, 255, 255, 18))

        # Tier name centred inside section
        mid_y = y + h // 2
        name_bbox = draw.textbbox((0, 0), badge.name, font=tier_font)
        nw = name_bbox[2] - name_bbox[0]
        nh = name_bbox[3] - name_bbox[1]
        draw.text(
            (FUNNEL_CX - nw // 2, mid_y - nh // 2), badge.name, font=tier_font, fill=TEXT_PRIMARY
        )

        # Conversion % to the right of the widest top edge
        pct_x = FUNNEL_CX + tw // 2 + 14
        pct_bbox = draw.textbbox((0, 0), badge.conversion_rate_pct, font=pct_font)
        ph = pct_bbox[3] - pct_bbox[1]
        draw.text((pct_x, mid_y - ph // 2), badge.conversion_rate_pct, font=pct_font, fill=col)

        y = bot_y

    # "LR AUC ≈ 0.88" caption below funnel
    caption = "LR AUC ≈ 0.88 across all tiers"
    cbbox = draw.textbbox((0, 0), caption, font=label_font)
    cw = cbbox[2] - cbbox[0]
    draw.text((FUNNEL_CX - cw // 2, y + 16), caption, font=label_font, fill=TEXT_DIM)


def _draw_bottom_strip(draw: ImageDraw.ImageDraw) -> None:
    """Thin teal strip at the very bottom edge."""

    draw.rectangle((0, CANVAS_HEIGHT - 5, CANVAS_WIDTH, CANVAS_HEIGHT), fill=(*ACCENT_TEAL, 128))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_cover(badges: Sequence[TierBadge] = TIER_BADGES) -> Image.Image:
    """Render the cover image as a fresh ``PIL.Image`` instance."""

    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND)

    # Bokeh background (composite via RGBA)
    image = _draw_bokeh_background(image)

    draw = ImageDraw.Draw(image, "RGBA")

    font_paths = {
        "regular": _find_font("DejaVu Sans", weight="normal"),
        "bold": _find_font("DejaVu Sans", weight="bold"),
    }

    _draw_vertical_accent(draw)
    _draw_title_block(draw, font_paths)
    _draw_funnel(draw, font_paths)

    # Three tier cards across the bottom-left panel
    card_top = 420
    card_bottom = 590
    card_count = len(badges)
    gap = 16
    available = 640 - LEFT_MARGIN - 10
    card_width = (available - gap * (card_count - 1)) // card_count
    for i, badge in enumerate(badges):
        left = LEFT_MARGIN + i * (card_width + gap)
        right = left + card_width
        _draw_tier_card(
            draw, badge=badge, box=(left, card_top, right, card_bottom), font_paths=font_paths
        )

    _draw_bottom_strip(draw)

    return image.convert("RGB")


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

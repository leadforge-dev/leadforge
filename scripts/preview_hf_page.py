#!/usr/bin/env python3
"""Local publication-readiness preview for the Hugging Face dataset page.

PR 7.2.  Reads the artefact the publish PR will upload
(``release/huggingface/README.md`` or ``release/huggingface-instructor/README.md``
per ``--variant=public|instructor``), parses the YAML frontmatter +
Markdown body, renders an offline HTML page that surfaces the
published structure (header pills, tag chips, configs dropdown,
README body, footer), and optionally serves it on
``http://localhost:8766``.

This is a *publication-readiness* preview — structured rendering of
the upload artefact that helps catch link / config / YAML-rendering
issues before the real ``huggingface-cli upload``.  It is
deliberately NOT an HF look-alike: pixel fidelity is out of scope
and the chrome is approximate.

Design rationale + decision log: ``docs/release/preview_pages_design.md``.

Usage::

    python scripts/preview_hf_page.py --open-browser              # public variant
    python scripts/preview_hf_page.py --variant=instructor        # companion repo
    python scripts/preview_hf_page.py --no-serve                  # build only

Exit codes: 0 success / 2 pre-flight error.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

# Make ``scripts/`` importable regardless of how this file is loaded.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _preview_common import (  # noqa: E402 — must follow sys.path insert
    escape,
    plural,
    render_cover,
    serve,
)
from _release_common import replace_file  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_DIR_PUBLIC: Final[Path] = Path("release/_preview/huggingface")
DEFAULT_OUT_DIR_INSTRUCTOR: Final[Path] = Path("release/_preview/huggingface-instructor")
DEFAULT_PORT: Final[int] = 8766

#: Per-variant relative path to the README (under ``release_dir``).
_VARIANT_README_REL: Final[dict[str, Path]] = {
    "public": Path("huggingface/README.md"),
    "instructor": Path("huggingface-instructor/README.md"),
}
VALID_VARIANTS: Final[tuple[str, ...]] = ("public", "instructor")


# ---------------------------------------------------------------------------
# Markdown rendering (markdown-it-py is in [dev] AND [publish])
# ---------------------------------------------------------------------------


def _render_markdown(text: str) -> str:
    """Render ``text`` to HTML.  See preview_kaggle_page._render_markdown."""

    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:  # pragma: no cover — dep is in [dev]
        raise ImportError(
            "markdown-it-py is required.  pip install -e '.[dev]' (or [publish])."
        ) from exc
    return MarkdownIt("gfm-like").disable("linkify").render(text)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

#: HF dataset cards open with a ``---`` block of YAML, then the body.
#: ``re.DOTALL`` matters because the YAML spans multiple lines.
_FRONTMATTER_RE: Final[re.Pattern[str]] = re.compile(
    r"\A---\n(?P<yaml>.*?)\n---\n(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class HuggingFaceDoc:
    """Parsed HF README — frontmatter dict + body markdown."""

    frontmatter: dict[str, Any]
    body: str


def parse_hf_readme(text: str) -> HuggingFaceDoc:
    """Split an HF README into YAML frontmatter + Markdown body.

    Raises ``ValueError`` if the document does not open with a
    ``---``-delimited frontmatter block, or if the YAML is not a
    mapping (every HF dataset card MUST satisfy both).
    """

    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "HF README is missing a YAML frontmatter block (expected '---\\n<yaml>\\n---\\n<body>')"
        )
    parsed = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(parsed, dict):
        raise ValueError(
            f"HF README frontmatter is not a YAML mapping (got {type(parsed).__name__})"
        )
    return HuggingFaceDoc(frontmatter=parsed, body=match.group("body"))


# ---------------------------------------------------------------------------
# Section renderers — pure, deterministic
# ---------------------------------------------------------------------------


def _render_header(frontmatter: dict[str, Any]) -> str:
    """Render the page header — pretty_name + licence / task / size pills."""

    pretty_name = escape(str(frontmatter.get("pretty_name", "")))
    license_id = escape(str(frontmatter.get("license", "")))
    languages = ", ".join(escape(str(x)) for x in frontmatter.get("language", []) or [])
    sizes = ", ".join(escape(str(x)) for x in frontmatter.get("size_categories", []) or [])
    tasks = ", ".join(escape(str(x)) for x in frontmatter.get("task_categories", []) or [])
    return f"""<header class="dataset-header">
  <div class="dataset-header__namespace">huggingface.co/datasets</div>
  <h1 class="dataset-header__title">{pretty_name}</h1>
  <ul class="dataset-header__pills">
    <li class="pill pill--license">License: {license_id}</li>
    <li class="pill pill--task">Task: {tasks}</li>
    <li class="pill pill--size">Size: {sizes}</li>
    <li class="pill pill--language">Language: {languages}</li>
  </ul>
</header>"""


def _render_tags(frontmatter: dict[str, Any]) -> str:
    """Render the tag chip row (omitted when no tags)."""

    tags = frontmatter.get("tags", []) or []
    if not tags:
        return ""
    chips = " ".join(f'<span class="chip">{escape(str(t))}</span>' for t in tags)
    return f'<section class="tags">\n  {chips}\n</section>'


def _render_configs(frontmatter: dict[str, Any]) -> str:
    """Render the configs dropdown — one entry per ``configs[]`` block.

    This is the load-bearing inventory of what the YAML declares: each
    config + its train/validation/test data_files.  HF's "Subset"
    selector at the top of the dataset viewer maps to this.  Default
    config is flagged with a single ``badge--default`` instance.
    """

    configs = frontmatter.get("configs", []) or []
    if not configs:
        return '<section class="configs"><p>No configs declared.</p></section>'
    blocks: list[str] = []
    for config in configs:
        config_name = escape(str(config.get("config_name", "")))
        is_default = bool(config.get("default"))
        default_badge = ' <span class="badge badge--default">default</span>' if is_default else ""
        data_files = config.get("data_files", []) or []
        rows = "\n".join(
            f"      <tr><td>{escape(str(df.get('split', '')))}</td>"
            f"<td><code>{escape(str(df.get('path', '')))}</code></td></tr>"
            for df in data_files
        )
        blocks.append(
            f'  <details class="config" open>\n'
            f'    <summary class="config__name"><code>{config_name}</code>{default_badge} '
            f'<span class="config__count">({plural(len(data_files), "split")})</span>'
            f"</summary>\n"
            f'    <table class="config__table">\n'
            f"      <thead><tr><th>Split</th><th>Path</th></tr></thead>\n"
            f"      <tbody>\n{rows}\n      </tbody>\n"
            f"    </table>\n"
            f"  </details>"
        )
    return f"""<section class="configs">
  <h2 class="section__heading">Configurations / Subsets <span class="section__count">({plural(len(configs), "config")})</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_readme_body(body_md: str) -> str:
    """Render the README body (everything after the YAML)."""

    return f'<section class="readme">\n{_render_markdown(body_md)}</section>'


def _render_footer(frontmatter: dict[str, Any], variant: str) -> str:
    """Render the licence + variant note footer."""

    license_id = escape(str(frontmatter.get("license", "")))
    return f"""<footer class="dataset-footer">
  <div class="dataset-footer__license">License: {license_id}</div>
  <div class="dataset-footer__variant">Variant: <code>{escape(variant)}</code></div>
  <div class="dataset-footer__note">Local Hugging Face publication-readiness preview rendered by scripts/preview_hf_page.py — not the live dataset page.</div>
</footer>"""


# ---------------------------------------------------------------------------
# HTML wrapper + minimal CSS
# ---------------------------------------------------------------------------

_PAGE_CSS: Final[str] = """\
:root { --bg:#fff; --fg:#1f2937; --muted:#6b7280; --accent:#ff9d00; --border:#e5e7eb; --pill-bg:#f3f4f6; --code-bg:#f9fafb; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 0; line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }
.dataset-header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
.dataset-header__namespace { color: var(--muted); font-size: 0.85em; font-family: monospace; margin-bottom: 4px; }
.dataset-header__title { font-size: 1.8em; margin: 0 0 12px 0; }
.dataset-header__pills { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { background: var(--pill-bg); border-radius: 12px; padding: 4px 12px; font-size: 0.85em; color: var(--fg); }
.cover { margin: 0 0 24px 0; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.cover__image { display: block; max-width: 100%; height: auto; }
.tags { margin: 0 0 24px 0; }
.chip { display: inline-block; background: var(--pill-bg); border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.85em; color: var(--fg); }
.section__heading { font-size: 1.3em; border-bottom: 2px solid var(--accent); padding-bottom: 4px; margin-top: 32px; }
.section__count { color: var(--muted); font-size: 0.7em; font-weight: normal; }
.config { border: 1px solid var(--border); border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.config__name { cursor: pointer; font-weight: 600; }
.config__count { color: var(--muted); font-weight: normal; font-size: 0.85em; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; vertical-align: middle; margin-left: 4px; }
.badge--default { background: var(--accent); color: white; }
.config__table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9em; }
.config__table th, .config__table td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.config__table th { background: var(--pill-bg); font-weight: 600; }
.readme { margin: 24px 0; }
.readme code { background: var(--code-bg); padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }
.readme pre { background: var(--code-bg); padding: 12px; border-radius: 4px; overflow-x: auto; }
.readme pre code { background: none; padding: 0; }
.readme table { border-collapse: collapse; margin: 12px 0; }
.readme th, .readme td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
.readme blockquote { border-left: 3px solid var(--accent); padding-left: 12px; color: var(--muted); margin: 12px 0; }
.dataset-footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.9em; }
.dataset-footer__note { font-style: italic; margin-top: 8px; }
"""


def _wrap_html(*, title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HF preview — {escape(title)}</title>
  <style>{_PAGE_CSS}</style>
</head>
<body>
<main class="container">
{body}
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


#: Cover-image filename in the HF upload tree.  Pinned (not derived
#: from the YAML — HF's dataset card doesn't reference the cover; the
#: file lives at the root of the upload directory and is consumed by
#: HF's UI, not the README body) so the preview's cover render is
#: deterministic given just the parsed doc.
HF_COVER_IMAGE_FILENAME: Final[str] = "dataset-cover-image.png"


def render_hf_html(
    doc: HuggingFaceDoc,
    *,
    variant: str,
    cover_image_filename: str = HF_COVER_IMAGE_FILENAME,
) -> str:
    """Render the full HF preview HTML.

    Pure: same ``(doc, variant, cover_image_filename)`` → byte-identical
    HTML.  No I/O, no clock, no random.  The cover-image block was
    added in self-review pass 4 (Copilot finding COPILOT-2 — the
    driver was copying the cover into the preview tree without ever
    rendering it).
    """

    body_parts = [
        _render_header(doc.frontmatter),
        render_cover(cover_image_filename),
        _render_tags(doc.frontmatter),
        _render_configs(doc.frontmatter),
        _render_readme_body(doc.body),
        _render_footer(doc.frontmatter, variant=variant),
    ]
    return _wrap_html(
        title=str(doc.frontmatter.get("pretty_name", "")),
        body="\n".join(p for p in body_parts if p),
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreviewConfig:
    """Frozen driver config — built from CLI args or test input."""

    release_dir: Path
    out_dir: Path
    port: int
    variant: str
    open_browser: bool
    serve: bool


@dataclass(frozen=True)
class PreviewOutcome:
    """Return value from :func:`run_preview`.

    ``cover_path`` is always set on success — the driver always
    copies the cover into the preview tree.
    """

    html_path: Path
    cover_path: Path


#: Required frontmatter keys the renderer indexes directly; validated
#: up-front in ``run_preview`` so a malformed README surfaces as
#: ``ValueError`` → CLI rc=2 rather than silently rendering empty
#: pretty_name / license pills (Copilot finding COPILOT-1, applied
#: symmetrically to the HF script).
_REQUIRED_FRONTMATTER_KEYS: Final[tuple[str, ...]] = ("pretty_name", "license")


def _validate_required_frontmatter(frontmatter: dict[str, Any], path: Path) -> None:
    """Raise ``ValueError`` if required HF frontmatter keys are missing.

    ``pretty_name`` and ``license`` are the two HF requires *and* the
    two we display prominently; missing or empty values would render
    a half-blank header that's easy to miss.
    """

    missing = sorted(
        k for k in _REQUIRED_FRONTMATTER_KEYS if not str(frontmatter.get(k, "")).strip()
    )
    if missing:
        raise ValueError(f"{path} frontmatter is missing required key(s): {', '.join(missing)}")


def _resolve_cover_image(release_dir: Path, variant: str) -> Path:
    """Locate the cover image for the variant.

    Lookup order: variant-specific upload tree (assembled by the HF
    packager — gitignored, absent on a fresh checkout) → committed
    master copy under ``release_dir``.
    """

    variant_dir = "huggingface" if variant == "public" else "huggingface-instructor"
    for candidate in (
        release_dir / variant_dir / "dataset-cover-image.png",
        release_dir / "dataset-cover-image.png",
    ):
        if candidate.is_file():
            return candidate
    return release_dir / variant_dir / "dataset-cover-image.png"


def run_preview(config: PreviewConfig) -> PreviewOutcome:
    """Render the preview HTML, optionally serve it.

    Pre-flight failures (missing README, malformed YAML, missing
    cover, unknown variant) raise; the CLI converts to rc=2.
    """

    if config.variant not in VALID_VARIANTS:
        raise ValueError(f"unknown --variant {config.variant!r}; expected one of {VALID_VARIANTS}")

    readme_path = config.release_dir / _VARIANT_README_REL[config.variant]
    if not readme_path.is_file():
        raise FileNotFoundError(
            f"HF README not found at {readme_path}; "
            f"regenerate via scripts/package_hf_release.py --variant={config.variant} first"
        )
    doc = parse_hf_readme(readme_path.read_text(encoding="utf-8"))
    _validate_required_frontmatter(doc.frontmatter, readme_path)

    cover_src = _resolve_cover_image(config.release_dir, config.variant)
    if not cover_src.is_file():
        raise FileNotFoundError(
            f"cover image not found at {cover_src} "
            f"(looked in the {config.variant} upload tree and {config.release_dir}/)"
        )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = config.out_dir / "index.html"
    html_path.write_text(render_hf_html(doc, variant=config.variant), encoding="utf-8")

    cover_dst = config.out_dir / "dataset-cover-image.png"
    replace_file(cover_src, cover_dst)

    if config.serve:
        serve(config.out_dir, config.port, open_browser=config.open_browser)

    return PreviewOutcome(html_path=html_path, cover_path=cover_dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI.  Free function so tests can build a Namespace."""

    parser = argparse.ArgumentParser(
        prog="preview_hf_page",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release tree containing huggingface[-instructor]/README.md (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "where to write the rendered preview "
            "(default: release/_preview/huggingface for variant=public, "
            "release/_preview/huggingface-instructor for variant=instructor)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="port for the local HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default="public",
        help="public (3-tier) or instructor (companion repo); default: %(default)s",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="pop a browser tab on the served URL after the page renders",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="render the HTML and exit; don't start the server (CI / inspection mode)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir or (
        DEFAULT_OUT_DIR_PUBLIC if args.variant == "public" else DEFAULT_OUT_DIR_INSTRUCTOR
    )
    config = PreviewConfig(
        release_dir=args.release_dir,
        out_dir=out_dir,
        port=args.port,
        variant=args.variant,
        open_browser=args.open_browser,
        serve=not args.no_serve,
    )
    try:
        outcome = run_preview(config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {outcome.html_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

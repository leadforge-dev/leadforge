#!/usr/bin/env python3
"""Local publication-readiness preview for the Kaggle dataset page.

PR 7.2.  Reads the artefacts the publish PR will upload
(``release/kaggle/dataset-metadata.json`` + cover image), renders an
offline HTML page that surfaces the published structure (header,
cover, description, file tree, schema tables, sources, footer), and
optionally serves it on ``http://localhost:8765``.

This is a *publication-readiness* preview — structured rendering of
the upload artefacts that helps catch link / config / column-listing
issues before the real ``kaggle datasets create`` upload.  It is
deliberately NOT a Kaggle look-alike: pixel fidelity is out of scope
and the chrome (CSS palette, layout) is approximate.

Design rationale + decision log: ``docs/release/preview_pages_design.md``.

Usage::

    python scripts/preview_kaggle_page.py --open-browser  # serve + browser
    python scripts/preview_kaggle_page.py --no-serve      # build only

Exit codes: 0 success / 2 pre-flight error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Make ``scripts/`` importable regardless of how this file is loaded
# (CLI entrypoint, ``importlib.util.spec_from_file_location`` from tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _preview_common import (  # noqa: E402 — must follow sys.path insert
    escape,
    plural,
    render_cover,
    render_jsonld_dataset,
    serve,
)
from _release_common import replace_file  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_DIR: Final[Path] = Path("release/_preview/kaggle")
DEFAULT_PORT: Final[int] = 8765


# ---------------------------------------------------------------------------
# Markdown rendering (markdown-it-py is in [dev] AND [publish])
# ---------------------------------------------------------------------------


def _render_markdown(text: str) -> str:
    """Render ``text`` (the inlined README body) to HTML.

    ``gfm-like`` preset gives tables / fenced code / strikethrough;
    ``linkify`` is explicitly disabled so the optional
    ``linkify-it-py`` transitive dep is not required.
    """

    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:  # pragma: no cover — dep is in [dev]
        raise ImportError(
            "markdown-it-py is required.  pip install -e '.[dev]' (or [publish])."
        ) from exc
    return MarkdownIt("gfm-like").disable("linkify").render(text)


# ---------------------------------------------------------------------------
# Tier inference
# ---------------------------------------------------------------------------


def _tier_of(resource_path: str) -> str:
    """Return the leading path segment of ``resource_path``, or ``""``.

    Used to bucket resources by tier in the file tree.  Empty string
    means top-level (none today, tolerated for forward compatibility).
    """

    parts = resource_path.split("/", 1)
    return parts[0] if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Section renderers — pure, deterministic
# ---------------------------------------------------------------------------


def _render_header(metadata: dict[str, Any]) -> str:
    """Render the page header — title, subtitle, id, licence, frequency.

    Visibility is intentionally NOT rendered: Kaggle's public dataset
    page does not display ``isPrivate``, so showing it here would
    misrepresent what public viewers see.
    """

    title = escape(metadata["title"])
    subtitle = escape(metadata["subtitle"])
    dataset_id = escape(metadata["id"])
    license_name = escape(metadata["licenses"][0]["name"]) if metadata.get("licenses") else ""
    update_freq = escape(metadata.get("expectedUpdateFrequency", ""))

    return f"""<header class="dataset-header">
  <div class="dataset-header__id">{dataset_id}</div>
  <h1 class="dataset-header__title">{title}</h1>
  <p class="dataset-header__subtitle">{subtitle}</p>
  <ul class="dataset-header__pills">
    <li class="pill pill--license">License: {license_name}</li>
    <li class="pill pill--frequency">Updates: {update_freq}</li>
  </ul>
</header>"""


def _render_description(description_md: str) -> str:
    """Render the inlined README body as HTML."""

    return f'<section class="description">\n{_render_markdown(description_md)}</section>'


def _render_file_tree(resources: list[dict[str, Any]]) -> str:
    """Render the file tree, grouped by tier (leading path segment)."""

    by_tier: dict[str, list[dict[str, Any]]] = {}
    for resource in resources:
        by_tier.setdefault(_tier_of(resource["path"]), []).append(resource)

    blocks: list[str] = []
    for tier, tier_resources in by_tier.items():
        tier_label = escape(tier) if tier else "(top-level)"
        items: list[str] = []
        for resource in tier_resources:
            path = escape(resource["path"])
            description = escape(resource.get("description", ""))
            items.append(
                f'    <li class="file"><code class="file__path">{path}</code>'
                f'<span class="file__desc">{description}</span></li>'
            )
        blocks.append(
            f'  <details class="tier" open>\n'
            f'    <summary class="tier__name">{tier_label}/ '
            f'<span class="tier__count">({plural(len(tier_resources), "file")})</span>'
            f"</summary>\n"
            f'    <ul class="tier__files">\n' + "\n".join(items) + "\n    </ul>\n"
            "  </details>"
        )
    return f"""<section class="files">
  <h2 class="section__heading">Data Files <span class="section__count">({len(resources)} total)</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_schema_tables(resources: list[dict[str, Any]]) -> str:
    """Render one schema/columns table per tabular resource."""

    blocks: list[str] = []
    total_columns = 0
    for resource in resources:
        schema = resource.get("schema")
        if not schema:
            continue
        fields = schema.get("fields", [])
        if not fields:
            continue
        total_columns += len(fields)
        path = escape(resource["path"])
        rows: list[str] = []
        for fd in fields:
            name = escape(fd.get("name", ""))
            ftype = escape(fd.get("type", ""))
            description = escape(fd.get("description", ""))
            rows.append(
                f"      <tr>"
                f'<td class="col__name"><code>{name}</code></td>'
                f'<td class="col__type">{ftype}</td>'
                f'<td class="col__desc">{description}</td>'
                f"</tr>"
            )
        blocks.append(
            f'  <details class="schema" open>\n'
            f'    <summary class="schema__path"><code>{path}</code> '
            f'<span class="schema__count">({plural(len(fields), "column")})</span>'
            f"</summary>\n"
            f'    <table class="schema__table">\n'
            f"      <thead><tr><th>Column</th><th>Type</th><th>Description</th></tr></thead>\n"
            f"      <tbody>\n" + "\n".join(rows) + "\n      </tbody>\n"
            "    </table>\n"
            "  </details>"
        )
    return f"""<section class="schemas">
  <h2 class="section__heading">Schema / Columns <span class="section__count">({plural(total_columns, "column")} across {plural(len(blocks), "tabular file")})</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_sources(metadata: dict[str, Any]) -> str:
    """Render the user-specified sources block (omitted when empty)."""

    sources = metadata.get("userSpecifiedSources", []) or []
    if not sources:
        return ""
    items = "\n".join(
        f'    <li><a href="{escape(s["url"])}" target="_blank" rel="noopener noreferrer">'
        f"{escape(s['title'])}</a></li>"
        for s in sources
    )
    return f"""<section class="sources">
  <h2 class="section__heading">Sources</h2>
  <ul class="sources__list">
{items}
  </ul>
</section>"""


def _render_footer(metadata: dict[str, Any]) -> str:
    """Render the licence + keywords footer."""

    keywords = metadata.get("keywords", []) or []
    keyword_chips = " ".join(f'<span class="chip">{escape(k)}</span>' for k in keywords)
    license_name = escape(metadata["licenses"][0]["name"]) if metadata.get("licenses") else ""
    return f"""<footer class="dataset-footer">
  <div class="dataset-footer__keywords">{keyword_chips}</div>
  <div class="dataset-footer__license">License: {license_name}</div>
  <div class="dataset-footer__note">Local Kaggle publication-readiness preview rendered by scripts/preview_kaggle_page.py — not the live dataset page.</div>
</footer>"""


# ---------------------------------------------------------------------------
# HTML wrapper + minimal CSS
# ---------------------------------------------------------------------------

#: Inlined for a single self-contained HTML file (easier inspection,
#: simpler byte-compare in the regeneration-discipline test, works
#: without a server).  Palette is approximate, not branded.
_PAGE_CSS: Final[str] = """\
:root { --bg:#fff; --fg:#202124; --muted:#5f6368; --accent:#20beff; --border:#e0e0e0; --pill-bg:#f1f3f4; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 0; line-height: 1.5; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }
.dataset-header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
.dataset-header__id { color: var(--muted); font-size: 0.85em; font-family: monospace; margin-bottom: 4px; }
.dataset-header__title { font-size: 1.8em; margin: 0 0 4px 0; }
.dataset-header__subtitle { color: var(--muted); margin: 0 0 12px 0; }
.dataset-header__pills { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { background: var(--pill-bg); border-radius: 12px; padding: 4px 12px; font-size: 0.85em; color: var(--fg); }
.cover { margin: 0 0 24px 0; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.cover__image { display: block; max-width: 100%; height: auto; }
.section__heading { font-size: 1.3em; border-bottom: 2px solid var(--accent); padding-bottom: 4px; margin-top: 32px; }
.section__count { color: var(--muted); font-size: 0.7em; font-weight: normal; }
.tier, .schema { border: 1px solid var(--border); border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.tier__name, .schema__path { cursor: pointer; font-weight: 600; }
.tier__count, .schema__count { color: var(--muted); font-weight: normal; font-size: 0.85em; }
.tier__files { list-style: none; padding: 8px 0 0 0; margin: 0; }
.file { display: flex; gap: 12px; padding: 4px 0; border-bottom: 1px dotted var(--border); }
.file:last-child { border-bottom: none; }
.file__path { color: var(--accent); flex-shrink: 0; }
.file__desc { color: var(--muted); font-size: 0.9em; }
.schema__table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9em; }
.schema__table th, .schema__table td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); vertical-align: top; }
.schema__table th { background: var(--pill-bg); font-weight: 600; }
.col__name code { background: none; }
.col__type { color: var(--muted); font-family: monospace; }
.description { margin: 24px 0; }
.description code { background: var(--pill-bg); padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }
.description pre { background: var(--pill-bg); padding: 12px; border-radius: 4px; overflow-x: auto; }
.description pre code { background: none; padding: 0; }
.description table { border-collapse: collapse; margin: 12px 0; }
.description th, .description td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
.description blockquote { border-left: 3px solid var(--accent); padding-left: 12px; color: var(--muted); margin: 12px 0; }
.sources__list { padding-left: 20px; }
.dataset-footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.9em; }
.dataset-footer__keywords { margin-bottom: 8px; }
.chip { display: inline-block; background: var(--pill-bg); border-radius: 12px; padding: 2px 10px; margin: 2px; font-size: 0.85em; }
.dataset-footer__note { font-style: italic; margin-top: 8px; }
"""


def _wrap_html(*, title: str, body: str, jsonld: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kaggle preview — {escape(title)}</title>
  <style>{_PAGE_CSS}</style>
  {jsonld}
</head>
<body>
<main class="container">
{body}
</main>
</body>
</html>
"""


#: SPDX-style URL for MIT (schema.org ``license`` is a URL, not the
#: SPDX short name).  Kept here so a future relicensing PR only has
#: to flip one constant per preview script.
_LICENSE_URL_MIT: Final[str] = "https://opensource.org/licenses/MIT"


def _jsonld_for_kaggle(metadata: dict[str, Any]) -> str:
    """Build the schema.org ``Dataset`` JSON-LD block for Kaggle.

    Sources: title / subtitle / id / keywords / image from the Kaggle
    metadata; license URL is pinned (Kaggle stores the license name,
    not the SPDX URL the JSON-LD spec wants).  ``distribution`` is a
    short representative list of file paths so an agent can see the
    bundle's shape without enumerating every parquet — the full list
    lives in ``resources[]`` lower on the page.
    """

    keywords = list(metadata.get("keywords", []))
    sources = metadata.get("userSpecifiedSources", []) or []
    same_as = [s["url"] for s in sources if isinstance(s, dict) and s.get("url")]

    resources = metadata.get("resources", [])
    representative_paths = [r["path"] for r in resources if isinstance(r, dict) and r.get("path")][
        :12
    ]

    return render_jsonld_dataset(
        name=str(metadata.get("title", "")),
        description=str(metadata.get("subtitle", "")),
        license_url=_LICENSE_URL_MIT,
        keywords=keywords,
        citation=(
            "Generated by leadforge (https://github.com/leadforge-dev/leadforge); "
            "recipe b2b_saas_procurement_v1, seed 42."
        ),
        distribution_paths=representative_paths,
        same_as=same_as,
        creator="leadforge",
        version="v1",
    )


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


def render_kaggle_html(metadata: dict[str, Any], cover_image_filename: str) -> str:
    """Render the full Kaggle preview HTML.

    Pure: same ``(metadata, cover_image_filename)`` → byte-identical
    HTML.  No I/O, no clock, no random.
    """

    body_parts = [
        _render_header(metadata),
        render_cover(cover_image_filename),
        _render_description(metadata.get("description", "")),
        _render_file_tree(metadata.get("resources", [])),
        _render_schema_tables(metadata.get("resources", [])),
        _render_sources(metadata),
        _render_footer(metadata),
    ]
    return _wrap_html(
        title=metadata.get("title", ""),
        body="\n".join(p for p in body_parts if p),
        jsonld=_jsonld_for_kaggle(metadata),
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


#: Required keys the renderer indexes directly (without ``.get``);
#: validated up-front in ``run_preview`` so a malformed metadata file
#: surfaces as ``ValueError`` → CLI rc=2 rather than a ``KeyError``
#: traceback mid-render (Copilot finding COPILOT-1).
_REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "title",
    "subtitle",
    "id",
    "expectedUpdateFrequency",
    "image",
)


def _validate_required_metadata(metadata: dict[str, Any], path: Path) -> None:
    """Raise ``ValueError`` if required Kaggle metadata keys are missing.

    Catches the case where ``dataset-metadata.json`` is hand-edited or
    produced by a future broken packager; the renderer's
    ``_render_header`` / ``_render_footer`` index these directly and
    would otherwise raise ``KeyError`` mid-render, bypassing
    ``main()``'s rc=2 handling.
    """

    missing = sorted(k for k in _REQUIRED_METADATA_KEYS if k not in metadata)
    licenses = metadata.get("licenses")
    if (
        not isinstance(licenses, list)
        or not licenses
        or not isinstance(licenses[0], dict)
        or "name" not in licenses[0]
    ):
        missing.append("licenses[0].name")
    if missing:
        raise ValueError(f"{path} is missing required key(s): {', '.join(missing)}")


def _resolve_cover_image(release_dir: Path, image_name: str) -> Path:
    """Locate the cover image referenced by the metadata's ``image``.

    Lookup order: ``release/kaggle/<image_name>`` (assembled
    upload-tree copy, present after the maintainer runs the Kaggle
    packager — gitignored, so absent on a fresh checkout) →
    ``release/<image_name>`` (the committed master copy).
    """

    for candidate in (release_dir / "kaggle" / image_name, release_dir / image_name):
        if candidate.is_file():
            return candidate
    return release_dir / "kaggle" / image_name  # surface the missing-file error here


def run_preview(config: PreviewConfig) -> PreviewOutcome:
    """Render the preview HTML, optionally serve it.

    Validation discipline: build → validate → write.  Pre-flight
    failures (missing metadata, malformed JSON, missing cover) raise;
    the CLI converts to rc=2.
    """

    metadata_path = config.release_dir / "kaggle" / "dataset-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Kaggle dataset metadata not found at {metadata_path}; "
            f"regenerate via scripts/package_kaggle_release.py first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} is not a JSON object")
    _validate_required_metadata(metadata, metadata_path)

    cover_name = metadata["image"]
    if not cover_name:
        raise ValueError(f"{metadata_path} declares no 'image' (cover image filename)")
    cover_src = _resolve_cover_image(config.release_dir, cover_name)
    if not cover_src.is_file():
        raise FileNotFoundError(f"cover image declared as {cover_name!r} not found at {cover_src}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = config.out_dir / "index.html"
    html_path.write_text(render_kaggle_html(metadata, cover_name), encoding="utf-8")

    cover_dst = config.out_dir / cover_name
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
        prog="preview_kaggle_page",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="release tree containing kaggle/dataset-metadata.json (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where to write the rendered preview (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="port for the local HTTP server (default: %(default)s)",
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
    config = PreviewConfig(
        release_dir=args.release_dir,
        out_dir=args.out_dir,
        port=args.port,
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

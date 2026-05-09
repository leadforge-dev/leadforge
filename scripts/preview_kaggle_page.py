#!/usr/bin/env python3
"""Render an offline mock of the Kaggle dataset page.

PR 7.2 — middle PR in Phase 7 (LLM critique + publish).  Reads the
artefacts the publish PR will upload (``release/kaggle/dataset-metadata.json``
+ ``release/dataset-cover-image.png``) and renders an HTML page that
mimics the public Kaggle dataset view: header (title / subtitle /
licence / id pill / update-frequency pill), cover image, rendered
description (the inlined README body), file tree of declared
resources, schema/columns tables for every tabular resource, and a
licence + sources footer.

The page exists for human click-through review BEFORE the maintainer
runs the real ``kaggle datasets create`` upload (PR 7.3).  Cached
previews on the live page are expensive to roll back, so the
publish runbook in PR 7.3 cites this script as a required pre-flight.

The rendered HTML is a deterministic function of the input artefacts
(no ``now()``, no random) — same metadata + cover-image filename →
byte-identical HTML.  The committed sample at
``release/_preview_committed/kaggle.html`` is the audit-artefact-sync
gate (mirrors PR 4.1 / 5.1 / 5.2 / 7.1).

Usage::

    # Render + serve on http://localhost:8765, pop a browser tab.
    python scripts/preview_kaggle_page.py --open-browser

    # Just build the HTML (CI / inspection); no server.
    python scripts/preview_kaggle_page.py --no-serve

Exit codes: 0 success / 2 pre-flight error (missing metadata,
missing cover image, malformed JSON).
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import sys
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

# Make ``scripts/`` importable regardless of how this file is loaded
# (CLI entrypoint, ``importlib.util.spec_from_file_location`` from
# tests).  Mirrors the pattern in ``package_kaggle_release.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _release_common import replace_file  # noqa: E402 — must follow sys.path insert

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_DIR: Final[Path] = Path("release/_preview/kaggle")
DEFAULT_PORT: Final[int] = 8765


# ---------------------------------------------------------------------------
# Markdown rendering (gated behind the [publish] extra)
# ---------------------------------------------------------------------------


def _render_markdown(text: str) -> str:
    """Render ``text`` (the inlined README body) to HTML.

    Uses ``markdown-it-py`` in GFM-like mode (tables, fenced code,
    autolink, strikethrough) — closest match to how Kaggle renders
    its description block.  The ``[publish]`` extra (alongside
    ``datasets`` / ``kaggle``) is the install path; absent dep
    raises a clear instruction rather than a cryptic ``ImportError``.
    Footnotes (``[^foo]``) render as literal text, which is faithful
    enough — Kaggle does not invest in footnote rendering either.
    """

    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:  # pragma: no cover — gated by extra
        raise ImportError(
            "markdown-it-py is required for the Kaggle preview page. "
            "Install the publish extra: pip install -e '.[publish]'"
        ) from exc
    # ``gfm-like`` enables linkify by default, which requires the
    # separate ``linkify-it-py`` package; we explicitly turn it off so
    # the preview does not pull a transitive dep beyond markdown-it-py.
    # Tables / fenced code / strikethrough remain on (the bits that
    # actually matter for faithful Kaggle/HF rendering).
    md = MarkdownIt("gfm-like").disable("linkify")
    return md.render(text)


# ---------------------------------------------------------------------------
# Tier inference + file tree
# ---------------------------------------------------------------------------

#: Kaggle's CLI emits resource paths like ``intro/lead_scoring.csv`` —
#: the leading path segment is the tier name.  We group resources by
#: this segment so the rendered file tree mirrors the bundle layout
#: the user will see on Kaggle.
_TIER_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^([^/]+)/")


def _tier_of(resource_path: str) -> str:
    """Return the leading path segment of ``resource_path``, or ``""``.

    Used to bucket resources by tier in the file tree.  An empty
    string indicates a top-level resource (none of these are emitted
    by the Kaggle packager today, but we tolerate them for forward
    compatibility).
    """

    match = _TIER_PATH_RE.match(resource_path)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Section renderers — pure, deterministic
# ---------------------------------------------------------------------------


def _render_header(metadata: dict[str, Any]) -> str:
    """Render the page header — title, subtitle, id pill, licence pill."""

    title = _escape(metadata["title"])
    subtitle = _escape(metadata["subtitle"])
    dataset_id = _escape(metadata["id"])
    license_name = _escape(metadata["licenses"][0]["name"]) if metadata.get("licenses") else ""
    update_freq = _escape(metadata.get("expectedUpdateFrequency", ""))
    visibility = "Private" if metadata.get("isPrivate") else "Public"

    return f"""<header class="dataset-header">
  <div class="dataset-header__id">{dataset_id}</div>
  <h1 class="dataset-header__title">{title}</h1>
  <p class="dataset-header__subtitle">{subtitle}</p>
  <ul class="dataset-header__pills">
    <li class="pill pill--license">License: {license_name}</li>
    <li class="pill pill--frequency">Updates: {update_freq}</li>
    <li class="pill pill--visibility">Visibility: {visibility}</li>
  </ul>
</header>"""


def _render_cover(cover_image_filename: str) -> str:
    """Render the cover-image block.

    The ``src`` is a sibling-relative path so the same HTML works
    against both the runtime preview tree (where the image was copied
    in) and the committed sample (used for byte-equality only — the
    sample is not served).
    """

    src = _escape(cover_image_filename)
    return f"""<section class="cover">
  <img class="cover__image" src="{src}" alt="Dataset cover image">
</section>"""


def _render_description(description_md: str) -> str:
    """Render the inlined README body as HTML."""

    body = _render_markdown(description_md)
    return f'<section class="description">\n{body}</section>'


def _render_file_tree(resources: list[dict[str, Any]]) -> str:
    """Render the file tree, grouped by tier (leading path segment).

    Inside each tier, files appear in declaration order — matches the
    order Kaggle renders the resources column.  Each entry is a
    monospace path + the resource description.
    """

    by_tier: dict[str, list[dict[str, Any]]] = {}
    for resource in resources:
        tier = _tier_of(resource["path"])
        by_tier.setdefault(tier, []).append(resource)

    blocks: list[str] = []
    for tier, tier_resources in by_tier.items():
        tier_label = _escape(tier) if tier else "(top-level)"
        items: list[str] = []
        for resource in tier_resources:
            path = _escape(resource["path"])
            description = _escape(resource.get("description", ""))
            items.append(
                f'    <li class="file"><code class="file__path">{path}</code>'
                f'<span class="file__desc">{description}</span></li>'
            )
        blocks.append(
            f'  <details class="tier" open>\n'
            f'    <summary class="tier__name">{tier_label}/ '
            f'<span class="tier__count">({len(tier_resources)} files)</span>'
            f"</summary>\n"
            f'    <ul class="tier__files">\n' + "\n".join(items) + "\n    </ul>\n"
            "  </details>"
        )
    file_count = len(resources)
    return f"""<section class="files">
  <h2 class="section__heading">Data Files <span class="section__count">({file_count} total)</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_schema_tables(resources: list[dict[str, Any]]) -> str:
    """Render one schema/columns table per tabular resource.

    Mimics Kaggle's "Data Card" expandable per-file column listing.
    Resources without a ``schema`` (markdown / JSON) are skipped —
    same posture as Kaggle.  Column count appears in the heading so
    the test can assert the table is exhaustive without parsing the
    DOM.
    """

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
        path = _escape(resource["path"])
        rows: list[str] = []
        for fd in fields:
            name = _escape(fd.get("name", ""))
            ftype = _escape(fd.get("type", ""))
            description = _escape(fd.get("description", ""))
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
            f'<span class="schema__count">({len(fields)} columns)</span>'
            f"</summary>\n"
            f'    <table class="schema__table">\n'
            f"      <thead><tr><th>Column</th><th>Type</th><th>Description</th></tr></thead>\n"
            f"      <tbody>\n" + "\n".join(rows) + "\n      </tbody>\n"
            "    </table>\n"
            "  </details>"
        )
    return f"""<section class="schemas">
  <h2 class="section__heading">Schema / Columns <span class="section__count">({total_columns} columns across {len(blocks)} tabular files)</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_sources(metadata: dict[str, Any]) -> str:
    """Render the user-specified sources block."""

    sources = metadata.get("userSpecifiedSources", []) or []
    if not sources:
        return ""
    items = "\n".join(
        f'    <li><a href="{_escape(s["url"])}" target="_blank" rel="noopener noreferrer">'
        f"{_escape(s['title'])}</a></li>"
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
    keyword_chips = " ".join(f'<span class="chip">{_escape(k)}</span>' for k in keywords)
    license_name = _escape(metadata["licenses"][0]["name"]) if metadata.get("licenses") else ""
    return f"""<footer class="dataset-footer">
  <div class="dataset-footer__keywords">{keyword_chips}</div>
  <div class="dataset-footer__license">License: {license_name}</div>
  <div class="dataset-footer__note">Local Kaggle preview rendered by scripts/preview_kaggle_page.py — not the live dataset page.</div>
</footer>"""


# ---------------------------------------------------------------------------
# HTML wrapper + minimal Kaggle-ish CSS
# ---------------------------------------------------------------------------

#: Kept inline rather than served as a separate ``style.css`` so the
#: rendered HTML is a single self-contained file — easier to inspect,
#: easier to byte-compare in the audit-artefact-sync test, and works
#: without a server (open the committed sample directly in a browser).
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


def _wrap_html(*, title: str, body: str) -> str:
    """Wrap rendered sections in the page chrome.

    Order: header → cover → description → files → schemas → sources →
    footer.  Description sits above files because Kaggle leads with
    the dataset card on the public page.
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Kaggle preview — {_escape(title)}</title>
  <style>{_PAGE_CSS}</style>
</head>
<body>
<main class="container">
{body}
</main>
</body>
</html>
"""


def _escape(value: str) -> str:
    """HTML-escape a single attribute / text value.

    Inlined rather than importing ``html.escape`` so the renderer's
    surface stays small and the (well-tested) substitution is local
    and obvious.
    """

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


# ---------------------------------------------------------------------------
# Top-level renderer
# ---------------------------------------------------------------------------


def render_kaggle_html(metadata: dict[str, Any], cover_image_filename: str) -> str:
    """Render the full Kaggle preview HTML.

    Pure function: same ``(metadata, cover_image_filename)`` →
    byte-identical HTML.  No I/O, no clock, no random.  Tests rely
    on this for the audit-artefact-sync gate.
    """

    body_parts = [
        _render_header(metadata),
        _render_cover(cover_image_filename),
        _render_description(metadata.get("description", "")),
        _render_file_tree(metadata.get("resources", [])),
        _render_schema_tables(metadata.get("resources", [])),
        _render_sources(metadata),
        _render_footer(metadata),
    ]
    return _wrap_html(title=metadata.get("title", ""), body="\n".join(p for p in body_parts if p))


# ---------------------------------------------------------------------------
# Driver — reads inputs, writes HTML, optionally serves
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreviewConfig:
    """Frozen driver config.

    Mirrors the ``DriverConfig`` posture in
    ``scripts/run_llm_critique.py`` — building this from CLI args
    keeps the test surface a Python-level call rather than an exec.
    """

    release_dir: Path
    out_dir: Path
    port: int
    open_browser: bool
    serve: bool


@dataclass(frozen=True)
class PreviewOutcome:
    """Return value from :func:`run_preview` — used by tests + CLI."""

    html_path: Path
    cover_path: Path | None


def _resolve_cover_image(release_dir: Path, image_name: str) -> Path:
    """Locate the cover image referenced by the metadata's ``image``.

    Lookup order: ``release/kaggle/<image_name>`` (assembled
    upload-tree copy, present after the maintainer runs the Kaggle
    packager — gitignored, so absent on a fresh checkout) →
    ``release/<image_name>`` (the committed master copy).  Returning
    the resolved path here mirrors ``_release_common.resolve_cover_image_path``
    so the assembler and inputs cannot disagree.
    """

    candidates = [
        release_dir / "kaggle" / image_name,
        release_dir / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]  # surface the missing-file error against the canonical location


def run_preview(config: PreviewConfig) -> PreviewOutcome:
    """Render the preview HTML, optionally serve it.

    Pre-flight failures (missing metadata, malformed JSON, missing
    cover image) raise — the CLI converts to rc=2.  Validation
    discipline mirrors the Phase 5 packagers: build → validate → write.
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

    cover_name = metadata.get("image", "")
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
        _serve(config.out_dir, config.port, open_browser=config.open_browser)

    return PreviewOutcome(html_path=html_path, cover_path=cover_dst)


def _serve(directory: Path, port: int, *, open_browser: bool) -> None:
    """Start a stdlib HTTP server rooted at ``directory`` and block.

    Uses ``http.server.ThreadingHTTPServer`` so the browser can fetch
    the cover image alongside the HTML without serialising requests.
    ``ThreadingHTTPServer`` (unlike bare ``socketserver.ThreadingTCPServer``)
    inherits ``allow_reuse_address = True`` from ``HTTPServer`` —
    matters because Ctrl-C → re-run within ~60s would otherwise
    raise ``OSError: [Errno 48] Address already in use`` while the
    socket sits in TIME_WAIT.

    Blocks on ``serve_forever()``; KeyboardInterrupt (Ctrl-C) is the
    documented exit path.  No coverage here — tests exercise the
    pure renderer and ``--no-serve`` path; serving is glue that
    requires a live socket.
    """

    handler_factory = _make_handler_factory(directory)
    url = f"http://localhost:{port}/"
    print(f"serving {directory} at {url} — Ctrl-C to stop", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    with http.server.ThreadingHTTPServer(("", port), handler_factory) as httpd:
        httpd.serve_forever()


def _make_handler_factory(directory: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    """Build a handler subclass that serves from ``directory``.

    ``SimpleHTTPRequestHandler`` ships a ``directory=`` kwarg in
    Python 3.7+, but threading the path through ``socketserver``'s
    ``RequestHandlerClass`` requires either a partial or a subclass.
    Subclassing keeps the import surface stdlib-only.
    """

    resolved = str(directory.resolve())

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=resolved, **kwargs)

    return _Handler


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

#!/usr/bin/env python3
"""Render an offline mock of the Hugging Face dataset page.

PR 7.2 — middle PR in Phase 7 (LLM critique + publish).  Reads the
artefact the publish PR will upload (``release/huggingface/README.md``
or ``release/huggingface-instructor/README.md``) and renders an HTML
page that mimics the public HF dataset view: header (pretty_name +
licence + size pill), tag chips, configs dropdown, file tree, the
README body, and a footer with sources.

Same rationale as ``preview_kaggle_page.py`` — cached previews on
the live HF page are expensive to roll back, so the publish runbook
in PR 7.3 cites this script as a required pre-flight.

The rendered HTML is a deterministic function of the input README
(no ``now()``, no random) — same input → byte-identical HTML.  The
committed samples at
``release/_preview_committed/huggingface_{public,instructor}.html``
are the audit-artefact-sync gate.

Usage::

    # Public variant on http://localhost:8766.
    python scripts/preview_hf_page.py --open-browser

    # Instructor companion variant (separate input README).
    python scripts/preview_hf_page.py --variant=instructor

    # Just build the HTML (CI / inspection).
    python scripts/preview_hf_page.py --no-serve

Exit codes: 0 success / 2 pre-flight error (missing README,
malformed YAML frontmatter, missing cover image).
"""

from __future__ import annotations

import argparse
import http.server
import re
import socketserver
import sys
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

# Make ``scripts/`` importable regardless of how this file is loaded.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _release_common import replace_file  # noqa: E402 — must follow sys.path insert

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RELEASE_DIR: Final[Path] = Path("release")
DEFAULT_OUT_DIR_PUBLIC: Final[Path] = Path("release/_preview/huggingface")
DEFAULT_OUT_DIR_INSTRUCTOR: Final[Path] = Path("release/_preview/huggingface-instructor")
DEFAULT_PORT: Final[int] = 8766

#: Per-variant relative paths to the README (under ``release_dir``)
#: and the committed sample HTML (under ``release/_preview_committed/``).
_VARIANT_README_REL: Final[dict[str, Path]] = {
    "public": Path("huggingface/README.md"),
    "instructor": Path("huggingface-instructor/README.md"),
}
_VARIANT_SAMPLE_PATH: Final[dict[str, Path]] = {
    "public": Path("release/_preview_committed/huggingface_public.html"),
    "instructor": Path("release/_preview_committed/huggingface_instructor.html"),
}
VALID_VARIANTS: Final[tuple[str, ...]] = ("public", "instructor")


# ---------------------------------------------------------------------------
# Markdown rendering (gated behind the [publish] extra)
# ---------------------------------------------------------------------------


def _render_markdown(text: str) -> str:
    """Render ``text`` to HTML using markdown-it-py in GFM-like mode.

    Same posture + dep gating as the Kaggle preview (markdown-it-py
    via the ``[publish]`` extra; ``linkify`` disabled so the
    transitive ``linkify-it-py`` dep is not required).  See
    ``preview_kaggle_page.py`` for the rationale.
    """

    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:  # pragma: no cover — gated by extra
        raise ImportError(
            "markdown-it-py is required for the Hugging Face preview page. "
            "Install the publish extra: pip install -e '.[publish]'"
        ) from exc
    md = MarkdownIt("gfm-like").disable("linkify")
    return md.render(text)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

#: HF dataset cards open with a ``---`` block of YAML, then the body.
#: This regex pulls them apart in one shot; ``re.DOTALL`` is essential
#: because the YAML spans multiple lines.
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
    ``---``-delimited frontmatter block (every HF dataset card MUST
    have one — the renderer cannot mock the page without it).
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


def _escape(value: str) -> str:
    """HTML-escape a single attribute / text value."""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_header(frontmatter: dict[str, Any]) -> str:
    """Render the page header — pretty_name, licence pill, sizes."""

    pretty_name = _escape(str(frontmatter.get("pretty_name", "")))
    license_id = _escape(str(frontmatter.get("license", "")))
    languages = ", ".join(_escape(str(x)) for x in frontmatter.get("language", []) or [])
    sizes = ", ".join(_escape(str(x)) for x in frontmatter.get("size_categories", []) or [])
    tasks = ", ".join(_escape(str(x)) for x in frontmatter.get("task_categories", []) or [])
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
    """Render the tag chip row (mimics HF tag pills under the header)."""

    tags = frontmatter.get("tags", []) or []
    if not tags:
        return ""
    chips = " ".join(f'<span class="chip">{_escape(str(t))}</span>' for t in tags)
    return f'<section class="tags">\n  {chips}\n</section>'


def _render_configs(frontmatter: dict[str, Any]) -> str:
    """Render the configs dropdown — one entry per ``configs[]`` block.

    Mirrors HF's "Subset" selector at the top of the dataset viewer.
    Each config lists its data_files (split → path) so the test can
    assert every config block from the YAML round-trips through to
    the rendered page.  The default config is flagged.
    """

    configs = frontmatter.get("configs", []) or []
    if not configs:
        return '<section class="configs"><p>No configs declared.</p></section>'
    blocks: list[str] = []
    for config in configs:
        config_name = _escape(str(config.get("config_name", "")))
        is_default = bool(config.get("default"))
        default_badge = ' <span class="badge badge--default">default</span>' if is_default else ""
        data_files = config.get("data_files", []) or []
        rows = "\n".join(
            f"      <tr><td>{_escape(str(df.get('split', '')))}</td>"
            f"<td><code>{_escape(str(df.get('path', '')))}</code></td></tr>"
            for df in data_files
        )
        blocks.append(
            f'  <details class="config" open>\n'
            f'    <summary class="config__name"><code>{config_name}</code>{default_badge} '
            f'<span class="config__count">({len(data_files)} splits)</span>'
            f"</summary>\n"
            f'    <table class="config__table">\n'
            f"      <thead><tr><th>Split</th><th>Path</th></tr></thead>\n"
            f"      <tbody>\n{rows}\n      </tbody>\n"
            f"    </table>\n"
            f"  </details>"
        )
    return f"""<section class="configs">
  <h2 class="section__heading">Configurations / Subsets <span class="section__count">({len(configs)} configs)</span></h2>
{chr(10).join(blocks)}
</section>"""


def _render_file_tree(frontmatter: dict[str, Any], variant: str) -> str:
    """Render the file tree.

    HF doesn't ship a structured file inventory in the dataset card
    YAML the way Kaggle does — ``data_files`` are the only paths
    declared in the frontmatter.  We list each declared path under
    its config heading.  The tree is therefore narrower than the
    real dataset (which also has ``manifest.json``, ``tables/``, etc.)
    but matches what the YAML knows about, which is what the publish
    runbook is trying to verify.
    """

    configs = frontmatter.get("configs", []) or []
    paths: list[tuple[str, str]] = []
    for config in configs:
        config_name = str(config.get("config_name", ""))
        for df in config.get("data_files", []) or []:
            paths.append((config_name, str(df.get("path", ""))))
    if not paths:
        return ""
    items = "\n".join(
        f'    <li class="file"><span class="file__config">[{_escape(c)}]</span> '
        f'<code class="file__path">{_escape(p)}</code></li>'
        for c, p in paths
    )
    return f"""<section class="files">
  <h2 class="section__heading">Files declared in YAML <span class="section__count">({len(paths)} files / variant: {_escape(variant)})</span></h2>
  <ul class="files__list">
{items}
  </ul>
</section>"""


def _render_readme_body(body_md: str) -> str:
    """Render the README body (everything after the YAML)."""

    return f'<section class="readme">\n{_render_markdown(body_md)}</section>'


def _render_footer(frontmatter: dict[str, Any], variant: str) -> str:
    """Render the licence + variant note footer."""

    license_id = _escape(str(frontmatter.get("license", "")))
    return f"""<footer class="dataset-footer">
  <div class="dataset-footer__license">License: {license_id}</div>
  <div class="dataset-footer__variant">Variant: <code>{_escape(variant)}</code></div>
  <div class="dataset-footer__note">Local Hugging Face preview rendered by scripts/preview_hf_page.py — not the live dataset page.</div>
</footer>"""


# ---------------------------------------------------------------------------
# HTML wrapper + minimal HF-ish CSS
# ---------------------------------------------------------------------------

#: Inlined for the same reasons as the Kaggle preview — single
#: self-contained file, simple byte-comparison in the audit-sync test,
#: works without the server.
_PAGE_CSS: Final[str] = """\
:root { --bg:#fff; --fg:#1f2937; --muted:#6b7280; --accent:#ff9d00; --border:#e5e7eb; --pill-bg:#f3f4f6; --code-bg:#f9fafb; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif; color: var(--fg); background: var(--bg); margin: 0; padding: 0; line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 32px; }
.dataset-header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
.dataset-header__namespace { color: var(--muted); font-size: 0.85em; font-family: monospace; margin-bottom: 4px; }
.dataset-header__title { font-size: 1.8em; margin: 0 0 12px 0; }
.dataset-header__pills { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { background: var(--pill-bg); border-radius: 12px; padding: 4px 12px; font-size: 0.85em; color: var(--fg); }
.tags { margin: 0 0 24px 0; }
.chip { display: inline-block; background: var(--pill-bg); border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.85em; color: var(--fg); }
.section__heading { font-size: 1.3em; border-bottom: 2px solid var(--accent); padding-bottom: 4px; margin-top: 32px; }
.section__count { color: var(--muted); font-size: 0.7em; font-weight: normal; }
.config, .file-tree { border: 1px solid var(--border); border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.config__name { cursor: pointer; font-weight: 600; }
.config__count { color: var(--muted); font-weight: normal; font-size: 0.85em; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; vertical-align: middle; margin-left: 4px; }
.badge--default { background: var(--accent); color: white; }
.config__table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9em; }
.config__table th, .config__table td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.config__table th { background: var(--pill-bg); font-weight: 600; }
.files__list { list-style: none; padding-left: 0; margin: 0; }
.file { padding: 4px 0; border-bottom: 1px dotted var(--border); }
.file:last-child { border-bottom: none; }
.file__config { color: var(--muted); font-size: 0.85em; margin-right: 8px; }
.file__path { color: var(--accent); }
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
    """Wrap the rendered sections in page chrome.

    Order: header → tags → configs → files → readme body → footer.
    Configs sit above the README because that's the primary affordance
    on the live HF dataset page (the user picks a subset before
    reading the body).
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HF preview — {_escape(title)}</title>
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


def render_hf_html(doc: HuggingFaceDoc, *, variant: str) -> str:
    """Render the full HF preview HTML.

    Pure function: same ``(doc, variant)`` → byte-identical HTML.
    No I/O, no clock, no random.  Tests rely on this for the
    audit-artefact-sync gate.
    """

    body_parts = [
        _render_header(doc.frontmatter),
        _render_tags(doc.frontmatter),
        _render_configs(doc.frontmatter),
        _render_file_tree(doc.frontmatter, variant=variant),
        _render_readme_body(doc.body),
        _render_footer(doc.frontmatter, variant=variant),
    ]
    return _wrap_html(
        title=str(doc.frontmatter.get("pretty_name", "")),
        body="\n".join(p for p in body_parts if p),
    )


# ---------------------------------------------------------------------------
# Driver — reads inputs, writes HTML, optionally serves
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
    """Return value from :func:`run_preview` — used by tests + CLI."""

    html_path: Path
    cover_path: Path | None


def _resolve_cover_image(release_dir: Path, variant: str) -> Path:
    """Locate the cover image for the variant.

    The HF packager (PR 5.2) copies the cover image into both
    ``release/huggingface/`` and ``release/huggingface-instructor/``
    next to each README.  Prefer the variant-tree copy (closest to
    the artefact the publish PR will upload); fall back to
    ``release_dir`` for the case where the assembler hasn't been
    run yet.
    """

    variant_dir = "huggingface" if variant == "public" else "huggingface-instructor"
    candidates = [
        release_dir / variant_dir / "dataset-cover-image.png",
        release_dir / "dataset-cover-image.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def run_preview(config: PreviewConfig) -> PreviewOutcome:
    """Render the preview HTML, optionally serve it.

    Pre-flight failures (missing README, malformed YAML, missing
    cover image, unknown variant) raise — the CLI converts to rc=2.
    """

    if config.variant not in VALID_VARIANTS:
        raise ValueError(f"unknown --variant {config.variant!r}; expected one of {VALID_VARIANTS}")

    readme_path = config.release_dir / _VARIANT_README_REL[config.variant]
    if not readme_path.is_file():
        raise FileNotFoundError(
            f"HF README not found at {readme_path}; "
            f"regenerate via scripts/package_hf_release.py "
            f"--variant={config.variant} first"
        )
    doc = parse_hf_readme(readme_path.read_text(encoding="utf-8"))

    cover_src = _resolve_cover_image(config.release_dir, config.variant)
    if not cover_src.is_file():
        raise FileNotFoundError(
            f"cover image not found at {cover_src} (looked in "
            f"{config.release_dir}/huggingface{'-instructor' if config.variant == 'instructor' else ''}/ "
            f"and {config.release_dir}/)"
        )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = config.out_dir / "index.html"
    html_path.write_text(render_hf_html(doc, variant=config.variant), encoding="utf-8")

    cover_dst = config.out_dir / "dataset-cover-image.png"
    replace_file(cover_src, cover_dst)

    if config.serve:
        _serve(config.out_dir, config.port, open_browser=config.open_browser)

    return PreviewOutcome(html_path=html_path, cover_path=cover_dst)


def _serve(directory: Path, port: int, *, open_browser: bool) -> None:
    """Start a stdlib HTTP server rooted at ``directory`` and block.

    Same posture as the Kaggle preview — see that module for rationale.
    """

    handler_factory = _make_handler_factory(directory)
    url = f"http://localhost:{port}/"
    print(f"serving {directory} at {url} — Ctrl-C to stop", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    with socketserver.ThreadingTCPServer(("", port), handler_factory) as httpd:
        httpd.serve_forever()


def _make_handler_factory(directory: Path) -> type[http.server.SimpleHTTPRequestHandler]:
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

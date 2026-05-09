"""Tests for ``scripts/preview_kaggle_page.py`` (PR 7.2).

Locks the local Kaggle preview-page contract:

* required field labels appear in the rendered HTML (title, subtitle,
  licence, file count, schema column count) — the four roadmap-mandated
  Kaggle checks;
* every Markdown link in the inlined description resolves to a
  non-404 URL pattern (no ``](../`` survives the rewrite, no
  ``](validation/...)`` lives at a relative path on the upload tree);
* the Kaggle schema table lists every CSV / parquet column declared
  in ``dataset-metadata.json::resources[].schema.fields``;
* the renderer is byte-deterministic and the committed sample at
  ``release/_preview_committed/kaggle.html`` matches a fresh
  regeneration (audit-artefact-sync gate, mirrors PR 5.1 / 5.2 / 7.1);
* the driver exits with rc=2 on missing artefacts (no live HTTP).

No network. No live HTTP. Everything goes through the pure
``render_kaggle_html()`` or the in-process ``run_preview()`` driver.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "preview_kaggle_page.py"
_spec = importlib.util.spec_from_file_location("preview_kaggle_page", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
preview = importlib.util.module_from_spec(_spec)
sys.modules["preview_kaggle_page"] = preview
_spec.loader.exec_module(preview)


_RELEASE_DIR = _REPO_ROOT / "release"
_COMMITTED_METADATA = _RELEASE_DIR / "kaggle" / "dataset-metadata.json"
_COMMITTED_COVER = _RELEASE_DIR / "dataset-cover-image.png"
_COMMITTED_SAMPLE = _REPO_ROOT / "release" / "_preview_committed" / "kaggle.html"
_RELEASE_PRESENT = _COMMITTED_METADATA.exists()

# Allow-listed link patterns the audit-sync test accepts.  Anything else
# in the rendered description is a regression — either the source
# README leaked a relative ``../`` link or the GitHub blob rewrite
# stopped firing.  The whitelist is intentionally narrow.
_LINK_OK_PREFIXES = (
    "https://github.com/leadforge-dev/leadforge",
    "https://huggingface.co/datasets/leadforge",
    "https://example.com",  # used by unit tests only
    "LICENSE",  # sibling-relative, resolves under the upload tree
    "#",  # in-document anchor (footnotes, etc.)
)


# ---------------------------------------------------------------------------
# Pure-renderer fixtures
# ---------------------------------------------------------------------------


def _minimal_metadata() -> dict[str, object]:
    """A minimum-viable metadata payload exercising every renderer
    branch (header pills, file tree, schema table, sources, footer)."""

    return {
        "title": "TestSet: Lead Scoring Mock",
        "id": "testorg/testset-lead-scoring",
        "subtitle": "A mock metadata payload exercising the renderer.",
        "description": (
            "# Mock dataset\n\n"
            "This is a [test link](https://github.com/leadforge-dev/leadforge).\n\n"
            "| Col | Notes |\n|---|---|\n| a | b |\n"
        ),
        "isPrivate": True,
        "licenses": [{"name": "MIT"}],
        "keywords": ["b2b", "tabular"],
        "collaborators": [],
        "expectedUpdateFrequency": "never",
        "userSpecifiedSources": [
            {"title": "source repo", "url": "https://github.com/leadforge-dev/leadforge"},
        ],
        "image": "dataset-cover-image.png",
        "resources": [
            {
                "path": "intro/lead_scoring.csv",
                "description": "Intro flat CSV.",
                "schema": {
                    "fields": [
                        {"name": "lead_id", "type": "string", "description": "Opaque id."},
                        {"name": "label", "type": "boolean", "description": "Outcome."},
                    ]
                },
            },
            {
                "path": "intro/manifest.json",
                "description": "Provenance manifest (no schema).",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Required field labels (one of the four roadmap-mandated Kaggle checks)
# ---------------------------------------------------------------------------


def test_render_includes_title_subtitle_id_and_license() -> None:
    html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    assert "TestSet: Lead Scoring Mock" in html
    assert "A mock metadata payload exercising the renderer." in html
    assert "testorg/testset-lead-scoring" in html
    assert "License: MIT" in html
    assert "Updates: never" in html


def test_render_does_not_include_visibility_pill() -> None:
    """Kaggle's public page does NOT display ``isPrivate``; rendering
    a ``Visibility:`` pill in the preview would misrepresent what
    public viewers see (folded back in self-review pass 3)."""

    private_html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    public_html = preview.render_kaggle_html(
        {**_minimal_metadata(), "isPrivate": False},
        "dataset-cover-image.png",
    )
    for html in (private_html, public_html):
        assert "Visibility:" not in html
        assert "pill--visibility" not in html


def test_render_file_tree_lists_every_resource_path() -> None:
    """File tree shows every resource path declared in metadata."""

    html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    assert "intro/lead_scoring.csv" in html
    assert "intro/manifest.json" in html
    assert "(2 total)" in html  # file count appears in the heading


def test_render_schema_table_lists_every_column() -> None:
    """The schema table lists every column from every tabular resource."""

    html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    assert "<code>lead_id</code>" in html
    assert "<code>label</code>" in html
    assert "Opaque id." in html
    assert "(2 columns)" in html  # per-table column count
    # Resources without a schema (manifest.json) do not appear in the table.
    assert "(2 columns across 1 tabular files)" in html


def test_render_keywords_appear_as_chips_in_footer() -> None:
    html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    assert '<span class="chip">b2b</span>' in html
    assert '<span class="chip">tabular</span>' in html


def test_render_sources_block_renders_when_present() -> None:
    html = preview.render_kaggle_html(_minimal_metadata(), "dataset-cover-image.png")
    assert "source repo" in html
    assert 'href="https://github.com/leadforge-dev/leadforge"' in html


def test_render_sources_block_omitted_when_empty() -> None:
    metadata = {**_minimal_metadata(), "userSpecifiedSources": []}
    html = preview.render_kaggle_html(metadata, "dataset-cover-image.png")
    assert '<h2 class="section__heading">Sources</h2>' not in html


def test_render_escapes_html_in_field_values() -> None:
    """User-controlled strings are HTML-escaped — guards against XSS
    if a recipe ever surfaces ``<script>`` in a description."""

    metadata = {**_minimal_metadata(), "title": "evil <script>alert(1)</script>"}
    html = preview.render_kaggle_html(metadata, "dataset-cover-image.png")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Schema-fields exhaustiveness (audit-style, against committed metadata)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_PRESENT, reason="release bundles not present")
def test_committed_metadata_schema_is_fully_listed() -> None:
    """The roadmap-mandated check: the Kaggle schema table lists every
    CSV / parquet column declared in dataset-metadata.json."""

    metadata = json.loads(_COMMITTED_METADATA.read_text(encoding="utf-8"))
    html = preview.render_kaggle_html(metadata, metadata["image"])
    for resource in metadata["resources"]:
        schema = resource.get("schema")
        if not schema:
            continue
        for field in schema["fields"]:
            name = field["name"]
            # Every column name appears as a ``<code>`` cell in the table.
            assert f"<code>{name}</code>" in html, (
                f"schema column {name!r} from {resource['path']!r} not rendered"
            )


# ---------------------------------------------------------------------------
# Markdown link resolution (the leakage / link-rewrite regression guard)
# ---------------------------------------------------------------------------

#: Match ``href="X"`` in the rendered HTML — markdown-it-py emits
#: double-quoted hrefs.  Inline ``](X)`` would slip past this and stay
#: as escaped text rather than a real link, so we also assert against
#: those separately.
_HREF_RE = re.compile(r'href="([^"]+)"')


@pytest.mark.skipif(not _RELEASE_PRESENT, reason="release bundles not present")
def test_committed_metadata_description_has_no_unrewritten_relative_links() -> None:
    """Source-side regression guard.

    The Kaggle packager runs ``rewrite_release_links()`` on the
    inlined README; if a future README adds a ``](../foo)`` link or a
    ``](validation/...)`` link AND someone updates the rewriter to
    miss it, the rendered description would carry a 404-bound href.
    Catch it here, before the publish runbook.
    """

    metadata = json.loads(_COMMITTED_METADATA.read_text(encoding="utf-8"))
    description = metadata["description"]
    # Source-form check: no parent-relative or validation-relative
    # markdown links remain in the inlined description.
    assert "](../" not in description, (
        "unrewritten parent-relative markdown link in inlined description"
    )
    assert "](validation/" not in description, (
        "unrewritten validation-relative markdown link in inlined description"
    )


@pytest.mark.skipif(not _RELEASE_PRESENT, reason="release bundles not present")
def test_committed_metadata_rendered_links_point_at_known_targets() -> None:
    """Every rendered href in the description body points at one of:

    * a GitHub blob URL (the rewriter's output);
    * a known external service (huggingface.co/datasets/leadforge);
    * a sibling-relative path that resolves under the upload tree
      (LICENSE), or an in-document anchor (#footnote-1 etc.).

    Anything else is a 404 risk on the live page.
    """

    metadata = json.loads(_COMMITTED_METADATA.read_text(encoding="utf-8"))
    html = preview.render_kaggle_html(metadata, metadata["image"])
    bad: list[str] = []
    for href in _HREF_RE.findall(html):
        if any(href.startswith(prefix) for prefix in _LINK_OK_PREFIXES):
            continue
        bad.append(href)
    assert not bad, (
        f"rendered HTML carries non-allowlisted hrefs that would 404 on Kaggle: {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# Determinism + audit-artefact-sync (against committed sample)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_PRESENT, reason="release bundles not present")
def test_render_is_byte_deterministic() -> None:
    """Two back-to-back renders against the same metadata produce
    byte-identical HTML — the determinism contract this script relies
    on for the sync test below."""

    metadata = json.loads(_COMMITTED_METADATA.read_text(encoding="utf-8"))
    a = preview.render_kaggle_html(metadata, metadata["image"])
    b = preview.render_kaggle_html(metadata, metadata["image"])
    assert a == b


@pytest.mark.skipif(
    not (_RELEASE_PRESENT and _COMMITTED_SAMPLE.exists()),
    reason="release bundles or committed preview sample missing",
)
def test_committed_sample_matches_fresh_regeneration() -> None:
    """The audit-artefact-sync gate.

    A fresh render of the committed Kaggle metadata must equal
    ``release/_preview_committed/kaggle.html`` byte-for-byte.  If
    this fails, either the renderer changed or the upstream metadata
    drifted without re-running the preview script.  Regenerate via::

        python scripts/preview_kaggle_page.py --no-serve
        cp release/_preview/kaggle/index.html release/_preview_committed/kaggle.html
    """

    metadata = json.loads(_COMMITTED_METADATA.read_text(encoding="utf-8"))
    fresh = preview.render_kaggle_html(metadata, metadata["image"])
    committed = _COMMITTED_SAMPLE.read_text(encoding="utf-8")
    assert fresh == committed


# ---------------------------------------------------------------------------
# Driver — pre-flight error paths (no server start)
# ---------------------------------------------------------------------------


def test_run_preview_raises_on_missing_metadata(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    fake_release.mkdir()
    config = preview.PreviewConfig(
        release_dir=fake_release,
        out_dir=tmp_path / "preview",
        port=8765,
        open_browser=False,
        serve=False,
    )
    with pytest.raises(FileNotFoundError, match="dataset metadata not found"):
        preview.run_preview(config)


def test_run_preview_raises_on_malformed_metadata(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    (fake_release / "kaggle").mkdir(parents=True)
    (fake_release / "kaggle" / "dataset-metadata.json").write_text(
        '"not-an-object"', encoding="utf-8"
    )
    config = preview.PreviewConfig(
        release_dir=fake_release,
        out_dir=tmp_path / "preview",
        port=8765,
        open_browser=False,
        serve=False,
    )
    with pytest.raises(ValueError, match="not a JSON object"):
        preview.run_preview(config)


def test_run_preview_raises_on_missing_cover_image(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    (fake_release / "kaggle").mkdir(parents=True)
    (fake_release / "kaggle" / "dataset-metadata.json").write_text(
        json.dumps({"image": "missing.png", "resources": []}), encoding="utf-8"
    )
    config = preview.PreviewConfig(
        release_dir=fake_release,
        out_dir=tmp_path / "preview",
        port=8765,
        open_browser=False,
        serve=False,
    )
    with pytest.raises(FileNotFoundError, match="cover image"):
        preview.run_preview(config)


def test_run_preview_writes_html_and_copies_cover(tmp_path: Path) -> None:
    """End-to-end no-serve path: HTML lands at ``out_dir/index.html``;
    cover image is copied as a real file (not a symlink)."""

    fake_release = tmp_path / "release"
    (fake_release / "kaggle").mkdir(parents=True)
    cover_src = fake_release / "kaggle" / "dataset-cover-image.png"
    cover_src.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (fake_release / "kaggle" / "dataset-metadata.json").write_text(
        json.dumps(_minimal_metadata()), encoding="utf-8"
    )
    out_dir = tmp_path / "preview"
    outcome = preview.run_preview(
        preview.PreviewConfig(
            release_dir=fake_release,
            out_dir=out_dir,
            port=8765,
            open_browser=False,
            serve=False,
        )
    )
    assert outcome.html_path == out_dir / "index.html"
    assert outcome.html_path.is_file()
    assert outcome.cover_path.is_file()
    assert not outcome.cover_path.is_symlink()
    # The HTML references the cover image by sibling-relative name.
    assert 'src="dataset-cover-image.png"' in outcome.html_path.read_text(encoding="utf-8")


def test_main_returns_2_on_missing_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = preview.main(
        [
            "--release-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(tmp_path / "preview"),
            "--no-serve",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "dataset metadata not found" in captured.err


def test_parse_args_defaults() -> None:
    """``parse_args`` is a free function so tests can exercise the
    flag wiring without invoking the full driver."""

    args = preview.parse_args(["--no-serve"])
    assert args.release_dir == preview.DEFAULT_RELEASE_DIR
    assert args.out_dir == preview.DEFAULT_OUT_DIR
    assert args.port == preview.DEFAULT_PORT
    assert args.open_browser is False
    assert args.no_serve is True


def test_tier_of_extracts_leading_path_segment() -> None:
    """``_tier_of`` is the load-bearing helper that buckets resources
    by tier in the file tree — pin its contract."""

    assert preview._tier_of("intro/lead_scoring.csv") == "intro"
    assert preview._tier_of("intermediate/tasks/converted/train.parquet") == "intermediate"
    assert preview._tier_of("toplevel.json") == ""


# ---------------------------------------------------------------------------
# Server smoke test — covers _preview_common.make_server / serve glue
# (folded back from self-review pass 3 — _serve was previously untested)
# ---------------------------------------------------------------------------


def test_make_server_binds_and_serves_index(tmp_path: Path) -> None:
    """Stand the server up on port 0 (kernel-picked), GET ``/``,
    assert 200 + body shape, shut down cleanly.

    Covers every path inside ``_preview_common.make_server`` and
    ``_make_handler_factory`` (handler subclass with ``directory=``,
    ``ThreadingHTTPServer`` instantiation, address-reuse posture,
    static-file serving).  ``serve`` itself is the blocking caller
    that wraps this and is exercised manually.
    """

    import threading
    import urllib.request

    import _preview_common  # noqa: PLC0415 — local import for the smoke test

    (tmp_path / "index.html").write_text(
        "<html><body><h1>preview-smoke-token</h1></body></html>", encoding="utf-8"
    )
    httpd = _preview_common.make_server(tmp_path, port=0)
    bound_port = httpd.server_address[1]
    assert bound_port > 0
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://localhost:{bound_port}/", timeout=5) as resp:  # noqa: S310 — localhost smoke
            assert resp.status == 200
            body = resp.read().decode("utf-8")
        assert "preview-smoke-token" in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()

"""Tests for ``scripts/preview_hf_page.py`` (PR 7.2).

Locks the local Hugging Face preview-page contract:

* required field labels appear in the rendered HTML (pretty_name,
  licence, configs, tags) — the four roadmap-mandated HF checks;
* every Markdown link in the README body resolves to a non-404 URL
  pattern (no ``](../`` survives, no ``](validation/...)``);
* every ``configs[]`` block in the YAML round-trips through to the
  rendered configs dropdown;
* the renderer is byte-deterministic and the committed samples at
  ``release/_preview_committed/huggingface_{public,instructor}.html``
  match a fresh regeneration (audit-artefact-sync gate);
* the ``--variant`` flag wires up the right input README, output
  dir, and footer label;
* the driver exits with rc=2 on missing artefacts (no live HTTP).

No network. No live HTTP.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "preview_hf_page.py"
_spec = importlib.util.spec_from_file_location("preview_hf_page", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
preview = importlib.util.module_from_spec(_spec)
sys.modules["preview_hf_page"] = preview
_spec.loader.exec_module(preview)


_RELEASE_DIR = _REPO_ROOT / "release"
_PUBLIC_README = _RELEASE_DIR / "huggingface" / "README.md"
_INSTRUCTOR_README = _RELEASE_DIR / "huggingface-instructor" / "README.md"
_PUBLIC_SAMPLE = _REPO_ROOT / "release" / "_preview_committed" / "huggingface_public.html"
_INSTRUCTOR_SAMPLE = _REPO_ROOT / "release" / "_preview_committed" / "huggingface_instructor.html"
_PUBLIC_PRESENT = _PUBLIC_README.exists()
_INSTRUCTOR_PRESENT = _INSTRUCTOR_README.exists()

# Same allow-list rule as the Kaggle preview tests — see
# ``test_preview_kaggle_page.py`` for rationale.
_LINK_OK_PREFIXES = (
    "https://github.com/leadforge-dev/leadforge",
    "https://huggingface.co/datasets/leadforge",
    "https://example.com",
    "LICENSE",
    "#",
)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


def test_parse_hf_readme_extracts_yaml_and_body() -> None:
    text = "---\npretty_name: Test\nlicense: mit\n---\n# Body\n\nText.\n"
    doc = preview.parse_hf_readme(text)
    assert doc.frontmatter == {"pretty_name": "Test", "license": "mit"}
    assert doc.body == "# Body\n\nText.\n"


def test_parse_hf_readme_rejects_missing_frontmatter() -> None:
    with pytest.raises(ValueError, match="missing a YAML frontmatter"):
        preview.parse_hf_readme("# No frontmatter here\n")


def test_parse_hf_readme_rejects_non_mapping_frontmatter() -> None:
    with pytest.raises(ValueError, match="not a YAML mapping"):
        preview.parse_hf_readme("---\n- 1\n- 2\n---\nbody\n")


# ---------------------------------------------------------------------------
# Pure-renderer fixtures
# ---------------------------------------------------------------------------


def _minimal_doc() -> preview.HuggingFaceDoc:
    """A minimum-viable HF doc exercising every renderer branch."""

    return preview.HuggingFaceDoc(
        frontmatter={
            "pretty_name": "TestSet: Mock HF Dataset",
            "license": "mit",
            "language": ["en"],
            "task_categories": ["tabular-classification"],
            "size_categories": ["1K<n<10K"],
            "tags": ["b2b", "tabular"],
            "configs": [
                {
                    "config_name": "intro",
                    "data_files": [
                        {"split": "train", "path": "intro/train.parquet"},
                        {"split": "validation", "path": "intro/valid.parquet"},
                        {"split": "test", "path": "intro/test.parquet"},
                    ],
                },
                {
                    "config_name": "intermediate",
                    "default": True,
                    "data_files": [
                        {"split": "train", "path": "intermediate/train.parquet"},
                    ],
                },
            ],
        },
        body="# Mock\n\nA [link](https://github.com/leadforge-dev/leadforge).\n",
    )


# ---------------------------------------------------------------------------
# Required field labels (the four roadmap-mandated HF checks)
# ---------------------------------------------------------------------------


def test_render_includes_pretty_name_and_license() -> None:
    html = preview.render_hf_html(_minimal_doc(), variant="public")
    assert "TestSet: Mock HF Dataset" in html
    assert "License: mit" in html
    assert "Task: tabular-classification" in html
    assert "Size: 1K&lt;n&lt;10K" in html  # HTML-escaped
    assert "Language: en" in html


def test_render_emits_one_chip_per_tag() -> None:
    html = preview.render_hf_html(_minimal_doc(), variant="public")
    assert '<span class="chip">b2b</span>' in html
    assert '<span class="chip">tabular</span>' in html


def test_render_configs_dropdown_lists_every_config() -> None:
    """The roadmap-mandated round-trip: every configs[] block from the
    YAML appears in the rendered dropdown."""

    html = preview.render_hf_html(_minimal_doc(), variant="public")
    assert "<code>intro</code>" in html
    assert "<code>intermediate</code>" in html
    assert "(2 configs)" in html


def test_render_configs_flags_the_default() -> None:
    html = preview.render_hf_html(_minimal_doc(), variant="public")
    # The default badge appears next to the default config.
    assert '<span class="badge badge--default">default</span>' in html
    # Exactly one badge instance — no other config gets it.
    assert html.count("badge badge--default") == 1


def test_render_data_files_appear_under_each_config() -> None:
    html = preview.render_hf_html(_minimal_doc(), variant="public")
    assert "intro/train.parquet" in html
    assert "intro/valid.parquet" in html
    assert "intro/test.parquet" in html
    assert "intermediate/train.parquet" in html


def test_render_does_not_emit_files_declared_section() -> None:
    """Real HF doesn't surface a "files declared in YAML" section —
    showing one would be an internal-concept leak that omits the bulk
    of the actual upload tree (manifest.json, tables/*.parquet, etc.).
    The configs dropdown already lists every YAML-declared path; a
    parallel files section would be misleading duplicate noise.
    Folded back from self-review pass 3.
    """

    html = preview.render_hf_html(_minimal_doc(), variant="public")
    assert "Files declared" not in html
    assert "files / variant:" not in html  # legacy heading text
    assert 'class="files"' not in html


def test_render_includes_variant_in_footer() -> None:
    public = preview.render_hf_html(_minimal_doc(), variant="public")
    instructor = preview.render_hf_html(_minimal_doc(), variant="instructor")
    assert "Variant: <code>public</code>" in public
    assert "Variant: <code>instructor</code>" in instructor
    # Variant differences are localised to the footer; the rest of
    # the output is identical between variants.  Replace via the
    # full ``Variant: <code>X</code>`` marker (not the bare word)
    # so this assertion does not match "public" inside "publication"
    # in the footer note (regression caught + folded back during
    # self-review pass 3 reframing).
    public_normalised = public.replace("Variant: <code>public</code>", "Variant: <code>X</code>")
    instructor_normalised = instructor.replace(
        "Variant: <code>instructor</code>", "Variant: <code>X</code>"
    )
    assert public_normalised == instructor_normalised


def test_render_handles_no_configs_gracefully() -> None:
    """Edge case: a malformed dataset card with no ``configs`` should
    still render rather than crash."""

    doc = preview.HuggingFaceDoc(
        frontmatter={"pretty_name": "X", "license": "mit"},
        body="body\n",
    )
    html = preview.render_hf_html(doc, variant="public")
    assert "No configs declared." in html


def test_render_escapes_html_in_field_values() -> None:
    """Same XSS-safety guard as the Kaggle preview."""

    doc = preview.HuggingFaceDoc(
        frontmatter={"pretty_name": "<script>x</script>", "license": "mit"},
        body="body\n",
    )
    html = preview.render_hf_html(doc, variant="public")
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html


# ---------------------------------------------------------------------------
# Markdown link resolution (the leakage / link-rewrite regression guard)
# ---------------------------------------------------------------------------

_HREF_RE = re.compile(r'href="([^"]+)"')


@pytest.mark.skipif(not _PUBLIC_PRESENT, reason="public README not present")
def test_public_readme_has_no_unrewritten_relative_links() -> None:
    """Same source-side regression guard as the Kaggle preview."""

    body = _PUBLIC_README.read_text(encoding="utf-8")
    assert "](../" not in body, "unrewritten parent-relative link in public README"
    assert "](validation/" not in body, "unrewritten validation-relative link in public README"


@pytest.mark.skipif(not _PUBLIC_PRESENT, reason="public README not present")
def test_public_rendered_links_point_at_known_targets() -> None:
    """Every rendered href in the public preview points at one of the
    allow-listed prefixes — anything else would 404 on the live HF
    page."""

    doc = preview.parse_hf_readme(_PUBLIC_README.read_text(encoding="utf-8"))
    html = preview.render_hf_html(doc, variant="public")
    bad: list[str] = []
    for href in _HREF_RE.findall(html):
        if any(href.startswith(prefix) for prefix in _LINK_OK_PREFIXES):
            continue
        bad.append(href)
    assert not bad, f"non-allowlisted hrefs would 404 on HF: {bad[:5]}"


@pytest.mark.skipif(not _INSTRUCTOR_PRESENT, reason="instructor README not present")
def test_instructor_rendered_links_point_at_known_targets() -> None:
    doc = preview.parse_hf_readme(_INSTRUCTOR_README.read_text(encoding="utf-8"))
    html = preview.render_hf_html(doc, variant="instructor")
    bad: list[str] = []
    for href in _HREF_RE.findall(html):
        if any(href.startswith(prefix) for prefix in _LINK_OK_PREFIXES):
            continue
        bad.append(href)
    assert not bad, f"non-allowlisted hrefs would 404 on HF: {bad[:5]}"


@pytest.mark.skipif(not _PUBLIC_PRESENT, reason="public README not present")
def test_public_yaml_configs_round_trip_into_html() -> None:
    """Every ``configs[].config_name`` declared in the YAML appears in
    the rendered HTML — the round-trip the roadmap mandates."""

    doc = preview.parse_hf_readme(_PUBLIC_README.read_text(encoding="utf-8"))
    html = preview.render_hf_html(doc, variant="public")
    for config in doc.frontmatter["configs"]:
        name = config["config_name"]
        assert f"<code>{name}</code>" in html, (
            f"config {name!r} declared in YAML but missing from rendered HTML"
        )


# ---------------------------------------------------------------------------
# Determinism + audit-artefact-sync (against committed samples)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _PUBLIC_PRESENT, reason="public README not present")
def test_render_is_byte_deterministic() -> None:
    doc = preview.parse_hf_readme(_PUBLIC_README.read_text(encoding="utf-8"))
    a = preview.render_hf_html(doc, variant="public")
    b = preview.render_hf_html(doc, variant="public")
    assert a == b


@pytest.mark.skipif(
    not (_PUBLIC_PRESENT and _PUBLIC_SAMPLE.exists()),
    reason="public README or committed sample missing",
)
def test_committed_public_sample_matches_fresh_regeneration() -> None:
    """Audit-sync gate for the public variant.

    Regenerate via::

        python scripts/preview_hf_page.py --no-serve
        cp release/_preview/huggingface/index.html \\
            release/_preview_committed/huggingface_public.html
    """

    doc = preview.parse_hf_readme(_PUBLIC_README.read_text(encoding="utf-8"))
    fresh = preview.render_hf_html(doc, variant="public")
    committed = _PUBLIC_SAMPLE.read_text(encoding="utf-8")
    assert fresh == committed


@pytest.mark.skipif(
    not (_INSTRUCTOR_PRESENT and _INSTRUCTOR_SAMPLE.exists()),
    reason="instructor README or committed sample missing",
)
def test_committed_instructor_sample_matches_fresh_regeneration() -> None:
    """Audit-sync gate for the instructor variant."""

    doc = preview.parse_hf_readme(_INSTRUCTOR_README.read_text(encoding="utf-8"))
    fresh = preview.render_hf_html(doc, variant="instructor")
    committed = _INSTRUCTOR_SAMPLE.read_text(encoding="utf-8")
    assert fresh == committed


# ---------------------------------------------------------------------------
# Driver — pre-flight error paths (no server start)
# ---------------------------------------------------------------------------


def _make_config(release_dir: Path, out_dir: Path, *, variant: str = "public") -> object:
    return preview.PreviewConfig(
        release_dir=release_dir,
        out_dir=out_dir,
        port=8766,
        variant=variant,
        open_browser=False,
        serve=False,
    )


def test_run_preview_raises_on_unknown_variant(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    fake_release.mkdir()
    config = _make_config(fake_release, tmp_path / "preview", variant="bogus")
    with pytest.raises(ValueError, match="unknown --variant"):
        preview.run_preview(config)  # type: ignore[arg-type]


def test_run_preview_raises_on_missing_readme(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    fake_release.mkdir()
    config = _make_config(fake_release, tmp_path / "preview")
    with pytest.raises(FileNotFoundError, match="HF README not found"):
        preview.run_preview(config)  # type: ignore[arg-type]


def test_run_preview_raises_on_malformed_readme(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    (fake_release / "huggingface").mkdir(parents=True)
    (fake_release / "huggingface" / "README.md").write_text("# No frontmatter\n", encoding="utf-8")
    config = _make_config(fake_release, tmp_path / "preview")
    with pytest.raises(ValueError, match="missing a YAML frontmatter"):
        preview.run_preview(config)  # type: ignore[arg-type]


def test_run_preview_raises_on_missing_cover(tmp_path: Path) -> None:
    fake_release = tmp_path / "release"
    (fake_release / "huggingface").mkdir(parents=True)
    (fake_release / "huggingface" / "README.md").write_text(
        "---\npretty_name: T\nlicense: mit\n---\nbody\n", encoding="utf-8"
    )
    config = _make_config(fake_release, tmp_path / "preview")
    with pytest.raises(FileNotFoundError, match="cover image"):
        preview.run_preview(config)  # type: ignore[arg-type]


def test_run_preview_writes_html_and_copies_cover(tmp_path: Path) -> None:
    """End-to-end no-serve: HTML lands at out_dir/index.html and the
    cover image is copied as a real file."""

    fake_release = tmp_path / "release"
    (fake_release / "huggingface").mkdir(parents=True)
    (fake_release / "huggingface" / "README.md").write_text(
        "---\npretty_name: T\nlicense: mit\n---\nbody\n", encoding="utf-8"
    )
    cover = fake_release / "huggingface" / "dataset-cover-image.png"
    cover.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out_dir = tmp_path / "preview"
    outcome = preview.run_preview(_make_config(fake_release, out_dir))  # type: ignore[arg-type]
    assert outcome.html_path == out_dir / "index.html"
    assert outcome.html_path.is_file()
    assert outcome.cover_path.is_file()
    assert not outcome.cover_path.is_symlink()


def test_run_preview_instructor_variant_uses_companion_paths(tmp_path: Path) -> None:
    """``--variant=instructor`` reads the companion README and writes
    to the companion-flavoured out_dir."""

    fake_release = tmp_path / "release"
    (fake_release / "huggingface-instructor").mkdir(parents=True)
    (fake_release / "huggingface-instructor" / "README.md").write_text(
        "---\npretty_name: I\nlicense: mit\n---\nbody\n", encoding="utf-8"
    )
    cover = fake_release / "huggingface-instructor" / "dataset-cover-image.png"
    cover.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    out_dir = tmp_path / "preview-instructor"
    outcome = preview.run_preview(
        _make_config(fake_release, out_dir, variant="instructor")  # type: ignore[arg-type]
    )
    assert outcome.html_path.is_file()
    assert "Variant: <code>instructor</code>" in outcome.html_path.read_text(encoding="utf-8")


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
    assert "HF README not found" in captured.err


def test_main_default_out_dir_depends_on_variant(tmp_path: Path) -> None:
    """``--out-dir`` defaults to the variant-flavoured location."""

    args_public = preview.parse_args(["--no-serve"])
    args_instructor = preview.parse_args(["--no-serve", "--variant=instructor"])
    assert args_public.out_dir is None  # resolved in main()
    assert args_instructor.out_dir is None
    # Sanity: ``main`` resolves the default per variant.
    rc = preview.main(
        [
            "--release-dir",
            str(tmp_path / "missing"),
            "--variant=instructor",
            "--no-serve",
        ]
    )
    assert rc == 2  # missing README; we just want to confirm CLI parsing didn't crash


def test_parse_args_defaults() -> None:
    args = preview.parse_args(["--no-serve"])
    assert args.release_dir == preview.DEFAULT_RELEASE_DIR
    assert args.out_dir is None  # variant-resolved in main()
    assert args.port == preview.DEFAULT_PORT
    assert args.variant == "public"
    assert args.open_browser is False
    assert args.no_serve is True


def test_parse_args_rejects_unknown_variant() -> None:
    with pytest.raises(SystemExit):
        preview.parse_args(["--variant=bogus"])

"""Tests for ``scripts/package_kaggle_release.py``.

Locks the Phase 5 Kaggle packaging contract:

* every Kaggle field constraint surfaced in chatgpt v2 §19 (G11.1)
* the cover-image dimension floor (G11.2)
* the README link-rewriting that lets the published dataset card on
  Kaggle keep working ``../`` links (rewritten to GitHub blob URLs)
  and a directory diagram that reflects the upload layout, plus a
  guard that the source ``SOURCE_TREE_BLOCK`` (in ``_release_common``)
  is still present verbatim in the README (silent-failure trap)
* the assembled upload tree resolves every declared resource path
  (so ``kaggle datasets create`` can find each file)
* the safety net that refuses to assemble into ``cwd`` /
  ``release_dir`` / its parent
* byte-equality + content-shape between the committed
  ``release/kaggle/dataset-metadata.json`` and a fresh regeneration
  (audit-artifact-sync pattern from PR 4.1)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "package_kaggle_release.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("package_kaggle_release", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
packager = importlib.util.module_from_spec(_spec)
sys.modules["package_kaggle_release"] = packager
_spec.loader.exec_module(packager)


_RELEASE_DIR = _REPO_ROOT / "release"
_RELEASE_BUNDLES_PRESENT = (_RELEASE_DIR / "intro" / "manifest.json").exists()
_COMMITTED_METADATA = _REPO_ROOT / "release" / "kaggle" / "dataset-metadata.json"
_COMMITTED_COVER = _REPO_ROOT / "release" / "dataset-cover-image.png"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_metadata() -> packager.DatasetMetadata:
    """A minimum-viable ``DatasetMetadata`` that should validate cleanly."""

    return packager.DatasetMetadata(
        title=packager.DEFAULT_TITLE,
        id=f"{packager.DEFAULT_USER_SLUG}/{packager.DEFAULT_DATASET_SLUG}",
        subtitle=packager.DEFAULT_SUBTITLE,
        description="Synthetic CRM lead-scoring dataset.",
        isPrivate=True,
        licenses=(packager.LicenseSpec(name=packager.DEFAULT_LICENSE_NAME),),
        keywords=packager.DEFAULT_KEYWORDS,
        collaborators=(),
        expectedUpdateFrequency=packager.DEFAULT_UPDATE_FREQUENCY,
        userSpecifiedSources=packager.DEFAULT_USER_SOURCES,
        image="dataset-cover-image.png",
        resources=(
            packager.Resource(
                path="intro/lead_scoring.csv",
                description="Intro flat CSV.",
                schema=packager.ResourceSchema(
                    fields=(
                        packager.FieldDescriptor(name="lead_id", type="string", description="ID."),
                    )
                ),
            ),
        ),
    )


def _make_valid_cover(path: Path) -> None:
    """Write a minimum-Kaggle-acceptable cover image at ``path``."""

    Image.new("RGB", (1280, 640), (0, 0, 0)).save(path)


# ---------------------------------------------------------------------------
# Field-constraint validation (G11.1)
# ---------------------------------------------------------------------------


def test_validate_metadata_accepts_canonical_v1_metadata() -> None:
    assert packager.validate_metadata(_minimal_metadata()) == []


def test_validate_metadata_reports_every_constraint_violation() -> None:
    """One bad metadata payload triggers every field check at once."""

    bad = packager.DatasetMetadata(
        title="Tiny",  # < 6 chars
        id="LeadForge Bad Slug!",  # missing '/' + invalid chars
        subtitle="short",  # < 20 chars
        description="x",
        isPrivate=True,
        licenses=(  # two entries, must be exactly one
            packager.LicenseSpec(name="MIT"),
            packager.LicenseSpec(name="Apache-2.0"),
        ),
        keywords=("synthetic-data",),
        collaborators=(),
        expectedUpdateFrequency="sometimes",  # not approved
        userSpecifiedSources=(),
        image="cover.bmp",  # disallowed extension
        resources=(),  # empty resource list
    )

    errors = packager.validate_metadata(bad)
    fields = {e.field for e in errors}
    assert "title" in fields
    assert "subtitle" in fields
    assert "id" in fields
    assert "licenses" in fields
    assert "expectedUpdateFrequency" in fields
    assert "image" in fields
    assert "resources" in fields


def test_validate_id_requires_user_slash_slug_format() -> None:
    """Slug-only ids are rejected — Kaggle's schema is ``user/slug``.

    Mirrors the design call recorded in the PR write-up: PR 7.2's
    publish script should not have to splice in a username at upload
    time.
    """

    slug_only = packager._validate_id("leadforge-lead-scoring-v1")
    assert any(e.field == "id" and "missing 'user/'" in e.message for e in slug_only)

    well_formed = packager._validate_id("leadforge/leadforge-lead-scoring-v1")
    assert well_formed == []

    invalid_slug = packager._validate_id("leadforge/Bad Slug!")
    assert any(e.field == "id (slug)" for e in invalid_slug)


def test_validate_metadata_flags_schema_fields_without_name_or_type() -> None:
    """Schema fields must declare both name and type to satisfy G11.1."""

    bad = _minimal_metadata()
    broken = packager.Resource(
        path="x.csv",
        description="x",
        schema=packager.ResourceSchema(
            fields=(packager.FieldDescriptor(name="", type="string"),),
        ),
    )
    bad = packager.DatasetMetadata(**{**bad.__dict__, "resources": (broken,)})
    errors = packager.validate_metadata(bad)
    assert any("name and type" in e.message for e in errors)


# ---------------------------------------------------------------------------
# Cover image (G11.2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _COMMITTED_COVER.exists(), reason="committed cover image not present")
def test_validate_cover_image_passes_for_committed_asset() -> None:
    assert packager.validate_cover_image(_COMMITTED_COVER) == []


def test_validate_cover_image_rejects_too_small_image(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.png"
    Image.new("RGB", (100, 50), (0, 0, 0)).save(tiny)
    errors = packager.validate_cover_image(tiny)
    assert errors
    assert errors[0].field == "cover_image"
    assert "below minimum" in errors[0].message


def test_validate_cover_image_reports_missing_file(tmp_path: Path) -> None:
    errors = packager.validate_cover_image(tmp_path / "no-such.png")
    assert errors
    assert errors[0].field == "cover_image"


# ---------------------------------------------------------------------------
# Schema fields — derive-from-source contract
#
# The flat-CSV schema is built by iterating the CSV header, so column-
# order parity with the CSV is a construction-time invariant.  The
# parquet schema comes straight from ``pq.read_schema``, same story.
# Re-checking either via a separate validator is tautological — the
# real coverage is the audit-artifact-sync test below
# (``test_committed_kaggle_metadata_matches_fresh_regeneration``),
# which fails the moment any tier's CSV header or parquet schema
# drifts without a matching metadata regeneration.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# README rewriting + description content
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_kaggle_readme_text_rewrites_links_and_tree_diagram() -> None:
    readme = (_RELEASE_DIR / "README.md").read_text(encoding="utf-8")
    rewritten = packager._kaggle_readme_text(readme)

    # Source-repo tree → upload tree.
    assert "intermediate_instructor/" not in rewritten
    assert "notebooks/01_baseline_lead_scoring.ipynb" not in rewritten
    assert "dataset-metadata.json             # Kaggle" in rewritten

    # Relative ../ links rewritten to GitHub blob URLs.
    assert "](../" not in rewritten
    assert packager.GITHUB_BLOB_BASE in rewritten

    # The validation-report link (which lives under release/, not under
    # the upload dir) must point at GitHub.
    assert "](validation/validation_report.md)" not in rewritten
    assert f"]({packager.GITHUB_BLOB_BASE}/release/validation/validation_report.md)" in rewritten


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_source_tree_block_is_present_in_release_readme() -> None:
    """Silent-failure guard.

    ``_kaggle_readme_text`` substitutes ``SOURCE_TREE_BLOCK`` →
    ``KAGGLE_UPLOAD_TREE_BLOCK`` via plain string replace.  If anyone
    tweaks the README's tree diagram by even one whitespace
    character, the substitution silently no-ops and the published
    Kaggle dataset card carries the source-repo tree.  This guard
    fires loudly the moment the constants drift apart.
    """

    readme = (_RELEASE_DIR / "README.md").read_text(encoding="utf-8")
    assert packager.SOURCE_TREE_BLOCK in readme, (
        "scripts/_release_common.py SOURCE_TREE_BLOCK no longer matches "
        "the tree diagram in release/README.md — reconcile the two before "
        "the next release-metadata regeneration."
    )


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_validate_readme_substitution_flags_drift(tmp_path: Path) -> None:
    """``validate_readme_substitution`` (now in ``_release_common``) is
    wired into the run-time validator, not just the static guard above."""

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    (fake_release / "README.md").write_text("# Some unrelated README\n", encoding="utf-8")
    errors = packager.validate_readme_substitution(fake_release, packager_name="Kaggle")
    assert errors
    assert errors[0].field == "release/README.md"
    assert "SOURCE_TREE_BLOCK" in errors[0].message

    # Sanity: the real release README does NOT trigger the validator.
    assert packager.validate_readme_substitution(_RELEASE_DIR, packager_name="Kaggle") == []


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_assembled_upload_dir_writes_rewritten_readme_copy(tmp_path: Path) -> None:
    """The README inside the upload tree is a real file with the
    rewrites — Kaggle reads this verbatim on the dataset page."""

    kaggle_dir = tmp_path / "kaggle"
    cover_image = tmp_path / "cover.png"
    _make_valid_cover(cover_image)
    packager.run_packager(_RELEASE_DIR, kaggle_dir=kaggle_dir, cover_image=cover_image)

    kaggle_readme = kaggle_dir / "README.md"
    assert kaggle_readme.is_file()
    assert not kaggle_readme.is_symlink()
    contents = kaggle_readme.read_text(encoding="utf-8")
    assert "](../" not in contents
    assert packager.GITHUB_BLOB_BASE in contents


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_assembled_upload_dir_resolves_every_declared_resource(tmp_path: Path) -> None:
    """Every ``resources[].path`` declared in the metadata must resolve
    to a real file (not a symlink, not a missing path) under the
    assembled upload directory.  Kaggle's CLI walks the directory at
    upload time; a declared resource that doesn't materialise is a
    silent upload-time failure.
    """

    kaggle_dir = tmp_path / "kaggle"
    cover_image = tmp_path / "cover.png"
    _make_valid_cover(cover_image)
    outcome = packager.run_packager(_RELEASE_DIR, kaggle_dir=kaggle_dir, cover_image=cover_image)

    # Every resource path resolves to a real file.
    for resource in outcome.metadata.resources:
        target = kaggle_dir / resource.path
        assert target.is_file(), f"declared resource missing from upload tree: {resource.path}"
        assert not target.is_symlink(), (
            f"declared resource is a symlink, not a real file: {resource.path} — "
            f"Kaggle's CLI may skip symlinked entries on upload"
        )

    # Top-level required artefacts.
    assert (kaggle_dir / "dataset-metadata.json").is_file()
    assert (kaggle_dir / "README.md").is_file()
    assert (kaggle_dir / cover_image.name).is_file()
    assert not (kaggle_dir / cover_image.name).is_symlink()


# ---------------------------------------------------------------------------
# Upload-dir assembly safety
# ---------------------------------------------------------------------------


def test_assemble_upload_dir_rejects_unsafe_kaggle_dir(tmp_path: Path) -> None:
    """Refuse to assemble into the release dir or its parent."""

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, fake_release)
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, fake_release.parent)


def test_assemble_upload_dir_rejects_kaggle_dir_equal_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse to assemble into the current working directory.

    A user passing ``--kaggle-dir .`` (or running from inside the
    intended ``kaggle_dir``) would otherwise rmtree-then-recopy
    arbitrary cwd contents.  This is the most-likely-to-trigger
    safety case and was missing test coverage in the initial PR.
    """

    fake_release = tmp_path / "release"
    fake_release.mkdir()
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with pytest.raises(ValueError, match="unsafe"):
        packager.assemble_upload_dir(fake_release, cwd)


def test_assemble_upload_dir_idempotent_against_existing_tree(tmp_path: Path) -> None:
    """Re-running the assembly over an already-populated upload tree
    succeeds — the previous PR's symlink-vs-file confusion is no
    longer possible because both passes call the same copy helpers."""

    if not _RELEASE_BUNDLES_PRESENT:
        pytest.skip("release bundles not present")

    kaggle_dir = tmp_path / "kaggle"
    cover_image = tmp_path / "cover.png"
    _make_valid_cover(cover_image)
    packager.run_packager(_RELEASE_DIR, kaggle_dir=kaggle_dir, cover_image=cover_image)
    # Second pass against the same kaggle_dir.
    outcome = packager.run_packager(_RELEASE_DIR, kaggle_dir=kaggle_dir, cover_image=cover_image)
    assert outcome.errors == ()
    for resource in outcome.metadata.resources:
        assert (kaggle_dir / resource.path).is_file()


# ---------------------------------------------------------------------------
# CLI driver — error paths
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_rejects_unsafe_kaggle_dir_in_dry_run(tmp_path: Path) -> None:
    """Copilot review on PR #72 item #1.

    The earlier draft only checked ``--kaggle-dir`` safety inside
    ``assemble_upload_dir``, which dry-run skips.  A user passing
    ``--kaggle-dir release`` (i.e. ``release_dir`` itself) in dry-run
    would write ``dataset-metadata.json`` into ``release/`` before
    the safety net fired.  With the hoisted check, dry-run also
    raises ``ValueError`` BEFORE any mkdir or write.
    """

    cover = tmp_path / "cover.png"
    Image.new("RGB", (1280, 640), (0, 0, 0)).save(cover)

    with pytest.raises(ValueError, match="unsafe"):
        packager.run_packager(
            _RELEASE_DIR,
            kaggle_dir=_RELEASE_DIR,  # release_dir itself
            cover_image=cover,
            dry_run=True,
        )

    # ``release/dataset-metadata.json`` must not exist at the top
    # level (it's gitignored anyway, but a stray write would still
    # show up in ``git status``).
    assert not (_RELEASE_DIR / "dataset-metadata.json").exists()


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_resolves_cover_image_via_release_fallback(tmp_path: Path) -> None:
    """Copilot review on PR #72 item #2.

    A bare-basename ``--cover-image`` that exists under ``release/``
    must validate (via ``resolve_cover_image_path`` fallback) and
    materialise into the assembled metadata's ``image`` field.
    """

    out_dir = tmp_path / "kaggle"
    bare = Path(_COMMITTED_COVER.name)
    assert not bare.is_absolute()

    outcome = packager.run_packager(
        _RELEASE_DIR,
        kaggle_dir=out_dir,
        cover_image=bare,
        dry_run=True,
    )
    assert outcome.errors == ()
    # Metadata's ``image`` field carries the resolved filename.
    parsed = json.loads((out_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert parsed["image"] == _COMMITTED_COVER.name


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_does_not_write_on_validation_failure(tmp_path: Path) -> None:
    """A failed validation must NOT leave a corrupt metadata file on
    disk (PR 5.2 self-review fix).

    Forces a validation failure by passing a cover image that is too
    small for Kaggle's 560×280 floor.  Asserts the metadata path
    doesn't materialise and ``outcome.errors`` is populated.
    """

    tiny_cover = tmp_path / "tiny.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(tiny_cover)
    out_dir = tmp_path / "kaggle"
    outcome = packager.run_packager(_RELEASE_DIR, kaggle_dir=out_dir, cover_image=tiny_cover)
    assert outcome.errors, "expected at least one validation error"
    assert not (out_dir / "dataset-metadata.json").exists(), (
        "metadata file must not be written when validation fails"
    )
    assert outcome.assembled is False


def test_main_reports_missing_release_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = packager.main(
        [
            "--release-dir",
            str(tmp_path / "missing"),
            "--kaggle-dir",
            str(tmp_path / "kaggle"),
            "--cover-image",
            str(tmp_path / "cover.png"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "release directory not found" in captured.err


# ---------------------------------------------------------------------------
# Determinism + sync with committed artefact
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_run_packager_metadata_is_byte_deterministic(tmp_path: Path) -> None:
    """Two back-to-back runs against the committed bundles must
    produce byte-identical metadata files."""

    cover = tmp_path / "cover.png"
    _make_valid_cover(cover)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    packager.run_packager(_RELEASE_DIR, kaggle_dir=out_a, cover_image=cover, dry_run=True)
    packager.run_packager(_RELEASE_DIR, kaggle_dir=out_b, cover_image=cover, dry_run=True)
    assert (out_a / "dataset-metadata.json").read_bytes() == (
        out_b / "dataset-metadata.json"
    ).read_bytes()


def test_render_metadata_emits_literal_unicode_not_escapes() -> None:
    """``ensure_ascii=False`` keeps em-dashes, ``×``, smart quotes etc.
    rendered literally so the committed JSON stays diffable."""

    metadata = _minimal_metadata()
    rendered = packager.render_metadata_json(
        packager.DatasetMetadata(**{**metadata.__dict__, "description": "a — b × c"})
    )
    assert "a — b × c" in rendered
    assert "\\u2014" not in rendered
    assert "\\u00d7" not in rendered


def test_render_metadata_keywords_are_sorted_at_render_time() -> None:
    """Keywords are sorted in the rendered JSON regardless of the
    order they were declared on the metadata object — locks the
    determinism contract independent of the ``DEFAULT_KEYWORDS``
    constant ordering."""

    base = _minimal_metadata()
    shuffled = packager.DatasetMetadata(
        **{**base.__dict__, "keywords": ("zebra", "alpha", "mango")},
    )
    parsed = json.loads(packager.render_metadata_json(shuffled))
    assert parsed["keywords"] == ["alpha", "mango", "zebra"]


# ---------------------------------------------------------------------------
# Kaggle CLI shape validation (G11.3) — gated, opt-in
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_kaggle_cli_accepts_assembled_metadata(tmp_path: Path) -> None:
    """G11.3 — feed the assembled tree to the actual Kaggle metadata
    validator and assert it accepts the shape.

    Skipped unless the optional ``kaggle`` package is installed
    (``pip install -e '.[publish]'``); we deliberately don't make
    that a hard dependency because the kaggle SDK pulls in a long
    transitive tail.  The Kaggle SDK exposes a metadata validator
    via ``kaggle.api.validate_dataset_metadata`` (path varies by
    version); we look it up dynamically and skip if absent rather
    than hard-couple to one CLI version.
    """

    kaggle = pytest.importorskip("kaggle", reason="kaggle SDK not installed")
    kaggle_dir = tmp_path / "kaggle"
    cover = tmp_path / "cover.png"
    _make_valid_cover(cover)
    packager.run_packager(_RELEASE_DIR, kaggle_dir=kaggle_dir, cover_image=cover)

    # Search for a metadata-validator entry point on the kaggle API.
    api = kaggle.api
    candidates = [
        getattr(api, name, None)
        for name in (
            "validate_dataset_metadata",
            "_validate_dataset_metadata",
            "process_resources",
        )
    ]
    validator = next((c for c in candidates if callable(c)), None)
    if validator is None:
        pytest.skip("no Kaggle metadata-validator entry point found on the installed SDK")

    # Different Kaggle SDK versions expose different signatures; try
    # the most common shapes.  We're treating "no exception raised"
    # as acceptance.
    try:
        validator(str(kaggle_dir))
    except TypeError:
        validator(str(kaggle_dir / "dataset-metadata.json"))


@pytest.mark.skipif(
    not (_RELEASE_BUNDLES_PRESENT and _COMMITTED_METADATA.exists()),
    reason="release bundles or committed metadata missing",
)
def test_committed_kaggle_metadata_matches_fresh_regeneration(tmp_path: Path) -> None:
    """A fresh metadata regeneration must match the committed
    ``release/kaggle/dataset-metadata.json`` byte-for-byte AND have
    a non-degenerate description / id / image.

    If this fails, ``release/`` drifted without re-running
    ``scripts/package_kaggle_release.py``.  Regenerate via that
    script from the repo root and commit the new metadata alongside
    the bundle change.
    """

    cover = _COMMITTED_COVER if _COMMITTED_COVER.exists() else tmp_path / "cover.png"
    if not _COMMITTED_COVER.exists():
        _make_valid_cover(cover)

    fresh_dir = tmp_path / "kaggle"
    packager.run_packager(_RELEASE_DIR, kaggle_dir=fresh_dir, cover_image=cover, dry_run=True)
    fresh_bytes = (fresh_dir / "dataset-metadata.json").read_bytes()
    committed_bytes = _COMMITTED_METADATA.read_bytes()
    assert fresh_bytes == committed_bytes

    # Positive content assertions — guard against the failure mode
    # where a code change accidentally produces empty / minimal
    # content that we then re-commit, leaving the byte-equality
    # check passing on broken output.
    parsed = json.loads(fresh_bytes)
    assert parsed["id"] == f"{packager.DEFAULT_USER_SLUG}/{packager.DEFAULT_DATASET_SLUG}"
    assert parsed["image"] == "dataset-cover-image.png"
    description = parsed["description"]
    # The description should carry the rewritten dataset card, not be
    # empty or stub content.
    assert "What's inside" in description
    assert "Why lead scoring matters" in description
    assert "Known limitations" in description
    # Rewrites fired (no source-tree leaks, no broken relative links).
    assert "intermediate_instructor/" not in description
    assert "](../" not in description
    assert "github.com/leadforge-dev/leadforge/blob/main" in description
    # Resources are non-trivial.
    assert len(parsed["resources"]) >= 30
    # Every flat CSV has a schema with the canonical 33-column shape.
    flat_csvs = [r for r in parsed["resources"] if r["path"].endswith("/lead_scoring.csv")]
    assert len(flat_csvs) == len(packager.DEFAULT_TIERS)
    for r in flat_csvs:
        assert r["schema"]["fields"][0]["name"] == "split"
        assert r["schema"]["fields"][-1]["name"] == "converted_within_90_days"

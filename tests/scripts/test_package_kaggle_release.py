"""Tests for ``scripts/package_kaggle_release.py``.

Locks the Phase 5 Kaggle packaging contract:

* every Kaggle field constraint surfaced in chatgpt v2 §19 (G11.1)
* the cover-image dimension floor (G11.2)
* schema-fields-in-column-order for every tabular resource — both
  flat CSVs (driven by ``feature_dictionary.csv``) and parquet files
  (driven by the Arrow schema)
* the README link-rewriting that lets the published dataset card on
  Kaggle keep working ``../`` links (rewritten to GitHub blob URLs)
  and a directory diagram that reflects the upload layout
* byte-equality between the committed ``release/kaggle/dataset-metadata.json``
  and a fresh regeneration (audit-artifact-sync pattern from PR 4.1)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow.parquet as pq
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
    assert "below Kaggle minimum" in errors[0].message


def test_validate_cover_image_reports_missing_file(tmp_path: Path) -> None:
    errors = packager.validate_cover_image(tmp_path / "no-such.png")
    assert errors
    assert errors[0].field == "cover_image"


# ---------------------------------------------------------------------------
# Schema fields — column-order parity for tabular resources
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_lead_scoring_resource_schema_follows_csv_column_order() -> None:
    """Field order in the metadata matches the flat CSV's column order
    for every tier (the constraint Kaggle's schema spec calls out)."""

    for tier in packager.DEFAULT_TIERS:
        resources = packager.build_tier_resources(_RELEASE_DIR, tier)
        flat = next(r for r in resources if r.path == f"{tier}/lead_scoring.csv")
        assert flat.schema is not None
        names = [f.name for f in flat.schema.fields]
        assert names[0] == "split"
        assert names[1] == "account_id"
        assert names[-1] == "converted_within_90_days"


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
def test_parquet_resource_schemas_match_arrow_column_order() -> None:
    """Parquet schemas in the metadata match the parquet file itself."""

    resources = packager.build_tier_resources(_RELEASE_DIR, "intro")
    train = next(
        r for r in resources if r.path.endswith("/tasks/converted_within_90_days/train.parquet")
    )
    assert train.schema is not None
    train_path = _RELEASE_DIR / "intro" / "tasks" / "converted_within_90_days" / "train.parquet"
    expected = list(pq.read_schema(train_path).names)
    assert [f.name for f in train.schema.fields] == expected


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
def test_assembled_upload_dir_writes_rewritten_readme_copy(tmp_path: Path) -> None:
    """The README inside ``release/kaggle/`` is a real file (not a
    symlink) and carries the rewrites — Kaggle reads this verbatim
    on the dataset page."""

    kaggle_dir = tmp_path / "kaggle"
    cover_image = tmp_path / "cover.png"
    Image.new("RGB", (1280, 640), (0, 0, 0)).save(cover_image)
    packager.run_packager(
        _RELEASE_DIR,
        kaggle_dir=kaggle_dir,
        cover_image=cover_image,
    )
    kaggle_readme = kaggle_dir / "README.md"
    assert kaggle_readme.exists()
    assert not kaggle_readme.is_symlink()
    contents = kaggle_readme.read_text(encoding="utf-8")
    assert "](../" not in contents
    assert packager.GITHUB_BLOB_BASE in contents


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


# ---------------------------------------------------------------------------
# CLI driver — error paths
# ---------------------------------------------------------------------------


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
    Image.new("RGB", (1280, 640), (0, 0, 0)).save(cover)

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    packager.run_packager(_RELEASE_DIR, kaggle_dir=out_a, cover_image=cover, dry_run=True)
    packager.run_packager(_RELEASE_DIR, kaggle_dir=out_b, cover_image=cover, dry_run=True)
    assert (out_a / "dataset-metadata.json").read_bytes() == (
        out_b / "dataset-metadata.json"
    ).read_bytes()


@pytest.mark.skipif(
    not (_RELEASE_BUNDLES_PRESENT and _COMMITTED_METADATA.exists()),
    reason="release bundles or committed metadata missing",
)
def test_committed_kaggle_metadata_matches_fresh_regeneration(tmp_path: Path) -> None:
    """A fresh metadata regeneration must match the committed
    ``release/kaggle/dataset-metadata.json`` byte-for-byte.

    If this fails, ``release/`` drifted without re-running
    ``scripts/package_kaggle_release.py``.  Regenerate via that script
    from the repo root and commit the new metadata alongside the
    bundle change.
    """

    cover = _COMMITTED_COVER if _COMMITTED_COVER.exists() else tmp_path / "cover.png"
    if not _COMMITTED_COVER.exists():
        Image.new("RGB", (1280, 640), (0, 0, 0)).save(cover)

    fresh_dir = tmp_path / "kaggle"
    packager.run_packager(_RELEASE_DIR, kaggle_dir=fresh_dir, cover_image=cover, dry_run=True)
    fresh = (fresh_dir / "dataset-metadata.json").read_bytes()
    committed = _COMMITTED_METADATA.read_bytes()
    assert fresh == committed

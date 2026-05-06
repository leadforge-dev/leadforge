"""Tests for ``scripts/package_kaggle_release.py``.

The packager is the Phase 5 Kaggle dry-run surface: it generates the
Kaggle upload directory, writes ``dataset-metadata.json``, and produces
the deterministic cover image.  These tests keep the committed metadata
in sync with the committed release bundles, following the PR 4.1 audit
artifact pattern.
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

_LOCAL_RELEASE_DIR = _REPO_ROOT / "release"
_COMMITTED_KAGGLE_DIR = _REPO_ROOT / "release" / "kaggle"
_RELEASE_SOURCE_DIR = (
    _LOCAL_RELEASE_DIR
    if (_LOCAL_RELEASE_DIR / "intro" / "manifest.json").exists()
    else _COMMITTED_KAGGLE_DIR
)


def _minimal_metadata() -> dict[str, object]:
    return {
        "title": packager.TITLE,
        "subtitle": packager.SUBTITLE,
        "id": packager.DATASET_ID,
        "licenses": [{"name": packager.LICENSE_NAME}],
        "expectedUpdateFrequency": packager.EXPECTED_UPDATE_FREQUENCY,
        "keywords": list(packager.KEYWORDS),
        "description": "Synthetic CRM lead scoring dataset.",
        "userSpecifiedSources": [{"title": "LeadForge", "url": "https://example.com"}],
        "image": packager.IMAGE_FILENAME,
        "resources": [
            {
                "path": "intro/lead_scoring.csv",
                "description": "Intro split.",
                "schema": {"fields": [{"name": "lead_id", "type": "string"}]},
            }
        ],
    }


def test_validate_metadata_accepts_expected_kaggle_constraints() -> None:
    assert packager.validate_metadata(_minimal_metadata()) == []


def test_validate_metadata_reports_field_constraint_errors() -> None:
    metadata = _minimal_metadata()
    metadata["title"] = "Tiny"
    metadata["subtitle"] = "short"
    metadata["id"] = "LeadForge Bad Slug!"
    metadata["licenses"] = [{"name": "MIT"}, {"name": "Apache 2.0"}]
    metadata["expectedUpdateFrequency"] = "sometimes"
    metadata["image"] = "wrong.png"
    metadata["resources"] = [{"path": "x.csv", "description": "x", "schema": {"fields": []}}]

    errors = packager.validate_metadata(metadata)

    assert "title must be 6-50 characters" in errors
    assert "subtitle must be 20-80 characters" in errors
    assert "id must be a 3-50 character lowercase slug" in errors
    assert "licenses must contain exactly one MIT entry" in errors
    assert "expectedUpdateFrequency is not approved" in errors
    assert "resources[0] must include schema.fields" in errors
    assert f"image must be {packager.IMAGE_FILENAME}" in errors


def test_cover_image_generation_meets_kaggle_minimum(tmp_path: Path) -> None:
    cover = tmp_path / "dataset-cover-image.png"

    packager.generate_cover_image(cover)

    with Image.open(cover) as image:
        assert image.size == (1120, 560)
    assert packager.validate_cover_image(cover) == []


def test_lead_scoring_resource_schema_follows_csv_column_order() -> None:
    resources = packager.discover_resources(_RELEASE_SOURCE_DIR, tiers=("intro",))
    lead_scoring = next(
        resource for resource in resources if resource.path == "intro/lead_scoring.csv"
    )

    names = [field.name for field in lead_scoring.schema_fields]

    assert names[:5] == ["split", "account_id", "industry", "region", "employee_band"]
    assert names[-1] == "converted_within_90_days"


def test_package_release_writes_upload_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "kaggle"
    cover = tmp_path / "dataset-cover-image.png"

    metadata = packager.package_release(_RELEASE_SOURCE_DIR, out_dir, cover)

    assert (out_dir / "dataset-metadata.json").exists()
    assert (out_dir / packager.IMAGE_FILENAME).exists()
    assert (out_dir / "intro" / "lead_scoring.csv").exists()
    assert packager.validate_upload_dir(out_dir, metadata) == []


def test_package_release_rewrites_kaggle_readme_links(tmp_path: Path) -> None:
    out_dir = tmp_path / "kaggle"
    cover = tmp_path / "dataset-cover-image.png"

    metadata = packager.package_release(_RELEASE_SOURCE_DIR, out_dir, cover)
    readme = (out_dir / "README.md").read_text(encoding="utf-8")

    assert "intermediate_instructor/" not in readme
    assert "notebooks/01_baseline_lead_scoring.ipynb" not in readme
    assert "](validation/validation_report.md)" not in readme
    assert "](../" not in readme
    assert (
        "github.com/leadforge-dev/leadforge/blob/main/release/validation/validation_report.md"
        in readme
    )
    assert "](validation/validation_report.md)" not in metadata["description"]


def test_package_release_rejects_unsafe_out_dir(tmp_path: Path) -> None:
    cover = tmp_path / "dataset-cover-image.png"

    with pytest.raises(ValueError, match="refusing to delete unsafe --out-dir"):
        packager.package_release(_RELEASE_SOURCE_DIR, _RELEASE_SOURCE_DIR.parent, cover)


def test_main_reports_missing_release_dir(tmp_path: Path, capsys) -> None:
    rc = packager.main(
        [
            "--release-dir",
            str(tmp_path / "missing"),
            "--out-dir",
            str(tmp_path / "kaggle"),
            "--cover-image",
            str(tmp_path / "cover.png"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "release directory not found" in captured.err


def test_committed_kaggle_metadata_matches_fresh_regeneration(tmp_path: Path) -> None:
    out_dir = tmp_path / "kaggle"
    cover = tmp_path / "dataset-cover-image.png"

    metadata = packager.package_release(_RELEASE_SOURCE_DIR, out_dir, cover)

    committed = json.loads(
        (_REPO_ROOT / "release" / "kaggle" / "dataset-metadata.json").read_text()
    )
    regenerated = json.loads((out_dir / "dataset-metadata.json").read_text())
    assert regenerated == committed
    assert packager.validate_metadata(metadata) == []

"""Tests for leadforge.validation.realism."""

from __future__ import annotations

from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.realism import check_realism

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("realism")
    Generator.from_recipe(
        "b2b_saas_procurement_v1", seed=42, exposure_mode="student_public"
    ).generate(**_SMALL).save(str(out))
    return out


class TestRealism:
    def test_valid_bundle_passes(self, bundle_dir: Path) -> None:
        errors = check_realism(bundle_dir)
        assert errors == [], f"Unexpected realism errors: {errors}"

    def test_detects_zero_row_table(self, tmp_path: Path, bundle_dir: Path) -> None:
        """A manifest claiming 0 rows for accounts should flag."""
        import json
        import shutil

        corrupt = tmp_path / "zero_rows"
        shutil.copytree(bundle_dir, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        manifest["tables"]["accounts"]["row_count"] = 0
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = check_realism(corrupt)
        assert any("0 rows" in e for e in errors)

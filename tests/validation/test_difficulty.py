"""Tests for leadforge.validation.difficulty."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.difficulty import check_difficulty, check_difficulty_ordering

_SMALL = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("difficulty")
    Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=42,
        exposure_mode="student_public",
        difficulty="intermediate",
    ).generate(**_SMALL).save(str(out))
    return out


class TestCheckDifficulty:
    def test_known_difficulty_passes(self, bundle_dir: Path) -> None:
        errors = check_difficulty(bundle_dir)
        assert errors == []

    def test_unknown_difficulty_fails(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "unknown_diff"
        shutil.copytree(bundle_dir, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        manifest["difficulty"] = "nightmare"
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = check_difficulty(corrupt)
        assert any("Unknown difficulty" in e for e in errors)

    def test_missing_difficulty_fails(self, tmp_path: Path, bundle_dir: Path) -> None:
        corrupt = tmp_path / "no_diff"
        shutil.copytree(bundle_dir, corrupt)
        manifest = json.loads((corrupt / "manifest.json").read_text())
        del manifest["difficulty"]
        (corrupt / "manifest.json").write_text(json.dumps(manifest, indent=2))

        errors = check_difficulty(corrupt)
        assert any("missing" in e.lower() for e in errors)


class TestDifficultyOrdering:
    def test_ordering_is_noop_for_v1(self, bundle_dir: Path) -> None:
        """Until the engine modulates by difficulty, ordering check is a no-op."""
        bundles = {"intro": bundle_dir, "intermediate": bundle_dir, "advanced": bundle_dir}
        errors = check_difficulty_ordering(bundles)
        assert errors == []

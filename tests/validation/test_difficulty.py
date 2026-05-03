"""Tests for leadforge.validation.difficulty."""

from __future__ import annotations

import json
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


@pytest.fixture(scope="module")
def manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "manifest.json").read_text())


class TestCheckDifficulty:
    def test_known_difficulty_passes(self, manifest: dict) -> None:
        errors = check_difficulty(manifest)
        assert errors == []

    def test_unknown_difficulty_fails(self, manifest: dict) -> None:
        corrupt = {**manifest, "difficulty": "nightmare"}
        errors = check_difficulty(corrupt)
        assert any("Unknown difficulty" in e for e in errors)

    def test_missing_difficulty_fails(self, manifest: dict) -> None:
        corrupt = {k: v for k, v in manifest.items() if k != "difficulty"}
        errors = check_difficulty(corrupt)
        assert any("missing" in e.lower() for e in errors)


class TestDifficultyOrdering:
    def test_same_bundle_fails_for_multiple_difficulties(self, bundle_dir: Path) -> None:
        """Same bundle cannot satisfy multiple difficulty ranges."""
        bundles = {"intro": bundle_dir, "intermediate": bundle_dir, "advanced": bundle_dir}
        errors = check_difficulty_ordering(bundles)
        # A single bundle's rate can't be in all three ranges at once,
        # so we expect at least one violation.
        assert len(errors) > 0

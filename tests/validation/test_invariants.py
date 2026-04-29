"""Tests for leadforge.validation.invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.invariants import check_determinism, check_exposure_monotonicity

_SMALL = {"n_leads": 20, "n_accounts": 10, "n_contacts": 30}


@pytest.fixture(scope="module")
def determinism_bundles(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate two bundles with the same seed."""
    a = tmp_path_factory.mktemp("det_a")
    b = tmp_path_factory.mktemp("det_b")
    for out in (a, b):
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=77, exposure_mode="student_public"
        )
        gen.generate(**_SMALL).save(str(out))
    return a, b


@pytest.fixture(scope="module")
def exposure_bundles(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate student_public and research_instructor bundles."""
    student = tmp_path_factory.mktemp("student")
    instructor = tmp_path_factory.mktemp("instructor")
    Generator.from_recipe(
        "b2b_saas_procurement_v1", seed=88, exposure_mode="student_public"
    ).generate(**_SMALL).save(str(student))
    Generator.from_recipe(
        "b2b_saas_procurement_v1", seed=88, exposure_mode="research_instructor"
    ).generate(**_SMALL).save(str(instructor))
    return student, instructor


class TestDeterminism:
    def test_same_seed_produces_identical_bundles(
        self, determinism_bundles: tuple[Path, Path]
    ) -> None:
        a, b = determinism_bundles
        errors = check_determinism(a, b)
        assert errors == []

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        a = tmp_path / "seed1"
        b = tmp_path / "seed2"
        Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=1, exposure_mode="student_public"
        ).generate(**_SMALL).save(str(a))
        Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=2, exposure_mode="student_public"
        ).generate(**_SMALL).save(str(b))
        errors = check_determinism(a, b)
        assert len(errors) > 0


class TestExposureMonotonicity:
    def test_valid_pair_passes(self, exposure_bundles: tuple[Path, Path]) -> None:
        student, instructor = exposure_bundles
        errors = check_exposure_monotonicity(student, instructor)
        assert errors == []

    def test_student_with_metadata_fails(self, exposure_bundles: tuple[Path, Path]) -> None:
        student, instructor = exposure_bundles
        # Swap args — instructor as "student" has metadata/, should fail
        errors = check_exposure_monotonicity(instructor, instructor)
        assert any("should not contain metadata" in e for e in errors)

    def test_instructor_without_metadata_fails(self, exposure_bundles: tuple[Path, Path]) -> None:
        student, _ = exposure_bundles
        # Student as "instructor" lacks metadata/
        errors = check_exposure_monotonicity(student, student)
        assert any("missing metadata" in e for e in errors)

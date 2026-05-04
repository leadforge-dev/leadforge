"""Tests for leadforge.validation.invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.validation.invariants import (
    check_determinism,
    check_exposure_monotonicity,
    compare_bundle_trees,
)

_SMALL = {"n_leads": 20, "n_accounts": 10, "n_contacts": 30}


_PINNED_TIMESTAMP = "2024-01-01T00:00:00+00:00"


@pytest.fixture(scope="module")
def determinism_bundles(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate two bundles with the same seed and pinned timestamp."""
    a = tmp_path_factory.mktemp("det_a")
    b = tmp_path_factory.mktemp("det_b")
    for out in (a, b):
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=77, exposure_mode="student_public"
        )
        gen.generate(**_SMALL).save(str(out), generation_timestamp=_PINNED_TIMESTAMP)
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


def _make_synthetic_bundle(
    root: Path,
    files: dict[str, str | bytes],
    manifest: dict | None = None,
) -> Path:
    """Write a fake bundle layout with the given files and optional manifest."""
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    return root


class TestCompareBundleTrees:
    """Synthetic-bundle unit tests for compare_bundle_trees.

    These avoid running the full generator so the verifier's logic is exercised
    independently of generation determinism.  Real end-to-end determinism is
    covered by TestDeterminism above.
    """

    def test_identical_trees_no_errors(self, tmp_path: Path) -> None:
        a = _make_synthetic_bundle(
            tmp_path / "a",
            files={"tables/x.parquet": b"\x01\x02", "dataset_card.md": "hello"},
        )
        b = _make_synthetic_bundle(
            tmp_path / "b",
            files={"tables/x.parquet": b"\x01\x02", "dataset_card.md": "hello"},
        )
        assert compare_bundle_trees(a, b) == []

    def test_only_in_a_reported(self, tmp_path: Path) -> None:
        a = _make_synthetic_bundle(
            tmp_path / "a",
            files={"tables/x.parquet": b"x", "tables/extra.parquet": b"y"},
        )
        b = _make_synthetic_bundle(tmp_path / "b", files={"tables/x.parquet": b"x"})
        errors = compare_bundle_trees(a, b)
        assert any("only in A" in e and "extra.parquet" in e for e in errors)

    def test_only_in_b_reported(self, tmp_path: Path) -> None:
        a = _make_synthetic_bundle(tmp_path / "a", files={"tables/x.parquet": b"x"})
        b = _make_synthetic_bundle(
            tmp_path / "b",
            files={"tables/x.parquet": b"x", "metadata/world_spec.json": "{}"},
        )
        errors = compare_bundle_trees(a, b)
        assert any("only in B" in e and "world_spec.json" in e for e in errors)

    def test_hash_mismatch_reported_with_sizes(self, tmp_path: Path) -> None:
        a = _make_synthetic_bundle(tmp_path / "a", files={"tables/x.parquet": b"abc"})
        b = _make_synthetic_bundle(tmp_path / "b", files={"tables/x.parquet": b"abcd"})
        errors = compare_bundle_trees(a, b)
        assert len(errors) == 1
        assert "hash mismatch" in errors[0]
        assert "x.parquet" in errors[0]
        assert "A=3B" in errors[0]
        assert "B=4B" in errors[0]

    def test_manifest_only_timestamp_diff_passes(self, tmp_path: Path) -> None:
        manifest_a = {"seed": 42, "generation_timestamp": "2026-01-01T00:00:00+00:00"}
        manifest_b = {"seed": 42, "generation_timestamp": "2026-12-31T23:59:59+00:00"}
        a = _make_synthetic_bundle(tmp_path / "a", files={}, manifest=manifest_a)
        b = _make_synthetic_bundle(tmp_path / "b", files={}, manifest=manifest_b)
        assert compare_bundle_trees(a, b) == []

    def test_manifest_real_diff_reported(self, tmp_path: Path) -> None:
        manifest_a = {"seed": 42, "generation_timestamp": "2026-01-01T00:00:00+00:00"}
        manifest_b = {"seed": 43, "generation_timestamp": "2026-01-01T00:00:00+00:00"}
        a = _make_synthetic_bundle(tmp_path / "a", files={}, manifest=manifest_a)
        b = _make_synthetic_bundle(tmp_path / "b", files={}, manifest=manifest_b)
        errors = compare_bundle_trees(a, b)
        assert len(errors) == 1
        assert "manifest payload mismatch" in errors[0]

    def test_manifest_key_reorder_only_passes(self, tmp_path: Path) -> None:
        # Same logical payload, different on-disk key order — must NOT be flagged
        # as a mismatch.  (json.dumps with sort_keys=True normalises both sides.)
        a_root = tmp_path / "a"
        b_root = tmp_path / "b"
        a_root.mkdir()
        b_root.mkdir()
        (a_root / "manifest.json").write_text(json.dumps({"seed": 42, "n_leads": 100}, indent=2))
        (b_root / "manifest.json").write_text(json.dumps({"n_leads": 100, "seed": 42}, indent=2))
        assert compare_bundle_trees(a_root, b_root) == []

    def test_nested_manifest_not_special_cased(self, tmp_path: Path) -> None:
        # Only the top-level bundle manifest.json gets timestamp-stripping.
        # A file named manifest.json deeper in the tree is compared byte-for-byte.
        a = _make_synthetic_bundle(
            tmp_path / "a",
            files={"tasks/foo/manifest.json": '{"generation_timestamp": "T1"}'},
        )
        b = _make_synthetic_bundle(
            tmp_path / "b",
            files={"tasks/foo/manifest.json": '{"generation_timestamp": "T2"}'},
        )
        errors = compare_bundle_trees(a, b)
        assert any("hash mismatch" in e for e in errors)

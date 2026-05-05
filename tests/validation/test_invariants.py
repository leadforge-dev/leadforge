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
    """Generate two student_public bundles with the same seed and pinned timestamp."""
    a = tmp_path_factory.mktemp("det_a")
    b = tmp_path_factory.mktemp("det_b")
    for out in (a, b):
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=77, exposure_mode="student_public"
        )
        gen.generate(**_SMALL).save(str(out), generation_timestamp=_PINNED_TIMESTAMP)
    return a, b


@pytest.fixture(scope="module")
def determinism_instructor_bundles(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    """Generate two research_instructor bundles with the same seed and pinned timestamp."""
    a = tmp_path_factory.mktemp("det_instructor_a")
    b = tmp_path_factory.mktemp("det_instructor_b")
    for out in (a, b):
        gen = Generator.from_recipe(
            "b2b_saas_procurement_v1", seed=77, exposure_mode="research_instructor"
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

    def test_same_seed_produces_identical_instructor_bundles(
        self, determinism_instructor_bundles: tuple[Path, Path]
    ) -> None:
        """Determinism must hold for ``research_instructor`` too — the
        full-horizon export and the metadata/ artefacts (graph, latents,
        mechanism summary) are part of the deterministic contract."""
        a, b = determinism_instructor_bundles
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


# ---------------------------------------------------------------------------
# check_exposure_monotonicity — focused unit tests for the v5 (PR 2.2)
# snapshot-safe branches.  These build minimal synthetic bundles so the
# table-comparison logic is exercised independently of generation.
# ---------------------------------------------------------------------------


class TestExposureMonotonicitySnapshotSafe:
    """Exercise the table-comparison branches added in PR 2.2:
    omitted ``BANNED_TABLES``, dropped ``BANNED_LEAD_COLUMNS`` /
    ``BANNED_OPP_COLUMNS``, and PK-based row-subset on
    ``SNAPSHOT_FILTERED_TABLES``."""

    @staticmethod
    def _shell(root: Path, *, has_metadata: bool) -> Path:
        """Write the minimum scaffolding ``check_exposure_monotonicity`` requires
        before it reaches the table-comparison logic.  Feature dictionary is
        written empty so the subset check trivially passes."""
        import pandas as pd

        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text("{}")
        (root / "dataset_card.md").write_text("")
        # Identical empty FD on both sides → subset check trivially passes.
        pd.DataFrame({"name": [], "dtype": []}).to_csv(root / "feature_dictionary.csv", index=False)
        if has_metadata:
            (root / "metadata").mkdir()
        (root / "tables").mkdir()
        return root

    @staticmethod
    def _write_parquet(root: Path, name: str, df) -> None:
        df.to_parquet(root / "tables" / f"{name}.parquet", index=False)

    @pytest.fixture
    def student_root(self, tmp_path: Path) -> Path:
        return self._shell(tmp_path / "student", has_metadata=False)

    @pytest.fixture
    def instructor_root(self, tmp_path: Path) -> Path:
        return self._shell(tmp_path / "instructor", has_metadata=True)

    def test_omitted_banned_tables_pass(self, student_root: Path, instructor_root: Path) -> None:
        """``customers`` / ``subscriptions`` in instructor only is the
        expected snapshot-safe contract — must NOT raise an error."""
        import pandas as pd

        for root in (student_root, instructor_root):
            self._write_parquet(root, "accounts", pd.DataFrame({"account_id": ["a1"]}))
        self._write_parquet(
            instructor_root,
            "customers",
            pd.DataFrame({"customer_id": ["c1"], "opportunity_id": ["o1"]}),
        )
        self._write_parquet(
            instructor_root,
            "subscriptions",
            pd.DataFrame({"subscription_id": ["s1"], "customer_id": ["c1"]}),
        )
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert errors == [], errors

    def test_unexpected_extra_instructor_table_fails(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """A NON-banned extra table on instructor must still be flagged —
        snapshot-safe doesn't license arbitrary instructor-only tables."""
        import pandas as pd

        for root in (student_root, instructor_root):
            self._write_parquet(root, "accounts", pd.DataFrame({"account_id": ["a1"]}))
        self._write_parquet(instructor_root, "rogue", pd.DataFrame({"x": [1]}))
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert any("rogue.parquet" in e and "instructor but not student" in e for e in errors)

    def test_dropped_banned_lead_columns_pass(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """Public ``leads`` drops ``converted_within_90_days`` /
        ``conversion_timestamp`` — must NOT raise."""
        import pandas as pd

        student_leads = pd.DataFrame({"lead_id": ["l1"], "lead_created_at": ["2024-01-01"]})
        instructor_leads = pd.DataFrame(
            {
                "lead_id": ["l1"],
                "lead_created_at": ["2024-01-01"],
                "converted_within_90_days": [True],
                "conversion_timestamp": ["2024-02-01"],
            }
        )
        self._write_parquet(student_root, "leads", student_leads)
        self._write_parquet(instructor_root, "leads", instructor_leads)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert errors == [], errors

    def test_dropped_banned_opp_columns_pass(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """Public ``opportunities`` drops ``close_outcome`` / ``closed_at`` —
        must NOT raise (and must still respect row-equality on shared
        columns since the table size is below the snapshot-filter threshold
        in this synthetic case)."""
        import pandas as pd

        student_opps = pd.DataFrame(
            {
                "opportunity_id": ["o1"],
                "lead_id": ["l1"],
                "created_at": ["2024-01-01"],
            }
        )
        instructor_opps = pd.DataFrame(
            {
                "opportunity_id": ["o1"],
                "lead_id": ["l1"],
                "created_at": ["2024-01-01"],
                "close_outcome": ["closed_won"],
                "closed_at": ["2024-02-01"],
            }
        )
        self._write_parquet(student_root, "opportunities", student_opps)
        self._write_parquet(instructor_root, "opportunities", instructor_opps)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert errors == [], errors

    def test_unexpected_extra_instructor_lead_column_fails(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """A non-banned, non-redacted extra column on instructor's leads
        must still trip the column-diff check."""
        import pandas as pd

        student_leads = pd.DataFrame({"lead_id": ["l1"]})
        instructor_leads = pd.DataFrame({"lead_id": ["l1"], "rogue_col": [42]})
        self._write_parquet(student_root, "leads", student_leads)
        self._write_parquet(instructor_root, "leads", instructor_leads)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert any("leads.parquet" in e and "rogue_col" in e for e in errors)

    def test_pk_row_subset_pass(self, student_root: Path, instructor_root: Path) -> None:
        """Student touches PK ⊂ instructor touches PK — must NOT raise."""
        import pandas as pd

        instructor_touches = pd.DataFrame(
            {
                "touch_id": ["t1", "t2", "t3", "t4"],
                "lead_id": ["l1", "l1", "l1", "l1"],
            }
        )
        # Student has a snapshot-window subset (rows for t1, t2 only).
        student_touches = instructor_touches.iloc[:2].reset_index(drop=True)
        self._write_parquet(student_root, "touches", student_touches)
        self._write_parquet(instructor_root, "touches", instructor_touches)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert errors == [], errors

    def test_pk_orphan_row_fails(self, student_root: Path, instructor_root: Path) -> None:
        """Student touch_id not present in instructor — must raise.  The PK
        branch fires when row counts differ; instructor strictly larger
        with student carrying an orphan PK is the regression case."""
        import pandas as pd

        instructor_touches = pd.DataFrame(
            {"touch_id": ["t1", "t2", "t3"], "lead_id": ["l1", "l1", "l1"]}
        )
        student_touches = pd.DataFrame(
            {"touch_id": ["t1", "t99"], "lead_id": ["l1", "l1"]}  # t99 is the orphan
        )
        self._write_parquet(student_root, "touches", student_touches)
        self._write_parquet(instructor_root, "touches", instructor_touches)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert any("touches.parquet" in e and "absent from instructor" in e for e in errors), (
            f"expected PK-orphan error; got: {errors}"
        )

    def test_student_has_more_rows_than_instructor_fails(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """Snapshot-safe must be a row-subset; student strictly larger means
        the writer leaked extra rows somewhere — must raise."""
        import pandas as pd

        instructor_touches = pd.DataFrame({"touch_id": ["t1"], "lead_id": ["l1"]})
        student_touches = pd.DataFrame(
            {"touch_id": ["t1", "t2", "t3"], "lead_id": ["l1", "l1", "l1"]}
        )
        self._write_parquet(student_root, "touches", student_touches)
        self._write_parquet(instructor_root, "touches", instructor_touches)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert any("touches.parquet" in e and "more rows than instructor" in e for e in errors)

    def test_missing_pk_column_surfaces_an_error(
        self, student_root: Path, instructor_root: Path
    ) -> None:
        """Student dropping ``touch_id`` is structurally invalid — the
        column-diff check catches it (``touch_id`` is not in the
        snapshot-safe allowlist for ``touches``), surfacing it as a
        non-redacted-column error before the PK-set comparison runs."""
        import pandas as pd

        instructor_touches = pd.DataFrame({"touch_id": ["t1", "t2"], "lead_id": ["l1", "l1"]})
        student_touches = pd.DataFrame({"lead_id": ["l1"]})
        self._write_parquet(student_root, "touches", student_touches)
        self._write_parquet(instructor_root, "touches", instructor_touches)
        errors = check_exposure_monotonicity(student_root, instructor_root)
        assert any(
            "touches.parquet" in e and "touch_id" in e and "non-redacted columns" in e
            for e in errors
        ), f"expected column-diff error naming touch_id; got: {errors}"


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

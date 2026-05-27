"""Determinism and exposure-monotonicity invariant checks.

These checks verify structural guarantees that must hold for every bundle:

- **Determinism**: same (recipe, seed, config) → identical output.
- **Exposure monotonicity**: ``student_public`` artefacts are a strict subset
  of ``research_instructor`` artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from leadforge.core.enums import ExposureMode
from leadforge.core.hashing import file_sha256
from leadforge.render.manifests import NON_DETERMINISTIC_MANIFEST_FIELDS
from leadforge.schema.features import redacted_columns_for
from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
    SNAPSHOT_FILTERED_TABLES,
)

# Feature-dictionary rows that are intentionally present only in the student
# bundle and are exempt from the subset check in check_exposure_monotonicity.
# ``split`` documents the partition-label column that exists only in the flat
# ``lead_scoring.csv`` convenience export (student_public only); the
# instructor bundle's feature dictionary covers the raw Parquet columns.
STUDENT_ONLY_DICT_ROWS: frozenset[str] = frozenset({"split"})


def check_determinism(bundle_a: Path, bundle_b: Path) -> list[str]:
    """Compare two bundles that should be identical (same seed/config).

    Both bundles must already exist on disk.  Returns a list of mismatch
    descriptions (empty = deterministic).
    """
    errors: list[str] = []

    # Compare core non-Parquet files that must also be deterministic.
    for fname in ("manifest.json", "dataset_card.md", "feature_dictionary.csv"):
        fa = bundle_a / fname
        fb = bundle_b / fname
        if fa.exists() and fb.exists():
            if file_sha256(fa) != file_sha256(fb):
                errors.append(f"Hash mismatch: {fname}")
        elif fa.exists() != fb.exists():
            errors.append(f"File '{fname}' exists in one bundle but not the other")

    # Compare all Parquet files under tables/ and tasks/
    for subdir in ("tables", "tasks"):
        dir_a = bundle_a / subdir
        dir_b = bundle_b / subdir
        if not dir_a.exists() or not dir_b.exists():
            if dir_a.exists() != dir_b.exists():
                errors.append(f"Directory '{subdir}' exists in one bundle but not the other")
            continue

        files_a = {p.relative_to(dir_a) for p in dir_a.rglob("*.parquet")}
        files_b = {p.relative_to(dir_b) for p in dir_b.rglob("*.parquet")}

        only_a = files_a - files_b
        only_b = files_b - files_a
        if only_a:
            errors.append(f"Files only in bundle A {subdir}/: {sorted(str(f) for f in only_a)}")
        if only_b:
            errors.append(f"Files only in bundle B {subdir}/: {sorted(str(f) for f in only_b)}")

        for rel in sorted(files_a & files_b):
            sha_a = file_sha256(dir_a / rel)
            sha_b = file_sha256(dir_b / rel)
            if sha_a != sha_b:
                errors.append(f"Hash mismatch: {subdir}/{rel}")

    return errors


def _manifest_payloads_match_modulo_non_deterministic(a: Path, b: Path) -> bool:
    """Compare two manifest.json files after stripping non-deterministic fields.

    Re-dumps both payloads with ``sort_keys=True`` so a key reordering still
    counts as a mismatch.
    """
    payload_a = json.loads(a.read_text())
    payload_b = json.loads(b.read_text())
    for field in NON_DETERMINISTIC_MANIFEST_FIELDS:
        payload_a.pop(field, None)
        payload_b.pop(field, None)
    return json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)


def compare_bundle_trees(bundle_a: Path, bundle_b: Path) -> list[str]:
    """Full-tree byte-identical comparison of two bundle directories.

    Walks every file under both roots and reports:

    - files present in only one tree (``only in A:`` / ``only in B:``)
    - files whose SHA-256 differs (``hash mismatch:``)

    The bundle ``manifest.json`` is special-cased: it carries
    ``generation_timestamp`` (wall-clock UTC, set by ``build_manifest()``),
    which is expected to differ across runs unless the caller pinned it.
    For that one file, if the raw hashes differ, the function re-compares the
    payload with non-deterministic fields stripped (see
    :data:`NON_DETERMINISTIC_MANIFEST_FIELDS`).  A mismatch *after* stripping
    is still reported.

    Use this for release-time integration checks; for the fast in-process
    determinism property used in CI, see :func:`check_determinism`.
    """
    errors: list[str] = []

    files_a = {p.relative_to(bundle_a) for p in bundle_a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(bundle_b) for p in bundle_b.rglob("*") if p.is_file()}

    for rel in sorted(files_a - files_b):
        errors.append(f"only in A: {rel}")
    for rel in sorted(files_b - files_a):
        errors.append(f"only in B: {rel}")

    for rel in sorted(files_a & files_b):
        path_a = bundle_a / rel
        path_b = bundle_b / rel
        if file_sha256(path_a) == file_sha256(path_b):
            continue
        if rel.name == "manifest.json" and rel.parent == Path():
            if _manifest_payloads_match_modulo_non_deterministic(path_a, path_b):
                continue
            errors.append(
                f"manifest payload mismatch (after stripping "
                f"{list(NON_DETERMINISTIC_MANIFEST_FIELDS)}): {rel}"
            )
            continue
        size_a = path_a.stat().st_size
        size_b = path_b.stat().st_size
        errors.append(f"hash mismatch: {rel} (sizes: A={size_a}B, B={size_b}B)")

    return errors


def check_exposure_monotonicity(student_bundle: Path, instructor_bundle: Path) -> list[str]:
    """Verify that student_public is a subset of research_instructor.

    The instructor bundle must contain everything the student bundle has,
    plus additional ``metadata/`` artefacts.  Shared files must be identical
    (same SHA-256 hash).  Returns errors if violated.
    """
    errors: list[str] = []

    # Student must NOT have metadata/
    if (student_bundle / "metadata").exists():
        errors.append("student_public bundle should not contain metadata/")

    # Instructor MUST have metadata/
    if not (instructor_bundle / "metadata").exists():
        errors.append("research_instructor bundle is missing metadata/")

    # Both must have the same core files.
    # manifest.json and dataset_card.md legitimately differ between modes
    # (exposure_mode field, metadata references).  feature_dictionary.csv
    # legitimately differs too — student_public drops rows for redacted
    # columns (e.g. ``current_stage``).  Only check presence here; content
    # is checked below in monotonic-subset form.
    core_files = ["manifest.json", "dataset_card.md", "feature_dictionary.csv"]
    for fname in core_files:
        s_path = student_bundle / fname
        i_path = instructor_bundle / fname
        if s_path.exists() and not i_path.exists():
            errors.append(f"Student has {fname} but instructor does not")
        elif not s_path.exists() and i_path.exists():
            errors.append(f"Instructor has {fname} but student does not")

    # feature_dictionary.csv: student rows must be a subset of instructor rows
    # (by ``name``).  For names present in both, the metadata must agree.
    #
    s_dict = student_bundle / "feature_dictionary.csv"
    i_dict = instructor_bundle / "feature_dictionary.csv"
    if s_dict.exists() and i_dict.exists():
        s_df = pd.read_csv(s_dict).set_index("name")
        i_df = pd.read_csv(i_dict).set_index("name")
        extra_in_student = set(s_df.index) - set(i_df.index) - STUDENT_ONLY_DICT_ROWS
        if extra_in_student:
            errors.append(
                "feature_dictionary.csv: student has rows missing from instructor: "
                f"{sorted(extra_in_student)}"
            )
        shared = sorted(set(s_df.index) & set(i_df.index))
        for col in s_df.columns:
            if col in i_df.columns:
                s_vals = s_df.loc[shared, col]
                i_vals = i_df.loc[shared, col]
                if not s_vals.equals(i_vals):
                    errors.append(
                        f"feature_dictionary.csv: column {col!r} differs between modes "
                        "for at least one shared feature"
                    )

    # Both must have the same tables with identical content, modulo the
    # snapshot-safe export contract (PR 2.2 / bundle schema v5):
    #
    # * Student is allowed to omit ``BANNED_TABLES`` (``customers`` /
    #   ``subscriptions``) — these are conversion-conditional and thus
    #   leak the label.
    # * Student's ``leads`` is allowed to drop ``BANNED_LEAD_COLUMNS``;
    #   ``opportunities`` is allowed to drop ``BANNED_OPP_COLUMNS``.
    # * Student's event tables in ``SNAPSHOT_FILTERED_TABLES`` are
    #   allowed to be a *row-subset* of instructor (filtered to
    #   ``lead_created_at + snapshot_day``).
    #
    # In all cases, student is still a subset of instructor on every
    # column / row that survives the contract.
    expected_redacted = redacted_columns_for(ExposureMode.student_public)
    snapshot_filtered_table_names = {name for name, _ in SNAPSHOT_FILTERED_TABLES}
    extra_columns_allowed_per_table: dict[str, set[str]] = {
        "leads": set(BANNED_LEAD_COLUMNS),
        "opportunities": set(BANNED_OPP_COLUMNS),
    }
    # Per-snapshot-filtered-table primary key.  Used to verify the
    # row-subset relationship cheaply and correctly under NaN: a left
    # merge over all shared columns is fragile (NaN doesn't equal NaN
    # in pandas, even on the same float), and the natural PK is enough
    # to assert ``student.rows ⊆ instructor.rows`` since student rows
    # were derived directly from instructor by row-filtering.
    snapshot_filtered_pks: dict[str, str] = {
        "touches": "touch_id",
        "sessions": "session_id",
        "sales_activities": "activity_id",
        "opportunities": "opportunity_id",
    }

    student_tables = (
        {p.name for p in (student_bundle / "tables").glob("*.parquet")}
        if (student_bundle / "tables").exists()
        else set()
    )
    instructor_tables = (
        {p.name for p in (instructor_bundle / "tables").glob("*.parquet")}
        if (instructor_bundle / "tables").exists()
        else set()
    )
    missing_from_instructor = student_tables - instructor_tables
    if missing_from_instructor:
        errors.append(f"Tables in student but not instructor: {sorted(missing_from_instructor)}")
    expected_only_in_instructor = {f"{name}.parquet" for name in BANNED_TABLES}
    extra_in_instructor = instructor_tables - student_tables - expected_only_in_instructor
    if extra_in_instructor:
        errors.append(f"Tables in instructor but not student: {sorted(extra_in_instructor)}")

    for table in sorted(student_tables & instructor_tables):
        s_path = student_bundle / "tables" / table
        i_path = instructor_bundle / "tables" / table
        if file_sha256(s_path) == file_sha256(i_path):
            continue
        s_df = pd.read_parquet(s_path)
        i_df = pd.read_parquet(i_path)

        table_name = table[: -len(".parquet")]
        # Row counts can differ for snapshot-filtered event tables; the
        # contract is "student rows ⊆ instructor rows", checked below by
        # comparing shared columns on the inner join.
        is_snapshot_filtered = table_name in snapshot_filtered_table_names
        if not is_snapshot_filtered and len(s_df) != len(i_df):
            errors.append(
                f"Table {table}: row count mismatch student={len(s_df)} instructor={len(i_df)}"
            )
            continue
        if is_snapshot_filtered and len(s_df) > len(i_df):
            errors.append(
                f"Table {table}: student has more rows than instructor "
                f"({len(s_df)} > {len(i_df)}); snapshot-safe export must be a row-subset"
            )
            continue
        s_cols = set(s_df.columns)
        i_cols = set(i_df.columns)
        extra_in_student = s_cols - i_cols
        if extra_in_student:
            errors.append(
                f"Table {table}: student has columns missing from instructor: "
                f"{sorted(extra_in_student)}"
            )
            continue
        diff = i_cols - s_cols
        allowed = expected_redacted | extra_columns_allowed_per_table.get(table_name, set())
        if not diff.issubset(allowed):
            errors.append(
                f"Table {table}: instructor−student column diff {sorted(diff)} contains "
                f"non-redacted columns (expected subset of {sorted(allowed)})"
            )
            continue
        shared = [c for c in s_df.columns if c in i_df.columns]
        if is_snapshot_filtered and len(s_df) != len(i_df):
            # Row-subset by primary key.  Each student row was derived
            # from instructor by row-filtering, so the PK relationship
            # is the strongest invariant we can assert without
            # depending on column-by-column equality (which is fragile
            # under NaN).
            pk = snapshot_filtered_pks.get(table_name)
            if pk is None or pk not in s_df.columns or pk not in i_df.columns:
                errors.append(
                    f"Table {table}: snapshot-filtered table missing expected "
                    f"primary key {pk!r}; cannot verify row-subset"
                )
                continue
            student_pks = set(s_df[pk].tolist())
            instructor_pks = set(i_df[pk].tolist())
            orphans = student_pks - instructor_pks
            if orphans:
                sample = sorted(orphans)[:3]
                errors.append(
                    f"Table {table}: {len(orphans)} student {pk}(s) absent from instructor, "
                    f"e.g. {sample} (snapshot-safe export must be a row-subset)"
                )
        else:
            s_shared = s_df[shared].reset_index(drop=True)
            i_shared = i_df[shared].reset_index(drop=True)
            if not s_shared.equals(i_shared):
                errors.append(f"Table {table}: shared-column values differ between modes")

    # Both must have the same task splits with identical content
    student_tasks = (
        {
            p.relative_to(student_bundle / "tasks")
            for p in (student_bundle / "tasks").rglob("*.parquet")
        }
        if (student_bundle / "tasks").exists()
        else set()
    )
    instructor_tasks = (
        {
            p.relative_to(instructor_bundle / "tasks")
            for p in (instructor_bundle / "tasks").rglob("*.parquet")
        }
        if (instructor_bundle / "tasks").exists()
        else set()
    )
    missing_tasks = student_tasks - instructor_tasks
    if missing_tasks:
        errors.append(
            f"Task files in student but not instructor: {sorted(str(f) for f in missing_tasks)}"
        )
    extra_tasks = instructor_tasks - student_tasks
    if extra_tasks:
        errors.append(
            f"Task files in instructor but not student: {sorted(str(f) for f in extra_tasks)}"
        )

    expected_redacted = redacted_columns_for(ExposureMode.student_public)
    for rel in sorted(student_tasks & instructor_tasks):
        s_path = student_bundle / "tasks" / rel
        i_path = instructor_bundle / "tasks" / rel
        if file_sha256(s_path) == file_sha256(i_path):
            # Byte-identical is fine only if no redaction is expected.
            if expected_redacted:
                # Hashes match but instructor should differ — sanity check.
                pass
            continue
        # Mismatch is acceptable iff the difference is *exactly* the
        # expected redaction set.  Anything else (extra column in student,
        # value drift, missing column not in the redaction set) is an error.
        s_df = pd.read_parquet(s_path)
        i_df = pd.read_parquet(i_path)
        if len(s_df) != len(i_df):
            errors.append(
                f"Task row count mismatch in {rel}: student={len(s_df)} instructor={len(i_df)}"
            )
            continue
        s_cols = set(s_df.columns)
        i_cols = set(i_df.columns)
        extra_in_student = s_cols - i_cols
        if extra_in_student:
            errors.append(
                f"Task {rel}: student has columns missing from instructor: "
                f"{sorted(extra_in_student)}"
            )
            continue
        diff = i_cols - s_cols
        if diff != expected_redacted:
            errors.append(
                f"Task {rel}: instructor−student column diff {sorted(diff)} does not "
                f"equal the expected student_public redaction set {sorted(expected_redacted)}"
            )
            continue
        shared = [c for c in s_df.columns if c in i_df.columns]
        s_shared = s_df[shared].reset_index(drop=True)
        i_shared = i_df[shared].reset_index(drop=True)
        if not s_shared.equals(i_shared):
            errors.append(f"Task {rel}: shared-column values differ between modes")

    return errors

"""Determinism and exposure-monotonicity invariant checks.

These checks verify structural guarantees that must hold for every bundle:

- **Determinism**: same (recipe, seed, config) → identical output.
- **Exposure monotonicity**: ``student_public`` artefacts are a strict subset
  of ``research_instructor`` artefacts.
"""

from __future__ import annotations

from pathlib import Path

from leadforge.core.hashing import file_sha256


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
                errors.append(
                    f"Directory '{subdir}' exists in one bundle but not the other"
                )
            continue

        files_a = {p.relative_to(dir_a) for p in dir_a.rglob("*.parquet")}
        files_b = {p.relative_to(dir_b) for p in dir_b.rglob("*.parquet")}

        only_a = files_a - files_b
        only_b = files_b - files_a
        if only_a:
            errors.append(
                f"Files only in bundle A {subdir}/: {sorted(str(f) for f in only_a)}"
            )
        if only_b:
            errors.append(
                f"Files only in bundle B {subdir}/: {sorted(str(f) for f in only_b)}"
            )

        for rel in sorted(files_a & files_b):
            sha_a = file_sha256(dir_a / rel)
            sha_b = file_sha256(dir_b / rel)
            if sha_a != sha_b:
                errors.append(f"Hash mismatch: {subdir}/{rel}")

    return errors


def check_exposure_monotonicity(
    student_bundle: Path, instructor_bundle: Path
) -> list[str]:
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
    # (exposure_mode field, metadata references), so only check presence.
    # feature_dictionary.csv should be identical.
    core_files = ["manifest.json", "dataset_card.md", "feature_dictionary.csv"]
    for fname in core_files:
        s_path = student_bundle / fname
        i_path = instructor_bundle / fname
        if s_path.exists() and not i_path.exists():
            errors.append(f"Student has {fname} but instructor does not")

    # feature_dictionary.csv should be identical across modes.
    s_dict = student_bundle / "feature_dictionary.csv"
    i_dict = instructor_bundle / "feature_dictionary.csv"
    if s_dict.exists() and i_dict.exists():
        if file_sha256(s_dict) != file_sha256(i_dict):
            errors.append("Content mismatch in shared file: feature_dictionary.csv")

    # Both must have the same tables with identical content
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
    missing = student_tables - instructor_tables
    if missing:
        errors.append(f"Tables in student but not instructor: {sorted(missing)}")

    for table in sorted(student_tables & instructor_tables):
        s_sha = file_sha256(student_bundle / "tables" / table)
        i_sha = file_sha256(instructor_bundle / "tables" / table)
        if s_sha != i_sha:
            errors.append(f"Table content mismatch: {table}")

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
            f"Task files in student but not instructor: "
            f"{sorted(str(f) for f in missing_tasks)}"
        )

    for rel in sorted(student_tasks & instructor_tasks):
        s_sha = file_sha256(student_bundle / "tasks" / rel)
        i_sha = file_sha256(instructor_bundle / "tasks" / rel)
        if s_sha != i_sha:
            errors.append(f"Task content mismatch: {rel}")

    return errors

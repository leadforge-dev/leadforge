"""Tests for ``release/docs/relational_table_schemas.csv``.

The CSV is hand-authored per-column documentation that the Kaggle
packager threads into ``resources[].schema.fields[].description`` and
that the preview's per-table schema panel renders.  These tests are
the only thing standing between the bundle and a ``description: TODO``
row that nobody notices on review.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CSV_PATH = _REPO_ROOT / "release" / "docs" / "relational_table_schemas.csv"
_INSTRUCTOR_TABLES = _REPO_ROOT / "release" / "intermediate_instructor" / "tables"

_REQUIRED_COLUMNS = ("table", "column", "dtype", "description", "bundle_visibility")
_ALLOWED_DTYPES = frozenset({"string", "int64", "bool", "float64"})
_ALLOWED_VISIBILITIES = frozenset({"public+instructor", "instructor_only"})
_MIN_DESCRIPTION_CHARS = 12
_EXPECTED_TABLES = frozenset(
    {
        "accounts",
        "contacts",
        "leads",
        "touches",
        "sessions",
        "sales_activities",
        "opportunities",
        "customers",
        "subscriptions",
    }
)


def _rows() -> list[dict[str, str]]:
    with _CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_csv_exists() -> None:
    assert _CSV_PATH.is_file()


def test_required_header_present() -> None:
    with _CSV_PATH.open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    for required in _REQUIRED_COLUMNS:
        assert required in header, f"missing required column {required!r}"


def test_every_table_documented() -> None:
    """All nine relational tables appear at least once."""

    tables_in_csv = {row["table"] for row in _rows()}
    missing = _EXPECTED_TABLES - tables_in_csv
    assert not missing, f"tables missing per-column docs: {sorted(missing)}"


def test_descriptions_are_non_trivial() -> None:
    """No empty or placeholder descriptions."""

    for row in _rows():
        desc = row["description"].strip()
        assert desc, f"{row['table']}.{row['column']}: empty description"
        assert len(desc) >= _MIN_DESCRIPTION_CHARS, (
            f"{row['table']}.{row['column']}: description too short ({len(desc)} chars)"
        )
        assert "TODO" not in desc.upper(), (
            f"{row['table']}.{row['column']}: description contains TODO"
        )


def test_dtypes_in_allowed_vocabulary() -> None:
    for row in _rows():
        dtype = row["dtype"].strip().lower()
        assert dtype in _ALLOWED_DTYPES, (
            f"{row['table']}.{row['column']}: dtype {dtype!r} not in {sorted(_ALLOWED_DTYPES)}"
        )


def test_bundle_visibility_in_allowed_vocabulary() -> None:
    for row in _rows():
        visibility = row["bundle_visibility"].strip()
        assert visibility in _ALLOWED_VISIBILITIES, (
            f"{row['table']}.{row['column']}: bundle_visibility {visibility!r} "
            f"not in {sorted(_ALLOWED_VISIBILITIES)}"
        )


def test_no_duplicate_rows() -> None:
    seen: set[tuple[str, str]] = set()
    for row in _rows():
        key = (row["table"], row["column"])
        assert key not in seen, f"duplicate row for {row['table']}.{row['column']}"
        seen.add(key)


@pytest.mark.skipif(not _INSTRUCTOR_TABLES.is_dir(), reason="instructor bundle not built")
def test_csv_matches_live_parquet_schemas() -> None:
    """Column-name + dtype parity with the actual parquet files.

    Uses the instructor bundle because it carries the full superset
    (public bundles drop some leads/opportunities columns and omit
    customers/subscriptions entirely — checking against public alone
    would miss those columns).
    """

    dtype_map = {
        "string": "string",
        "int64": "int64",
        "bool": "bool",
        "float64": "double",
    }

    csv_by_table: dict[str, dict[str, str]] = {}
    for row in _rows():
        csv_by_table.setdefault(row["table"], {})[row["column"]] = row["dtype"].strip().lower()

    for table, csv_cols in csv_by_table.items():
        parquet_path = _INSTRUCTOR_TABLES / f"{table}.parquet"
        if not parquet_path.is_file():
            continue
        arrow_schema = pq.read_schema(parquet_path)
        arrow_cols = {f.name: str(f.type) for f in arrow_schema}

        csv_set = set(csv_cols)
        arrow_set = set(arrow_cols)
        only_csv = csv_set - arrow_set
        only_arrow = arrow_set - csv_set
        assert not only_csv, f"{table}: CSV has columns not in parquet: {sorted(only_csv)}"
        assert not only_arrow, f"{table}: parquet has columns not in CSV: {sorted(only_arrow)}"

        for col, csv_dtype in csv_cols.items():
            expected_arrow = dtype_map.get(csv_dtype)
            actual_arrow = arrow_cols[col]
            if expected_arrow is None:
                continue
            assert actual_arrow == expected_arrow, (
                f"{table}.{col}: CSV dtype {csv_dtype!r} → expected arrow {expected_arrow!r}, "
                f"got {actual_arrow!r}"
            )

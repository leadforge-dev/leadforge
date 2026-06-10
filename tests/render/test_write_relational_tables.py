"""Tests for the shared ``write_relational_tables`` envelope helper (LTV-Pe)."""

from pathlib import Path

import pandas as pd

from leadforge.render.relational_io import write_relational_tables
from leadforge.schema.tables import read_parquet


def _dfs() -> dict[str, pd.DataFrame]:
    return {
        "accounts": pd.DataFrame({"account_id": ["a1", "a2"], "secret": [1, 2]}),
        "leads": pd.DataFrame({"lead_id": ["l1"]}),
    }


def test_writes_one_parquet_per_table_and_counts_rows(tmp_path: Path) -> None:
    counts = write_relational_tables(_dfs(), tmp_path / "tables")
    assert counts == {"accounts": 2, "leads": 1}
    assert (tmp_path / "tables" / "accounts.parquet").exists()
    assert (tmp_path / "tables" / "leads.parquet").exists()


def test_creates_nested_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "tables"
    assert not target.exists()
    write_relational_tables(_dfs(), target)
    assert target.is_dir()


def test_drops_redacted_columns_present(tmp_path: Path) -> None:
    write_relational_tables(_dfs(), tmp_path / "tables", redacted={"secret", "absent"})
    accounts = read_parquet(tmp_path / "tables" / "accounts.parquet")
    assert "secret" not in accounts.columns
    assert "account_id" in accounts.columns


def test_no_redaction_keeps_all_columns(tmp_path: Path) -> None:
    write_relational_tables(_dfs(), tmp_path / "tables")
    accounts = read_parquet(tmp_path / "tables" / "accounts.parquet")
    assert list(accounts.columns) == ["account_id", "secret"]


def test_preserves_iteration_order_in_counts(tmp_path: Path) -> None:
    counts = write_relational_tables(_dfs(), tmp_path / "tables")
    assert list(counts.keys()) == ["accounts", "leads"]

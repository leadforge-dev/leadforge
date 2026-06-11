"""Shared entity-row primitives.

This module exposes only the **scheme-agnostic** building blocks:

- :class:`EntityRowProtocol` — the structural protocol all entity row
  dataclasses satisfy.
- :func:`make_empty_dataframe` — construct a zero-row DataFrame with the right
  column types from a ``DTYPE_MAP``.
- :class:`AccountRow` — the ``accounts`` entity, shared between the
  lead-scoring and lifecycle schemes (accounts are the same real-world entity
  in both).

Lead-scoring entity rows (``ContactRow``, ``LeadRow``, …) and the lead-scoring
catalog (``ALL_ROW_TYPES``, ``TABLE_NAMES``) live in
:mod:`leadforge.schemes.lead_scoring.entities`.

Lifecycle entity rows live in :mod:`leadforge.schemes.lifecycle.entities`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Protocol

import pandas as pd


class EntityRowProtocol(Protocol):
    """Structural protocol shared by all entity row dataclasses.

    Allows typed dispatch in render code without coupling to concrete classes.
    """

    TABLE_NAME: ClassVar[str]
    DTYPE_MAP: ClassVar[dict[str, str]]

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame: ...


def make_empty_dataframe(dtype_map: dict[str, str]) -> pd.DataFrame:
    """Return a zero-row DataFrame with columns ordered as *dtype_map*."""
    return pd.DataFrame({col: pd.array([], dtype=dtype) for col, dtype in dtype_map.items()})


# ---------------------------------------------------------------------------
# accounts (shared entity — present in both lead-scoring and lifecycle bundles)
# ---------------------------------------------------------------------------


@dataclass
class AccountRow:
    """One row in the ``accounts`` table."""

    TABLE_NAME: ClassVar[str] = "accounts"
    DTYPE_MAP: ClassVar[dict[str, str]] = {
        "account_id": "string",
        "company_name": "string",
        "industry": "string",
        "region": "string",
        "employee_band": "string",
        "estimated_revenue_band": "string",
        "process_maturity_band": "string",
        "created_at": "string",
    }

    account_id: str
    company_name: str
    industry: str
    region: str
    employee_band: str
    estimated_revenue_band: str
    process_maturity_band: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def empty_dataframe(cls) -> pd.DataFrame:
        return make_empty_dataframe(cls.DTYPE_MAP)

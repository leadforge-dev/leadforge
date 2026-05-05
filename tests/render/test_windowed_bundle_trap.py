"""End-to-end test that the leakage trap remains pedagogically meaningful
once the bundle uses a windowed snapshot.

The pedagogical contract is that ``total_touches_all`` (full-horizon counts)
diverges from ``touch_count`` (windowed counts) for at least some leads.
If a careless refactor accidentally widened ``touch_count`` back to the full
horizon, the trap would silently collapse — students could no longer see
the gap that the v4 dataset card promises.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadforge.api.generator import Generator


@pytest.fixture(scope="module")
def windowed_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small student_public bundle generated with the recipe default
    ``snapshot_day`` (currently 30)."""
    out = tmp_path_factory.mktemp("windowed_trap")
    gen = Generator.from_recipe(
        "b2b_saas_procurement_v1",
        seed=42,
        exposure_mode="student_public",
    )
    gen.generate(n_leads=200, n_accounts=80, n_contacts=200).save(str(out))
    return out


def test_windowed_bundle_uses_recipe_snapshot_day(windowed_bundle: Path) -> None:
    import json

    manifest = json.loads((windowed_bundle / "manifest.json").read_text())
    assert manifest["snapshot_day"] == 30


def test_total_touches_all_dominates_touch_count(windowed_bundle: Path) -> None:
    """Invariant: ``total_touches_all >= touch_count`` for every lead."""
    train = pd.read_parquet(
        windowed_bundle / "tasks" / "converted_within_90_days" / "train.parquet",
        columns=["touch_count", "total_touches_all"],
    )
    diff = train["total_touches_all"].astype("Float64") - train["touch_count"].astype("Float64")
    assert (diff >= 0).all(), (
        "total_touches_all < touch_count for some leads — the windowed feature "
        "exceeded the full-horizon count, which contradicts the trap contract."
    )


def test_trap_gap_is_meaningful(windowed_bundle: Path) -> None:
    """At ``snapshot_day < horizon_days``, the gap must be non-trivial: some
    leads must have ``total_touches_all > touch_count``.  If not, the trap is
    pedagogically dead and the windowed snapshot wiring is broken."""
    train = pd.read_parquet(
        windowed_bundle / "tasks" / "converted_within_90_days" / "train.parquet",
        columns=["touch_count", "total_touches_all"],
    )
    diff = train["total_touches_all"].astype("Float64") - train["touch_count"].astype("Float64")
    n_with_gap = int((diff > 0).sum())
    assert n_with_gap > 0, (
        "No lead has total_touches_all > touch_count — the trap gap collapsed.\n"
        "Either snapshot_day was set to horizon_days (bundle is no longer windowed) "
        "or the build_snapshot() windowing logic regressed."
    )

"""Tests for windowed snapshot features (v4 engine changes).

Covers: snapshot_day parameter, touches_week_1, days_since_first_touch,
expected_acv, total_touches_all (leakage trap), opportunity_created.
"""

from __future__ import annotations

import pandas as pd
import pytest

from leadforge.core.models import GenerationConfig
from leadforge.core.rng import RNGRoot
from leadforge.render.snapshots import build_snapshot
from leadforge.simulation.engine import simulate_world
from leadforge.simulation.population import build_population
from leadforge.structure.sampler import sample_hidden_graph


def _make_narrative(seed: int = 42):
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed)
    assert gen.world_spec.narrative is not None
    return gen.world_spec.narrative


@pytest.fixture(scope="module")
def sim_data():
    """Run a small simulation once; share across all tests in this module."""
    config = GenerationConfig(seed=42, n_accounts=30, n_contacts=90, n_leads=80)
    narrative = _make_narrative(config.seed)
    graph = sample_hidden_graph(RNGRoot(42))
    population = build_population(config, narrative, graph)
    result = simulate_world(config, population, graph)
    return config, population, result


# ---------------------------------------------------------------------------
# Windowed snapshot basics
# ---------------------------------------------------------------------------


class TestWindowedSnapshot:
    def test_snapshot_day_produces_valid_dataframe(self, sim_data):
        config, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert len(snap) == config.n_leads
        assert "touches_week_1" in snap.columns

    def test_windowed_touch_counts_leq_full(self, sim_data):
        """Windowed touch counts should be ≤ full-horizon counts."""
        _, population, result = sim_data
        full = build_snapshot(result, population)
        windowed = build_snapshot(result, population, snapshot_day=14)
        assert (windowed["touch_count"] <= full["touch_count"]).all()

    def test_windowed_session_counts_leq_full(self, sim_data):
        _, population, result = sim_data
        full = build_snapshot(result, population)
        windowed = build_snapshot(result, population, snapshot_day=14)
        assert (windowed["session_count"] <= full["session_count"]).all()

    def test_snapshot_day_none_equals_default(self, sim_data):
        """snapshot_day=None should produce same result as omitting it."""
        _, population, result = sim_data
        default = build_snapshot(result, population)
        explicit_none = build_snapshot(result, population, snapshot_day=None)
        pd.testing.assert_frame_equal(default, explicit_none)


# ---------------------------------------------------------------------------
# touches_week_1
# ---------------------------------------------------------------------------


class TestTouchesWeek1:
    def test_non_negative(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert (snap["touches_week_1"] >= 0).all()

    def test_leq_total_touches(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert (snap["touches_week_1"] <= snap["touch_count"]).all()


# ---------------------------------------------------------------------------
# days_since_first_touch
# ---------------------------------------------------------------------------


class TestDaysSinceFirstTouch:
    def test_non_negative_when_present(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        valid = snap["days_since_first_touch"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_nan_when_no_touches(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        no_touch = snap[snap["touch_count"] == 0]
        if len(no_touch) > 0:
            assert no_touch["days_since_first_touch"].isna().all()


# ---------------------------------------------------------------------------
# total_touches_all (leakage trap)
# ---------------------------------------------------------------------------


class TestTotalTouchesAll:
    def test_present_with_snapshot_day(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert "total_touches_all" in snap.columns

    def test_geq_windowed_touch_count(self, sim_data):
        """Leakage trap uses full horizon, so should be ≥ windowed counts."""
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert (snap["total_touches_all"] >= snap["touch_count"]).all()

    def test_non_negative(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population, snapshot_day=14)
        assert (snap["total_touches_all"] >= 0).all()


# ---------------------------------------------------------------------------
# opportunity_created
# ---------------------------------------------------------------------------


class TestOpportunityCreated:
    def test_is_boolean_dtype(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population)
        assert snap["opportunity_created"].dtype.name == "boolean"

    def test_superset_of_has_open(self, sim_data):
        """Every lead with has_open_opportunity must also have opportunity_created."""
        _, population, result = sim_data
        snap = build_snapshot(result, population)
        has_open = snap["has_open_opportunity"].fillna(False)
        opp_created = snap["opportunity_created"].fillna(False)
        assert (has_open <= opp_created).all()


# ---------------------------------------------------------------------------
# expected_acv
# ---------------------------------------------------------------------------


class TestExpectedAcv:
    def test_present_in_snapshot(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population)
        assert "expected_acv" in snap.columns

    def test_positive_when_present(self, sim_data):
        _, population, result = sim_data
        snap = build_snapshot(result, population)
        valid = snap["expected_acv"].dropna()
        if len(valid) > 0:
            assert (valid > 0).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestWindowedDeterminism:
    def test_same_seed_same_output(self):
        """Windowed snapshots must be deterministic given the same seed."""

        def _snap(seed):
            cfg = GenerationConfig(seed=seed, n_accounts=15, n_contacts=45, n_leads=40)
            narr = _make_narrative(seed)
            g = sample_hidden_graph(RNGRoot(seed))
            pop = build_population(cfg, narr, g)
            res = simulate_world(cfg, pop, g)
            return build_snapshot(res, pop, snapshot_day=14)

        s1 = _snap(99)
        s2 = _snap(99)
        pd.testing.assert_frame_equal(s1, s2, check_like=False)

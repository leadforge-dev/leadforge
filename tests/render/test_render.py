"""Tests for the render layer: relational.py, snapshots.py, tasks.py, manifests.py."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from leadforge.core.models import GenerationConfig
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES
from leadforge.simulation.engine import simulate_world
from leadforge.simulation.population import build_population
from leadforge.structure.sampler import sample_hidden_graph

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SNAPSHOT_COLUMNS = [f.name for f in LEAD_SNAPSHOT_FEATURES]
_SNAPSHOT_DTYPES = {f.name: f.dtype for f in LEAD_SNAPSHOT_FEATURES}


def _make_config(seed: int = 42, n_leads: int = 80) -> GenerationConfig:
    return GenerationConfig(seed=seed, n_accounts=30, n_contacts=90, n_leads=n_leads)


def _make_narrative():
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=42)
    assert gen.world_spec.narrative is not None
    return gen.world_spec.narrative


@pytest.fixture(scope="module")
def sim_outputs():
    """Run a small simulation once; share across all tests in this module."""
    config = _make_config()
    narrative = _make_narrative()
    graph = sample_hidden_graph(42)
    population = build_population(config, narrative, graph)
    result = simulate_world(config, population, graph)
    return config, population, result, graph


# ---------------------------------------------------------------------------
# render/relational.py
# ---------------------------------------------------------------------------


class TestToDataframes:
    def test_returns_all_table_names(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.relational import to_dataframes

        dfs = to_dataframes(result, population)
        expected = {
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
        assert set(dfs.keys()) == expected

    def test_lead_count_matches(self, sim_outputs):
        config, population, result, _ = sim_outputs
        from leadforge.render.relational import to_dataframes

        dfs = to_dataframes(result, population)
        assert len(dfs["leads"]) == config.n_leads

    def test_account_and_contact_counts(self, sim_outputs):
        config, population, result, _ = sim_outputs
        from leadforge.render.relational import to_dataframes

        dfs = to_dataframes(result, population)
        assert len(dfs["accounts"]) == config.n_accounts
        assert len(dfs["contacts"]) == config.n_contacts

    def test_dataframes_are_dataframes(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.relational import to_dataframes

        dfs = to_dataframes(result, population)
        for name, df in dfs.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"

    def test_empty_tables_have_schema(self, sim_outputs):
        """Tables with zero rows must still expose the correct column names."""
        _, population, result, _ = sim_outputs
        from leadforge.render.relational import to_dataframes
        from leadforge.schema.entities import CustomerRow

        dfs = to_dataframes(result, population)
        # customers may or may not be empty, but its columns must be a superset
        # of the entity's DTYPE_MAP keys.
        assert set(CustomerRow.DTYPE_MAP.keys()).issubset(set(dfs["customers"].columns))

    def test_deterministic_under_same_seed(self):
        """Same seed → identical relational DataFrames."""
        from leadforge.render.relational import to_dataframes

        def _run(seed):
            cfg = _make_config(seed=seed)
            narr = _make_narrative()
            g = sample_hidden_graph(seed)
            pop = build_population(cfg, narr, g)
            res = simulate_world(cfg, pop, g)
            return to_dataframes(res, pop)

        dfs1 = _run(77)
        dfs2 = _run(77)
        for tbl in ("leads", "accounts", "touches"):
            pd.testing.assert_frame_equal(dfs1[tbl], dfs2[tbl], check_like=False)


# ---------------------------------------------------------------------------
# render/snapshots.py
# ---------------------------------------------------------------------------


class TestBuildSnapshot:
    def test_row_count_equals_lead_count(self, sim_outputs):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        assert len(snap) == config.n_leads

    def test_all_snapshot_columns_present(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        for col in _SNAPSHOT_COLUMNS:
            assert col in snap.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        assert set(snap.columns) == set(_SNAPSHOT_COLUMNS)

    def test_target_column_is_boolean(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        assert snap["converted_within_90_days"].dtype.name == "boolean"

    def test_touch_counts_non_negative(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        assert (snap["touch_count"].dropna() >= 0).all()
        assert (snap["inbound_touch_count"].dropna() >= 0).all()
        assert (snap["outbound_touch_count"].dropna() >= 0).all()

    def test_inbound_plus_outbound_le_total(self, sim_outputs):
        """inbound + outbound ≤ touch_count (can be less if other directions exist)."""
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        valid = snap[["touch_count", "inbound_touch_count", "outbound_touch_count"]].dropna()
        combined = valid["inbound_touch_count"] + valid["outbound_touch_count"]
        assert (combined <= valid["touch_count"]).all()

    def test_days_since_last_touch_finite_when_touches_exist(self, sim_outputs):
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        has_touch = snap["touch_count"] > 0
        if has_touch.any():
            assert snap.loc[has_touch, "days_since_last_touch"].notna().all()

    def test_no_leakage_target_not_derived_from_future(self, sim_outputs):
        """converted_within_90_days must match SimulationResult's own flag."""
        _, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot

        snap = build_snapshot(result, population)
        lead_flags = {row.lead_id: row.converted_within_90_days for row in result.leads}
        # Map lead_id → snapshot label
        snap_flags = dict(zip(snap["lead_id"], snap["converted_within_90_days"], strict=False))
        for lid, flag in lead_flags.items():
            assert snap_flags[lid] == flag, f"Mismatch on {lid}"

    def test_deterministic_under_same_seed(self):
        """Same seed → identical snapshots."""
        from leadforge.render.snapshots import build_snapshot

        def _snap(seed):
            cfg = _make_config(seed=seed)
            narr = _make_narrative()
            g = sample_hidden_graph(seed)
            pop = build_population(cfg, narr, g)
            res = simulate_world(cfg, pop, g)
            return build_snapshot(res, pop, horizon_days=cfg.horizon_days)

        s1 = _snap(13)
        s2 = _snap(13)
        pd.testing.assert_frame_equal(s1, s2, check_like=False)


# ---------------------------------------------------------------------------
# render/tasks.py
# ---------------------------------------------------------------------------


class TestWriteTaskSplits:
    def test_three_files_written(self, sim_outputs, tmp_path):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        write_task_splits(snap, tmp_path, seed=config.seed)

        task_dir = tmp_path / "converted_within_90_days"
        for split in ("train", "valid", "test"):
            assert (task_dir / f"{split}.parquet").exists(), f"{split}.parquet missing"

    def test_task_manifest_written(self, sim_outputs, tmp_path):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        write_task_splits(snap, tmp_path, seed=config.seed)

        manifest_path = tmp_path / "converted_within_90_days" / "task_manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert "task_id" in data

    def test_row_counts_sum_to_total(self, sim_outputs, tmp_path):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        counts = write_task_splits(snap, tmp_path, seed=config.seed)

        assert counts["train"] + counts["valid"] + counts["test"] == len(snap)

    def test_split_ratios_approx(self, sim_outputs, tmp_path):
        """Train ≈ 70%, valid ≈ 15%, test ≈ 15% (±5% tolerance for small samples)."""
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        counts = write_task_splits(snap, tmp_path, seed=config.seed)
        n = len(snap)
        assert counts["train"] / n == pytest.approx(0.70, abs=0.05)
        assert counts["valid"] / n == pytest.approx(0.15, abs=0.05)
        assert counts["test"] / n == pytest.approx(0.15, abs=0.05)

    def test_splits_are_disjoint(self, sim_outputs, tmp_path):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        write_task_splits(snap, tmp_path, seed=config.seed)

        task_dir = tmp_path / "converted_within_90_days"
        dfs = {s: pd.read_parquet(task_dir / f"{s}.parquet") for s in ("train", "valid", "test")}
        ids = {s: set(dfs[s]["lead_id"]) for s in dfs}
        assert ids["train"].isdisjoint(ids["valid"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["valid"].isdisjoint(ids["test"])

    def test_deterministic_under_same_seed(self, sim_outputs, tmp_path):
        config, population, result, _ = sim_outputs
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)

        p1 = tmp_path / "run1"
        p2 = tmp_path / "run2"
        c1 = write_task_splits(snap, p1, seed=config.seed)
        c2 = write_task_splits(snap, p2, seed=config.seed)
        assert c1 == c2

        task_id = "converted_within_90_days"
        for split in ("train", "valid", "test"):
            df1 = pd.read_parquet(p1 / task_id / f"{split}.parquet")
            df2 = pd.read_parquet(p2 / task_id / f"{split}.parquet")
            pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# render/manifests.py
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def _make_manifest(self, sim_outputs, tmp_path):
        config, population, result, world_graph = sim_outputs
        from leadforge.render.manifests import build_manifest
        from leadforge.render.relational import to_dataframes
        from leadforge.render.snapshots import build_snapshot
        from leadforge.render.tasks import write_task_splits
        from leadforge.schema.tables import write_parquet

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        dfs = to_dataframes(result, population)
        table_row_counts = {}
        for name, df in dfs.items():
            write_parquet(df, tables_dir / f"{name}.parquet")
            table_row_counts[name] = len(df)

        snap = build_snapshot(result, population, horizon_days=config.horizon_days)
        task_counts = write_task_splits(snap, tmp_path / "tasks", seed=config.seed)

        manifest = build_manifest(
            config=config,
            world_graph=world_graph,
            table_row_counts=table_row_counts,
            task_row_counts={"converted_within_90_days": task_counts},
            bundle_root=tmp_path,
        )
        return manifest

    def test_required_top_level_keys(self, sim_outputs, tmp_path):
        manifest = self._make_manifest(sim_outputs, tmp_path)
        required = {
            "bundle_schema_version",
            "package_version",
            "recipe_id",
            "seed",
            "generation_timestamp",
            "exposure_mode",
            "difficulty",
            "n_accounts",
            "n_contacts",
            "n_leads",
            "horizon_days",
            "motif_family",
            "tables",
            "tasks",
        }
        assert required.issubset(set(manifest.keys()))

    def test_table_row_counts_match(self, sim_outputs, tmp_path):
        config, _, _, _ = sim_outputs
        manifest = self._make_manifest(sim_outputs, tmp_path)
        assert manifest["tables"]["leads"]["row_count"] == config.n_leads
        assert manifest["tables"]["accounts"]["row_count"] == config.n_accounts
        assert manifest["tables"]["contacts"]["row_count"] == config.n_contacts

    def test_sha256_populated(self, sim_outputs, tmp_path):
        manifest = self._make_manifest(sim_outputs, tmp_path)
        for tbl, entry in manifest["tables"].items():
            assert isinstance(entry["sha256"], str), f"{tbl} sha256 is not a string"
            assert len(entry["sha256"]) == 64, f"{tbl} sha256 has wrong length"

    def test_task_split_counts_present(self, sim_outputs, tmp_path):
        manifest = self._make_manifest(sim_outputs, tmp_path)
        task = manifest["tasks"]["converted_within_90_days"]
        assert "train_rows" in task
        assert "valid_rows" in task
        assert "test_rows" in task

    def test_seed_and_recipe_recorded(self, sim_outputs, tmp_path):
        config, _, _, _ = sim_outputs
        manifest = self._make_manifest(sim_outputs, tmp_path)
        assert manifest["seed"] == config.seed
        assert manifest["recipe_id"] == config.recipe_id

    def test_manifest_is_json_serialisable(self, sim_outputs, tmp_path):
        manifest = self._make_manifest(sim_outputs, tmp_path)
        dumped = json.dumps(manifest)
        reloaded = json.loads(dumped)
        assert reloaded["seed"] == manifest["seed"]


# ---------------------------------------------------------------------------
# api/bundle.py — integration smoke test
# ---------------------------------------------------------------------------


class TestWriteBundle:
    def test_full_bundle_written(self, sim_outputs, tmp_path):
        config, population, result, world_graph = sim_outputs
        from leadforge.api.bundle import write_bundle
        from leadforge.core.models import WorldBundle, WorldSpec

        bundle = WorldBundle(
            spec=WorldSpec(config=config),
            population=population,
            simulation_result=result,
            world_graph=world_graph,
        )
        write_bundle(bundle, str(tmp_path))

        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "dataset_card.md").exists()
        assert (tmp_path / "feature_dictionary.csv").exists()
        assert (tmp_path / "tables").is_dir()
        assert (tmp_path / "tasks" / "converted_within_90_days").is_dir()

    def test_manifest_is_valid_json(self, sim_outputs, tmp_path):
        config, population, result, world_graph = sim_outputs
        from leadforge.api.bundle import write_bundle
        from leadforge.core.models import WorldBundle, WorldSpec

        bundle = WorldBundle(
            spec=WorldSpec(config=config),
            population=population,
            simulation_result=result,
            world_graph=world_graph,
        )
        write_bundle(bundle, str(tmp_path))

        data = json.loads((tmp_path / "manifest.json").read_text())
        assert data["seed"] == config.seed

    def test_unpopulated_bundle_raises(self):
        from leadforge.api.bundle import write_bundle
        from leadforge.core.models import WorldBundle

        with pytest.raises(RuntimeError, match="not fully populated"):
            write_bundle(WorldBundle(), "/tmp/leadforge_test_empty")

    def test_generator_generate_and_save(self, tmp_path):
        """End-to-end: Generator.from_recipe → generate → save."""
        from leadforge.api.generator import Generator

        gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=7)
        bundle = gen.generate(n_leads=60, n_accounts=20, n_contacts=60)
        bundle.save(str(tmp_path))

        assert (tmp_path / "manifest.json").exists()
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["n_leads"] == 60
        assert manifest["seed"] == 7

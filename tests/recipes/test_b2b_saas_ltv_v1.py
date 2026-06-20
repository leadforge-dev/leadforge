"""End-to-end tests for the b2b_saas_ltv_v1 (lifecycle / pLTV) recipe (LTV-Po.2b).

These exercise the recipe assets — recipe.yaml, narrative.yaml,
difficulty_profiles.yaml — through the public Generator API: discovery, config
resolution, the build_world round-trip, and a full write_bundle in both
exposure modes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from leadforge.api.generator import Generator
from leadforge.api.recipes import Recipe
from leadforge.recipes.registry import list_recipes, load_recipe
from leadforge.schemes.lifecycle.artifacts import LifecycleArtifacts

_RECIPE_ID = "b2b_saas_ltv_v1"
_TS = "2026-01-01T00:00:00+00:00"
_SMALL = 120

_PUBLIC_TASKS = {
    "pltv_revenue_90d",
    "pltv_revenue_365d",
    "pltv_revenue_730d",
    "churned_within_180d",
}
_INSTRUCTOR_TASKS = _PUBLIC_TASKS | {
    "early_pltv_revenue_90d",
    "early_pltv_revenue_365d",
    "early_pltv_revenue_730d",
    "early_churned_within_180d",
}


# ---------------------------------------------------------------------------
# Discovery + recipe asset shape
# ---------------------------------------------------------------------------


def test_recipe_is_discoverable() -> None:
    ids = [r["id"] for r in list_recipes()]
    assert _RECIPE_ID in ids


def test_recipe_declares_lifecycle_scheme() -> None:
    recipe = Recipe.from_dict(load_recipe(_RECIPE_ID))
    assert recipe.scheme == "lifecycle"
    assert recipe.primary_task == "pltv_revenue_365d"
    assert recipe.default_population == {"n_customers": 1500}


def test_recipe_supports_both_modes_and_all_tiers() -> None:
    from leadforge.core.enums import DifficultyProfile, ExposureMode

    recipe = Recipe.from_dict(load_recipe(_RECIPE_ID))
    assert set(recipe.supported_modes) == {
        ExposureMode.student_public,
        ExposureMode.research_instructor,
    }
    assert set(recipe.supported_difficulty) == set(DifficultyProfile)


def test_narrative_declares_multi_value_firmographics() -> None:
    """student_public invariant #6: industry/region must not be zero-variance,
    so the recipe narrative must declare >= 2 industries and >= 2 geographies."""
    recipe = Recipe.from_dict(load_recipe(_RECIPE_ID))
    market = recipe.load_narrative()["market"]
    assert len(market["icp_industries"]) >= 2
    assert len(market["geographies"]) >= 2


def test_difficulty_profiles_present_for_every_tier() -> None:
    recipe = Recipe.from_dict(load_recipe(_RECIPE_ID))
    profiles = recipe.load_difficulty_profiles()
    assert {"intro", "intermediate", "advanced"} <= set(profiles)
    for tier in ("intro", "intermediate", "advanced"):
        for key in ("signal_strength", "noise_scale", "missing_rate", "outlier_rate"):
            assert key in profiles[tier], f"{tier} missing {key}"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_resolve_config_carries_n_customers_from_default_population() -> None:
    recipe = Recipe.from_dict(load_recipe(_RECIPE_ID))
    config = recipe.resolve_config(seed=42)
    assert config.recipe_id == _RECIPE_ID
    assert config.n_customers == 1500


# ---------------------------------------------------------------------------
# build_world round-trip via Generator
# ---------------------------------------------------------------------------


def test_generate_returns_lifecycle_artifacts() -> None:
    gen = Generator.from_recipe(_RECIPE_ID, seed=42, n_customers=_SMALL)
    assert gen.world_spec.scheme == "lifecycle"
    bundle = gen.generate()
    assert isinstance(bundle.artifacts, LifecycleArtifacts)
    assert bundle.spec.scheme == "lifecycle"
    assert len(bundle.artifacts.population.customers) == _SMALL


def test_generate_resolves_difficulty_params_from_recipe() -> None:
    gen = Generator.from_recipe(_RECIPE_ID, seed=42, n_customers=_SMALL, difficulty="advanced")
    bundle = gen.generate()
    params = bundle.spec.config.difficulty_params
    assert params is not None
    # The advanced tier's knobs (from difficulty_profiles.yaml) flowed through.
    assert params.noise_scale == 0.55
    assert params.missing_rate == 0.18


def test_narrative_drives_population_firmographics() -> None:
    gen = Generator.from_recipe(_RECIPE_ID, seed=42, n_customers=_SMALL)
    accounts = gen.generate().artifacts.population.accounts
    market = Recipe.from_dict(load_recipe(_RECIPE_ID)).load_narrative()["market"]
    seen_industries = {a.industry for a in accounts}
    seen_regions = {a.region for a in accounts}
    assert seen_industries <= set(market["icp_industries"])
    assert seen_regions <= set(market["geographies"])
    # A 120-customer world should surface variety from the multi-value vocab.
    assert len(seen_industries) >= 2
    assert len(seen_regions) >= 2


def test_generate_is_deterministic() -> None:
    a = Generator.from_recipe(_RECIPE_ID, seed=7, n_customers=_SMALL).generate()
    b = Generator.from_recipe(_RECIPE_ID, seed=7, n_customers=_SMALL).generate()
    assert a.artifacts.motif_family == b.artifacts.motif_family
    assert [s.to_dict() for s in a.artifacts.simulation_result.subscriptions] == [
        s.to_dict() for s in b.artifacts.simulation_result.subscriptions
    ]


# ---------------------------------------------------------------------------
# Full bundle round-trip on disk (both exposure modes)
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, *, mode: str, difficulty: str = "intermediate") -> Path:
    gen = Generator.from_recipe(
        _RECIPE_ID, seed=42, n_customers=150, exposure_mode=mode, difficulty=difficulty
    )
    bundle = gen.generate()
    out = tmp_path / mode
    bundle.save(str(out), generation_timestamp=_TS)
    return out


def test_public_bundle_round_trip(tmp_path) -> None:
    out = _write(tmp_path, mode="student_public")
    assert (out / "manifest.json").is_file()
    assert not (out / "metadata").exists()
    task_dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert task_dirs == _PUBLIC_TASKS  # early-pLTV family omitted from public
    m = json.loads((out / "manifest.json").read_text())
    assert m["generation_scheme"] == "lifecycle"
    assert m["relational_snapshot_safe"] is True


def test_instructor_bundle_round_trip(tmp_path) -> None:
    out = _write(tmp_path, mode="research_instructor")
    assert (out / "metadata").is_dir()
    task_dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert task_dirs == _INSTRUCTOR_TASKS
    m = json.loads((out / "manifest.json").read_text())
    assert m["relational_snapshot_safe"] is False


def test_public_industry_region_features_have_variance(tmp_path) -> None:
    """The multi-value narrative vocab must produce >= 2 distinct values in the
    public snapshot's firmographic features (student_public invariant #6)."""
    out = _write(tmp_path, mode="student_public")
    train = pd.read_parquet(out / "tasks" / "pltv_revenue_365d" / "train.parquet")
    for col in ("industry", "region"):
        if col in train.columns:
            assert train[col].nunique(dropna=True) >= 2, f"{col} is zero-variance"


def test_full_bundle_byte_deterministic(tmp_path) -> None:
    def hashes(root: Path) -> dict[str, str]:
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    a = _write(tmp_path / "a", mode="research_instructor")
    b = _write(tmp_path / "b", mode="research_instructor")
    assert hashes(a) == hashes(b)

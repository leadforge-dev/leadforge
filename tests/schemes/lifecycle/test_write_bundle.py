"""Tests for LifecycleScheme.write_bundle — instructor-mode bundle (LTV-Pn.4b)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from leadforge.core.models import DifficultyParams, GenerationConfig
from leadforge.schemes import get_scheme
from leadforge.schemes.lifecycle.snapshots import CHURN_WINDOW_DAYS, FORWARD_WINDOWS_DAYS

_TS = "2026-01-01T00:00:00+00:00"
_EXPECTED_TASK_IDS = {
    "pltv_revenue_90d",
    "pltv_revenue_365d",
    "pltv_revenue_730d",
    "churned_within_180d",
    "early_pltv_revenue_90d",
    "early_pltv_revenue_365d",
    "early_pltv_revenue_730d",
    "early_churned_within_180d",
}


def _config(**kw) -> GenerationConfig:
    base = {
        "seed": 42,
        "n_customers": 150,
        "recipe_id": "b2b_saas_ltv_v1",
        "exposure_mode": "research_instructor",
    }
    base.update(kw)
    return GenerationConfig(**base)


def _write(tmp_path: Path, config: GenerationConfig | None = None) -> Path:
    config = config or _config()
    scheme = get_scheme("lifecycle")
    bundle = scheme.build_world(config, narrative=None)
    out = tmp_path / "bundle"
    scheme.write_bundle(bundle, str(out), generation_timestamp=_TS)
    return out


# ---------------------------------------------------------------------------
# Bundle shape
# ---------------------------------------------------------------------------


def test_required_bundle_files_present(tmp_path) -> None:
    out = _write(tmp_path)
    for f in ("manifest.json", "dataset_card.md", "feature_dictionary.csv"):
        assert (out / f).is_file(), f"missing {f}"
    assert (out / "tables").is_dir()
    assert (out / "tasks").is_dir()
    assert (out / "metadata").is_dir()  # research_instructor


def test_six_relational_tables(tmp_path) -> None:
    out = _write(tmp_path)
    tables = {p.stem for p in (out / "tables").glob("*.parquet")}
    assert tables == {
        "accounts",
        "customers",
        "subscriptions",
        "subscription_events",
        "health_signals",
        "invoices",
    }


def test_eight_task_directories(tmp_path) -> None:
    out = _write(tmp_path)
    task_dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert task_dirs == _EXPECTED_TASK_IDS
    for td in (out / "tasks").iterdir():
        for split in ("train", "valid", "test"):
            assert (td / f"{split}.parquet").is_file(), f"{td.name} missing {split}"
        assert (td / "task_manifest.json").is_file()


def test_task_manifest_types(tmp_path) -> None:
    out = _write(tmp_path)
    for td in (out / "tasks").iterdir():
        m = json.loads((td / "task_manifest.json").read_text())
        if "pltv_revenue" in td.name:
            assert m["task_type"] == "regression"
            assert m["label_column"].startswith("ltv_revenue_")
        else:
            assert m["task_type"] == "binary_classification"
            assert m["label_column"] == "churned_within_180d"
            assert m["label_window_days"] == CHURN_WINDOW_DAYS


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_records_scheme_and_lifecycle_fields(tmp_path) -> None:
    out = _write(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    assert m["generation_scheme"] == "lifecycle"
    assert m["bundle_schema_version"] == "6"
    assert m["motif_family"] in {
        "product_led_retention",
        "relationship_led_retention",
        "expansion_led_growth",
        "payment_fragile",
        "churner_dominated",
    }
    assert m["forward_windows_days"] == list(FORWARD_WINDOWS_DAYS)
    assert m["observation_date"]  # non-empty ISO date
    assert set(m["tasks"]) == _EXPECTED_TASK_IDS
    assert len(m["tables"]) == 6


# ---------------------------------------------------------------------------
# Hidden-truth metadata
# ---------------------------------------------------------------------------


def test_metadata_files(tmp_path) -> None:
    out = _write(tmp_path)
    meta = out / "metadata"
    # world_spec.json (shared, generic) + lifecycle hidden truth; NO graph.
    assert (meta / "world_spec.json").is_file()
    assert (meta / "latent_registry.json").is_file()
    assert (meta / "mechanism_summary.json").is_file()
    assert not (meta / "graph.json").exists()

    latents = json.loads((meta / "latent_registry.json").read_text())
    assert set(latents) == {"account_latents", "customer_latents"}
    mech = json.loads((meta / "mechanism_summary.json").read_text())
    assert set(mech) == {
        "motif_family",
        "churn_hazard",
        "expansion_propensity",
        "payment_failure",
    }


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_bundle_deterministic(tmp_path) -> None:
    import hashlib

    def hashes(root: Path) -> dict[str, str]:
        return {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    a = _write(tmp_path / "a")
    b = _write(tmp_path / "b")
    assert hashes(a) == hashes(b)


# ---------------------------------------------------------------------------
# Difficulty threading (the LTV-Pn.4a pinned obligation)
# ---------------------------------------------------------------------------


def test_difficulty_params_thread_into_snapshots(tmp_path) -> None:
    """config.difficulty_params must reach the snapshot builders — with strong
    distortion knobs the task features differ from an undistorted bundle.
    (Recipe-driven resolution of difficulty_params lands in LTV-Po; this proves
    the wiring so that resolution will take effect.)"""
    params = DifficultyParams(
        signal_strength=1.0,
        noise_scale=1.0,
        missing_rate=0.3,
        outlier_rate=0.05,
        conversion_rate_lo=0.02,
        conversion_rate_hi=0.4,
        committee_friction=0.5,
    )
    plain = _write(tmp_path / "plain")
    distorted = _write(tmp_path / "distorted", config=_config(difficulty_params=params))

    plain_df = pd.read_parquet(plain / "tasks" / "pltv_revenue_365d" / "train.parquet")
    dist_df = pd.read_parquet(distorted / "tasks" / "pltv_revenue_365d" / "train.parquet")
    # A numeric feature column should differ once distortions are applied.
    assert not plain_df["avg_active_users_l12w"].equals(dist_df["avg_active_users_l12w"])
    # Targets are never distorted (the distortion helper excludes them).
    assert plain_df["ltv_revenue_365d"].equals(dist_df["ltv_revenue_365d"])


# ---------------------------------------------------------------------------
# Exposure guard
# ---------------------------------------------------------------------------


def test_student_public_refused_until_pn4c(tmp_path) -> None:
    scheme = get_scheme("lifecycle")
    bundle = scheme.build_world(_config(exposure_mode="student_public"), narrative=None)
    with pytest.raises(NotImplementedError, match="LTV-Pn.4c"):
        scheme.write_bundle(bundle, str(tmp_path / "public"), generation_timestamp=_TS)


def test_unpopulated_bundle_refused(tmp_path) -> None:
    from leadforge.core.models import WorldBundle

    scheme = get_scheme("lifecycle")
    with pytest.raises(RuntimeError, match="lifecycle artifacts"):
        scheme.write_bundle(WorldBundle(), str(tmp_path / "x"), generation_timestamp=_TS)


def test_lead_scoring_artifacts_rejected_by_lifecycle_writer(tmp_path) -> None:
    # Defensive: a bundle from the wrong scheme must not silently half-write.
    from leadforge.api.generator import Generator

    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=1)
    ls_bundle = gen.generate(n_accounts=10, n_contacts=30, n_leads=30, difficulty="intro")
    # Re-label it as lifecycle to force the dispatch onto the lifecycle writer.
    mislabeled = dataclasses.replace(
        ls_bundle, spec=dataclasses.replace(ls_bundle.spec, scheme="lifecycle")
    )
    with pytest.raises(RuntimeError, match="lifecycle artifacts"):
        get_scheme("lifecycle").write_bundle(
            mislabeled, str(tmp_path / "y"), generation_timestamp=_TS
        )

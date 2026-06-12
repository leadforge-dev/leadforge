"""Tests for the scheme-agnostic WorldBundle + metadata hook seam (LTV-Pn.2)."""

from __future__ import annotations

import json

import pytest

from leadforge.api.generator import Generator
from leadforge.core.models import WorldBundle, WorldSpec
from leadforge.exposure.metadata import write_world_spec_json
from leadforge.schemes import get_scheme
from leadforge.schemes.lead_scoring.artifacts import LeadScoringArtifacts


def _populated_bundle() -> WorldBundle:
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=7)
    return gen.generate(n_accounts=20, n_contacts=60, n_leads=60, difficulty="intro")


# ---------------------------------------------------------------------------
# WorldBundle holds scheme-owned artifacts (cleanup #3)
# ---------------------------------------------------------------------------


def test_build_world_populates_scheme_artifacts() -> None:
    bundle = _populated_bundle()
    assert isinstance(bundle.artifacts, LeadScoringArtifacts)
    assert bundle.artifacts.population is not None
    assert bundle.artifacts.simulation_result is not None
    assert bundle.artifacts.world_graph is not None


def test_worldbundle_has_no_scheme_typed_fields() -> None:
    """The generalized bundle exposes only spec + opaque artifacts; the old
    lead-scoring-typed fields are gone (the core->scheme layering inversion)."""
    field_names = {f.name for f in WorldBundle.__dataclass_fields__.values()}
    assert field_names == {"spec", "artifacts"}


# ---------------------------------------------------------------------------
# Generic world_spec writer is scheme-agnostic
# ---------------------------------------------------------------------------


def test_write_world_spec_json_needs_only_spec(tmp_path) -> None:
    # Depends on WorldSpec alone — no scheme artifacts required.
    spec = WorldSpec()
    write_world_spec_json(spec, tmp_path)
    data = json.loads((tmp_path / "world_spec.json").read_text())
    assert set(data) == {"config", "narrative"}
    assert data["narrative"] is None  # default spec has no narrative


# ---------------------------------------------------------------------------
# apply_exposure dispatches to the producing scheme's write_metadata hook
# ---------------------------------------------------------------------------


def test_lead_scoring_write_metadata_emits_hidden_truth(tmp_path) -> None:
    bundle = _populated_bundle()
    meta = tmp_path / "metadata"
    meta.mkdir()
    get_scheme(bundle.spec.scheme).write_metadata(bundle, meta)
    for fname in ("graph.json", "graph.graphml", "latent_registry.json", "mechanism_summary.json"):
        assert (meta / fname).exists(), f"hook did not emit {fname}"


def test_write_metadata_rejects_unpopulated_bundle(tmp_path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    with pytest.raises(RuntimeError, match="lead-scoring artifacts"):
        get_scheme("lead_scoring").write_metadata(WorldBundle(), meta)


def test_lifecycle_metadata_hook_is_stubbed(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        get_scheme("lifecycle").write_metadata(WorldBundle(), tmp_path)

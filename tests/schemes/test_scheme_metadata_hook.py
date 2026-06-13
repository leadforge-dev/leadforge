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


# ---------------------------------------------------------------------------
# apply_exposure starts from a clean metadata/ (Copilot review on #122)
# ---------------------------------------------------------------------------


def test_apply_exposure_clears_stale_metadata(tmp_path) -> None:
    """A reused output path must not retain hidden-truth files that the current
    bundle did not write — critical once a different scheme (with a different
    file set) regenerates over the same path."""
    from leadforge.core.enums import ExposureMode
    from leadforge.exposure.modes import apply_exposure

    bundle = _populated_bundle()

    # Pre-seed a stale metadata/ with a file no scheme writes.
    meta = tmp_path / "metadata"
    meta.mkdir()
    stale = meta / "stale_graph.graphml"
    stale.write_text("<orphan/>")

    apply_exposure(bundle, tmp_path, ExposureMode.research_instructor)

    assert not stale.exists(), "stale metadata file survived the rewrite"
    # The current bundle's hidden-truth files are present.
    assert (meta / "graph.json").exists()
    assert (meta / "world_spec.json").exists()


def test_apply_exposure_student_public_removes_metadata(tmp_path) -> None:
    from leadforge.core.enums import ExposureMode
    from leadforge.exposure.modes import apply_exposure

    bundle = _populated_bundle()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "graph.json").write_text("{}")

    apply_exposure(bundle, tmp_path, ExposureMode.student_public)
    assert not meta.exists(), "student_public must not retain a metadata/ dir"

"""Tests for leadforge.exposure — ExposureMode filtering and metadata writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leadforge.api.generator import Generator
from leadforge.core.enums import ExposureMode
from leadforge.exposure.filters import FILTERS, BundleFilter, get_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SMALL_GENERATE_KWARGS: dict[str, int] = {"n_leads": 30, "n_accounts": 15, "n_contacts": 45}


def _make_bundle(mode: str, seed: int = 42):
    gen = Generator.from_recipe("b2b_saas_procurement_v1", seed=seed, exposure_mode=mode)
    return gen.generate(**_SMALL_GENERATE_KWARGS)


# ---------------------------------------------------------------------------
# Unit tests — BundleFilter / FILTERS
# ---------------------------------------------------------------------------


class TestFilters:
    def test_all_modes_have_filter(self) -> None:
        for mode in ExposureMode:
            assert mode in FILTERS, f"{mode!r} has no entry in FILTERS"

    def test_student_public_no_metadata(self) -> None:
        f = get_filter(ExposureMode.student_public)
        assert isinstance(f, BundleFilter)
        assert f.write_metadata is False

    def test_research_instructor_writes_metadata(self) -> None:
        f = get_filter(ExposureMode.research_instructor)
        assert f.write_metadata is True

    def test_unknown_mode_raises(self) -> None:
        """get_filter must raise KeyError for an unregistered mode string."""
        with pytest.raises(KeyError):
            get_filter("totally_fake_mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests — write_bundle via WorldBundle.save
# ---------------------------------------------------------------------------


class TestStudentPublicMode:
    def test_no_metadata_dir(self, tmp_path: Path) -> None:
        bundle = _make_bundle("student_public")
        bundle.save(str(tmp_path))
        assert not (tmp_path / "metadata").exists()

    def test_core_files_present(self, tmp_path: Path) -> None:
        bundle = _make_bundle("student_public")
        bundle.save(str(tmp_path))
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "dataset_card.md").exists()
        assert (tmp_path / "feature_dictionary.csv").exists()
        assert (tmp_path / "tables").is_dir()
        assert (tmp_path / "tasks").is_dir()


class TestResearchInstructorMode:
    def test_metadata_dir_created(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        assert (tmp_path / "metadata").is_dir()

    def test_all_metadata_files_present(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        meta = tmp_path / "metadata"
        for fname in (
            "graph.json",
            "graph.graphml",
            "world_spec.json",
            "latent_registry.json",
            "mechanism_summary.json",
        ):
            assert (meta / fname).exists(), f"Missing metadata file: {fname}"

    def test_graph_json_valid(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "graph.json").read_text())
        assert "nodes" in data
        assert "edges" in data
        assert "motif_family" in data

    def test_graph_graphml_valid_xml(self, tmp_path: Path) -> None:
        import xml.etree.ElementTree as ET  # stdlib

        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        text = (tmp_path / "metadata" / "graph.graphml").read_text()
        # Must parse without error.
        ET.fromstring(text)  # noqa: S314 — bundle data we generated, not external input

    def test_latent_registry_keys(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "latent_registry.json").read_text())
        assert set(data.keys()) == {"account_latents", "contact_latents", "lead_latents"}

    def test_latent_registry_populated(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "latent_registry.json").read_text())
        # Each registry should be non-empty.
        assert len(data["account_latents"]) > 0
        assert len(data["contact_latents"]) > 0
        assert len(data["lead_latents"]) > 0

    def test_latent_registry_values_in_unit_interval(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "latent_registry.json").read_text())
        for registry_key in ("account_latents", "contact_latents", "lead_latents"):
            for entity_id, traits in data[registry_key].items():
                for trait_name, value in traits.items():
                    assert 0.0 <= value <= 1.0, (
                        f"{registry_key}[{entity_id!r}][{trait_name!r}] = {value} out of [0, 1]"
                    )

    def test_world_spec_json_keys(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "world_spec.json").read_text())
        assert "config" in data
        assert "narrative" in data

    def test_world_spec_config_matches_bundle(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor", seed=77)
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "world_spec.json").read_text())
        assert data["config"]["seed"] == 77
        assert data["config"]["recipe_id"] == "b2b_saas_procurement_v1"

    def test_mechanism_summary_keys(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        data = json.loads((tmp_path / "metadata" / "mechanism_summary.json").read_text())
        assert "motif_family" in data
        assert "conversion_hazard" in data
        assert "stage_transition" in data
        assert "touch_intensity" in data
        assert "measurement" in data

    def test_mechanism_summary_motif_matches_graph(self, tmp_path: Path) -> None:
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        graph_data = json.loads((tmp_path / "metadata" / "graph.json").read_text())
        mech_data = json.loads((tmp_path / "metadata" / "mechanism_summary.json").read_text())
        assert graph_data["motif_family"] == mech_data["motif_family"]

    def test_core_files_still_present(self, tmp_path: Path) -> None:
        """Metadata write must not replace or skip the standard bundle files."""
        bundle = _make_bundle("research_instructor")
        bundle.save(str(tmp_path))
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "dataset_card.md").exists()
        assert (tmp_path / "feature_dictionary.csv").exists()
        assert (tmp_path / "tables").is_dir()
        assert (tmp_path / "tasks").is_dir()


class TestModeDeterminism:
    def test_same_seed_same_latent_registry(self, tmp_path: Path) -> None:
        p1 = tmp_path / "run1"
        p2 = tmp_path / "run2"
        _make_bundle("research_instructor", seed=42).save(str(p1))
        _make_bundle("research_instructor", seed=42).save(str(p2))
        d1 = json.loads((p1 / "metadata" / "latent_registry.json").read_text())
        d2 = json.loads((p2 / "metadata" / "latent_registry.json").read_text())
        assert d1 == d2

    def test_different_seeds_different_latent_registries(self, tmp_path: Path) -> None:
        p1 = tmp_path / "run1"
        p2 = tmp_path / "run2"
        _make_bundle("research_instructor", seed=1).save(str(p1))
        _make_bundle("research_instructor", seed=2).save(str(p2))
        d1 = json.loads((p1 / "metadata" / "latent_registry.json").read_text())
        d2 = json.loads((p2 / "metadata" / "latent_registry.json").read_text())
        assert d1 != d2

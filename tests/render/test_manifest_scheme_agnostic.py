"""Unit tests for the scheme-agnostic build_manifest contract (LTV-Pn.1)."""

from __future__ import annotations

import pytest

from leadforge.core.models import GenerationConfig
from leadforge.render.manifests import BUNDLE_SCHEMA_VERSION, build_manifest


def _config() -> GenerationConfig:
    return GenerationConfig(seed=1, n_accounts=2, n_contacts=3, n_leads=4)


def _manifest(tmp_path, **kwargs):
    # Empty row-count dicts → no parquet files to hash; isolates the
    # scheme-agnostic header fields under test.
    return build_manifest(
        config=_config(),
        table_row_counts={},
        task_row_counts={},
        bundle_root=tmp_path,
        generation_timestamp="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


def test_records_generation_scheme(tmp_path) -> None:
    m = _manifest(tmp_path, generation_scheme="lifecycle")
    assert m["generation_scheme"] == "lifecycle"
    assert m["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION


def test_motif_family_defaults_to_none(tmp_path) -> None:
    # A scheme without a single named motif (e.g. lifecycle) omits it.
    m = _manifest(tmp_path, generation_scheme="lifecycle")
    assert m["motif_family"] is None


def test_motif_family_passthrough(tmp_path) -> None:
    m = _manifest(tmp_path, generation_scheme="lead_scoring", motif_family="fit_dominant")
    assert m["motif_family"] == "fit_dominant"


def test_extra_fields_merged(tmp_path) -> None:
    m = _manifest(
        tmp_path,
        generation_scheme="lifecycle",
        extra_fields={"observation_date": "2026-06-01", "forward_windows_days": [90, 365, 730]},
    )
    assert m["observation_date"] == "2026-06-01"
    assert m["forward_windows_days"] == [90, 365, 730]


def test_extra_fields_cannot_overwrite_core_keys(tmp_path) -> None:
    with pytest.raises(ValueError, match="overwrite core manifest keys"):
        _manifest(
            tmp_path,
            generation_scheme="lifecycle",
            extra_fields={"seed": 999, "generation_scheme": "evil"},
        )


def test_generation_scheme_is_required(tmp_path) -> None:
    # Positional/keyword required argument — omitting it is a TypeError.
    with pytest.raises(TypeError):
        build_manifest(  # type: ignore[call-arg]
            config=_config(),
            table_row_counts={},
            task_row_counts={},
            bundle_root=tmp_path,
        )

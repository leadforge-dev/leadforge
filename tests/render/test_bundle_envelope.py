"""Tests for the shared bundle-writing envelope (LTV-Pn.4d).

The byte-identity of both schemes' full bundles is exercised by their own
suites + the generator round-trip; these tests pin the envelope's own contract
(ordering, all artefacts written, manifest fields) directly on a minimal bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from leadforge.core.models import GenerationConfig, WorldBundle, WorldSpec
from leadforge.render.bundle import TaskExport, write_bundle_envelope
from leadforge.schema.features import FeatureSpec
from leadforge.schema.tasks import SplitSpec, TaskManifest

_TS = "2026-01-01T00:00:00+00:00"


def _spec() -> WorldSpec:
    # student_public (the default) keeps this focused on envelope I/O: no
    # metadata/ is written, so no scheme hidden-truth hook is exercised.
    return WorldSpec(
        config=GenerationConfig(seed=1, n_accounts=2, n_contacts=3, n_leads=4),
        scheme="lead_scoring",
    )


def _features() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="x", dtype="Int64", description="", category="lead_meta"),
        FeatureSpec(name="y", dtype="Float64", description="", category="engagement"),
    )


def _task_frame() -> pd.DataFrame:
    return pd.DataFrame({"x": list(range(20)), "y": [float(i) for i in range(20)]})


def _write(tmp_path: Path, **overrides) -> Path:
    # Default exposure_mode is student_public → relational_snapshot_safe path,
    # so apply_exposure writes no metadata (no scheme hidden-truth hook needed).
    bundle = WorldBundle(spec=_spec(), artifacts=None)
    task = TaskManifest(
        task_id="demo_task",
        label_column="x",
        label_window_days=90,
        primary_table="t",
        split=SplitSpec(0.5, 0.25, 0.25),
    )
    kwargs = {
        "relational": {"t": pd.DataFrame({"id": [1, 2, 3]})},
        "tasks": [TaskExport(manifest=task, frame=_task_frame())],
        "dataset_card": "# card\n",
        "feature_specs": _features(),
        "generation_scheme": "lead_scoring",
        "relational_snapshot_safe": True,  # student_public default → no metadata hook
        "generation_timestamp": _TS,
    }
    kwargs.update(overrides)
    out = tmp_path / "b"
    write_bundle_envelope(bundle, out, **kwargs)
    return out


def test_envelope_writes_all_artifacts(tmp_path) -> None:
    out = _write(tmp_path)
    assert (out / "manifest.json").is_file()
    assert (out / "dataset_card.md").read_text() == "# card\n"
    assert (out / "feature_dictionary.csv").is_file()
    assert (out / "tables" / "t.parquet").is_file()
    for split in ("train", "valid", "test"):
        assert (out / "tasks" / "demo_task" / f"{split}.parquet").is_file()
    assert (out / "tasks" / "demo_task" / "task_manifest.json").is_file()
    # student_public → no metadata
    assert not (out / "metadata").exists()


def test_envelope_manifest_records_passthrough_fields(tmp_path) -> None:
    out = _write(
        tmp_path,
        motif_family="fit_dominant",
        extra_fields={"observation_date": "2026-06-01"},
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["generation_scheme"] == "lead_scoring"
    assert m["motif_family"] == "fit_dominant"
    assert m["observation_date"] == "2026-06-01"
    assert m["tables"]["t"]["row_count"] == 3
    assert set(m["tasks"]["demo_task"]) >= {"train_rows", "valid_rows", "test_rows"}
    # 20 rows split 0.5/0.25/0.25.
    assert m["tasks"]["demo_task"]["train_rows"] == 10


def test_envelope_writes_multiple_task_dirs(tmp_path) -> None:
    t1 = TaskManifest("a", "x", 90, "t", SplitSpec(0.5, 0.25, 0.25))
    t2 = TaskManifest("b", "y", 90, "t", SplitSpec(0.5, 0.25, 0.25), task_type="regression")
    out = _write(
        tmp_path,
        tasks=[TaskExport(t1, _task_frame()), TaskExport(t2, _task_frame())],
    )
    dirs = {p.name for p in (out / "tasks").iterdir() if p.is_dir()}
    assert dirs == {"a", "b"}

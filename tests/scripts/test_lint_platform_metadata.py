"""Tests for ``scripts/lint_platform_metadata.py``.

The lint gate compares the two canonical platform artifacts used by
real publication and by the local preview pages:

* ``release/kaggle/dataset-metadata.json``
* ``release/huggingface/README.md``

It is intentionally metadata-only, so it runs on a fresh checkout even
when the heavy per-tier bundle directories are not materialised.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "lint_platform_metadata.py"
_spec = importlib.util.spec_from_file_location("lint_platform_metadata", _SCRIPT_PATH)
assert _spec is not None
assert _spec.loader is not None
lint = importlib.util.module_from_spec(_spec)
sys.modules["lint_platform_metadata"] = lint
_spec.loader.exec_module(lint)

EXPECTED_ROOT_RESOURCES = (
    "metrics.json",
    "claims_register.md",
    "claims_register.json",
    "docs/README.md",
    "docs/feature_dictionary.md",
    "docs/generation_method.md",
    "docs/break_me_guide.md",
    "docs/relational_table_schemas.csv",
)
EXPECTED_TIER_RESOURCES = (
    "lead_scoring.csv",
    "feature_dictionary.csv",
    "dataset_card.md",
    "metrics.json",
    "manifest.json",
    "tasks/converted_within_90_days/train.parquet",
    "tasks/converted_within_90_days/valid.parquet",
    "tasks/converted_within_90_days/test.parquet",
    "tables/accounts.parquet",
    "tables/contacts.parquet",
    "tables/leads.parquet",
    "tables/touches.parquet",
    "tables/sessions.parquet",
    "tables/sales_activities.parquet",
    "tables/opportunities.parquet",
)


def _field(name: str, field_type: str = "string") -> dict[str, str]:
    return {"name": name, "type": field_type, "description": f"{name} description"}


def _resource(path: str, fields: list[dict[str, str]] | None = None) -> dict[str, object]:
    resource: dict[str, object] = {"path": path, "description": f"{path} resource"}
    if fields is not None:
        resource["schema"] = {"fields": fields}
    return resource


def _minimal_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    flat_fields = [
        _field("split"),
        _field("account_id"),
        _field("score", "number"),
        _field("converted_within_90_days", "boolean"),
    ]
    task_fields = flat_fields[1:]
    resources: list[dict[str, object]] = []
    for path in EXPECTED_ROOT_RESOURCES:
        resources.append(_resource(path))
    for tier in lint.DEFAULT_TIERS:
        for suffix in EXPECTED_TIER_RESOURCES:
            path = f"{tier}/{suffix}"
            if suffix == "lead_scoring.csv":
                resources.append(_resource(path, flat_fields))
            elif suffix.startswith(f"tasks/{lint.DEFAULT_TASK}/"):
                resources.append(_resource(path, task_fields))
            else:
                resources.append(_resource(path))

    kaggle = {
        "title": "LeadForge test",
        "id": "leadforge/leadforge-lead-scoring-v1",
        "subtitle": "A metadata lint fixture",
        "description": "body",
        "isPrivate": False,
        "licenses": [{"name": "MIT"}],
        "keywords": [
            "classification",
            "education",
            "tabular",
        ],
        "expectedUpdateFrequency": "never",
        "image": "dataset-cover-image.png",
        "resources": resources,
    }
    hf = {
        "pretty_name": "LeadForge test",
        "license": "mit",
        "language": ["en"],
        "task_categories": ["tabular-classification"],
        "size_categories": ["1K<n<10K"],
        "tags": ["b2b", "crm", "datasets", "lead-scoring", "pandas", "synthetic-data", "tabular"],
        "configs": [
            {
                "config_name": "intro",
                "default": True,
                "data_files": [
                    {
                        "split": "train",
                        "path": f"intro/tasks/{lint.DEFAULT_TASK}/train.parquet",
                    },
                    {
                        "split": "validation",
                        "path": f"intro/tasks/{lint.DEFAULT_TASK}/valid.parquet",
                    },
                    {
                        "split": "test",
                        "path": f"intro/tasks/{lint.DEFAULT_TASK}/test.parquet",
                    },
                ],
            },
            {
                "config_name": "intermediate",
                "data_files": [
                    {
                        "split": "train",
                        "path": f"intermediate/tasks/{lint.DEFAULT_TASK}/train.parquet",
                    },
                    {
                        "split": "validation",
                        "path": f"intermediate/tasks/{lint.DEFAULT_TASK}/valid.parquet",
                    },
                    {
                        "split": "test",
                        "path": f"intermediate/tasks/{lint.DEFAULT_TASK}/test.parquet",
                    },
                ],
            },
            {
                "config_name": "advanced",
                "data_files": [
                    {
                        "split": "train",
                        "path": f"advanced/tasks/{lint.DEFAULT_TASK}/train.parquet",
                    },
                    {
                        "split": "validation",
                        "path": f"advanced/tasks/{lint.DEFAULT_TASK}/valid.parquet",
                    },
                    {
                        "split": "test",
                        "path": f"advanced/tasks/{lint.DEFAULT_TASK}/test.parquet",
                    },
                ],
            },
        ],
    }
    return kaggle, hf


def _messages(outcome: object) -> list[str]:
    return [f"{finding.field}: {finding.message}" for finding in outcome.findings]


def test_lint_accepts_matching_platform_metadata() -> None:
    kaggle, hf = _minimal_artifacts()
    outcome = lint.lint_metadata(kaggle, hf)
    assert outcome.ok
    assert outcome.findings == ()


def test_lint_catches_private_kaggle_metadata() -> None:
    kaggle, hf = _minimal_artifacts()
    kaggle["isPrivate"] = True
    outcome = lint.lint_metadata(kaggle, hf)
    assert not outcome.ok
    assert any("kaggle.isPrivate" in msg for msg in _messages(outcome))


def test_lint_catches_license_task_and_tag_mismatches() -> None:
    kaggle, hf = _minimal_artifacts()
    kaggle["licenses"] = [{"name": "CC0"}]
    kaggle["keywords"] = ["crm"]
    hf["license"] = "apache-2.0"
    hf["task_categories"] = ["text-classification"]
    hf["tags"] = ["crm"]
    outcome = lint.lint_metadata(kaggle, hf)
    messages = "\n".join(_messages(outcome))
    assert "kaggle.licenses[0].name" in messages
    assert "hf.license" in messages
    assert "hf.task_categories" in messages
    assert "kaggle.keywords" in messages
    assert "hf.tags" in messages


def test_lint_catches_hf_split_path_absent_from_kaggle_resources() -> None:
    kaggle, hf = _minimal_artifacts()
    hf["configs"][0]["data_files"][0]["path"] = "intro/tasks/converted_within_90_days/oops.parquet"
    outcome = lint.lint_metadata(kaggle, hf)
    messages = "\n".join(_messages(outcome))
    assert "hf.configs.data_files" in messages
    assert "oops.parquet" in messages


def test_lint_catches_missing_hf_split_entry() -> None:
    kaggle, hf = _minimal_artifacts()
    hf["configs"][0]["data_files"] = hf["configs"][0]["data_files"][:1]
    outcome = lint.lint_metadata(kaggle, hf)
    messages = "\n".join(_messages(outcome))
    assert "intro data_files expected" in messages
    assert "valid.parquet" in messages
    assert "test.parquet" in messages


def test_lint_catches_schema_mismatch_between_flat_csv_and_task_splits() -> None:
    kaggle, hf = _minimal_artifacts()
    mutated = deepcopy(kaggle["resources"])
    for resource in mutated:
        if resource["path"] == f"intro/tasks/{lint.DEFAULT_TASK}/train.parquet":
            resource["schema"]["fields"] = [_field("account_id"), _field("unexpected")]
            break
    kaggle["resources"] = mutated
    outcome = lint.lint_metadata(kaggle, hf)
    messages = "\n".join(_messages(outcome))
    assert "schema differs from 'intro/lead_scoring.csv' minus split" in messages


def test_lint_catches_missing_per_tier_review_artifact() -> None:
    kaggle, hf = _minimal_artifacts()
    kaggle["resources"] = [r for r in kaggle["resources"] if r["path"] != "advanced/metrics.json"]
    outcome = lint.lint_metadata(kaggle, hf)
    messages = "\n".join(_messages(outcome))
    assert "missing per-tier review artifact 'advanced/metrics.json'" in messages


def test_lint_compares_metadata_schema_to_actual_files_when_present(tmp_path: Path) -> None:
    kaggle, hf = _minimal_artifacts()
    release_dir = tmp_path / "release"
    task_dir = release_dir / "intro" / "tasks" / lint.DEFAULT_TASK
    task_dir.mkdir(parents=True)
    (release_dir / "intro" / "lead_scoring.csv").write_text(
        "split,account_id,score,converted_within_90_days\ntrain,acct_1,0.1,false\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "account_id": pa.array(["acct_1"], pa.string()),
                "score": pa.array([0.1], pa.float64()),
                "unexpected": pa.array([False], pa.bool_()),
            }
        ),
        task_dir / "train.parquet",
    )
    outcome = lint.lint_metadata(
        kaggle,
        hf,
        release_dir=release_dir,
        tiers=("intro",),
    )
    messages = "\n".join(_messages(outcome))
    assert "metadata schema differs from actual parquet schema" in messages
    assert "unexpected" in messages


def test_strict_files_fails_when_release_files_are_missing(tmp_path: Path) -> None:
    kaggle, hf = _minimal_artifacts()
    outcome = lint.lint_metadata(
        kaggle,
        hf,
        release_dir=tmp_path / "release",
        tiers=("intro",),
        strict_files=True,
    )
    messages = "\n".join(_messages(outcome))
    assert "missing release file required for strict schema lint" in messages
    assert "intro/lead_scoring.csv" in messages


def test_committed_release_artifacts_pass_lint() -> None:
    outcome = lint.run_lint(_REPO_ROOT / "release")
    assert outcome.ok, _messages(outcome)


def test_main_returns_1_on_lint_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    release_dir = tmp_path / "release"
    (release_dir / "kaggle").mkdir(parents=True)
    (release_dir / "huggingface").mkdir()
    kaggle, hf = _minimal_artifacts()
    kaggle["isPrivate"] = True
    (release_dir / "kaggle" / "dataset-metadata.json").write_text(
        json.dumps(kaggle), encoding="utf-8"
    )
    (release_dir / "huggingface" / "README.md").write_text(
        "---\n" + lint.yaml.safe_dump(hf, sort_keys=False) + "---\nbody\n",
        encoding="utf-8",
    )
    rc = lint.main(["--release-dir", str(release_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "kaggle.isPrivate" in captured.err


def test_main_returns_2_on_missing_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = lint.main(["--release-dir", str(tmp_path / "missing")])
    assert rc == 2
    captured = capsys.readouterr()
    assert "missing JSON artifact" in captured.err

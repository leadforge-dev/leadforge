"""Tests for ``scripts/build_release_metrics.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "build_release_metrics.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_release_metrics", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_report() -> dict:
    """Hand-rolled validation_report.json with the keys the script reads."""

    return {
        "release_id": "leadforge-lead-scoring-v1",
        "package_version": "1.0.0",
        "generation_timestamp": "2026-05-06T07:38:31+00:00",
        "seeds": [42, 43, 44, 45, 46],
        "tiers": {
            "intro": {
                "medians": {
                    "lr_auc": 0.879,
                    "lr_average_precision": 0.761,
                    "brier_score": 0.130,
                    "conversion_rate_test": 0.427,
                    "gbm_auc": 0.873,
                    "gbm_minus_lr_auc": -0.0045,
                    "log_loss": 0.4,
                    "calibration_max_bin_error": 0.25,
                    "gbm_average_precision": 0.75,
                    "top_decile_rate": 0.77,
                },
                "spreads": {
                    "lr_auc": 0.027,
                    "conversion_rate_test": 0.092,
                },
                "seeds": [42, 43, 44, 45, 46],
                "per_seed": [{"seed": s, "precision_at_k": {"100": 0.80}} for s in range(42, 47)],
            },
            "intermediate": {
                "medians": {"lr_auc": 0.886, "lr_average_precision": 0.575},
                "spreads": {"lr_auc": 0.023},
                "seeds": [42, 43, 44, 45, 46],
                "per_seed": [{"seed": s, "precision_at_k": {"100": 0.59}} for s in range(42, 47)],
            },
            "advanced": {
                "medians": {"lr_auc": 0.886, "lr_average_precision": 0.351},
                "spreads": {"lr_auc": 0.040},
                "seeds": [42, 43, 44, 45, 46],
                "per_seed": [{"seed": s, "precision_at_k": {"100": 0.34}} for s in range(42, 47)],
            },
        },
        "cohort_shift": {
            "intro": {
                "random_split_auc": 0.873,
                "cohort_split_auc": 0.857,
                "auc_degradation": 0.016,
                "seed": 42,
            },
        },
        "cross_tier_ordering": {
            "by_conversion_rate": ["intro", "intermediate", "advanced"],
            "by_average_precision": ["intro", "intermediate", "advanced"],
        },
    }


def _write_minimal_release(tmp_path: Path) -> tuple[Path, Path]:
    release_dir = tmp_path / "release"
    (release_dir / "validation").mkdir(parents=True)
    report_path = release_dir / "validation" / "validation_report.json"
    report_path.write_text(json.dumps(_minimal_report()), encoding="utf-8")
    for tier in ("intro", "intermediate", "advanced"):
        (release_dir / tier).mkdir()
    return release_dir, report_path


def test_top_level_payload_contains_expected_keys(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    stale, top = mod.write_metrics(release_dir, report_path, check_only=False)
    assert "tiers" in top
    assert set(top["tiers"]) == {"intro", "intermediate", "advanced"}
    assert top["release_id"] == "leadforge-lead-scoring-v1"
    assert top["seeds"] == [42, 43, 44, 45, 46]
    assert top["cohort_shift"]["intro"]["auc_degradation"] == 0.016


def test_per_tier_files_written_when_dir_exists(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    mod.write_metrics(release_dir, report_path, check_only=False)
    for tier in ("intro", "intermediate", "advanced"):
        path = release_dir / tier / "metrics.json"
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["tier"] == tier
        assert payload["medians"]["lr_auc"] is not None
        assert payload["source_of_truth"]["file"] == "release/validation/validation_report.json"


def test_precision_at_100_median_attached_to_per_tier_metrics(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    mod.write_metrics(release_dir, report_path, check_only=False)
    intro = json.loads((release_dir / "intro" / "metrics.json").read_text(encoding="utf-8"))
    assert intro["medians"]["precision_at_100"] == 0.80


def test_idempotent_writes(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    mod.write_metrics(release_dir, report_path, check_only=False)
    stale, _ = mod.write_metrics(release_dir, report_path, check_only=False)
    assert stale == []


def test_check_mode_flags_drift_on_missing_files(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    stale, _ = mod.write_metrics(release_dir, report_path, check_only=True)
    assert stale  # nothing written yet
    assert not (release_dir / "metrics.json").is_file()


def test_skips_tier_dir_when_absent(tmp_path: Path) -> None:
    """Per-tier bundle dirs are gitignored on fresh checkouts; the script
    must skip silently rather than error."""

    mod = _load_module()
    release_dir, report_path = _write_minimal_release(tmp_path)
    # Remove the bundle dirs so only the top-level path can be written.
    for tier in ("intro", "intermediate", "advanced"):
        (release_dir / tier).rmdir()
    stale, _ = mod.write_metrics(release_dir, report_path, check_only=False)
    # Top-level file is the only one stale (and now written).
    assert (release_dir / "metrics.json").is_file()
    for tier in ("intro", "intermediate", "advanced"):
        assert not (release_dir / tier / "metrics.json").is_file()


def test_missing_report_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError):
        mod.write_metrics(tmp_path, tmp_path / "no.json", check_only=False)


def test_non_object_report_raises(tmp_path: Path) -> None:
    mod = _load_module()
    report_path = tmp_path / "validation_report.json"
    report_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        mod.write_metrics(tmp_path, report_path, check_only=False)


def test_committed_release_metrics_match_validation_report() -> None:
    """The real repo's ``release/metrics.json`` is in sync with
    ``release/validation/validation_report.json``."""

    mod = _load_module()
    release_dir = _REPO_ROOT / "release"
    report_path = release_dir / "validation" / "validation_report.json"
    if not report_path.is_file():
        pytest.skip("validation_report.json missing on this checkout")
    stale, _ = mod.write_metrics(release_dir, report_path, check_only=True)
    assert stale == [], f"metrics drift: {stale}"

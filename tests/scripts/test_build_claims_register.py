"""Tests for ``scripts/build_claims_register.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "build_claims_register.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_claims_register", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_claims_yaml() -> str:
    return """\
claims:
  - id: a01
    text: Composition claim.
    category: composition
    backing_artifact: release/<tier>/manifest.json
    backing_path: $.n_leads
    verifier: leadforge validate
  - id: a02
    text: Calibration claim.
    category: calibration
    backing_artifact: release/metrics.json
    backing_path: $.tiers.<tier>.medians.lr_auc
    verifier: scripts/validate_release_candidate.py
"""


def _write_source(tmp_path: Path, text: str | None = None) -> tuple[Path, Path]:
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    source = release_dir / "claims_register_source.yaml"
    source.write_text(text or _minimal_claims_yaml(), encoding="utf-8")
    return release_dir, source


def test_renders_both_files(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, source = _write_source(tmp_path)
    mod.write_register(release_dir, source, check_only=False)
    assert (release_dir / "claims_register.json").is_file()
    assert (release_dir / "claims_register.md").is_file()


def test_json_payload_includes_schema_block(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, source = _write_source(tmp_path)
    mod.write_register(release_dir, source, check_only=False)
    payload = json.loads((release_dir / "claims_register.json").read_text(encoding="utf-8"))
    assert "schema" in payload
    assert "claims" in payload
    assert len(payload["claims"]) == 2
    assert payload["claims"][0]["id"] == "a01"


def test_markdown_groups_claims_by_category(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, source = _write_source(tmp_path)
    mod.write_register(release_dir, source, check_only=False)
    md = (release_dir / "claims_register.md").read_text(encoding="utf-8")
    assert "## calibration" in md
    assert "## composition" in md
    # Claim text is present, escaped or not.
    assert "Composition claim." in md


def test_idempotent_writes(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, source = _write_source(tmp_path)
    mod.write_register(release_dir, source, check_only=False)
    stale = mod.write_register(release_dir, source, check_only=False)
    assert stale == []


def test_check_mode_flags_drift(tmp_path: Path) -> None:
    mod = _load_module()
    release_dir, source = _write_source(tmp_path)
    stale = mod.write_register(release_dir, source, check_only=True)
    assert stale
    assert not (release_dir / "claims_register.json").is_file()


def test_missing_required_keys_rejected(tmp_path: Path) -> None:
    mod = _load_module()
    bad_yaml = """\
claims:
  - id: missing_text
    category: composition
    backing_artifact: x
    backing_path: y
    verifier: z
"""
    release_dir, source = _write_source(tmp_path, bad_yaml)
    with pytest.raises(ValueError, match="missing required key"):
        mod.write_register(release_dir, source, check_only=False)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    mod = _load_module()
    bad_yaml = """\
claims:
  - id: dup
    text: a
    category: composition
    backing_artifact: x
    backing_path: y
    verifier: z
  - id: dup
    text: b
    category: composition
    backing_artifact: x
    backing_path: y
    verifier: z
"""
    release_dir, source = _write_source(tmp_path, bad_yaml)
    with pytest.raises(ValueError, match="duplicate claim id"):
        mod.write_register(release_dir, source, check_only=False)


def test_invalid_category_rejected(tmp_path: Path) -> None:
    mod = _load_module()
    bad_yaml = """\
claims:
  - id: x01
    text: bad category
    category: not_in_vocab
    backing_artifact: x
    backing_path: y
    verifier: z
"""
    release_dir, source = _write_source(tmp_path, bad_yaml)
    with pytest.raises(ValueError, match="not in"):
        mod.write_register(release_dir, source, check_only=False)


def test_missing_source_raises(tmp_path: Path) -> None:
    mod = _load_module()
    with pytest.raises(FileNotFoundError):
        mod.write_register(tmp_path, tmp_path / "nope.yaml", check_only=False)


def test_committed_claims_register_is_in_sync() -> None:
    """The real repo's ``release/claims_register.{md,json}`` is in sync
    with ``claims_register_source.yaml``."""

    mod = _load_module()
    release_dir = _REPO_ROOT / "release"
    source = release_dir / "claims_register_source.yaml"
    if not source.is_file():
        pytest.skip("claims_register_source.yaml missing on this checkout")
    stale = mod.write_register(release_dir, source, check_only=True)
    assert stale == [], f"claims register drift: {stale}"


def test_every_categories_token_is_in_valid_set() -> None:
    """The source-file categories all match VALID_CATEGORIES (guards
    silent drift in the source if a future contributor invents a
    category)."""

    mod = _load_module()
    source = _REPO_ROOT / "release" / "claims_register_source.yaml"
    if not source.is_file():
        pytest.skip("claims_register_source.yaml missing on this checkout")
    claims = mod.load_claims(source)
    for claim in claims:
        assert claim["category"] in mod.VALID_CATEGORIES

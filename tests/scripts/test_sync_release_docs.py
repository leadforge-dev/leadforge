"""Tests for ``scripts/sync_release_docs.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "sync_release_docs.py"


def _load_module() -> ModuleType:
    # Register in ``sys.modules`` BEFORE ``exec_module`` so dataclasses
    # declared inside the module resolve their own module reference
    # (CPython's ``dataclass`` machinery reads ``sys.modules[cls.__module__]``
    # during ``InitVar``/``ClassVar`` handling, and crashes with
    # ``AttributeError: 'NoneType' object has no attribute '__dict__'``
    # if the entry doesn't yet exist).
    spec = importlib.util.spec_from_file_location("sync_release_docs", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_release_docs"] = module
    spec.loader.exec_module(module)
    return module


def _build_fake_repo(tmp_path: Path) -> Path:
    docs_src = tmp_path / "docs" / "release"
    docs_src.mkdir(parents=True)
    (docs_src / "break_me_guide.md").write_text("break me\n", encoding="utf-8")
    (docs_src / "channel_signal_audit.md").write_text("channel\n", encoding="utf-8")
    (docs_src / "feature_dictionary.md").write_text("features\n", encoding="utf-8")
    (docs_src / "generation_method.md").write_text("how\n", encoding="utf-8")
    (docs_src / "v1_acceptance_gates_bands.yaml").write_text("bands: {}\n", encoding="utf-8")
    (docs_src / "v2_decision_log.md").write_text("v2 decisions\n", encoding="utf-8")
    return tmp_path


def test_sync_copies_all_declared_pairs(tmp_path: Path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    result = mod.sync_docs(repo, check_only=False)
    assert not result.missing_sources
    assert set(result.stale) == {dst for _src, dst in mod.VENDORED_DOCS}
    for _src, dst in mod.VENDORED_DOCS:
        assert (repo / dst).is_file()


def test_sync_is_idempotent(tmp_path: Path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    mod.sync_docs(repo, check_only=False)
    result = mod.sync_docs(repo, check_only=False)
    assert not result.missing_sources
    assert result.stale == []


def test_check_mode_reports_drift_without_writing(tmp_path: Path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    result = mod.sync_docs(repo, check_only=True)
    assert not result.missing_sources
    assert result.stale  # destinations don't exist yet
    for _src, dst in mod.VENDORED_DOCS:
        assert not (repo / dst).is_file()


def test_missing_source_returns_in_missing_list(tmp_path: Path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    (repo / "docs" / "release" / "break_me_guide.md").unlink()
    result = mod.sync_docs(repo, check_only=False)
    assert any("break_me_guide.md" in str(p) for p in result.missing_sources)


def test_refuses_to_overwrite_locally_edited_destination(tmp_path: Path) -> None:
    """An edit to a vendored copy (mtime newer than source) is the
    sentinel that someone touched the wrong file.  Without --force the
    sync must refuse to clobber it."""

    import os
    import time

    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    mod.sync_docs(repo, check_only=False)  # populate destinations
    # Edit one destination AND bump its mtime past the source's.
    edited = repo / "release" / "docs" / "break_me_guide.md"
    edited.write_text("locally edited\n", encoding="utf-8")
    src = repo / "docs" / "release" / "break_me_guide.md"
    future = src.stat().st_mtime + 10
    os.utime(edited, (future, future))
    time.sleep(0)  # no-op, kept for documentation that we rely on mtime

    result = mod.sync_docs(repo, check_only=False)
    assert any("break_me_guide.md" in str(p) for p in result.orphan_destinations)
    # The edit was preserved (not clobbered).
    assert edited.read_text(encoding="utf-8") == "locally edited\n"


def test_force_overwrites_locally_edited_destination(tmp_path: Path) -> None:
    """``--force`` explicitly opts in to discarding the local edit."""

    import os

    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    mod.sync_docs(repo, check_only=False)
    edited = repo / "release" / "docs" / "break_me_guide.md"
    edited.write_text("locally edited\n", encoding="utf-8")
    src = repo / "docs" / "release" / "break_me_guide.md"
    future = src.stat().st_mtime + 10
    os.utime(edited, (future, future))

    mod.sync_docs(repo, check_only=False, force=True)
    assert edited.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


@pytest.mark.parametrize(("argv", "expected"), [(["--check"], 0), ([], 0)])
def test_cli_passes_on_clean_tree(monkeypatch, argv, expected, tmp_path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    # First populate the destinations.
    mod.sync_docs(repo, check_only=False)
    rc = mod.main(argv)
    assert rc == expected


def test_cli_check_mode_returns_1_on_drift(monkeypatch, tmp_path) -> None:
    mod = _load_module()
    repo = _build_fake_repo(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", repo)
    rc = mod.main(["--check"])
    assert rc == 1


def test_committed_release_docs_match_sources() -> None:
    """The real repo's ``release/docs/`` is in sync with ``docs/release/``."""

    mod = _load_module()
    result = mod.sync_docs(_REPO_ROOT, check_only=True)
    assert not result.missing_sources
    assert result.stale == [], f"release/docs/ drift: {result.stale}"
    assert result.orphan_destinations == [], (
        f"release/docs/ destinations look locally edited: {result.orphan_destinations}"
    )

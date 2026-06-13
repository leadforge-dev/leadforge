"""Tests for the render half of the scheme seam (LTV-Pe).

Covers ``WorldBundle.save`` / ``api.bundle.write_bundle`` dispatching to the
producing scheme, the registration footgun fix (``base``-direct resolution),
and render-path determinism.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from leadforge.api.bundle import write_bundle
from leadforge.api.generator import Generator
from leadforge.core.models import WorldBundle
from leadforge.schemes import UnknownSchemeError

_SMALL = {"n_accounts": 20, "n_contacts": 40, "n_leads": 60, "difficulty": "intro"}
_TS = "2026-01-01T00:00:00Z"


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _gen():
    return Generator.from_recipe("b2b_saas_procurement_v1", seed=42).generate(**_SMALL)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_save_dispatches_to_scheme_and_writes_bundle(tmp_path: Path) -> None:
    _gen().save(str(tmp_path), generation_timestamp=_TS)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "tables").is_dir()
    assert (tmp_path / "tasks").is_dir()
    assert (tmp_path / "feature_dictionary.csv").exists()
    assert (tmp_path / "dataset_card.md").exists()


def test_write_bundle_unknown_scheme_raises(tmp_path: Path) -> None:
    bundle = _gen()
    bundle.spec.scheme = "nope"
    with pytest.raises(UnknownSchemeError):
        write_bundle(bundle, str(tmp_path))


def test_write_bundle_unpopulated_raises(tmp_path: Path) -> None:
    # Default bundle has spec.scheme == "lead_scoring" → dispatches, then the
    # lead-scoring write_bundle rejects the unpopulated bundle.
    with pytest.raises(RuntimeError, match="not populated with lead-scoring artifacts"):
        write_bundle(WorldBundle(), str(tmp_path))


# ---------------------------------------------------------------------------
# Render-path determinism
# ---------------------------------------------------------------------------


def test_save_is_byte_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _gen().save(str(a), generation_timestamp=_TS)
    _gen().save(str(b), generation_timestamp=_TS)
    assert _hash_tree(a) == _hash_tree(b)


# ---------------------------------------------------------------------------
# Registration footgun fix — resolution must not depend on import order
# ---------------------------------------------------------------------------


def test_resolution_works_via_base_without_package_import() -> None:
    # A fresh interpreter that imports ONLY leadforge.schemes.base (never the
    # leadforge.schemes package) must still resolve built-in schemes, because
    # get_scheme lazily triggers builtin registration.
    code = "from leadforge.schemes.base import get_scheme; print(get_scheme('lead_scoring').name)"
    out = subprocess.run(  # noqa: S603 — controlled args, sys.executable + literal code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "lead_scoring"

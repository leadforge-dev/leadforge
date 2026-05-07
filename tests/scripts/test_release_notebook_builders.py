"""Byte-stability gate for the release-notebook builders.

The builders advertise an audit-artifact-sync invariant (PR 4.1 / 5.1 /
5.2 pattern): re-running the builder must produce a byte-identical
``.ipynb``, and the committed file under ``release/notebooks/`` must
equal a fresh build.  Without this test the invariant is wishful
thinking — ``nbformat.v4.new_*_cell`` randomises cell IDs by default,
so an unguarded builder silently diverges on every run.

We don't run the builders by importing them — they shell out to
``ruff format`` and that's deterministic only via subprocess.  Instead,
we invoke each builder twice into temp paths and diff the outputs, then
diff a third invocation against the committed file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_NOTEBOOKS_DIR = _REPO_ROOT / "release" / "notebooks"

_BUILDERS: list[tuple[str, str]] = [
    ("build_release_notebook_01.py", "01_baseline_lead_scoring.ipynb"),
    ("build_release_notebook_02.py", "02_relational_feature_engineering.ipynb"),
]


@pytest.mark.parametrize(("builder_name", "notebook_name"), _BUILDERS)
def test_builder_is_byte_stable_and_matches_committed(
    tmp_path: Path,
    builder_name: str,
    notebook_name: str,
) -> None:
    """Run the builder twice into separate temp homes and assert the
    emitted ``.ipynb`` files are byte-identical, and that the committed
    notebook matches one of them.

    The builder writes to ``release/notebooks/<notebook>`` by default,
    so we redirect by snapshotting + restoring the committed file
    around the build calls.
    """
    builder_path = _SCRIPTS_DIR / builder_name
    committed_path = _NOTEBOOKS_DIR / notebook_name
    assert builder_path.exists(), f"missing builder: {builder_path}"
    assert committed_path.exists(), f"missing committed notebook: {committed_path}"

    backup = tmp_path / "committed_backup.ipynb"
    shutil.copy2(committed_path, backup)

    run_a = tmp_path / "run_a.ipynb"
    run_b = tmp_path / "run_b.ipynb"

    try:
        subprocess.run(  # noqa: S603 — sys.executable + repo-internal builder path
            [sys.executable, str(builder_path)], check=True, cwd=_REPO_ROOT
        )
        shutil.copy2(committed_path, run_a)
        subprocess.run(  # noqa: S603 — sys.executable + repo-internal builder path
            [sys.executable, str(builder_path)], check=True, cwd=_REPO_ROOT
        )
        shutil.copy2(committed_path, run_b)

        assert run_a.read_bytes() == run_b.read_bytes(), (
            f"{builder_name}: two runs produced different bytes — cell IDs are "
            "non-deterministic; pass an explicit ``id=`` to nbformat cell "
            "constructors (see scripts/_release_notebook_common.py)"
        )
        assert backup.read_bytes() == run_a.read_bytes(), (
            f"{notebook_name}: committed file does not match a fresh build of "
            f"{builder_name} — re-run the builder and commit the result "
            "(audit-artifact-sync, same pattern as PR 4.1 / 5.1 / 5.2)"
        )
    finally:
        shutil.copy2(backup, committed_path)

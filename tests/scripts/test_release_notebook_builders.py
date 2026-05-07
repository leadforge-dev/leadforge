"""Byte-stability gate for the release-notebook builders.

The builders advertise an audit-artifact-sync invariant (PR 4.1 / 5.1 /
5.2 pattern): re-running the builder must produce a byte-identical
``.ipynb``, and the committed file under ``release/notebooks/`` must
equal a fresh build.  Without this test the invariant is wishful
thinking — ``nbformat.v4.new_*_cell`` randomises cell IDs by default,
so an unguarded builder silently diverges on every run.

The builders accept ``--out PATH`` so this test can build into
``tmp_path`` instead of mutating the committed ``release/notebooks/``
file.  That keeps the test parallel-safe (pytest-xdist running both
parametrised cases at once won't race), interrupt-safe (an interrupted
run can't leave the working tree dirty), and side-effect-free (the
committed notebook is never touched).
"""

from __future__ import annotations

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
    ("build_release_notebook_03.py", "03_leakage_and_time_windows.ipynb"),
    ("build_release_notebook_04.py", "04_lift_calibration_value_ranking.ipynb"),
]


@pytest.mark.parametrize(("builder_name", "notebook_name"), _BUILDERS)
def test_builder_is_byte_stable_and_matches_committed(
    tmp_path: Path,
    builder_name: str,
    notebook_name: str,
) -> None:
    """Build twice into ``tmp_path`` via ``--out``; assert the two runs
    produce byte-identical output and that the committed notebook
    matches them.
    """
    # ``nbformat`` lives in the optional ``[notebooks]`` extra; the
    # main ``test`` CI job installs only ``[dev]`` and would otherwise
    # see the subprocess-invoked builders crash with
    # ``ModuleNotFoundError: nbformat``.  The dedicated ``notebooks``
    # CI job installs ``[dev,scripts,notebooks]`` and runs this test
    # alongside ``test_execute_notebooks.py``.
    pytest.importorskip("nbformat", reason="nbformat not installed (use [notebooks] extra)")

    builder_path = _SCRIPTS_DIR / builder_name
    committed_path = _NOTEBOOKS_DIR / notebook_name
    assert builder_path.exists(), f"missing builder: {builder_path}"
    assert committed_path.exists(), f"missing committed notebook: {committed_path}"

    run_a = tmp_path / "run_a.ipynb"
    run_b = tmp_path / "run_b.ipynb"

    subprocess.run(  # noqa: S603 — sys.executable + repo-internal builder path
        [sys.executable, str(builder_path), "--out", str(run_a)],
        check=True,
        cwd=_REPO_ROOT,
    )
    subprocess.run(  # noqa: S603 — sys.executable + repo-internal builder path
        [sys.executable, str(builder_path), "--out", str(run_b)],
        check=True,
        cwd=_REPO_ROOT,
    )

    assert run_a.read_bytes() == run_b.read_bytes(), (
        f"{builder_name}: two runs produced different bytes — cell IDs are "
        "non-deterministic; pass an explicit ``id=`` to nbformat cell "
        "constructors (see scripts/_release_notebook_common.py)"
    )
    assert committed_path.read_bytes() == run_a.read_bytes(), (
        f"{notebook_name}: committed file does not match a fresh build of "
        f"{builder_name} — re-run the builder and commit the result "
        "(audit-artifact-sync, same pattern as PR 4.1 / 5.1 / 5.2)"
    )

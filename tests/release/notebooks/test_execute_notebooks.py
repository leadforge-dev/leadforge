"""End-to-end execution gate for the public release notebooks (G13.1).

Each notebook under ``release/notebooks/*.ipynb`` is executed top to
bottom with ``nbclient`` against the committed public release bundles.
A clean run is the contract:

* G13.1 — every cell executes from a clean kernel without raising.
* G13.2 — notebook 01's ``assert_within_tolerance`` cell pins notebook
  metrics to the cross-seed-median targets in
  ``release/notebooks/_release_targets.json`` (per-metric tolerances;
  AUC/Brier ±0.02, AP / top-decile ±0.05). The targets file is itself
  audit-synced against ``release/validation/validation_report.json``
  by ``test_release_targets_match_report.py``, so the gate
  transitively pins to the validation report.
* G13.3 — neither notebook touches ``release/intermediate_instructor``
  or any other instructor artefact (enforced by the notebooks' own
  ``BUNDLE = Path("../intermediate")`` path discipline; an instructor
  load would fail the public-mode manifest assertion in cell 1).

The test is gated on the public release bundles being on disk (matches
the HF/Kaggle smoke-test pattern in ``tests/scripts/test_package_*``).
``nbclient`` lives in the optional ``[notebooks]`` extra; ``importorskip``
keeps the dev install lean while letting the dedicated CI job run the
gate against ``pip install -e ".[dev,scripts,notebooks]"``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NOTEBOOKS_DIR = _REPO_ROOT / "release" / "notebooks"
_RELEASE_BUNDLES_PRESENT = (_REPO_ROOT / "release" / "intermediate" / "manifest.json").exists()

_NOTEBOOKS = [
    "01_baseline_lead_scoring.ipynb",
    "02_relational_feature_engineering.ipynb",
    "03_leakage_and_time_windows.ipynb",
    "04_lift_calibration_value_ranking.ipynb",
]


@pytest.mark.skipif(not _RELEASE_BUNDLES_PRESENT, reason="release bundles not present")
@pytest.mark.parametrize("notebook_name", _NOTEBOOKS)
def test_notebook_executes_end_to_end(notebook_name: str) -> None:
    """Execute the notebook with nbclient and surface any cell error.

    Each notebook hard-codes ``BUNDLE = Path("../intermediate")`` and
    ``sys.path.insert(0, str(Path.cwd()))`` to import the sibling
    ``_notebook_utils`` module — both work iff the kernel cwd is the
    notebook directory, so we set ``resources={"metadata": {"path": ...}}``
    accordingly.
    """
    nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")
    nbclient = pytest.importorskip("nbclient", reason="nbclient not installed")
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    pytest.importorskip("matplotlib", reason="matplotlib not installed")

    notebook_path = _NOTEBOOKS_DIR / notebook_name
    nb = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(_NOTEBOOKS_DIR)}},
    )
    client.execute()

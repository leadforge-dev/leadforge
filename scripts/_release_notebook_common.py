"""Shared scaffolding for the ``scripts/build_release_notebook_*.py`` builders.

The release-notebook builders share ~80% of their plumbing: imports,
``md`` / ``code`` cell wrappers, the metadata block, and the
``write_notebook`` step that serializes JSON and shells out to
``ruff format`` so builder output matches the project's pre-commit hook
byte-for-byte.  Putting them here keeps each per-notebook builder
focused on its cell list.

Cell IDs are assigned deterministically (``cell_000``, ``cell_001``,
...) so re-running a builder produces a byte-identical ``.ipynb`` —
audit-artifact-sync, the same invariant PR 4.1 / 5.1 / 5.2 use for
``release/`` artifacts.  Nondeterministic IDs are the default in
``nbformat.v4.new_*_cell``; without an explicit override every build
diverges and the byte-equality test in
``tests/scripts/test_release_notebook_builders.py`` fails.

Each builder exposes an ``--out PATH`` flag (see ``builder_arg_parser``)
so the byte-stability test can build into ``tmp_path`` rather than
mutating the committed ``release/notebooks/<name>.ipynb`` in place.
That removes a pytest-xdist race and the worktree-dirtying failure
mode under interrupted runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

_KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}


def md(source: str) -> nbf.NotebookNode:
    """Markdown cell with ``source`` dedented and surrounding blank lines stripped."""
    return nbf.v4.new_markdown_cell(dedent(source).strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    """Cleared code cell (no execution_count, no outputs)."""
    cell = nbf.v4.new_code_cell(dedent(source).strip("\n"))
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def assemble_notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    """Build a notebook from ``cells`` with deterministic cell IDs and stable metadata."""
    nb = nbf.v4.new_notebook()
    for index, cell in enumerate(cells):
        cell["id"] = f"cell_{index:03d}"
    nb.cells = cells
    nb.metadata = _KERNEL_METADATA
    return nb


def write_notebook(out_path: Path, nb: nbf.NotebookNode) -> None:
    """Write ``nb`` to ``out_path`` then run ``ruff format`` for hook-parity.

    Without the post-write format step a contributor running the builder
    would see pre-commit reformat their notebook on commit, defeating the
    audit-artifact-sync invariant.  ``json.dumps`` parameters are pinned
    so the pre-format bytes are deterministic.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False)
    out_path.write_text(text + "\n", encoding="utf-8")
    subprocess.run(["ruff", "format", str(out_path)], check=True)  # noqa: S603, S607
    print(f"wrote {out_path}")


def builder_arg_parser(*, default_out: Path, description: str) -> argparse.ArgumentParser:
    """Return the argparse parser shared by every notebook builder.

    Exposes ``--out PATH`` so callers (notably the byte-stability test)
    can redirect the build into a temp directory.  Defaults to the
    canonical committed ``release/notebooks/<name>.ipynb`` path so the
    no-arg invocation contributors run today keeps working.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"Path to write the notebook to (default: {default_out})",
    )
    return parser

"""Lock the LTV-Pf module move (hard break, no shims — decision D12).

These tests pin the *layout* decision in code: the lead-scoring compute core
lives under ``leadforge.schemes.lead_scoring.*`` and the old flat paths are
gone (no back-compat re-export shims).  If a future change accidentally
reintroduces a shim at an old path, or fails to move a module, these fail.
"""

import importlib

import pytest

# (old flat path, new scheme-owned path) for modules moved in LTV-Pf.1 (compute
# core) and LTV-Pf.2 (lead-scoring render).
_MOVED = [
    # LTV-Pf.1 — compute core
    ("leadforge.simulation.engine", "leadforge.schemes.lead_scoring.simulation.engine"),
    ("leadforge.simulation.population", "leadforge.schemes.lead_scoring.simulation.population"),
    ("leadforge.simulation.state", "leadforge.schemes.lead_scoring.simulation.state"),
    ("leadforge.mechanisms.policies", "leadforge.schemes.lead_scoring.mechanisms.policies"),
    ("leadforge.structure.sampler", "leadforge.schemes.lead_scoring.structure.sampler"),
    ("leadforge.structure.graph", "leadforge.schemes.lead_scoring.structure.graph"),
    # LTV-Pf.2 — lead-scoring render
    ("leadforge.render.snapshots", "leadforge.schemes.lead_scoring.render.snapshots"),
    (
        "leadforge.render.relational_snapshot_safe",
        "leadforge.schemes.lead_scoring.render.relational_snapshot_safe",
    ),
    ("leadforge.render.tasks", "leadforge.schemes.lead_scoring.render.tasks"),
]


@pytest.mark.parametrize(("_old", "new"), _MOVED)
def test_new_path_importable(_old: str, new: str) -> None:
    assert importlib.import_module(new) is not None


@pytest.mark.parametrize(("old", "_new"), _MOVED)
def test_old_path_is_gone(old: str, _new: str) -> None:
    # Hard break: the old flat module path must no longer resolve.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old)


@pytest.mark.parametrize("pkg", ["simulation", "mechanisms", "structure"])
def test_old_top_level_package_is_gone(pkg: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"leadforge.{pkg}")


def test_render_envelope_package_stays() -> None:
    # LTV-Pf.2 moved the lead-scoring render modules, but `leadforge.render`
    # remains the shared envelope (manifests + the relational-table writer).
    import leadforge.render.manifests  # noqa: F401
    import leadforge.render.relational as shared_relational

    assert hasattr(shared_relational, "write_relational_tables")


def test_relational_split_to_dataframes_moved_to_scheme() -> None:
    # The 9-table assembler moved to the scheme; the shared writer did not.
    import leadforge.render.relational as shared_relational
    from leadforge.schemes.lead_scoring.render.relational import to_dataframes  # noqa: F401

    assert not hasattr(shared_relational, "to_dataframes")


def test_public_api_unchanged_by_the_move() -> None:
    # The documented public surface must keep importing from its stable home.
    from leadforge.api import Generator, list_recipes  # noqa: F401

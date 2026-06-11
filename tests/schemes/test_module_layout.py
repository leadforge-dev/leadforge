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
    # remains the shared envelope: manifests + the relational-table writer
    # (renamed to relational_io to avoid a basename clash with the scheme's
    # relational.py assembler).
    import leadforge.render.manifests  # noqa: F401
    import leadforge.render.relational_io as shared_writer

    assert hasattr(shared_writer, "write_relational_tables")


def test_relational_split_to_dataframes_moved_to_scheme() -> None:
    # The 9-table assembler moved to the scheme; the shared writer did not.
    import leadforge.render.relational_io as shared_writer
    from leadforge.schemes.lead_scoring.render.relational import to_dataframes  # noqa: F401

    assert not hasattr(shared_writer, "to_dataframes")
    # The ambiguous flat `leadforge.render.relational` module is gone.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("leadforge.render.relational")


def test_schema_split_primitives_stay_in_schema() -> None:
    # LTV-Pg.2: shared primitives kept in schema/, not moved with the rows.
    from leadforge.schema.entities import (  # noqa: F401
        AccountRow,
        EntityRowProtocol,
        make_empty_dataframe,
    )
    from leadforge.schema.features import FeatureSpec  # noqa: F401
    from leadforge.schema.relationships import FKConstraint, validate_fk  # noqa: F401
    from leadforge.schema.tasks import SplitSpec, TaskManifest  # noqa: F401


def test_schema_split_lead_scoring_specifics_in_scheme() -> None:
    # LTV-Pg.2: lead-scoring-specific symbols live in the scheme package.
    from leadforge.schemes.lead_scoring.entities import (  # noqa: F401
        ALL_ROW_TYPES,
        ContactRow,
        LeadRow,
    )
    from leadforge.schemes.lead_scoring.features import LEAD_SNAPSHOT_FEATURES  # noqa: F401
    from leadforge.schemes.lead_scoring.relationships import ALL_CONSTRAINTS  # noqa: F401
    from leadforge.schemes.lead_scoring.tasks import CONVERTED_WITHIN_90_DAYS  # noqa: F401


def test_schema_split_lead_scoring_removed_from_shared_schema() -> None:
    # LTV-Pg.2: moved symbols are gone from the shared schema namespace.
    import leadforge.schema.entities as shared_entities
    import leadforge.schema.features as shared_features
    import leadforge.schema.relationships as shared_relationships
    import leadforge.schema.tasks as shared_tasks

    assert not hasattr(shared_entities, "LeadRow")
    assert not hasattr(shared_entities, "ALL_ROW_TYPES")
    assert not hasattr(shared_features, "LEAD_SNAPSHOT_FEATURES")
    assert not hasattr(shared_relationships, "ALL_CONSTRAINTS")
    assert not hasattr(shared_tasks, "CONVERTED_WITHIN_90_DAYS")


def test_public_api_unchanged_by_the_move() -> None:
    # The documented public surface must keep importing from its stable home.
    from leadforge.api import Generator, list_recipes  # noqa: F401

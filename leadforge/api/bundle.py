"""Bundle writer — assembles and serialises the full output bundle.

:func:`write_bundle` is called by :meth:`WorldBundle.save` and orchestrates
all rendering steps:

1. Write relational Parquet tables (``tables/``).
2. Build the lead snapshot and write task splits (``tasks/``).
3. Write ``dataset_card.md`` and ``feature_dictionary.csv``.
4. Build and write ``manifest.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from leadforge.render.manifests import build_manifest, write_manifest
from leadforge.render.relational import to_dataframes
from leadforge.render.snapshots import build_snapshot
from leadforge.render.tasks import write_task_splits
from leadforge.schema.dictionaries import write_feature_dictionary
from leadforge.schema.tables import write_parquet

if TYPE_CHECKING:
    from leadforge.core.models import WorldBundle


def write_bundle(bundle: WorldBundle, path: str) -> None:
    """Write *bundle* to disk at *path*.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        path: Destination directory (created if absent).

    Raises:
        RuntimeError: if any of ``bundle.simulation_result``,
            ``bundle.population``, or ``bundle.world_graph`` are ``None``.
    """
    if bundle.simulation_result is None or bundle.population is None or bundle.world_graph is None:
        raise RuntimeError("WorldBundle is not fully populated. Call Generator.generate() first.")

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)

    config = bundle.spec.config
    result = bundle.simulation_result
    population = bundle.population
    world_graph = bundle.world_graph

    # ------------------------------------------------------------------
    # 1. Relational tables → tables/
    # ------------------------------------------------------------------
    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)

    dfs = to_dataframes(result, population)
    table_row_counts: dict[str, int] = {}
    for table_name, df in dfs.items():
        write_parquet(df, tables_dir / f"{table_name}.parquet")
        table_row_counts[table_name] = len(df)

    # ------------------------------------------------------------------
    # 2. Snapshot + task splits → tasks/
    # ------------------------------------------------------------------
    snapshot = build_snapshot(result, population, horizon_days=config.horizon_days)
    task_row_counts = write_task_splits(snapshot, root / "tasks", seed=config.seed)

    # ------------------------------------------------------------------
    # 3. Dataset card and feature dictionary
    # ------------------------------------------------------------------
    from leadforge.narrative.dataset_card import render_dataset_card

    (root / "dataset_card.md").write_text(render_dataset_card(bundle.spec))
    write_feature_dictionary(root / "feature_dictionary.csv")

    # ------------------------------------------------------------------
    # 4. Manifest
    # ------------------------------------------------------------------
    manifest = build_manifest(
        config=config,
        world_graph=world_graph,
        table_row_counts=table_row_counts,
        task_row_counts={"converted_within_90_days": task_row_counts},
        bundle_root=root,
    )
    write_manifest(manifest, root)

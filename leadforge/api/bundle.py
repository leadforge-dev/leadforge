"""Bundle writer — assembles and serialises the full output bundle.

:func:`write_bundle` is called by :meth:`WorldBundle.save` and orchestrates
all rendering steps:

1. Write relational Parquet tables (``tables/``).
2. Build the lead snapshot and write task splits (``tasks/``).
3. Write ``dataset_card.md`` and ``feature_dictionary.csv``.
4. Apply exposure filtering — write ``metadata/`` for ``research_instructor``
   mode; skip it for ``student_public``.
5. Build and write ``manifest.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from leadforge.exposure.modes import apply_exposure
from leadforge.narrative.dataset_card import render_dataset_card
from leadforge.render.manifests import build_manifest, write_manifest
from leadforge.render.relational import to_dataframes
from leadforge.render.snapshots import build_snapshot
from leadforge.render.tasks import write_task_splits
from leadforge.schema.dictionaries import write_feature_dictionary
from leadforge.schema.tables import write_parquet
from leadforge.schema.tasks import task_manifest_for_config

if TYPE_CHECKING:
    from leadforge.core.models import WorldBundle


def write_bundle(
    bundle: WorldBundle,
    path: str,
    generation_timestamp: str | None = None,
) -> None:
    """Write *bundle* to disk at *path*.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        path: Destination directory (created if absent).
        generation_timestamp: ISO-8601 UTC timestamp.  Defaults to now.
            Pass a fixed value to produce byte-identical manifests.

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
    task = task_manifest_for_config(config.primary_task, config.label_window_days)
    task_row_counts = write_task_splits(snapshot, root / "tasks", seed=config.seed, task=task)

    # ------------------------------------------------------------------
    # 3. Dataset card and feature dictionary
    # ------------------------------------------------------------------
    (root / "dataset_card.md").write_text(render_dataset_card(bundle.spec))
    write_feature_dictionary(root / "feature_dictionary.csv")

    # ------------------------------------------------------------------
    # 4. Exposure metadata (research_instructor only)
    # ------------------------------------------------------------------
    apply_exposure(bundle, root, config.exposure_mode)

    # ------------------------------------------------------------------
    # 5. Manifest
    # ------------------------------------------------------------------
    manifest = build_manifest(
        config=config,
        world_graph=world_graph,
        table_row_counts=table_row_counts,
        task_row_counts={task.task_id: task_row_counts},
        bundle_root=root,
        generation_timestamp=generation_timestamp,
    )
    write_manifest(manifest, root)

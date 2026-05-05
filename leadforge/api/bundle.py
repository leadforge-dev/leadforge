"""Bundle writer — assembles and serialises the full output bundle.

:func:`write_bundle` is called by :meth:`WorldBundle.save` and orchestrates
all rendering steps:

1. Project the relational dict (snapshot-safe for ``student_public``,
   full-horizon for ``research_instructor``) and write ``tables/``.
2. Build the lead snapshot and write task splits (``tasks/``).
3. Write ``dataset_card.md`` and ``feature_dictionary.csv``.
4. Apply exposure filtering — write ``metadata/`` for ``research_instructor``
   mode; skip it for ``student_public``.
5. Build and write ``manifest.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from leadforge.exposure.filters import get_filter
from leadforge.exposure.modes import apply_exposure
from leadforge.narrative.dataset_card import render_dataset_card
from leadforge.render.manifests import build_manifest, write_manifest
from leadforge.render.relational import to_dataframes
from leadforge.render.relational_snapshot_safe import to_dataframes_snapshot_safe
from leadforge.render.snapshots import build_snapshot
from leadforge.render.tasks import write_task_splits
from leadforge.schema.dictionaries import write_feature_dictionary
from leadforge.schema.features import LEAD_SNAPSHOT_FEATURES, redacted_columns_for
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

    # The redaction set comes from the canonical feature spec — the same
    # source of truth the validator uses.  It is applied uniformly to
    # every published parquet file (relational tables AND task splits) so
    # users doing feature engineering off the raw tables (per the
    # README's "Option 3") cannot trivially reintroduce a redacted
    # column by joining ``tables/leads.parquet`` to their feature set.
    redacted = redacted_columns_for(config.exposure_mode)
    bundle_filter = get_filter(config.exposure_mode)

    # ------------------------------------------------------------------
    # 1. Relational tables → tables/
    #
    # For ``student_public`` (``relational_snapshot_safe = True``) we
    # project the full-horizon dict onto the snapshot-safe shape:
    # ``BANNED_LEAD_COLUMNS`` / ``BANNED_OPP_COLUMNS`` are dropped, event
    # tables are filtered per-lead to ``lead_created_at + snapshot_day``,
    # and ``BANNED_TABLES`` (``customers`` / ``subscriptions``) are
    # omitted entirely.  The feature-level redaction below still applies
    # on top — the two policies operate on disjoint columns
    # (snapshot-safe owns the structural reconstruction surface;
    # ``redacted_columns_for`` owns near-deterministic snapshot
    # features), so they neither double-emit nor overlap.
    # ------------------------------------------------------------------
    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)

    dfs = to_dataframes(result, population)
    if bundle_filter.relational_snapshot_safe:
        if config.snapshot_day is None:
            raise ValueError(
                f"exposure_mode={config.exposure_mode.value!r} requires "
                "config.snapshot_day to be set (the snapshot-safe relational "
                "export filters event tables to lead_created_at + snapshot_day); "
                "got snapshot_day=None.  Pin a snapshot_day on the recipe or "
                "pass it explicitly."
            )
        dfs = to_dataframes_snapshot_safe(dfs, snapshot_day=config.snapshot_day)
    table_row_counts: dict[str, int] = {}
    for table_name, df in dfs.items():
        if redacted:
            cols_to_drop = [c for c in redacted if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        write_parquet(df, tables_dir / f"{table_name}.parquet")
        table_row_counts[table_name] = len(df)

    # ------------------------------------------------------------------
    # 2. Snapshot + task splits → tasks/
    #
    # Same redaction rule applied to the snapshot DataFrame before the
    # task splits are written, so manifest SHA-256 hashes reflect the
    # published column set without a post-write rewrite step.
    # ------------------------------------------------------------------
    snapshot = build_snapshot(
        result,
        population,
        horizon_days=config.horizon_days,
        snapshot_day=config.snapshot_day,
        difficulty_params=config.difficulty_params,
        seed=config.seed,
    )
    if redacted:
        drop_cols = [c for c in redacted if c in snapshot.columns]
        if drop_cols:
            snapshot = snapshot.drop(columns=drop_cols)
    visible_features = tuple(f for f in LEAD_SNAPSHOT_FEATURES if f.name not in redacted)

    task = task_manifest_for_config(config.primary_task, config.label_window_days)
    task_row_counts = write_task_splits(snapshot, root / "tasks", seed=config.seed, task=task)

    # ------------------------------------------------------------------
    # 3. Dataset card and feature dictionary
    # ------------------------------------------------------------------
    (root / "dataset_card.md").write_text(
        render_dataset_card(
            bundle.spec,
            task_manifest=task,
            table_counts=table_row_counts,
            features=visible_features,
        )
    )
    write_feature_dictionary(root / "feature_dictionary.csv", features=visible_features)

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
        redacted_columns=sorted(redacted),
        relational_snapshot_safe=bundle_filter.relational_snapshot_safe,
    )
    write_manifest(manifest, root)

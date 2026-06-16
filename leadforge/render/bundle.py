"""Shared bundle-writing envelope for all generation schemes (LTV-Pn.4d).

Every scheme's ``write_bundle`` performs the same on-disk sequence — create the
root, write the relational tables, split each task, write the dataset card and
feature dictionary, apply exposure-mode metadata, and build + write the
manifest.  Only the *content* differs (which tables, which tasks, which card),
and each scheme computes that content itself.

:func:`write_bundle_envelope` is that shared sequence.  A scheme computes its
final, exposure-projected relational frames, its per-task ``(manifest, frame)``
exports, its rendered dataset card, and its visible feature catalog, then hands
them here.  This keeps the I/O orchestration — and the file-ordering that the
manifest's hashing depends on — in one place, with no scheme-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from leadforge.exposure.modes import apply_exposure
from leadforge.render.manifests import build_manifest, write_manifest
from leadforge.render.relational_io import write_relational_tables
from leadforge.render.tasks import write_task_splits
from leadforge.schema.dictionaries import write_feature_dictionary

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    import pandas as pd

    from leadforge.core.models import WorldBundle
    from leadforge.schema.features import FeatureSpec
    from leadforge.schema.tasks import TaskManifest

__all__ = ["TaskExport", "write_bundle_envelope"]


@dataclass(frozen=True)
class TaskExport:
    """One task's manifest plus the (already exposure-projected) snapshot frame
    to split into train/valid/test."""

    manifest: TaskManifest
    frame: pd.DataFrame


def write_bundle_envelope(
    bundle: WorldBundle,
    root: Path,
    *,
    relational: dict[str, pd.DataFrame],
    tasks: Sequence[TaskExport],
    dataset_card: str,
    feature_specs: Sequence[FeatureSpec],
    generation_scheme: str,
    redacted: Collection[str] = frozenset(),
    motif_family: str | None = None,
    relational_snapshot_safe: bool = False,
    structural_redactions: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
    generation_timestamp: str | None = None,
) -> None:
    """Write *bundle* to *root* given the scheme's already-computed content.

    Steps (order fixed — the manifest hashes the written files, so they must
    exist first): relational tables → task splits → dataset card → feature
    dictionary → exposure metadata → manifest.

    Args:
        bundle: The fully populated bundle (its ``spec.config`` supplies seed,
            exposure mode, and the manifest's provenance fields).
        root: Destination directory (created if absent).
        relational: ``{table_name: DataFrame}`` already projected for the
            exposure mode (snapshot-safe where required).
        tasks: One :class:`TaskExport` per task directory to write.
        dataset_card: Rendered ``dataset_card.md`` contents.
        feature_specs: The visible feature catalog for ``feature_dictionary.csv``.
        generation_scheme: Producing scheme name (recorded in the manifest).
        redacted: Columns to drop from every written table/split (lead-scoring
            feature redactions; empty for schemes without column redaction).
        motif_family / relational_snapshot_safe / structural_redactions /
        extra_fields: Passed through to :func:`build_manifest`.
        generation_timestamp: ISO timestamp; defaults to now in the manifest.
    """
    config = bundle.spec.config
    root.mkdir(parents=True, exist_ok=True)

    table_row_counts = write_relational_tables(relational, root / "tables", redacted=redacted)

    task_row_counts: dict[str, dict[str, int]] = {}
    for export in tasks:
        task_row_counts[export.manifest.task_id] = write_task_splits(
            export.frame, root / "tasks", seed=config.seed, task=export.manifest
        )

    (root / "dataset_card.md").write_text(dataset_card)
    write_feature_dictionary(root / "feature_dictionary.csv", features=tuple(feature_specs))

    apply_exposure(bundle, root, config.exposure_mode)

    manifest = build_manifest(
        config=config,
        generation_scheme=generation_scheme,
        motif_family=motif_family,
        table_row_counts=table_row_counts,
        task_row_counts=task_row_counts,
        bundle_root=root,
        generation_timestamp=generation_timestamp,
        redacted_columns=sorted(redacted),
        relational_snapshot_safe=relational_snapshot_safe,
        structural_redactions=structural_redactions,
        extra_fields=extra_fields,
    )
    write_manifest(manifest, root)

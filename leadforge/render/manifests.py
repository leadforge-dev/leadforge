"""Bundle manifest builder.

:func:`build_manifest` constructs the ``manifest.json`` dict that is written
at the root of every output bundle.  The manifest records provenance (recipe,
seed, version, generation timestamp) and integrity metadata (row counts and
SHA-256 hashes) for the Parquet data files: relational tables and task splits.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leadforge.core.hashing import file_sha256
from leadforge.validation.leakage_probes import (
    BANNED_LEAD_COLUMNS,
    BANNED_OPP_COLUMNS,
    BANNED_TABLES,
)

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig

# Bump this whenever the bundle layout or manifest schema changes.
# History:
#   "1" — initial layout (pre-M8)
#   "2" — M8 render layer: tables/, tasks/, dataset_card.md,
#         feature_dictionary.csv, manifest.json structure
#   "3" — issue #57 follow-up: ``is_mql`` removed from the canonical
#         feature list (zero-variance); ``is_sql`` redacted in
#         ``student_public`` mode (near-deterministic for non-conversion).
#         ``manifest.redacted_columns`` was already added in PR #56.
#   "4" — issue #57 sub-item 1: windowed snapshot.  Event-aggregate
#         features (touch_count, session_count, expected_acv, ...) now
#         aggregate only events within ``[lead_created_at, lead_created_at
#         + snapshot_day]``.  Column SET unchanged from v3, but column
#         VALUES are no longer full-horizon — consumers pinning v3 and
#         assuming "features computed over full horizon" must update.
#         ``manifest.snapshot_day`` recorded so the contract is
#         self-describing (``null`` means full-horizon, legacy behaviour).
#   "5" — PR 2.2: ``student_public`` bundles route through the
#         snapshot-safe relational export (
#         :mod:`leadforge.schemes.lead_scoring.render.relational_snapshot_safe`).  Public
#         ``leads`` drops ``converted_within_90_days`` /
#         ``conversion_timestamp``; public ``opportunities`` drops
#         ``close_outcome`` / ``closed_at``; public bundles omit
#         ``customers`` / ``subscriptions``; event tables filtered
#         per-lead to ``lead_created_at + snapshot_day``.
#         ``manifest.relational_snapshot_safe`` records the contract so
#         consumers / validators can tell from the bundle alone whether
#         the tables are snapshot-safe.  ``research_instructor`` bundles
#         keep the full-horizon export (``relational_snapshot_safe = false``).
#   "6" — LTV-Pn.1: every manifest now records ``generation_scheme`` (which
#         peer generation scheme produced the bundle — ``lead_scoring`` or
#         ``lifecycle``).  ``build_manifest`` is scheme-agnostic: it takes a
#         ``motif_family`` string (or ``None``) instead of the lead-scoring
#         ``world_graph``, and an ``extra_fields`` mapping for scheme-specific
#         keys (the lifecycle scheme adds ``observation_date`` / forward
#         windows in a later PR).  Data files (tables/, tasks/) are unchanged
#         from v5 for the lead-scoring path; only ``manifest.json`` gains the
#         ``generation_scheme`` field.
BUNDLE_SCHEMA_VERSION = "6"

# Manifest fields whose value is non-deterministic by design (wall-clock,
# host metadata, etc.).  Determinism checks must ignore these fields when
# comparing two bundles produced from the same (recipe, config, seed, version).
NON_DETERMINISTIC_MANIFEST_FIELDS: tuple[str, ...] = ("generation_timestamp",)


def build_manifest(
    config: GenerationConfig,
    generation_scheme: str,
    table_row_counts: dict[str, int],
    task_row_counts: dict[str, dict[str, int]],
    bundle_root: Path,
    generation_timestamp: str | None = None,
    redacted_columns: list[str] | None = None,
    relational_snapshot_safe: bool = False,
    motif_family: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the bundle manifest dict.

    SHA-256 hashes are computed by reading the written Parquet files from
    *bundle_root*, so all table and task files must already exist on disk
    before calling this function.

    Args:
        config: The resolved generation configuration.
        generation_scheme: Name of the peer generation scheme that produced
            the bundle (``lead_scoring`` / ``lifecycle``).  Recorded so a
            consumer can tell which pipeline shape a bundle came from without
            inspecting its tables.
        table_row_counts: Mapping of table name → row count.
        task_row_counts: Mapping of task_id → {split_name → row count}.
        bundle_root: Root directory of the written bundle.
        generation_timestamp: ISO-8601 UTC timestamp string.  Defaults to now.
        redacted_columns: Sorted list of column names that the bundle writer
            removed from snapshot / task splits / feature dictionary for
            this exposure mode.  Recorded in the manifest so consumers
            (and the validator) can audit redaction without inspecting
            package internals.  Defaults to ``[]`` (nothing redacted).
        relational_snapshot_safe: ``True`` if the relational ``tables/``
            were projected through
            :func:`leadforge.schemes.lead_scoring.render.relational_snapshot_safe.to_dataframes_snapshot_safe`
            before being written.  Recorded in the manifest so a tool
            reading a v5+ bundle can tell from the manifest alone whether
            ``tables/`` is the snapshot-safe (public) shape or the
            full-horizon (instructor) shape.  Defaults to ``False``.
        motif_family: The hidden-world motif family, when the scheme has one
            (lead-scoring passes ``world_graph.motif_family``).  ``None`` for
            schemes without a single named motif.  Recorded as
            ``manifest.motif_family``.
        extra_fields: Optional scheme-specific top-level manifest keys merged
            into the result (e.g. the lifecycle scheme's ``observation_date``
            and forward windows).  Must not collide with a core manifest key.

    Returns:
        A JSON-serialisable dict ready to be written as ``manifest.json``.
    """
    if generation_timestamp is None:
        generation_timestamp = datetime.now(UTC).isoformat(timespec="seconds")

    redacted_columns_list = sorted(redacted_columns) if redacted_columns else []

    # Build table entries with row counts and file hashes.
    tables: dict[str, Any] = {}
    for table_name, row_count in table_row_counts.items():
        rel_path = f"tables/{table_name}.parquet"
        abs_path = bundle_root / rel_path
        sha = file_sha256(abs_path)
        tables[table_name] = {"row_count": row_count, "file": rel_path, "sha256": sha}

    # Build task entries.
    tasks: dict[str, Any] = {}
    for task_id, split_counts in task_row_counts.items():
        entry: dict[str, Any] = {}
        for split_name, row_count in split_counts.items():
            rel_path = f"tasks/{task_id}/{split_name}.parquet"
            abs_path = bundle_root / rel_path
            sha = file_sha256(abs_path)
            entry[f"{split_name}_rows"] = row_count
            entry[f"{split_name}_sha256"] = sha
        tasks[task_id] = entry

    manifest: dict[str, Any] = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "package_version": config.package_version,
        "generation_scheme": generation_scheme,
        "recipe_id": config.recipe_id,
        "seed": config.seed,
        "generation_timestamp": generation_timestamp,
        "exposure_mode": config.exposure_mode.value,
        "difficulty": config.difficulty.value,
        "n_accounts": config.n_accounts,
        "n_contacts": config.n_contacts,
        "n_leads": config.n_leads,
        "horizon_days": config.horizon_days,
        "primary_task": config.primary_task,
        "label_window_days": config.label_window_days,
        "snapshot_day": config.snapshot_day,
        "motif_family": motif_family,
        "redacted_columns": redacted_columns_list,
        "relational_snapshot_safe": bool(relational_snapshot_safe),
        "structural_redactions": _build_structural_redactions(bool(relational_snapshot_safe)),
        "tables": tables,
        "tasks": tasks,
    }
    if extra_fields:
        collisions = set(extra_fields) & set(manifest)
        if collisions:
            raise ValueError(
                f"extra_fields would overwrite core manifest keys: {sorted(collisions)}"
            )
        manifest.update(extra_fields)
    return manifest


def _build_structural_redactions(relational_snapshot_safe: bool) -> dict[str, Any]:
    """Self-describing record of the table-level redactions applied at write.

    For snapshot-safe (public) bundles this enumerates every column the
    snapshot-safe export drops from ``leads`` / ``opportunities`` and the
    tables it omits entirely.  For full-horizon (instructor) bundles
    every list is empty.  Together with ``manifest.redacted_columns``
    (the snapshot-feature redactions) and ``manifest.relational_snapshot_safe``
    (the contract flag) the manifest fully describes what the writer
    dropped without the consumer needing to consult package internals.
    """
    if not relational_snapshot_safe:
        return {"columns": {}, "omitted_tables": []}
    return {
        "columns": {
            "leads": sorted(BANNED_LEAD_COLUMNS),
            "opportunities": sorted(BANNED_OPP_COLUMNS),
        },
        "omitted_tables": sorted(BANNED_TABLES),
    }


def write_manifest(manifest: dict[str, Any], bundle_root: Path) -> Path:
    """Serialise *manifest* to ``bundle_root/manifest.json`` and return the path."""
    path = bundle_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path

"""Bundle manifest builder.

:func:`build_manifest` constructs the ``manifest.json`` dict that is written
at the root of every output bundle.  The manifest is the authoritative record
of provenance: it identifies the recipe, seed, version, and every file in the
bundle along with its SHA-256 hash and row count.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig
    from leadforge.structure.graph import WorldGraph

# Bump this whenever the bundle layout or manifest schema changes.
BUNDLE_SCHEMA_VERSION = "1"


def build_manifest(
    config: GenerationConfig,
    world_graph: WorldGraph,
    table_row_counts: dict[str, int],
    task_row_counts: dict[str, dict[str, int]],
    bundle_root: Path,
    generation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the bundle manifest dict.

    SHA-256 hashes are computed by reading the written Parquet files from
    *bundle_root*, so all table and task files must already exist on disk
    before calling this function.

    Args:
        config: The resolved generation configuration.
        world_graph: The sampled hidden world graph (provides motif_family).
        table_row_counts: Mapping of table name → row count.
        task_row_counts: Mapping of task_id → {split_name → row count}.
        bundle_root: Root directory of the written bundle.
        generation_timestamp: ISO-8601 UTC timestamp string.  Defaults to now.

    Returns:
        A JSON-serialisable dict ready to be written as ``manifest.json``.
    """
    if generation_timestamp is None:
        generation_timestamp = datetime.now(UTC).isoformat(timespec="seconds")

    # Build table entries with row counts and file hashes.
    tables: dict[str, Any] = {}
    for table_name, row_count in table_row_counts.items():
        rel_path = f"tables/{table_name}.parquet"
        abs_path = bundle_root / rel_path
        sha = _sha256(abs_path) if abs_path.exists() else ""
        tables[table_name] = {"row_count": row_count, "file": rel_path, "sha256": sha}

    # Build task entries.
    tasks: dict[str, Any] = {}
    for task_id, split_counts in task_row_counts.items():
        entry: dict[str, Any] = {}
        for split_name, row_count in split_counts.items():
            rel_path = f"tasks/{task_id}/{split_name}.parquet"
            abs_path = bundle_root / rel_path
            sha = _sha256(abs_path) if abs_path.exists() else ""
            entry[f"{split_name}_rows"] = row_count
            entry[f"{split_name}_sha256"] = sha
        tasks[task_id] = entry

    return {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "package_version": config.package_version,
        "recipe_id": config.recipe_id,
        "seed": config.seed,
        "generation_timestamp": generation_timestamp,
        "exposure_mode": config.exposure_mode.value,
        "difficulty": config.difficulty.value,
        "n_accounts": config.n_accounts,
        "n_contacts": config.n_contacts,
        "n_leads": config.n_leads,
        "horizon_days": config.horizon_days,
        "motif_family": world_graph.motif_family,
        "tables": tables,
        "tasks": tasks,
    }


def write_manifest(manifest: dict[str, Any], bundle_root: Path) -> Path:
    """Serialise *manifest* to ``bundle_root/manifest.json`` and return the path."""
    path = bundle_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

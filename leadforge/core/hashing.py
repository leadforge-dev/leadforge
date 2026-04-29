"""Deterministic config hashing and file digest helpers.

A config hash uniquely identifies a (recipe, config, seed, version) tuple and
is embedded in every generated manifest so that bundles can be traced back to
the exact parameters that produced them.

:func:`file_sha256` provides a reusable SHA-256 file digest used by the
manifest builder and the bundle validator.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from leadforge.core.models import GenerationConfig


def _canonical(obj: Any) -> Any:
    """Recursively convert to a JSON-stable form (sorted keys, enums → str)."""
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):  # noqa: UP038
        return [_canonical(v) for v in obj]
    # StrEnum values are already strings; this handles plain Enum too
    if hasattr(obj, "value"):
        return obj.value
    return obj


def file_sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_config(config: "GenerationConfig") -> str:
    """Return a stable hex-encoded SHA-256 digest of *config*.

    The digest is derived from a canonicalised JSON representation of the
    dataclass fields, ensuring it is stable across Python processes and
    platform endianness.
    """
    canonical = _canonical(asdict(config))
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()

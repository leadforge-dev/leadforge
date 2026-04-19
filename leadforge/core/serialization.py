"""JSON and YAML read/write helpers used across the package."""

import json
from pathlib import Path
from typing import Any

import yaml

from leadforge.core.exceptions import LeadforgeError


def load_yaml(path: Path) -> Any:
    """Parse a YAML file and return the raw Python object."""
    try:
        with path.open() as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise LeadforgeError(f"Failed to parse YAML at '{path}': {exc}") from exc
    except OSError as exc:
        raise LeadforgeError(f"Cannot read YAML file '{path}': {exc}") from exc


def load_json(path: Path) -> Any:
    """Parse a JSON file and return the raw Python object."""
    try:
        with path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise LeadforgeError(f"Failed to parse JSON at '{path}': {exc}") from exc
    except OSError as exc:
        raise LeadforgeError(f"Cannot read JSON file '{path}': {exc}") from exc


def _json_default(value: Any) -> str:
    """Convert explicitly supported non-JSON types for ``json.dump``.

    Only ``pathlib.Path`` is handled.  Any other non-serialisable type raises
    ``TypeError`` immediately so bugs are caught at serialisation time rather
    than producing silently coerced strings.
    """
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dump_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise *data* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w") as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=True, default=_json_default)
            fh.write("\n")
    except OSError as exc:
        raise LeadforgeError(f"Cannot write JSON file '{path}': {exc}") from exc

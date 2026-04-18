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


def load_json(path: Path) -> Any:
    """Parse a JSON file and return the raw Python object."""
    try:
        with path.open() as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise LeadforgeError(f"Failed to parse JSON at '{path}': {exc}") from exc


def dump_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Serialise *data* to *path* as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=True, default=str)
        fh.write("\n")

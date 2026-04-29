"""Difficulty profile adherence checks.

Verifies that a bundle's manifest declares a known difficulty profile.

NOTE: The v1 simulation engine does not yet modulate conversion rates by
difficulty profile — all profiles currently produce the same rate.  The
``check_difficulty_ordering`` function is therefore a no-op.  Once the
engine wires in difficulty-dependent parameters, it can be extended with
per-profile rate assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Known difficulty profiles.
_KNOWN_DIFFICULTIES = {"intro", "intermediate", "advanced"}


def check_difficulty(manifest: dict[str, Any]) -> list[str]:
    """Check that the manifest declares a known difficulty profile.

    Args:
        manifest: Parsed manifest dict.

    Returns a list of error strings (empty = pass).
    """
    errors: list[str] = []
    difficulty = manifest.get("difficulty")
    if difficulty is None:
        errors.append("Manifest missing 'difficulty' field")
    elif difficulty not in _KNOWN_DIFFICULTIES:
        errors.append(f"Unknown difficulty profile: '{difficulty}'")
    return errors


def check_difficulty_ordering(bundles: dict[str, Path]) -> list[str]:
    """Check that conversion rates decrease as difficulty increases.

    Args:
        bundles: Mapping of difficulty name → bundle path.

    Returns:
        Error strings if the ordering is violated.

    NOTE: This check is a no-op until the simulation engine modulates
    conversion rates by difficulty.  Currently all difficulties produce
    the same rate so we return an empty list unconditionally.
    """
    return []

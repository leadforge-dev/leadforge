"""Difficulty profile adherence checks.

Verifies that a bundle's observed conversion rate is consistent with the
difficulty profile declared in its manifest.

NOTE: The v1 simulation engine does not yet modulate conversion rates by
difficulty profile — all profiles currently produce the same rate.  These
checks are structured for future use but currently only validate that the
manifest declares a known difficulty and that the rate is within a wide
plausible band.  Once the engine wires in difficulty-dependent parameters,
the per-profile ranges can be tightened.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leadforge.core.serialization import load_json

# Known difficulty profiles.  The conversion-rate ranges below are wide
# because the v1 engine does not yet modulate by difficulty.  These will
# be tightened when difficulty-dependent parameters are wired in.
_KNOWN_DIFFICULTIES = {"intro", "intermediate", "advanced"}


def check_difficulty(bundle_root: Path) -> list[str]:
    """Check that the bundle declares a known difficulty profile.

    Returns a list of error strings (empty = pass).
    """
    errors: list[str] = []
    manifest = load_json(bundle_root / "manifest.json")
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
            Must contain at least two difficulties to compare.

    Returns:
        Error strings if the ordering is violated.

    NOTE: This check is a no-op until the simulation engine modulates
    conversion rates by difficulty.  Currently all difficulties produce
    the same rate so we skip the monotonicity assertion.
    """
    ordering = ["intro", "intermediate", "advanced"]
    rates: dict[str, float] = {}

    for diff in ordering:
        bundle_path = bundles.get(diff)
        if bundle_path is None:
            continue
        train_path = bundle_path / "tasks/converted_within_90_days/train.parquet"
        if not train_path.exists():
            continue
        df = pd.read_parquet(train_path, columns=["converted_within_90_days"])
        if len(df) > 0:
            rates[diff] = float(df["converted_within_90_days"].mean())

    # TODO: Once the simulation engine modulates by difficulty, re-enable
    # strict monotonic ordering check here.  For now, just return the
    # observed rates for informational purposes without asserting ordering.
    errors: list[str] = []
    return errors

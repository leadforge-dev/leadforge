"""Difficulty profile adherence checks.

Verifies that a bundle's manifest declares a known difficulty profile and
that the actual conversion rate falls within the declared range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Known difficulty profiles and their expected conversion rate ranges.
_KNOWN_DIFFICULTIES = {"intro", "intermediate", "advanced"}

_CONVERSION_RATE_RANGES: dict[str, tuple[float, float]] = {
    "intro": (0.30, 0.45),
    "intermediate": (0.18, 0.28),
    "advanced": (0.08, 0.15),
}

# Tolerance applied to range bounds for validation (accounts for stochastic variance).
_RATE_TOLERANCE = 0.05


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

    Reads the task train split from each bundle to compute the actual
    conversion rate and verifies:
    1. Each rate falls within the declared range (with tolerance).
    2. Rates are ordered: intro > intermediate > advanced.

    Args:
        bundles: Mapping of difficulty name → bundle path.

    Returns:
        Error strings if any check is violated.
    """
    import pandas as pd

    errors: list[str] = []
    rates: dict[str, float] = {}

    for name, bundle_path in bundles.items():
        # Try all task split files to compute conversion rate.
        task_dir = bundle_path / "tasks" / "converted_within_90_days"
        for split in ("train", "valid", "test"):
            split_path = task_dir / f"{split}.parquet"
            if split_path.exists():
                df = pd.read_parquet(split_path)
                if "converted_within_90_days" in df.columns:
                    if name not in rates:
                        rates[name] = float(df["converted_within_90_days"].mean())
                    break

    # Check each rate is within the declared range (with tolerance).
    for name, rate in rates.items():
        if name in _CONVERSION_RATE_RANGES:
            lo, hi = _CONVERSION_RATE_RANGES[name]
            if rate < lo - _RATE_TOLERANCE:
                errors.append(
                    f"Difficulty '{name}' conversion rate {rate:.3f} "
                    f"is below expected range [{lo:.2f}, {hi:.2f}] "
                    f"(tolerance {_RATE_TOLERANCE})"
                )
            elif rate > hi + _RATE_TOLERANCE:
                errors.append(
                    f"Difficulty '{name}' conversion rate {rate:.3f} "
                    f"is above expected range [{lo:.2f}, {hi:.2f}] "
                    f"(tolerance {_RATE_TOLERANCE})"
                )

    # Check ordering: intro > intermediate > advanced.
    ordering = ["intro", "intermediate", "advanced"]
    for i in range(len(ordering) - 1):
        higher = ordering[i]
        lower = ordering[i + 1]
        if higher in rates and lower in rates:
            if rates[lower] >= rates[higher]:
                errors.append(
                    f"Conversion rate for '{lower}' ({rates[lower]:.3f}) "
                    f"should be less than '{higher}' ({rates[higher]:.3f})"
                )

    return errors

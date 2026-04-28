"""Static latent mechanisms — used for trait sampling at population time.

These mechanisms draw a single value from a fixed distribution given only an
RNG; they do not depend on parent state in the graph.  They are provided as a
library for higher-level code (e.g. future mechanism assignment passes) rather
than being called directly by the simulation engine.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


class CategoricalDraw(Mechanism):
    """Draw one category from a weighted categorical distribution.

    Args:
        categories: Ordered list of category labels.
        weights: Non-negative weights parallel to *categories*; need not sum
            to 1 (normalised internally).
    """

    def __init__(self, categories: list[str], weights: list[float]) -> None:
        if len(categories) != len(weights):
            raise ValueError("categories and weights must have the same length")
        if not categories:
            raise ValueError("categories must not be empty")
        if any(w < 0 for w in weights):
            raise ValueError("all weights must be non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self._categories = list(categories)
        self._weights = [w / total for w in weights]

    @property
    def name(self) -> str:
        return "categorical_draw"

    def sample(self, context: MechanismContext, rng: random.Random) -> str:
        return rng.choices(self._categories, weights=self._weights, k=1)[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "categories": self._categories,
            "weights": self._weights,
        }


class BoundedNumericDraw(Mechanism):
    """Draw a float from a clipped Gaussian in [*lo*, *hi*].

    Args:
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).
        mean: Mean of the underlying Gaussian (clamped to [lo, hi]).
        std: Standard deviation of the underlying Gaussian.
    """

    def __init__(
        self,
        lo: float = 0.0,
        hi: float = 1.0,
        mean: float = 0.5,
        std: float = 0.2,
    ) -> None:
        if lo >= hi:
            raise ValueError(f"lo ({lo}) must be < hi ({hi})")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self._lo = lo
        self._hi = hi
        self._mean = max(lo, min(hi, mean))
        self._std = std

    @property
    def name(self) -> str:
        return "bounded_numeric_draw"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        return max(self._lo, min(self._hi, rng.gauss(self._mean, self._std)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lo": self._lo,
            "hi": self._hi,
            "mean": self._mean,
            "std": self._std,
        }


class MixtureDraw(Mechanism):
    """Draw from a finite mixture of :class:`BoundedNumericDraw` components.

    Args:
        components: List of ``(mean, std)`` pairs, one per mixture component.
            All components share the same ``[lo, hi]`` range.
        mix_weights: Mixture weights (need not sum to 1).
        lo: Shared lower bound.
        hi: Shared upper bound.
    """

    def __init__(
        self,
        components: list[tuple[float, float]],
        mix_weights: list[float],
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> None:
        if not components:
            raise ValueError("components must not be empty")
        if len(components) != len(mix_weights):
            raise ValueError("components and mix_weights must have the same length")
        total = sum(mix_weights)
        if total <= 0:
            raise ValueError("mix_weights must sum to a positive value")
        self._drawers = [BoundedNumericDraw(lo, hi, m, s) for m, s in components]
        self._mix_weights = [w / total for w in mix_weights]

    @property
    def name(self) -> str:
        return "mixture_draw"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        drawer = rng.choices(self._drawers, weights=self._mix_weights, k=1)[0]
        return drawer.sample(context, rng)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "components": [d.to_dict() for d in self._drawers],
            "mix_weights": self._mix_weights,
        }

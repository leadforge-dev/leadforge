"""Influence mechanisms — latent-to-latent propagation along graph edges.

Each mechanism maps a subset of latent traits from ``context.latents`` to a
single float output in [0, 1], representing the influenced child node's value.
"""

from __future__ import annotations

import math
import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


def _weighted_sum(latents: dict[str, float], weights: dict[str, float], bias: float) -> float:
    return bias + sum(weights.get(k, 0.0) * latents.get(k, 0.0) for k in weights)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid that avoids overflow for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class AdditiveInfluence(Mechanism):
    """Weighted sum of parent latents, clipped to [0, 1].

    Args:
        weights: Mapping of latent-key → weight.
        bias: Additive intercept before clipping.
    """

    def __init__(self, weights: dict[str, float], bias: float = 0.0) -> None:
        self._weights = dict(weights)
        self._bias = bias

    @property
    def name(self) -> str:
        return "additive_influence"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        raw = _weighted_sum(context.latents, self._weights, self._bias)
        return max(0.0, min(1.0, raw))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": self._weights, "bias": self._bias}


class LogisticInfluence(Mechanism):
    """Logistic (sigmoid) transform of a weighted latent sum.

    Args:
        weights: Mapping of latent-key → weight.
        bias: Additive intercept inside the sigmoid.
        temperature: Inverse scale applied to the linear combination
            (higher = sharper decision boundary; default 1.0).
    """

    def __init__(
        self,
        weights: dict[str, float],
        bias: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self._weights = dict(weights)
        self._bias = bias
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "logistic_influence"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        raw = _weighted_sum(context.latents, self._weights, self._bias)
        return _sigmoid(raw * self._temperature)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self._weights,
            "bias": self._bias,
            "temperature": self._temperature,
        }


class SaturatingInfluence(Mechanism):
    """Saturating (tanh) transform of a weighted latent sum, mapped to [0, 1].

    Args:
        weights: Mapping of latent-key → weight.
        bias: Additive intercept inside the tanh.
        scale: Pre-tanh multiplier controlling curvature.
    """

    def __init__(
        self,
        weights: dict[str, float],
        bias: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self._weights = dict(weights)
        self._bias = bias
        self._scale = scale

    @property
    def name(self) -> str:
        return "saturating_influence"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        raw = _weighted_sum(context.latents, self._weights, self._bias)
        return (math.tanh(raw * self._scale) + 1.0) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self._weights,
            "bias": self._bias,
            "scale": self._scale,
        }


class ThresholdInfluence(Mechanism):
    """Hard threshold: returns 1.0 if weighted sum ≥ *threshold*, else 0.0.

    Args:
        weights: Mapping of latent-key → weight.
        threshold: Decision boundary.
        bias: Additive intercept before threshold comparison.
    """

    def __init__(
        self,
        weights: dict[str, float],
        threshold: float = 0.5,
        bias: float = 0.0,
    ) -> None:
        self._weights = dict(weights)
        self._threshold = threshold
        self._bias = bias

    @property
    def name(self) -> str:
        return "threshold_influence"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        raw = _weighted_sum(context.latents, self._weights, self._bias)
        return 1.0 if raw >= self._threshold else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": self._weights,
            "threshold": self._threshold,
            "bias": self._bias,
        }


class InteractionTerm(Mechanism):
    """Product of two latent keys, optionally scaled and biased.

    Captures synergy/antagonism effects (e.g. fit × authority).

    Args:
        key_a: First latent key.
        key_b: Second latent key.
        weight: Scalar multiplier on the product.
        bias: Additive term; result clipped to [0, 1].
    """

    def __init__(
        self,
        key_a: str,
        key_b: str,
        weight: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        self._key_a = key_a
        self._key_b = key_b
        self._weight = weight
        self._bias = bias

    @property
    def name(self) -> str:
        return "interaction_term"

    def sample(self, context: MechanismContext, rng: random.Random) -> float:
        a = context.latents.get(self._key_a, 0.0)
        b = context.latents.get(self._key_b, 0.0)
        return max(0.0, min(1.0, self._weight * a * b + self._bias))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key_a": self._key_a,
            "key_b": self._key_b,
            "weight": self._weight,
            "bias": self._bias,
        }

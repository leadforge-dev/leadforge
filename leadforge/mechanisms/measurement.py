"""Measurement mechanisms — hidden truth → noisy CRM observations.

These mechanisms transform latent float values into the imperfect proxies
that appear in CRM exports.  They are applied during the rendering layer
(M9) to introduce realistic data-quality issues.
"""

from __future__ import annotations

import random
from typing import Any

from leadforge.mechanisms.base import Mechanism, MechanismContext


class NoisyProxy(Mechanism):
    """Add Gaussian noise to a latent value and optionally inject missingness.

    Reads ``context.latents[latent_key]``, adds noise, clips to [0, 1], then
    returns ``None`` with probability *missing_rate*.

    Args:
        latent_key: Key to read from ``context.latents``.
        noise_std: Standard deviation of the Gaussian noise term.
        missing_rate: Probability that the observed value is ``None``
            (simulates incomplete enrichment / data gaps).
    """

    def __init__(
        self,
        latent_key: str,
        noise_std: float = 0.10,
        missing_rate: float = 0.05,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        if not (0.0 <= missing_rate <= 1.0):
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        self._key = latent_key
        self._noise_std = noise_std
        self._missing_rate = missing_rate

    @property
    def name(self) -> str:
        return "noisy_proxy"

    def sample(self, context: MechanismContext, rng: random.Random) -> float | None:
        if rng.random() < self._missing_rate:
            return None
        true_val = context.latents.get(self._key, 0.5)
        noisy = true_val + rng.gauss(0.0, self._noise_std) if self._noise_std > 0 else true_val
        return max(0.0, min(1.0, noisy))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latent_key": self._key,
            "noise_std": self._noise_std,
            "missing_rate": self._missing_rate,
        }


class NoisyCategorization(Mechanism):
    """Randomly flip a categorical value to simulate CRM data-entry noise.

    With probability *confusion_prob*, replaces the value found in
    ``context.extra[context_key]`` with a uniformly drawn alternative from
    *categories*.  With probability ``1 - confusion_prob``, returns the
    true value unchanged.

    Args:
        context_key: Key to read from ``context.extra``.
        categories: All valid category labels.
        confusion_prob: Per-record mislabelling probability.
        missing_rate: Probability the field is ``None``.
    """

    def __init__(
        self,
        context_key: str,
        categories: list[str],
        confusion_prob: float = 0.05,
        missing_rate: float = 0.03,
    ) -> None:
        if not categories:
            raise ValueError("categories must not be empty")
        if not (0.0 <= confusion_prob <= 1.0):
            raise ValueError(f"confusion_prob must be in [0, 1], got {confusion_prob}")
        if not (0.0 <= missing_rate <= 1.0):
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        self._key = context_key
        self._categories = list(categories)
        self._confusion_prob = confusion_prob
        self._missing_rate = missing_rate

    @property
    def name(self) -> str:
        return "noisy_categorization"

    def sample(self, context: MechanismContext, rng: random.Random) -> str | None:
        if rng.random() < self._missing_rate:
            return None
        true_val = context.extra.get(self._key)
        if rng.random() < self._confusion_prob or true_val not in self._categories:
            return rng.choice(self._categories)
        return str(true_val)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context_key": self._key,
            "categories": self._categories,
            "confusion_prob": self._confusion_prob,
            "missing_rate": self._missing_rate,
        }


class ProxyCompression(Mechanism):
    """Compress a continuous latent to a coarse ordinal label.

    Partitions [0, 1] into bands using *thresholds* and maps each band to the
    corresponding label in *labels*.  Simulates CRM fields like lead score
    tiers ("low" / "medium" / "high") that collapse a continuous signal.

    Args:
        latent_key: Key to read from ``context.latents``.
        thresholds: Strictly increasing cut-points in (0, 1).  With *k*
            thresholds, *k+1* labels are required.
        labels: Ordered labels, one per band (lowest band first).
        missing_rate: Probability the field is ``None``.
    """

    def __init__(
        self,
        latent_key: str,
        thresholds: list[float],
        labels: list[str],
        missing_rate: float = 0.05,
    ) -> None:
        if len(labels) != len(thresholds) + 1:
            raise ValueError(
                f"Expected {len(thresholds) + 1} labels for {len(thresholds)} thresholds, "
                f"got {len(labels)}"
            )
        for i in range(len(thresholds) - 1):
            if thresholds[i] >= thresholds[i + 1]:
                raise ValueError("thresholds must be strictly increasing")
        if any(not (0.0 < t < 1.0) for t in thresholds):
            raise ValueError("all thresholds must be in (0, 1)")
        if not (0.0 <= missing_rate <= 1.0):
            raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")
        self._key = latent_key
        self._thresholds = list(thresholds)
        self._labels = list(labels)
        self._missing_rate = missing_rate

    @property
    def name(self) -> str:
        return "proxy_compression"

    def sample(self, context: MechanismContext, rng: random.Random) -> str | None:
        if rng.random() < self._missing_rate:
            return None
        val = context.latents.get(self._key, 0.5)
        for i, threshold in enumerate(self._thresholds):
            if val < threshold:
                return self._labels[i]
        return self._labels[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latent_key": self._key,
            "thresholds": self._thresholds,
            "labels": self._labels,
            "missing_rate": self._missing_rate,
        }

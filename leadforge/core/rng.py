"""Seeded RNG root and deterministic substream utilities.

Every stochastic component in leadforge must derive its RNG from a single
seeded root so that (recipe, config, seed, version) fully determines all outputs.

Usage::

    root = RNGRoot(seed=42)
    account_rng = root.child("accounts")
    contact_rng = root.child("contacts")
    # Each child is an independent random.Random with a deterministically
    # derived seed — re-creating from the same root seed always gives
    # the same sequence.

    # For numpy/pandas operations:
    np_rng = root.numpy_child("subsample")
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class RNGRoot:
    """Single seeded RNG root for a generation run.

    All stochastic substreams must be obtained via ``child(name)`` so that
    the full generation is reproducible from the seed alone.
    """

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {type(seed).__name__!r}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    def child(self, name: str) -> random.Random:
        """Return a deterministic ``random.Random`` instance for the named substream.

        The derived seed is SHA-256(``<root_seed>:<name>``) truncated to 8 bytes,
        ensuring each named stream is independent and reproducible.
        """
        digest = hashlib.sha256(f"{self._seed}:{name}".encode()).digest()
        derived_seed = int.from_bytes(digest[:8], "little")
        return random.Random(derived_seed)  # noqa: S311

    def numpy_child(self, name: str) -> np.random.RandomState:
        """Return a deterministic ``np.random.RandomState`` for the named substream.

        Uses the same SHA-256 hash derivation as ``child()`` but truncates to
        4 bytes (``RandomState`` requires a seed in ``[0, 2**32)``).  This means
        ``child("x")`` and ``numpy_child("x")`` produce **different** derived
        seeds — they are independent substreams that happen to share a name.

        .. note::

            ``np.random.RandomState`` is legacy numpy API.  We use it here
            because ``pd.DataFrame.sample(random_state=...)`` and the rest of
            the pipeline code rely on it.  A migration to
            ``np.random.Generator`` is tracked but out of scope for now.
        """
        import numpy as np  # lazy — keeps core/rng.py importable without numpy

        digest = hashlib.sha256(f"{self._seed}:{name}".encode()).digest()
        # RandomState seed must be in [0, 2**32); use 4 bytes.
        derived_seed = int.from_bytes(digest[:4], "little")
        return np.random.RandomState(derived_seed)  # noqa: NPY002

    def __repr__(self) -> str:
        return f"RNGRoot(seed={self._seed})"

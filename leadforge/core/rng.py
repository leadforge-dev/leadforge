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
"""

import hashlib
import random


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

    def __repr__(self) -> str:
        return f"RNGRoot(seed={self._seed})"

"""Tests for leadforge.core.hashing."""

from leadforge.core.hashing import hash_config
from leadforge.core.models import GenerationConfig


def test_same_config_same_hash() -> None:
    c1 = GenerationConfig(seed=42)
    c2 = GenerationConfig(seed=42)
    assert hash_config(c1) == hash_config(c2)


def test_different_seed_different_hash() -> None:
    c1 = GenerationConfig(seed=42)
    c2 = GenerationConfig(seed=99)
    assert hash_config(c1) != hash_config(c2)


def test_hash_is_hex_string() -> None:
    digest = hash_config(GenerationConfig())
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 → 32 bytes → 64 hex chars
    int(digest, 16)  # must be valid hex


def test_hash_stable_across_calls() -> None:
    config = GenerationConfig(seed=7, n_leads=1000)
    h1 = hash_config(config)
    h2 = hash_config(config)
    assert h1 == h2


def test_different_exposure_mode_different_hash() -> None:
    from leadforge.core.enums import ExposureMode

    c1 = GenerationConfig(exposure_mode=ExposureMode.student_public)
    c2 = GenerationConfig(exposure_mode=ExposureMode.research_instructor)
    assert hash_config(c1) != hash_config(c2)

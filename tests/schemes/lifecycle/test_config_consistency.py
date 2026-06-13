"""Pin the lifecycle config defaults to the scheme's canonical constants.

``GenerationConfig`` lives in the shared ``core`` layer, which must not import a
scheme (the LTV-Pn.2 layering cleanup).  So the lifecycle window / tenure
defaults are *duplicated* literals: one copy on ``GenerationConfig`` and the
authoritative copy in ``schemes.lifecycle``.  Until LTV-Pn.4 threads the config
through, these must stay numerically equal — otherwise a bundle generated with
config defaults would carry windows/tenure that disagree with the columns the
snapshot builder actually produces.  This test (which, unlike ``core``, *may*
import both layers) is the guard against that drift.
"""

from __future__ import annotations

import inspect

from leadforge.core.models import GenerationConfig
from leadforge.schemes.lifecycle.engine import simulate_lifecycle
from leadforge.schemes.lifecycle.snapshots import (
    DEFAULT_EARLY_TENURE_WEEKS,
    FORWARD_WINDOWS_DAYS,
)


def test_config_forward_windows_match_snapshot_constant() -> None:
    assert GenerationConfig().forward_windows_days == FORWARD_WINDOWS_DAYS


def test_config_early_tenure_matches_snapshot_constant() -> None:
    assert GenerationConfig().early_tenure_weeks == DEFAULT_EARLY_TENURE_WEEKS


def test_engine_early_tenure_default_matches_snapshot_constant() -> None:
    # The engine carries its own early_tenure_weeks default (the horizon it
    # simulates); it must agree with the snapshot anchor default so a
    # default-config run is fully covered.
    default = inspect.signature(simulate_lifecycle).parameters["early_tenure_weeks"].default
    assert default == DEFAULT_EARLY_TENURE_WEEKS

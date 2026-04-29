"""Exposure-mode dispatch for bundle publication.

:func:`apply_exposure` is the single entry point called by
:func:`~leadforge.api.bundle.write_bundle`.  It reads the resolved
:class:`~leadforge.exposure.filters.BundleFilter` for the requested mode
and performs the corresponding writes (or skips them).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from leadforge.core.enums import ExposureMode
from leadforge.exposure.filters import get_filter
from leadforge.exposure.metadata import write_metadata_dir

if TYPE_CHECKING:
    from leadforge.core.models import WorldBundle


def apply_exposure(bundle: WorldBundle, bundle_root: Path, mode: ExposureMode) -> None:
    """Apply exposure filtering for *mode* to the bundle at *bundle_root*.

    For ``research_instructor`` mode this writes the ``metadata/``
    directory with all hidden-truth files.  For ``student_public`` mode any
    pre-existing ``metadata/`` directory is removed so that hidden truth
    is never accidentally published when reusing an output path.

    Args:
        bundle: Fully populated :class:`~leadforge.core.models.WorldBundle`.
        bundle_root: Root directory of the written bundle (must already exist).
        mode: Exposure mode that controls which artefacts are published.
    """
    filt = get_filter(mode)
    meta_dir = bundle_root / "metadata"
    if filt.write_metadata:
        write_metadata_dir(bundle, bundle_root)
    elif meta_dir.exists():
        shutil.rmtree(meta_dir)

"""Single-source package version, read from installed metadata."""

from importlib.metadata import version

__version__: str = version("leadforge")

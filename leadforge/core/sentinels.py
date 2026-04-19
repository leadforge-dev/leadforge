"""Package-level sentinel objects.

Sentinels are used to distinguish "kwarg was not explicitly provided by
the caller" from any real value (including the package default).  This is
necessary in config-resolution functions where the override dict must be
able to supply a value that explicit kwargs can then beat, but where the
mere presence of a function-signature default must not silently win.
"""


class _MissingType:
    """Type of the :data:`_MISSING` sentinel.

    A named class gives the sentinel a stable, readable representation in
    ``help()`` output and generated documentation (``<default>`` rather than
    the opaque ``<object object at 0x...>`` you get from a bare ``object()``).
    """

    _instance: "_MissingType | None" = None

    def __new__(cls) -> "_MissingType":
        # Singleton — there is exactly one _MISSING value.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<default>"

    def __bool__(self) -> bool:
        return False


# Single shared sentinel — import this rather than defining local object()
# sentinels to avoid tight coupling between modules.
_MISSING: _MissingType = _MissingType()

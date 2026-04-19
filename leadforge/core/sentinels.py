"""Package-level sentinel objects.

Sentinels are used to distinguish "kwarg was not explicitly provided by
the caller" from any real value (including the package default).  This is
necessary in config-resolution functions where the override dict must be
able to supply a value that explicit kwargs can then beat, but where the
mere presence of a function-signature default must not silently win.
"""

from typing import Any

# Single shared sentinel — import this rather than defining local object()
# sentinels to avoid tight coupling between modules.
_MISSING: Any = object()

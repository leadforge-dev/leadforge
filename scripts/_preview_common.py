"""Shared primitives for the local Kaggle / HF preview-page scripts.

PR 7.2 — both ``scripts/preview_kaggle_page.py`` and
``scripts/preview_hf_page.py`` need to:

* HTML-escape user-controlled strings the same way (and emit the
  same entity form so committed sample HTML doesn't churn between
  scripts);
* construct an ``http.server.ThreadingHTTPServer`` rooted at a
  preview-output directory (chosen for ``allow_reuse_address=True``
  inheritance from ``HTTPServer``);
* start serving + optionally pop a browser tab.

Splitting ``make_server`` away from ``serve`` is what lets the test
suite stand the server up on port 0 in a thread, GET ``/``, and
shut down cleanly — the alternative (calling ``serve_forever``
directly) would require subprocess management and a real port
allocation race.
"""

from __future__ import annotations

import http.server
import sys
import webbrowser
from pathlib import Path
from typing import Any


def escape(value: str) -> str:
    """HTML-escape a single attribute / text value.

    Hand-rolled rather than using ``html.escape`` so the committed
    sample HTML uses the decimal ``&#39;`` entity for ``'`` (matching
    what the preview scripts emitted at PR-open time) — switching to
    ``html.escape``'s ``&#x27;`` would force a regen of every
    committed sample with no observable rendering difference.
    """

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _make_handler_factory(directory: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    """Build a handler subclass that serves from ``directory``.

    ``SimpleHTTPRequestHandler`` accepts a ``directory=`` kwarg in
    Python 3.7+, but threading the path through ``ThreadingHTTPServer``'s
    ``RequestHandlerClass`` requires either a ``functools.partial`` or
    a subclass; subclassing keeps the import surface stdlib-only.
    """

    resolved = str(directory.resolve())

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=resolved, **kwargs)

    return _Handler


def make_server(directory: Path, port: int) -> http.server.ThreadingHTTPServer:
    """Build (don't start) an HTTP server rooted at ``directory``.

    ``ThreadingHTTPServer`` (unlike bare ``socketserver.ThreadingTCPServer``)
    inherits ``allow_reuse_address = True`` from ``HTTPServer`` —
    matters because Ctrl-C → re-run within ~60s would otherwise raise
    ``OSError [Errno 48] Address already in use`` while the socket
    sits in TIME_WAIT.

    Pass ``port=0`` to let the kernel pick a free port; the bound
    port is then on ``server.server_address[1]``.  This is the seam
    that makes ``_serve`` testable (test starts the server in a
    thread, fetches one URL, shuts down).
    """

    return http.server.ThreadingHTTPServer(("", port), _make_handler_factory(directory))


def serve(directory: Path, port: int, *, open_browser: bool) -> None:
    """Start the HTTP server rooted at ``directory`` and block.

    Blocks on ``serve_forever()``; KeyboardInterrupt (Ctrl-C) is the
    documented exit path.  Untested by unit tests because it blocks;
    ``make_server`` is the testable seam.
    """

    httpd = make_server(directory, port)
    bound_port = httpd.server_address[1]
    url = f"http://localhost:{bound_port}/"
    print(f"serving {directory} at {url} — Ctrl-C to stop", file=sys.stderr)
    if open_browser:
        webbrowser.open(url)
    httpd.serve_forever()

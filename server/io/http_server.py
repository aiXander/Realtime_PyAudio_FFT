"""Tiny static-file HTTP server for the UI.

Browsers refuse ES module imports from file:// URLs (cross-origin/CORS),
so serving ui/ over http://127.0.0.1:<port>/ is the simplest fix.
Runs in its own thread so it stays out of the asyncio loop's way.
"""
from __future__ import annotations

import http.server
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)


class StaticHTTPServer:
    def __init__(self, host: str, port: int, root: Path):
        self.host = host
        self.port = port
        self.root = root.resolve()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        root_str = str(self.root)

        class Handler(http.server.SimpleHTTPRequestHandler):
            # Force correct JS / WASM / JSON mime types regardless of host OS.
            # On Windows, mimetypes is registry-driven and frequently maps .js
            # to text/plain or application/x-javascript — browsers then refuse
            # to load ES modules and the UI silently dies. SimpleHTTPRequestHandler
            # consults extensions_map first, so this overrides the registry path.
            extensions_map = {
                **http.server.SimpleHTTPRequestHandler.extensions_map,
                ".js":   "text/javascript",
                ".mjs":  "text/javascript",
                ".css":  "text/css",
                ".json": "application/json",
                ".wasm": "application/wasm",
                ".svg":  "image/svg+xml",
                ".html": "text/html",
            }

            def __init__(self, *a, **kw):
                super().__init__(*a, directory=root_str, **kw)

            def end_headers(self):
                # Revalidate on every load (a cheap 304 when unchanged).
                # Without it browsers heuristically cache the ES modules from
                # Last-Modified, so a refresh after editing ui/ can keep
                # running stale code.
                self.send_header("Cache-Control", "no-cache")
                super().end_headers()

            def log_message(self, fmt, *args):
                # silence access logs; errors still surface via log_error
                pass

        self._server = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="http-static", daemon=True)
        self._thread.start()
        log.info("static http server listening on http://%s:%d (serving %s)",
                 self.host, self.port, self.root)

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

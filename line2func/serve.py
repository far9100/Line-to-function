"""Local web viewer for a ``demo`` output folder.

    python -m line2func.serve out/

Serves the viewer (``line2func/viewer/``) and a fixed set of output files from
the folder under ``/data/``. It binds to 127.0.0.1 by default and needs no
internet connection. (``python -m line2func`` starts the full app, which can
also import and trace images; both use the same viewer.)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VIEWER_DIR = Path(__file__).with_name("viewer")
VIEWER = VIEWER_DIR / "index.html"

# the viewer's other files, served next to index.html: name -> content type
ASSETS = {
    "app.js": "text/javascript; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
    "i18n.js": "text/javascript; charset=utf-8",
    "i18n.json": "application/json; charset=utf-8",
    "engine.js": "text/javascript; charset=utf-8",  # the online page's engine (see line2func.web)
    "worker.js": "text/javascript; charset=utf-8",
}

# name -> content type; nothing else in the folder is ever served
DATA_FILES = {
    "curves.json": "application/json; charset=utf-8",
    "out.svg": "image/svg+xml",
    "desmos.txt": "text/plain; charset=utf-8",
    "desmos.js": "text/javascript; charset=utf-8",
    "equations.tex": "text/plain; charset=utf-8",
    "overlay.png": "image/png",
    "source.png": "image/png",
    "lineart.png": "image/png",
    "quality.png": "image/png",
    "quality.json": "application/json; charset=utf-8",
}

LOOPBACK = ("127.0.0.1", "localhost", "::1")


class LocalServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` that owns its port.

    On Windows, the standard ``SO_REUSEADDR`` lets a second server bind a port
    that is already in use, and the two then share its connections. There the
    port is bound with ``SO_EXCLUSIVEADDRUSE`` instead, so a busy port fails
    with an ``OSError`` as on other systems.
    """

    allow_reuse_address = os.name != "nt"
    daemon_threads = True
    block_on_close = False  # don't wait for open event streams when closing
    request_queue_size = 64

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address) -> None:
        # a browser closing a connection early (reload, closed window) is not an error
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class BaseHandler(BaseHTTPRequestHandler):
    """Shared plumbing: responses, the viewer's files and the Host check."""

    server_version = "line2func"

    def host_allowed(self) -> bool:
        """Only requests addressed to this machine by name (blocks DNS-rebinding pages)."""
        bound = self.server.server_address[0]
        if bound not in LOOPBACK:
            return True  # deliberately served to the network (--host)
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def send_body(self, body: bytes, content_type: str, status: int = HTTPStatus.OK,
                  headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        extra = dict(headers or {})
        extra.setdefault("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, obj, status: int = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_body(body, "application/json; charset=utf-8", status, headers)

    def send_status(self, status: int) -> None:
        self.send_body(f"{int(status)} {HTTPStatus(status).phrase}\n".encode(), "text/plain; charset=utf-8", status)

    def serve_viewer(self, path: str) -> bool:
        """Serve ``index.html`` or one of :data:`ASSETS`; False if ``path`` is neither."""
        if path in ("/", "/index.html"):
            self.send_body(VIEWER.read_bytes(), "text/html; charset=utf-8")
            return True
        name = path.lstrip("/")
        if name in ASSETS:
            self.send_body((VIEWER_DIR / name).read_bytes(), ASSETS[name])
            return True
        return False

    def log_message(self, format: str, *args) -> None:  # quiet
        pass


def make_handler(data_dir: Path):
    data_dir = data_dir.resolve()

    class Handler(BaseHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if not self.host_allowed():
                self.send_status(HTTPStatus.FORBIDDEN)
                return
            url = urlparse(self.path)
            if self.serve_viewer(url.path):
                return
            if url.path == "/api/info":
                from line2func import __version__

                files = sorted(name for name in DATA_FILES if (data_dir / name).is_file())
                self.send_json({"app": "line2func", "mode": "static", "version": __version__, "files": files})
                return
            if url.path.startswith("/data/"):
                name = url.path[len("/data/") :]
                path = data_dir / name
                if name in DATA_FILES and path.is_file():
                    extra = {}
                    query = parse_qs(url.query, keep_blank_values=True)
                    if "download" in query:
                        extra["Content-Disposition"] = f'attachment; filename="{name}"'
                    body = path.read_bytes()
                    curves = data_dir / "curves.json"
                    if ("color" in query or "width" in query) and curves.is_file():
                        # the page styles the lines itself; re-export so a download matches what it shows
                        from line2func.jobs import RESTYLED, ApiError, restyled

                        if name in RESTYLED:  # the others have no line style, and ignore the query
                            try:
                                body = restyled(curves.read_bytes(), name, query.get("color", ["measured"])[0],
                                                query.get("seed", ["0"])[0], query.get("width", ["measured"])[0])
                            except ApiError:
                                self.send_status(HTTPStatus.BAD_REQUEST)
                                return
                    self.send_body(body, DATA_FILES[name], headers=extra)
                    return
            self.send_status(HTTPStatus.NOT_FOUND)

        do_HEAD = do_GET

    return Handler


def make_server(data_dir: str | Path, host: str = "127.0.0.1", port: int = 8000) -> LocalServer:
    return LocalServer((host, port), make_handler(Path(data_dir)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.serve", description="Open the line2func viewer.")
    p.add_argument("folder", type=Path, help="output folder written by line2func.demo")
    p.add_argument("--port", type=int, default=8000, help="port (default: 8000; 0 = any free port)")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = p.parse_args(argv)
    if not (args.folder / "curves.json").is_file():
        print(f"error: {args.folder / 'curves.json'} not found; run line2func.demo first", file=sys.stderr)
        return 2
    try:
        server = make_server(args.folder, args.host, args.port)
    except OSError as exc:
        print(f"error: cannot listen on {args.host}:{args.port} ({exc}); try --port 0", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"line2func viewer: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

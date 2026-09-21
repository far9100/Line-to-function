"""Build the online page: the local web page as static files that trace in the browser.

    python -m line2func.website --out _site            # build into _site/
    python -m line2func.website --out _site --serve    # build, then preview it at http://127.0.0.1:8000/

The page is the one ``python -m line2func`` serves (``line2func/viewer``). Next
to it go a static ``api/info``, which puts the page in "web" mode, and a ZIP of
the line2func package (``py/line2func-<version>-<hash>.zip``). In web mode the
page runs line2func in the browser with Pyodide (``viewer/worker.js``,
:mod:`line2func.web`); Pyodide and its numpy, scipy and Pillow come from the
jsDelivr CDN (:data:`PYODIDE_URL`, ``--pyodide-url`` for another copy). Images
never leave the browser. GitHub Pages publishes the folder
(``.github/workflows/pages.yml``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

from line2func import web
from line2func.serve import ASSETS, VIEWER, VIEWER_DIR, LocalServer

PYODIDE_VERSION = "314.0.7"  # = tests/pyodide/package.json (tests check it)
PYODIDE_URL = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
PACKAGES = ("numpy", "scipy", "pillow")  # what line2func.web needs from Pyodide
PACKAGE_DIR = Path(__file__).parent
# Modules the browser never imports, left out of the ZIP below. worker.js imports
# line2func.web and line2func.quality and nothing else: app/serve/browser/website are the
# local server, __main__/demo/eval/synth are commands, and lineart_model/optimize/weights
# belong to the two PyTorch features - Pyodide has no torch, so shipping those three could
# only ever turn one import error into another. lineart.extract still reaches lineart_model
# and pipeline.trace still reaches optimize, but only for a model lineart method and for
# optimize=True, and web.py can produce neither (tests/test_website.py checks that these
# two stay the only way out). A list of what to leave out rather than what to keep, so a
# module added later is merely shipped when it need not be, never missing when it is needed.
NOT_IN_BROWSER = frozenset({"__main__", "app", "browser", "demo", "eval", "serve", "synth",
                            "website", "lineart_model", "optimize", "weights"})
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".json": "application/json; charset=utf-8", ".zip": "application/zip", "": "application/json; charset=utf-8"}


def package_zip() -> bytes:
    """The line2func package as the browser installs it: the modules it imports, no viewer.

    :data:`NOT_IN_BROWSER` is left out. The same files always give the same
    bytes (sorted, fixed dates), so the name built from its hash only changes
    with the code.
    """
    files = sorted(p for p in PACKAGE_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                   and p.suffix == ".py" and p.stem not in NOT_IN_BROWSER)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            info = zipfile.ZipInfo("line2func/" + path.relative_to(PACKAGE_DIR).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return buf.getvalue()


def _is_build(out: Path) -> bool:
    try:
        info = json.loads((out / "api" / "info").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(info, dict) and info.get("app") == "line2func" and info.get("mode") == "web"


def build(out: str | Path, pyodide_url: str = PYODIDE_URL) -> dict:
    """Write the online page into ``out`` (empty, or an earlier build, which is replaced); returns its api/info."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not _is_build(out):
            raise FileExistsError(f"{out} is not empty and holds no earlier build of the online page")
        shutil.rmtree(out)
    if not pyodide_url.endswith("/"):
        pyodide_url += "/"
    page = {"index.html": VIEWER.read_bytes(), **{name: (VIEWER_DIR / name).read_bytes() for name in ASSETS}}
    package = package_zip()
    digest = hashlib.sha256(package)
    for name in sorted(page):
        digest.update(name.encode() + b"\0" + page[name])
    build_id = digest.hexdigest()[:12]
    package_path = f"py/line2func-{web.VERSION}-{hashlib.sha256(package).hexdigest()[:12]}.zip"
    info = {**web.info(), "build": build_id,
            "engine": {"pyodide": pyodide_url, "package": package_path, "packages": list(PACKAGES), "build": build_id}}
    (out / "api").mkdir(parents=True)
    (out / "py").mkdir()
    for name, body in page.items():
        (out / name).write_bytes(body)
    (out / package_path).write_bytes(package)
    (out / "api" / "info").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
    return info


class _Handler(SimpleHTTPRequestHandler):
    """The built folder with the types browsers need (Windows may map .js to text/plain), never cached."""

    def guess_type(self, path):
        return TYPES.get(Path(str(path)).suffix, "application/octet-stream")

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # quiet
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.website",
                                description="Build the online page: line2func's web page as static files that trace "
                                            "in the browser (Pyodide).")
    p.add_argument("--out", type=Path, default=Path("_site"), help="output folder (default: _site)")
    p.add_argument("--pyodide-url", default=PYODIDE_URL,
                   help=f"where Pyodide {PYODIDE_VERSION} and its packages are loaded from (default: {PYODIDE_URL})")
    p.add_argument("--serve", type=int, nargs="?", const=8000, default=None, metavar="PORT",
                   help="then serve the page on 127.0.0.1 to try it (default port 8000; 0 = any free port)")
    args = p.parse_args(argv)
    try:
        info = build(args.out, args.pyodide_url)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"line2func {info['version']} online page (build {info['build']}) in {args.out}")
    if args.serve is None:
        return 0
    try:
        server = LocalServer(("127.0.0.1", args.serve), partial(_Handler, directory=str(args.out)))
    except OSError as exc:
        print(f"error: cannot listen on port {args.serve} ({exc}); try --serve 0", file=sys.stderr)
        return 2
    print(f"  http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

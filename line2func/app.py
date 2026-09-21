"""line2func in the browser: trace images in a local web page.

    python -m line2func                    # opens the page in a tab of the default browser
    python -m line2func --browser none     # only print the address
    python -m line2func --browser chrome   # an app window instead (no tabs, no address bar)

The page is the curve viewer. Drop a line drawing onto it (or choose or paste
one), choose functions or parametric equations and convert: the drawing is
traced like ``demo`` (up to 5,000 curves) with a quality check, and the result
opens in the same page (equations, Desmos, SVG, LaTeX, ZIP). The API can also
extract line art from photos first (jobs of kind "lineart"); the page does not
offer that.

Everything runs on this computer. The server binds to 127.0.0.1, answers only
requests addressed to this machine by name, and every POST must carry a
per-run token that only its own page can read. Ctrl+C or the page's Quit
button ends the program. The page keeps an event stream open
(``/api/events``); in an app window (``--browser chrome`` / ``edge``), once the
last one closes the server waits a short grace period for a reload, then exits
(``--keep-running`` keeps it serving).

Reading images, checking a job's settings and tracing are in
:mod:`line2func.jobs`, which the online page runs in the browser too
(:mod:`line2func.web`); this module adds the HTTP server, the job queue and the
settings file.

Routes (errors are ``{"error": {"code", "detail", "field"}}``; the page translates the codes)::

    GET  /  /app.js  /viewer.js  /i18n.js  /i18n.json   the page (line2func/viewer/)
    GET  /api/info                        version, token, available line-art methods, limits, settings
    GET  /api/events                      server-sent events: hello, job (snapshots), ping, bye
    POST /api/images                      raw image bytes (X-Filename header) -> image info
    GET  /api/images/<id>[/preview]       image info, or its upright preview (JPEG / PNG)
    POST /api/jobs                        {image_id, kind: "lineart" | "trace", method, scale, form, curves, denoise,
                                           faint_sensitivity, ...}
                                          -> snapshot
    GET  /api/jobs/<id>                   job snapshot (state, stage, summary, files)
    POST /api/jobs/<id>/cancel            cancel (a running job stops at its next stage)
    GET  /api/jobs/<id>/data/<file>       curves.json, out.svg, desmos.txt, desmos.js, equations.tex, lineart.png,
                                          quality.json, quality.png  (?download: as an attachment;
                                          out.svg and desmos.js also take ?color=&seed=&width=, the
                                          line style the page shows)
    GET  /api/jobs/<id>/zip               all of them
    POST /api/settings                    {lang, options}, kept in $LINE2FUNC_HOME/app-settings.json
    POST /api/shutdown                    end the program
"""

from __future__ import annotations

import argparse
import hmac
import importlib.util
import json
import os
import re
import secrets
import select
import socket
import sys
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from line2func import __version__, browser, jobs, lineart, pipeline, weights
from line2func.export import DESMOS_CURVE_LIMIT
from line2func.jobs import ApiError, StoredImage
from line2func.serve import DATA_FILES, BaseHandler, LocalServer

MAX_UPLOAD = 64 << 20  # bytes per uploaded file
MAX_PIXELS = 50_000_000  # decoded size limit (big JPEGs are decoded at a reduced scale first)
STORE_SIDE = 4096  # uploads are kept at most this large (long side)
AUTO_SIDE = 2048  # "auto" resolution: long side at most this, never enlarged
MAX_WORK_PIXELS = 12_000_000  # largest image that is traced (about 10 s per megapixel)
QUALITY_MAX_PIXELS = 4_000_000  # the quality check needs ~160 bytes per pixel
PREVIEW_SIDE = 2048  # images sent to the page for display
KEEP_IMAGES = 3
KEEP_JOBS = 12
DERIVED_BYTES = 300 << 20  # resized images and ink maps kept for reuse
GRACE = 10.0  # seconds to wait for a reload after the last window closed
PING = 5.0
LANGS = ("en", "zh-TW")
IMMUTABLE = "private, max-age=31536000, immutable"  # id-addressed files never change
MODEL_METHODS = ("informative", "informative-coarse")
OPTION_KEYS = {"method", "scale", "tolerance", "threshold", "refine", "named", "shape_tolerance", "upscale",
               "faint", "quality", "form", "denoise", "denoise_on", "faint_sensitivity",
               "line_color", "color_seed", "line_width"}
DOWNLOAD_NAMES = {
    "curves.json": "{stem}.json",
    "out.svg": "{stem}.svg",
    "desmos.txt": "{stem}-desmos.txt",
    "desmos.js": "{stem}-desmos.js",
    "equations.tex": "{stem}.tex",
    "lineart.png": "{stem}-lineart.png",
    "quality.json": "{stem}-quality.json",
    "quality.png": "{stem}-quality.png",
}
_ID = r"[0-9a-f]{16}"


def _limits() -> jobs.Limits:
    """The server's limits, read when they are needed (tests change the constants above)."""
    return jobs.Limits(max_pixels=MAX_PIXELS, store_side=STORE_SIDE, auto_side=AUTO_SIDE,
                       max_work_pixels=MAX_WORK_PIXELS, quality_max_pixels=QUALITY_MAX_PIXELS,
                       preview_side=PREVIEW_SIDE)


def _new_id() -> str:
    return secrets.token_hex(8)


def _disposition(filename: str) -> str:
    """``attachment`` header with an ASCII fallback and the UTF-8 name (RFC 6266 / 5987)."""
    ascii_name = "".join(ch if 32 <= ord(ch) < 127 and ch not in '"\\' else "_" for ch in filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class Derived:
    """Resized images and ink maps, shared by line-art previews and tracing (LRU within a byte budget)."""

    def __init__(self, budget: int = DERIVED_BYTES):
        self.budget = budget
        self.lock = threading.Lock()
        self.items: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.nbytes = 0

    def get(self, key: tuple) -> np.ndarray | None:
        with self.lock:
            value = self.items.get(key)
            if value is not None:
                self.items.move_to_end(key)
            return value

    def put(self, key: tuple, value: np.ndarray) -> None:
        with self.lock:
            old = self.items.pop(key, None)
            if old is not None:
                self.nbytes -= old.nbytes
            self.items[key] = value
            self.nbytes += value.nbytes
            while self.nbytes > self.budget and len(self.items) > 1:
                _, dropped = self.items.popitem(last=False)
                self.nbytes -= dropped.nbytes

    def drop_image(self, image_id: str) -> None:
        with self.lock:
            for key in [k for k in self.items if k[1] == image_id]:
                self.nbytes -= self.items.pop(key).nbytes


@dataclass
class Job:
    id: str
    kind: str  # "lineart" (preview) or "trace"
    image_id: str
    image_name: str
    params: dict
    state: str = "queued"  # queued -> running -> done | error | cancelled
    stage: str | None = None
    error: dict | None = None
    summary: dict | None = None
    files: dict[str, bytes] = field(default_factory=dict)
    started: float | None = None
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict:
        end = self.finished if self.finished is not None else time.monotonic()
        return {
            "job_id": self.id, "kind": self.kind, "image_id": self.image_id, "state": self.state,
            "stage": self.stage, "elapsed": round(end - self.started, 2) if self.started is not None else 0.0,
            "error": self.error, "summary": self.summary, "files": sorted(self.files), "params": self.params,
        }


class Board:
    """The job list, one worker thread with a one-slot mailbox, and the event log for the pages."""

    def __init__(self, app: App):
        self.app = app
        self.cond = threading.Condition()
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.pending: Job | None = None
        self.running: Job | None = None
        self.events: list[tuple[int, str, dict]] = []
        self.seq = 0
        self.stopping = False
        self.thread = threading.Thread(target=self._work, name="line2func-worker", daemon=True)
        self.thread.start()

    def publish(self, name: str, data: dict) -> None:
        """Append an event (the caller holds ``cond``)."""
        self.seq += 1
        self.events.append((self.seq, name, data))
        del self.events[:-200]
        self.cond.notify_all()

    def get(self, job_id: str) -> Job | None:
        with self.cond:
            return self.jobs.get(job_id)

    def submit(self, job: Job) -> None:
        """Queue ``job``; a job still waiting in the mailbox is replaced (and cancelled)."""
        with self.cond:
            if self.pending is not None:
                old, self.pending = self.pending, None
                old.state, old.finished = "cancelled", time.monotonic()
                self.publish("job", old.snapshot())
            self.pending = job
            self.jobs[job.id] = job
            for old_id in [k for k, j in self.jobs.items() if j.state in ("done", "error", "cancelled")]:
                if len(self.jobs) <= KEEP_JOBS:
                    break
                del self.jobs[old_id]
            self.publish("job", job.snapshot())

    def cancel(self, job_id: str) -> Job | None:
        with self.cond:
            job = self.jobs.get(job_id)
            if job is self.pending:
                self.pending = None
                job.state, job.finished = "cancelled", time.monotonic()
                self.publish("job", job.snapshot())
            elif job is not None and job is self.running:
                job.cancel.set()  # takes effect at the next stage
            return job

    def set_stage(self, job: Job, stage: str) -> None:
        with self.cond:
            job.stage = stage
            self.publish("job", job.snapshot())

    def stop(self) -> None:
        with self.cond:
            self.stopping = True
            self.cond.notify_all()

    def _work(self) -> None:
        while True:
            with self.cond:
                while self.pending is None and not self.stopping:
                    self.cond.wait()
                if self.stopping:
                    return
                job, self.pending = self.pending, None
                self.running = job
                job.state, job.started = "running", time.monotonic()
                self.publish("job", job.snapshot())
            state, error = "done", None
            try:
                self.app.run_job(job)
            except pipeline.Cancelled:
                state = "cancelled"
            except ApiError as exc:
                state, error = "error", {"code": exc.code, "detail": exc.detail}
            except MemoryError:
                state, error = "error", {"code": "out_of_memory", "detail": None}
            except Exception as exc:  # noqa: BLE001 - reported to the page, logged here
                traceback.print_exc()
                state, error = "error", {"code": "internal", "detail": f"{type(exc).__name__}: {exc}"}
            with self.cond:
                job.state, job.error, job.finished = state, error, time.monotonic()
                if state == "done":
                    job.stage = None
                self.running = None
                self.publish("job", job.snapshot())


class Lifecycle:
    """Counts open event streams (pages); with ``auto_exit`` (an app window) ends the program a grace period
    after the last one closed."""

    def __init__(self, app: App, grace: float = GRACE):
        self.app = app
        self.grace = grace
        self.auto_exit = False
        self.streams = 0
        self.seen = False
        self.timer: threading.Timer | None = None
        self.lock = threading.Lock()

    def opened(self) -> None:
        with self.lock:
            self.streams += 1
            self.seen = True
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None

    def closed(self) -> None:
        with self.lock:
            self.streams -= 1
            if self.auto_exit and self.seen and self.streams == 0 and self.timer is None:
                self.timer = threading.Timer(self.grace, self._expire)
                self.timer.daemon = True
                self.timer.start()

    def _expire(self) -> None:
        with self.lock:
            self.timer = None
            if self.streams:
                return
        self.app.stop("window_closed")


class App:
    def __init__(self, server: LocalServer, *, auto_exit: bool = False, grace: float = GRACE):
        self.server = server
        server.app = self
        self.token = secrets.token_urlsafe(24)
        self.images: OrderedDict[str, StoredImage] = OrderedDict()
        self.images_lock = threading.Lock()
        self.derived = Derived()
        self.models_loaded: set[str] = set()
        self.lifecycle = Lifecycle(self, grace)
        self.lifecycle.auto_exit = auto_exit
        self.stopped = threading.Event()
        self.stop_reason: str | None = None
        self.settings_path = weights.cache_dir() / "app-settings.json"
        self.settings = self._load_settings()
        self.board = Board(self)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    # -- lifetime --------------------------------------------------------------------------------

    def stop(self, reason: str) -> None:
        """End the program: pages get ``bye``, then ``serve_forever`` returns."""
        if self.stopped.is_set():
            return
        self.stop_reason = reason
        self.stopped.set()
        self.board.stop()  # also wakes the event streams
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def close(self) -> None:
        """After ``serve_forever`` returned: say goodbye to open pages, then release the port."""
        self.stop(self.stop_reason or "stopped")
        deadline = time.monotonic() + 0.5
        while self.lifecycle.streams > 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        self.server.server_close()

    # -- settings --------------------------------------------------------------------------------

    def _load_settings(self) -> dict:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out = {}
        if data.get("lang") in LANGS:
            out["lang"] = data["lang"]
        if isinstance(data.get("options"), dict):
            out["options"] = {k: v for k, v in data["options"].items() if k in OPTION_KEYS}
        return out

    def save_settings(self, patch) -> dict:
        if not isinstance(patch, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params")
        settings = dict(self.settings)
        if "lang" in patch:
            settings["lang"] = jobs.check_choice(patch, "lang", LANGS)
        if "options" in patch:
            options = patch["options"]
            if not isinstance(options, dict) or len(json.dumps(options)) > 2000:
                raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="options")
            settings["options"] = {k: v for k, v in options.items()
                                   if k in OPTION_KEYS and (v is None or isinstance(v, (bool, int, float, str)))}
        self.settings = settings
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.settings_path.with_name(self.settings_path.name + ".tmp")
            tmp.write_text(json.dumps(settings, indent=1), encoding="utf-8", newline="\n")
            os.replace(tmp, self.settings_path)
        except OSError:
            pass  # settings are a convenience; never fail the request for them
        return settings

    # -- info and images -------------------------------------------------------------------------

    @staticmethod
    def method_status(method: str) -> dict:
        if method in MODEL_METHODS:
            if importlib.util.find_spec("torch") is None:
                return {"available": False, "reason": "torch_missing"}
            if not weights.path_for(method).is_file():
                return {"available": False, "reason": "weights_missing"}
        return {"available": True, "reason": None}

    def info(self) -> dict:
        return {
            "app": "line2func", "mode": "app", "version": __version__, "token": self.token,
            "methods": {m: self.method_status(m) for m in lineart.METHODS},
            "limits": {"max_bytes": MAX_UPLOAD, "max_pixels": MAX_PIXELS, "max_work_pixels": MAX_WORK_PIXELS,
                       "quality_max_pixels": QUALITY_MAX_PIXELS, "store_side": STORE_SIDE, "auto_side": AUTO_SIDE},
            "desmos_limit": DESMOS_CURVE_LIMIT,
            "weights_command": "python -m line2func.weights fetch informative",
            "settings": self.settings,
            "windows": self.lifecycle.streams,  # open pages (event streams)
        }

    def add_image(self, data: bytes, name: str) -> StoredImage:
        img = jobs.read_image(data, name, _limits(), _new_id())
        with self.images_lock:
            self.images[img.id] = img
            while len(self.images) > KEEP_IMAGES:
                old_id, _ = self.images.popitem(last=False)
                self.derived.drop_image(old_id)
        return img

    def image(self, image_id) -> StoredImage:
        with self.images_lock:
            img = self.images.get(image_id) if isinstance(image_id, str) else None
        if img is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown_image")
        return img

    # -- jobs ------------------------------------------------------------------------------------

    def make_job(self, p) -> Job:
        if not isinstance(p, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params")
        img = self.image(p.get("image_id"))
        kind = jobs.check_choice(p, "kind", ("lineart", "trace"))
        method = jobs.check_choice(p, "method", lineart.METHODS, "none")
        if kind == "lineart" and method == "none":
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="method")
        status = self.method_status(method)
        if not status["available"]:
            raise ApiError(HTTPStatus.CONFLICT, status["reason"], field="method")
        params = {"method": method, "scale": jobs.resolve_scale(img, p.get("scale", "auto"))}
        if kind == "trace":
            params.update(jobs.trace_options(p, img.size, params["scale"], QUALITY_MAX_PIXELS))
        return Job(_new_id(), kind, img.id, img.name, params)

    def _working_rgb(self, img: StoredImage, scale: float) -> np.ndarray:
        if scale >= 1.0:
            return img.rgb
        key = ("rgb", img.id, scale)
        rgb = self.derived.get(key)
        if rgb is None:
            rgb = jobs.scaled(img.rgb, scale)
            self.derived.put(key, rgb)
        return rgb

    def _ink(self, img: StoredImage, scale: float, method: str, rgb: np.ndarray, step) -> np.ndarray:
        key = ("ink", img.id, scale, method)
        ink = self.derived.get(key)
        if ink is not None:
            return ink
        if method in MODEL_METHODS and method not in self.models_loaded:
            step("load_model")
            try:
                from line2func import lineart_model

                lineart_model.load_generator(method)
            except ImportError as exc:
                raise ApiError(HTTPStatus.CONFLICT, "torch_missing", str(exc)) from None
            except weights.WeightsError as exc:
                raise ApiError(HTTPStatus.CONFLICT, "weights_missing", str(exc)) from None
            self.models_loaded.add(method)
        step("lineart")
        try:
            ink = lineart.extract(rgb, method)
        except (ImportError, weights.WeightsError):
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. CUDA out of memory
            if method not in MODEL_METHODS:
                raise
            traceback.print_exc()
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "model_failed", f"{type(exc).__name__}: {exc}") from None
        self.derived.put(key, ink)
        return ink

    def run_job(self, job: Job) -> None:
        """Run one job in the worker thread (``job.files`` and ``job.summary`` are filled in)."""
        p = job.params

        def step(stage: str) -> None:
            if job.cancel.is_set():
                raise pipeline.Cancelled
            self.board.set_stage(job, stage)

        img = self.image(job.image_id)
        step("resize")
        rgb = self._working_rgb(img, p["scale"])
        method = p["method"]
        if job.kind == "lineart":
            ink = self._ink(img, p["scale"], method, rgb, step)
            step("encode")
            h, w = rgb.shape[:2]
            job.files = {"lineart.png": jobs.lineart_png(ink, PREVIEW_SIDE)}
            job.summary = {"width": w, "height": h, "scale": p["scale"]}
            return
        ink = self._ink(img, p["scale"], method, rgb, step) if method != "none" else None
        job.files, job.summary = jobs.run_trace(rgb, img.name, p, step, ink=ink, started=job.started)

    def zip_bytes(self, job: Job) -> bytes:
        return jobs.zip_files(job.files)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHandler):
    """The app's routes (see the module docstring); ``/`` and the viewer's files come from :class:`BaseHandler`."""

    @property
    def app(self) -> App:
        return self.server.app

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        try:
            if not self.host_allowed():
                raise ApiError(HTTPStatus.FORBIDDEN, "forbidden")
            url = urlparse(self.path)
            if method == "POST":
                token = self.headers.get("X-Line2func-Token", "")
                if not hmac.compare_digest(token.encode("latin-1", "replace"), self.app.token.encode()):
                    raise ApiError(HTTPStatus.FORBIDDEN, "bad_token")
                self._post(url.path)
            elif not self.serve_viewer(url.path):
                self._get(url.path, parse_qs(url.query, keep_blank_values=True), head=method == "HEAD")
        except ApiError as exc:
            self.close_connection = True
            self.send_json(exc.payload(), exc.status)
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    # GET routes
    def _get(self, path: str, query: dict, head: bool) -> None:
        app = self.app
        if path == "/api/info":
            self.send_json(app.info())
            return
        if path == "/api/events" and not head:
            self._events()
            return
        m = re.fullmatch(rf"/api/images/({_ID})(/preview)?", path)
        if m:
            img = app.image(m.group(1))
            if m.group(2):
                self.send_body(img.preview, img.preview_type, headers={"Cache-Control": IMMUTABLE})
            else:
                self.send_json(img.info())
            return
        m = re.fullmatch(rf"/api/jobs/({_ID})(?:/(zip|data/([\w.-]+)))?", path)
        if m:
            job = app.board.get(m.group(1))
            if job is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "unknown_job")
            if m.group(2) is None:
                self.send_json(job.snapshot())
                return
            if job.state != "done":
                raise ApiError(HTTPStatus.CONFLICT, "not_ready")
            stem = jobs.stem(job.image_name)
            if m.group(2) == "zip":
                headers = {"Cache-Control": IMMUTABLE}
                if "download" in query:
                    headers["Content-Disposition"] = _disposition(f"{stem}-line2func.zip")
                self.send_body(app.zip_bytes(job), "application/zip", headers=headers)
                return
            name = m.group(3)
            if name not in job.files:
                raise ApiError(HTTPStatus.NOT_FOUND, "not_found")
            body = job.files[name]
            headers = {"Cache-Control": IMMUTABLE}
            if name in jobs.RESTYLED and ("color" in query or "width" in query):
                # the page styles the lines itself; re-export so a download matches what it shows
                body = jobs.restyled(job.files["curves.json"], name, query.get("color", ["measured"])[0],
                                     query.get("seed", ["0"])[0], query.get("width", ["measured"])[0])
                headers["Cache-Control"] = "no-store"  # the style depends on the query, not only on the job id
            if "download" in query:
                headers["Content-Disposition"] = _disposition(DOWNLOAD_NAMES[name].format(stem=stem))
            self.send_body(body, DATA_FILES[name], headers=headers)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    # POST routes
    def _post(self, path: str) -> None:
        app = self.app
        if path == "/api/images":
            data = self._body(MAX_UPLOAD)
            img = app.add_image(data, jobs.clean_name(unquote(self.headers.get("X-Filename", ""))))
            self.send_json(img.info(), HTTPStatus.CREATED)
            return
        if path == "/api/jobs":
            job = app.make_job(self._json())
            app.board.submit(job)
            self.send_json(job.snapshot(), HTTPStatus.ACCEPTED)
            return
        m = re.fullmatch(rf"/api/jobs/({_ID})/cancel", path)
        if m:
            job = app.board.cancel(m.group(1))
            if job is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "unknown_job")
            self.send_json(job.snapshot())
            return
        if path == "/api/settings":
            self.send_json({"settings": app.save_settings(self._json())})
            return
        if path == "/api/shutdown":
            self._body(1 << 10)
            self.send_json({"ok": True})
            app.stop("quit")
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    def _body(self, limit: int) -> bytes:
        length = self.headers.get("Content-Length")
        if length is None:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "length_required")
        try:
            n = int(length)
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request") from None
        if n < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request")
        if n > limit:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large")
        data = self.rfile.read(n)
        if len(data) != n:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_request")
        return data

    def _json(self):
        try:
            return json.loads(self._body(64 << 10) or b"{}")
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_json") from None

    # event stream: job updates, and the signal that the window is still open
    def _events(self) -> None:
        app, board = self.app, self.app.board
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        app.lifecycle.opened()
        try:
            with board.cond:
                last = board.seq
                snapshots = [j.snapshot() for j in board.jobs.values()]
            self._send_event("hello", {"version": __version__, "jobs": snapshots})
            next_ping = time.monotonic() + PING
            while True:
                with board.cond:
                    board.cond.wait_for(lambda: board.seq > last or app.stopped.is_set(), timeout=0.25)
                    events = [e for e in board.events if e[0] > last]
                for seq, name, data in events:
                    self._send_event(name, data)
                    last = seq
                if app.stopped.is_set():
                    self._send_event("bye", {"reason": app.stop_reason})
                    break
                if time.monotonic() >= next_ping:
                    self._send_event("ping", {})
                    next_ping = time.monotonic() + PING
                if self._peer_gone():
                    break
        except OSError:
            pass  # the page went away mid-write
        finally:
            app.lifecycle.closed()

    def _send_event(self, name: str, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _peer_gone(self) -> bool:
        """True once the page closed the stream (a closed socket reads as empty)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True


def make_app(port: int = 0, *, auto_exit: bool = False, grace: float = GRACE, host: str = "127.0.0.1") -> App:
    """Create the server (not yet serving). A busy or reserved ``port`` falls back to any free port."""
    try:
        server = LocalServer((host, port), Handler)
    except OSError:
        if port == 0:
            raise
        server = LocalServer((host, 0), Handler)
    return App(server, auto_exit=auto_exit, grace=grace)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m line2func",
        description="Open line2func in the browser: drop a line drawing onto the page and turn its lines "
                    "into equations.")
    p.add_argument("--port", type=int, default=0, help="port to use (default: any free port)")
    p.add_argument("--browser", choices=browser.CHOICES, default="default",
                   help="default: a tab in the default browser (the default); chrome, edge: an app window "
                        "without tabs or address bar, which ends the program when closed; auto: Chrome's app "
                        "window, else Edge's, else a tab; none: only print the address")
    p.add_argument("--keep-running", action="store_true",
                   help="with an app window: keep running after it is closed (stop with Ctrl+C)")
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # Chinese text on any console code page; lines appear at once even when redirected to a file
            stream.reconfigure(errors="replace", line_buffering=True)
    try:
        app = make_app(args.port)
    except OSError as exc:
        print(f"error: cannot start the local server ({exc})", file=sys.stderr)
        return 2
    if args.port and app.port != args.port:
        print(f"note: port {args.port} is not available, using {app.port}", file=sys.stderr)
    print(f"line2func {__version__}: {app.url}")
    opened = browser.launch(app.url, args.browser)
    app.lifecycle.auto_exit = opened in browser.BROWSERS and not args.keep_running
    if app.lifecycle.auto_exit:
        print("  已開啟 App 視窗；關閉視窗或按 Ctrl+C 即結束。")
        print("  Opened an app window. Close it, or press Ctrl+C, to stop.")
    elif opened == "none":
        print("  請用瀏覽器開啟上面的網址；按 Ctrl+C 結束。")
        print("  Open the address above in a browser. Press Ctrl+C to stop.")
    else:
        print("  已在瀏覽器開啟；按 Ctrl+C 或頁面上的〔結束〕即結束。")
        print("  Opened in the browser. Press Ctrl+C, or Quit on the page, to stop.")
    try:
        app.server.serve_forever()
    except KeyboardInterrupt:
        app.stop_reason = app.stop_reason or "interrupted"
    finally:
        app.close()
    print("line2func: 已結束 / stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

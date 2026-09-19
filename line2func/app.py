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
    GET  /api/jobs/<id>/data/<file>       curves.json, out.svg, desmos.txt, equations.tex, lineart.png,
                                          quality.json, quality.png  (?download: as an attachment)
    GET  /api/jobs/<id>/zip               all of them
    POST /api/settings                    {lang, options}, kept in $LINE2FUNC_HOME/app-settings.json
    POST /api/shutdown                    end the program
"""

from __future__ import annotations

import argparse
import hmac
import importlib.util
import io
import json
import math
import os
import re
import secrets
import select
import socket
import sys
import threading
import time
import traceback
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from http import HTTPStatus
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image

from line2func import __version__, browser, lineart, pipeline, weights
from line2func.export import DESMOS_CURVE_LIMIT, FORMS, output_texts
from line2func.functions import attach
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
               "faint", "quality", "form", "denoise", "denoise_on", "faint_sensitivity"}
DOWNLOAD_NAMES = {
    "curves.json": "{stem}.json",
    "out.svg": "{stem}.svg",
    "desmos.txt": "{stem}-desmos.txt",
    "equations.tex": "{stem}.tex",
    "lineart.png": "{stem}-lineart.png",
    "quality.json": "{stem}-quality.json",
    "quality.png": "{stem}-quality.png",
}
_ID = r"[0-9a-f]{16}"


class ApiError(Exception):
    """An error answer: HTTP status plus a code the page translates."""

    def __init__(self, status: int, code: str, detail: str | None = None, field: str | None = None):
        super().__init__(code)
        self.status, self.code, self.detail, self.field = int(status), code, detail, field

    def payload(self) -> dict:
        return {"error": {"code": self.code, "detail": self.detail, "field": self.field}}


def _new_id() -> str:
    return secrets.token_hex(8)


def _finite(obj):
    """``obj`` with NaN / infinity replaced by None (browsers' JSON.parse rejects NaN)."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _png(image: np.ndarray, max_side: int | None = None) -> bytes:
    im = Image.fromarray(np.ascontiguousarray(image))
    if max_side and max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def _lineart_png(ink: np.ndarray) -> bytes:
    """The ink map as a drawing (dark lines on white), at most :data:`PREVIEW_SIDE` px."""
    return _png(np.rint((1.0 - np.clip(ink, 0.0, 1.0)) * 255.0).astype(np.uint8), PREVIEW_SIDE)


def _clean_name(raw: str) -> str:
    name = re.split(r"[\\/]", raw or "")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:120]
    return name or "image"


def _stem(name: str) -> str:
    stem = name.rsplit(".", 1)[0] if "." in name.strip(".") else name
    return stem.strip() or "image"


def _disposition(filename: str) -> str:
    """``attachment`` header with an ASCII fallback and the UTF-8 name (RFC 6266 / 5987)."""
    ascii_name = "".join(ch if 32 <= ord(ch) < 127 and ch not in '"\\' else "_" for ch in filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _bool(p: dict, key: str, default: bool) -> bool:
    v = p.get(key, default)
    if not isinstance(v, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return v


def _number(p: dict, key: str, default: float | None, lo: float, hi: float, optional: bool = False) -> float | None:
    v = p.get(key, default)
    if v is None and optional:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return float(v)


def _choice(p: dict, key: str, choices: tuple, default=None):
    v = p.get(key, default)
    if isinstance(v, bool) or v not in choices:
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return v


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class StoredImage:
    id: str
    name: str
    rgb: np.ndarray  # at most STORE_SIDE, EXIF orientation applied
    source_size: tuple[int, int]
    suggested: str  # "lineart" or "photo"
    preview: bytes
    preview_type: str

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]

    def max_scale(self) -> float:
        w, h = self.size
        return min(1.0, math.sqrt(MAX_WORK_PIXELS / (w * h)))

    def auto_scale(self) -> float:
        w, h = self.size
        return round(min(1.0, AUTO_SIDE / max(w, h), self.max_scale()), 4)

    def info(self) -> dict:
        w, h = self.size
        return {
            "image_id": self.id, "name": self.name, "width": w, "height": h,
            "source_width": self.source_size[0], "source_height": self.source_size[1],
            "suggested": self.suggested, "auto_scale": self.auto_scale(), "max_scale": round(self.max_scale(), 4),
        }


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
            settings["lang"] = _choice(patch, "lang", LANGS)
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
        try:
            rgb = lineart.open_image(data, max_side=STORE_SIDE, max_pixels=MAX_PIXELS)
        except lineart.ImageTooLarge as exc:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_many_pixels", str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - Pillow raises many kinds of errors for bad files
            raise ApiError(HTTPStatus.BAD_REQUEST, "not_an_image", f"{type(exc).__name__}: {exc}") from None
        try:
            with Image.open(io.BytesIO(data)) as im:
                sw, sh = im.size
                if im.getexif().get(0x0112) in (5, 6, 7, 8):  # rotated by 90 degrees
                    sw, sh = sh, sw
        except Exception:  # noqa: BLE001
            sw, sh = rgb.shape[1], rgb.shape[0]
        suggested = lineart.suggest_mode(rgb)
        if suggested == "photo":
            im = Image.fromarray(rgb)
            if max(im.size) > PREVIEW_SIDE:
                f = PREVIEW_SIDE / max(im.size)
                im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            preview, ptype = buf.getvalue(), "image/jpeg"
        else:
            preview, ptype = _png(rgb, PREVIEW_SIDE), "image/png"
        img = StoredImage(_new_id(), name, rgb, (sw, sh), suggested, preview, ptype)
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

    def resolve_scale(self, img: StoredImage, value) -> float:
        if value is None or value == "auto":
            scale = img.auto_scale()
        else:
            scale = _number({"scale": value}, "scale", None, 0.01, 1.0)
            if scale > img.max_scale() + 1e-6:
                raise ApiError(HTTPStatus.BAD_REQUEST, "too_many_pixels", field="scale")
        w, h = img.size
        if min(w, h) * scale < 8:
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="scale")
        return round(scale, 4)

    def make_job(self, p) -> Job:
        if not isinstance(p, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params")
        img = self.image(p.get("image_id"))
        kind = _choice(p, "kind", ("lineart", "trace"))
        method = _choice(p, "method", lineart.METHODS, "none")
        if kind == "lineart" and method == "none":
            raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="method")
        status = self.method_status(method)
        if not status["available"]:
            raise ApiError(HTTPStatus.CONFLICT, status["reason"], field="method")
        params = {"method": method, "scale": self.resolve_scale(img, p.get("scale", "auto"))}
        if kind == "trace":
            named = _bool(p, "named", False)
            # a number of curves (as demo makes them) instead of the fitting tolerance
            curves = _number(p, "curves", None, 1, 50_000, optional=True)
            if curves is not None and not curves.is_integer():
                raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="curves")
            params.update(
                tolerance=_number(p, "tolerance", 1.0, 0.05, 20.0),
                curves=None if curves is None else int(curves),
                threshold=_number(p, "threshold", None, 0.01, 0.99, optional=True),
                refine=_bool(p, "refine", True),
                form=_choice(p, "form", FORMS, "named" if named else "parametric"),
                shape_tolerance=_number(p, "shape_tolerance", 0.5, 0.05, 10.0),
                upscale=_choice(p, "upscale", ("auto", 1, 2), "auto"),
                faint=_bool(p, "faint", True),
                denoise=_number(p, "denoise", pipeline.DENOISE, 0.0, 100.0),
                faint_sensitivity=_number(p, "faint_sensitivity", pipeline.FAINT_SENSITIVITY, 0.0, 100.0),
                quality=_bool(p, "quality", False),
            )
            if params["quality"]:
                w, h = img.size
                if w * h * params["scale"] ** 2 > QUALITY_MAX_PIXELS:
                    params["quality"] = False  # reported as the "quality_skipped" warning
                    params["quality_skipped"] = True
        return Job(_new_id(), kind, img.id, img.name, params)

    def _working_rgb(self, img: StoredImage, scale: float) -> np.ndarray:
        if scale >= 1.0:
            return img.rgb
        key = ("rgb", img.id, scale)
        rgb = self.derived.get(key)
        if rgb is None:
            w, h = img.size
            rgb = pipeline.resize(img.rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
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
        h, w = rgb.shape[:2]
        method = p["method"]
        if job.kind == "lineart":
            ink = self._ink(img, p["scale"], method, rgb, step)
            step("encode")
            job.files = {"lineart.png": _lineart_png(ink)}
            job.summary = {"width": w, "height": h, "scale": p["scale"]}
            return

        ink = self._ink(img, p["scale"], method, rgb, step) if method != "none" else None
        curves, full_ink = pipeline.trace(
            rgb, lineart_method=method, fit_tolerance=p["tolerance"], threshold=p["threshold"],
            refine=p["refine"], upscale=p["upscale"], shape_tolerance=p["shape_tolerance"],
            faint_lines=p["faint"], ink=ink, progress=step, curve_count=p["curves"], denoise=p["denoise"],
            faint_sensitivity=p["faint_sensitivity"],
        )
        n_shapes = sum(c.shape is not None for c in curves)
        curves.meta.update(source=img.name, lineart=method, vectorizer="baseline", refined=p["refine"],
                           named=p["form"] == "named", form=p["form"], scale=p["scale"],
                           seconds=round(time.monotonic() - job.started, 3))
        if p["threshold"] is not None:
            curves.meta["threshold"] = p["threshold"]
        step("export")
        functions = attach(curves) if p["form"] == "function" else None
        if functions is not None:
            curves.meta["functions"] = functions
        files = {name: text.encode("utf-8") for name, text in output_texts(curves, form=p["form"]).items()}
        if method != "none":
            files["lineart.png"] = _lineart_png(full_ink)
        equations = len(curves) if functions is None else functions["count"]  # lines of desmos.txt
        warnings = []
        if len(curves) == 0:
            warnings.append("no_lines")
        if equations > DESMOS_CURVE_LIMIT:
            warnings.append("over_desmos_limit")
        if p.get("quality_skipped"):
            warnings.append("quality_skipped")
        headline = None
        if p["quality"]:
            step("quality")
            from line2func import quality

            report, maps = quality.assess(curves, full_ink, p["threshold"])
            report = _finite(report)
            files["quality.json"] = json.dumps(report, indent=1, allow_nan=False).encode("utf-8")
            files["quality.png"] = _png(quality.quality_map(maps))
            headline = {"kept": report["recall"]["line"]["within_2px"],
                        "faint": report["recall"]["with_faint"]["within_2px"],
                        "stray": report["flags"]["stray_curves"]}
        job.files = files
        job.summary = {
            "curves": len(curves), "strokes": curves.num_strokes, "shapes": n_shapes, "form": p["form"],
            "equations": equations,
            "width": curves.width, "height": curves.height, "scale": p["scale"],
            "upscale": curves.meta.get("upscale", 1), "line_width": curves.meta.get("line_width"),
            "seconds": curves.meta["seconds"], "warnings": warnings, "quality": headline,
        }

    def zip_bytes(self, job: Job) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(job.files):
                zf.writestr(name, job.files[name])
        return buf.getvalue()


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
            stem = _stem(job.image_name)
            if m.group(2) == "zip":
                headers = {"Cache-Control": IMMUTABLE}
                if "download" in query:
                    headers["Content-Disposition"] = _disposition(f"{stem}-line2func.zip")
                self.send_body(app.zip_bytes(job), "application/zip", headers=headers)
                return
            name = m.group(3)
            if name not in job.files:
                raise ApiError(HTTPStatus.NOT_FOUND, "not_found")
            headers = {"Cache-Control": IMMUTABLE}
            if "download" in query:
                headers["Content-Disposition"] = _disposition(DOWNLOAD_NAMES[name].format(stem=stem))
            self.send_body(job.files[name], DATA_FILES[name], headers=headers)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "not_found")

    # POST routes
    def _post(self, path: str) -> None:
        app = self.app
        if path == "/api/images":
            data = self._body(MAX_UPLOAD)
            img = app.add_image(data, _clean_name(unquote(self.headers.get("X-Filename", ""))))
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
                jobs = [j.snapshot() for j in board.jobs.values()]
            self._send_event("hello", {"version": __version__, "jobs": jobs})
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

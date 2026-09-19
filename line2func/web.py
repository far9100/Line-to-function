"""line2func in the browser: the engine of the online web page.

The online page is the local web page (``line2func/viewer``) served as static
files (:mod:`line2func.website`). It runs Python in the browser with Pyodide,
in a Web Worker (``viewer/worker.js``), which calls :func:`open_image` and
:func:`trace`. They do what the local server does for ``POST /api/images`` and
``POST /api/jobs`` (:mod:`line2func.jobs`), within :data:`LIMITS`: a browser
tab has less memory than the local server, and the browser's WebAssembly is
slower. The image never leaves the browser.

Both answer with JSON text, the same JSON the server sends, plus the files as
bytes. The worker keeps no state of its own: every call brings the image's
bytes (a JavaScript ``Uint8Array`` or bytes); :func:`trace` reuses the image
decoded last when the ``key`` is the same.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
from collections.abc import Callable
from http import HTTPStatus

from line2func import __version__, jobs
from line2func.export import DESMOS_CURVE_LIMIT

VERSION = __version__
MAX_BYTES = 32 << 20  # bytes per image file
# smaller than the local server's (line2func.app): about 0.6 MP took 17-56 s and 460 MB of memory in the browser
LIMITS = jobs.Limits(max_pixels=25_000_000, store_side=2048, auto_side=2048, max_work_pixels=1_500_000,
                     quality_max_pixels=1_500_000)

_last: tuple[str, jobs.StoredImage] | None = None  # the image decoded last, by key


def info() -> dict:
    """The page's ``api/info`` online (written by :mod:`line2func.website`): the server's keys, these limits."""
    return {
        "app": "line2func", "mode": "web", "version": VERSION,
        "limits": {"max_bytes": MAX_BYTES, "max_pixels": LIMITS.max_pixels, "max_work_pixels": LIMITS.max_work_pixels,
                   "quality_max_pixels": LIMITS.quality_max_pixels, "store_side": LIMITS.store_side,
                   "auto_side": LIMITS.auto_side},
        "desmos_limit": DESMOS_CURVE_LIMIT,
    }


def open_image(data, name: str, key: str = "") -> dict:
    """Read an image file as ``POST /api/images`` does: ``{"answer": JSON text, "preview": bytes or None}``.

    The answer is ``{"image": info, "preview_type": ...}`` (info as the server
    sends it, with ``key`` as its ``image_id``) or ``{"error": {...}}``.
    """
    try:
        img = _image(data, name, key)
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error answer
        return {"answer": _answer(_error(exc)), "preview": None}
    return {"answer": _answer({"image": img.info(), "preview_type": img.preview_type}), "preview": img.preview}


def trace(data, name: str, params: str, progress: Callable[[str], None] | None = None, key: str = "") -> dict:
    """Trace an image as a ``POST /api/jobs`` trace job does.

    ``params`` is the job's JSON (the page sends ``kind`` "trace" and ``method``
    "none"); ``progress(stage)`` is called before every stage. Returns
    ``{"answer": JSON text, "files": {name: bytes}, "zip": bytes or None}``; the
    answer is ``{"summary": ..., "params": ...}`` as in the server's job
    snapshot, or ``{"error": {...}}``.
    """
    started = time.monotonic()
    step = progress or (lambda stage: None)
    try:
        p = _params(params)
        jobs.check_choice(p, "kind", ("trace",), "trace")
        jobs.check_choice(p, "method", ("none",), "none")  # the online page traces line art only
        step("resize")
        img = _image(data, name, key)
        scale = jobs.resolve_scale(img, p.get("scale", "auto"))
        options = {"method": "none", "scale": scale, **jobs.trace_options(p, img.size, scale, LIMITS.quality_max_pixels)}
        files, summary = jobs.run_trace(jobs.scaled(img.rgb, scale), img.name, options, step, started=started)
        return {"answer": _answer({"summary": summary, "params": options}), "files": files,
                "zip": jobs.zip_files(files)}
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error answer
        return {"answer": _answer(_error(exc)), "files": {}, "zip": None}
    finally:
        gc.collect()


def _image(data, name: str, key: str) -> jobs.StoredImage:
    global _last
    if key and _last is not None and _last[0] == key:
        return _last[1]
    _last = None  # let the previous image go before decoding the next one
    raw = data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)  # a JavaScript Uint8Array, or bytes
    if len(raw) > MAX_BYTES:
        raise jobs.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large")
    img = jobs.read_image(raw, jobs.clean_name(name), LIMITS, key)
    if key:
        _last = (key, img)
    return img


def _params(text) -> dict:
    try:
        p = json.loads(text) if isinstance(text, (str, bytes)) else text
    except ValueError:
        raise jobs.ApiError(HTTPStatus.BAD_REQUEST, "bad_json") from None
    if not isinstance(p, dict):
        raise jobs.ApiError(HTTPStatus.BAD_REQUEST, "bad_params")
    return p


def _error(exc: BaseException) -> dict:
    """The error answer for ``exc``, with the codes the local server's jobs report."""
    if isinstance(exc, jobs.ApiError):
        return exc.payload()
    if isinstance(exc, MemoryError):
        return {"error": {"code": "out_of_memory", "detail": None, "field": None}}
    traceback.print_exc()
    return {"error": {"code": "internal", "detail": f"{type(exc).__name__}: {exc}", "field": None}}


def _answer(obj) -> str:
    return json.dumps(jobs.finite(obj), ensure_ascii=False, allow_nan=False)

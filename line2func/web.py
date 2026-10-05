"""line2func in the browser: the engine of the online web page.

The online page is the local web page (``line2func/viewer``) served as static
files (:mod:`line2func.website`). It runs Python in the browser with Pyodide,
in a Web Worker (``viewer/worker.js``), which calls :func:`open_image`,
:func:`lineart_preview`, :func:`trace` and :func:`export_svg`. They do what the
local server does for ``POST /api/images``, ``POST /api/jobs`` (its "lineart"
and "trace" jobs) and ``out.svg?color=&width=``
(:mod:`line2func.jobs`), within :data:`LIMITS`: a browser tab has less memory
than the local server, and the browser's WebAssembly is slower. The image never
leaves the browser.

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

import numpy as np

from line2func import __version__, jobs, lineart
from line2func.export import DESMOS_CURVE_LIMIT

VERSION = __version__
MAX_BYTES = 32 << 20  # bytes per image file
# Smaller than the local server's (line2func.app). Measured in real Pyodide on photographs, with the
# quality check the page asks for, extracting line art and then tracing it:
#
#     1.5 MP   2.4 s + 7.4 s    338 MB of WebAssembly heap
#     2.0 MP   3.0 s + 6.2 s    406 MB
#     2.5 MP   3.8 s + 7.6 s    487 MB
#
# so 2.5 MP costs about 11 s and half a gigabyte - lighter than what this page already put up with, as
# 0.6 MP once took 17-56 s and 460 MB. The cap is not where memory runs out; it is where more pixels
# stop buying anything, because budget.reduce_to merges the extra curves away again. WebAssembly memory
# never shrinks, and viewer/engine.js replaces a worker past 1 GB, which 2.5 MP leaves room under.
# quality_max_pixels stops lower because the quality check needs ~160 bytes a pixel on top of all this;
# over it the check is skipped and the page says so (the "quality_skipped" warning).
LIMITS = jobs.Limits(max_pixels=25_000_000, store_side=2048, auto_side=2048, max_work_pixels=2_500_000,
                     quality_max_pixels=2_000_000)

_last: tuple[str, jobs.StoredImage] | None = None  # the image decoded last, by key
_last_ink: tuple[tuple, np.ndarray] | None = None  # the ink map made last, by (key, method, detail, scale)


def info() -> dict:
    """The page's ``api/info`` online (written by :mod:`line2func.website`): the server's keys, these limits."""
    return {
        "app": "line2func", "mode": "web", "version": VERSION,
        # the same shape the local server publishes, so the page reads one thing in both
        "methods": {m: {"available": m in lineart.PURE_METHODS,
                        "reason": None if m in lineart.PURE_METHODS else "torch_missing"}
                    for m in lineart.METHODS},
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

    ``params`` is the job's JSON (the page sends ``kind`` "trace" and the
    ``method`` it previewed, "none" for line art); ``progress(stage)`` is called
    before every stage. Returns
    ``{"answer": JSON text, "files": {name: bytes}, "zip": bytes or None}``; the
    answer is ``{"summary": ..., "params": ...}`` as in the server's job
    snapshot, or ``{"error": {...}}``.
    """
    started = time.monotonic()
    step = progress or (lambda stage: None)
    try:
        p = _params(params)
        jobs.check_choice(p, "kind", ("trace",), "trace")
        art = jobs.lineart_options(p, lineart.PURE_METHODS)  # no PyTorch in a browser: no model methods
        step("resize")
        img = _image(data, name, key)
        scale = jobs.resolve_scale(img, p.get("scale", "auto"))
        options = {**art, "scale": scale, **jobs.trace_options(p, img.size, scale, LIMITS.quality_max_pixels)}
        rgb = jobs.scaled(img.rgb, scale)
        ink = _ink(key, rgb, art, scale, step) if art["method"] != "none" else None
        files, summary = jobs.run_trace(rgb, img.name, options, step, ink=ink, started=started)
        return {"answer": _answer({"summary": summary, "params": options}), "files": files,
                "zip": jobs.zip_files(files)}
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error answer
        return {"answer": _answer(_error(exc)), "files": {}, "zip": None}
    finally:
        gc.collect()


def lineart_preview(data, name: str, params: str, progress: Callable[[str], None] | None = None,
                    key: str = "") -> dict:
    """Extract line art as a ``POST /api/jobs`` job of kind "lineart" does.

    ``params`` is the job's JSON (``kind`` "lineart", ``method`` one of
    :data:`line2func.lineart.PURE_METHODS` other than "none", ``lineart_detail``
    0..100, ``scale``); ``progress(stage)`` is called before every stage.
    Returns ``{"answer": JSON text, "files": {"lineart.png": bytes}}``; the
    answer is ``{"summary": ..., "params": ...}`` as in the server's job
    snapshot, or ``{"error": {...}}``.

    It is named apart from the :mod:`line2func.lineart` module this reads; the
    page's message type and the job's kind are both still "lineart".
    """
    started = time.monotonic()
    step = progress or (lambda stage: None)
    try:
        p = _params(params)
        jobs.check_choice(p, "kind", ("lineart",), "lineart")
        art = jobs.lineart_options(p, lineart.PURE_METHODS)
        if art["method"] == "none":
            raise jobs.ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="method")
        step("resize")
        img = _image(data, name, key)
        scale = jobs.resolve_scale(img, p.get("scale", "auto"))
        rgb = jobs.scaled(img.rgb, scale)
        ink = _ink(key, rgb, art, scale, step)
        step("encode")
        h, w = rgb.shape[:2]
        summary = {"width": w, "height": h, "scale": scale, "seconds": round(time.monotonic() - started, 3)}
        return {"answer": _answer({"summary": summary, "params": {**art, "scale": scale}}),
                "files": {"lineart.png": jobs.lineart_png(ink, LIMITS.preview_side)}}
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error answer
        return {"answer": _answer(_error(exc)), "files": {}}
    finally:
        gc.collect()


def export_svg(curves, color_mode: str = "measured", seed=0, width_mode: str = "measured",
               name: str = "out.svg") -> dict:
    """Write a styled output again in a chosen line style, as the servers do for ``?color=&width=``.

    ``name`` is one of :data:`line2func.jobs.RESTYLED` (``out.svg`` or ``desmos.js``)
    and ``curves`` is the result's own ``curves.json`` (a JavaScript ``Uint8Array``
    or bytes), so nothing is traced again. Returns ``{"answer": JSON text,
    "svg": bytes or None}``; the answer is ``{"ok": true}`` or ``{"error": {...}}``.
    """
    try:
        raw = curves.to_bytes() if hasattr(curves, "to_bytes") else bytes(curves)
        svg = jobs.restyled(raw, name, color_mode, seed, width_mode)
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error answer
        return {"answer": _answer(_error(exc)), "svg": None}
    return {"answer": _answer({"ok": True}), "svg": svg}


def _image(data, name: str, key: str) -> jobs.StoredImage:
    global _last
    if key and _last is not None and _last[0] == key:
        return _last[1]
    global _last_ink
    _last = _last_ink = None  # let the previous image and its ink go before decoding the next one
    raw = data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)  # a JavaScript Uint8Array, or bytes
    if len(raw) > MAX_BYTES:
        raise jobs.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too_large")
    img = jobs.read_image(raw, jobs.clean_name(name), LIMITS, key)
    if key:
        _last = (key, img)
    return img


def _ink(key: str, rgb: np.ndarray, art: dict, scale: float, step) -> np.ndarray:
    """The ink map for these settings, reusing the one made last when it is the same one.

    This is what makes the line art the user confirmed the line art that is traced: the page previews,
    then traces, and the second call finds the first call's answer. It is only ever an optimisation -
    the worker is replaced when it grows too large (``viewer/engine.js``), and the cache goes with it -
    so nothing may depend on a hit.
    """
    global _last_ink
    wanted = (key, art["method"], art["lineart_detail"], scale, rgb.shape)
    if key and _last_ink is not None and _last_ink[0] == wanted:
        return _last_ink[1]
    _last_ink = None  # let the previous one go before making the next
    step("lineart")
    ink = lineart.extract(rgb, art["method"], art["lineart_detail"])
    if key:
        _last_ink = (wanted, ink)
    return ink


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

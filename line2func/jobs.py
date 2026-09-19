"""The web page's tracing jobs, shared by the local server and the in-browser version.

Reading an uploaded image, checking a job's settings and tracing the image into
the files the page shows and downloads: what ``python -m line2func``
(:mod:`line2func.app`) does for ``/api/images`` and ``/api/jobs``, and what the
online page does in the browser (:mod:`line2func.web`, run by Pyodide). Nothing
here serves HTTP or starts threads, so it runs in both places. The limits come
from the caller (:class:`Limits`): a browser tab has less memory than the local
server.

Errors are :class:`ApiError`: an HTTP status and a code that the page translates.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus

import numpy as np
from PIL import Image

from line2func import lineart, pipeline
from line2func.export import DESMOS_CURVE_LIMIT, FORMS, output_texts
from line2func.functions import attach

PREVIEW_SIDE = 2048  # images sent to the page for display


@dataclass(frozen=True)
class Limits:
    """How large an image may be, and how large it is traced."""

    max_pixels: int  # decoded size limit (big JPEGs are decoded at a reduced scale first)
    store_side: int  # uploads are kept at most this large (long side)
    auto_side: int  # "auto" resolution: long side at most this, never enlarged
    max_work_pixels: int  # largest image that is traced
    quality_max_pixels: int  # the quality check needs ~160 bytes per pixel
    preview_side: int = PREVIEW_SIDE  # images sent to the page for display


class ApiError(Exception):
    """An error answer: HTTP status plus a code the page translates."""

    def __init__(self, status: int, code: str, detail: str | None = None, field: str | None = None):
        super().__init__(code)
        self.status, self.code, self.detail, self.field = int(status), code, detail, field

    def payload(self) -> dict:
        return {"error": {"code": self.code, "detail": self.detail, "field": self.field}}


def finite(obj):
    """``obj`` with NaN / infinity replaced by None (browsers' JSON.parse rejects NaN)."""
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    return obj


def png(image: np.ndarray, max_side: int | None = None) -> bytes:
    im = Image.fromarray(np.ascontiguousarray(image))
    if max_side and max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def lineart_png(ink: np.ndarray, max_side: int = PREVIEW_SIDE) -> bytes:
    """The ink map as a drawing (dark lines on white), at most ``max_side`` px."""
    return png(np.rint((1.0 - np.clip(ink, 0.0, 1.0)) * 255.0).astype(np.uint8), max_side)


def clean_name(raw: str) -> str:
    name = re.split(r"[\\/]", raw or "")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:120]
    return name or "image"


def stem(name: str) -> str:
    stem = name.rsplit(".", 1)[0] if "." in name.strip(".") else name
    return stem.strip() or "image"


def zip_files(files: dict[str, bytes]) -> bytes:
    """All of a job's files in one ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            zf.writestr(name, files[name])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def check_bool(p: dict, key: str, default: bool) -> bool:
    v = p.get(key, default)
    if not isinstance(v, bool):
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return v


def check_number(p: dict, key: str, default: float | None, lo: float, hi: float,
                 optional: bool = False) -> float | None:
    v = p.get(key, default)
    if v is None and optional:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return float(v)


def check_choice(p: dict, key: str, choices: tuple, default=None):
    v = p.get(key, default)
    if isinstance(v, bool) or v not in choices:
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field=key)
    return v


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


@dataclass
class StoredImage:
    id: str
    name: str
    rgb: np.ndarray  # at most limits.store_side, EXIF orientation applied
    source_size: tuple[int, int]
    suggested: str  # "lineart" or "photo"
    preview: bytes
    preview_type: str
    limits: Limits

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]

    def max_scale(self) -> float:
        w, h = self.size
        return min(1.0, math.sqrt(self.limits.max_work_pixels / (w * h)))

    def auto_scale(self) -> float:
        w, h = self.size
        return round(min(1.0, self.limits.auto_side / max(w, h), self.max_scale()), 4)

    def info(self) -> dict:
        w, h = self.size
        return {
            "image_id": self.id, "name": self.name, "width": w, "height": h,
            "source_width": self.source_size[0], "source_height": self.source_size[1],
            "suggested": self.suggested, "auto_scale": self.auto_scale(), "max_scale": round(self.max_scale(), 4),
        }


def read_image(data: bytes, name: str, limits: Limits, image_id: str = "") -> StoredImage:
    """Decode an uploaded image file: upright, at most ``limits.store_side``, with the preview the page shows."""
    try:
        rgb = lineart.open_image(data, max_side=limits.store_side, max_pixels=limits.max_pixels)
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
        if max(im.size) > limits.preview_side:
            f = limits.preview_side / max(im.size)
            im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        preview, ptype = buf.getvalue(), "image/jpeg"
    else:
        preview, ptype = png(rgb, limits.preview_side), "image/png"
    return StoredImage(image_id, name, rgb, (sw, sh), suggested, preview, ptype, limits)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def resolve_scale(img: StoredImage, value) -> float:
    """The scale to trace ``img`` at: "auto" (or None) or a number up to the largest the limits allow."""
    if value is None or value == "auto":
        scale = img.auto_scale()
    else:
        scale = check_number({"scale": value}, "scale", None, 0.01, 1.0)
        if scale > img.max_scale() + 1e-6:
            raise ApiError(HTTPStatus.BAD_REQUEST, "too_many_pixels", field="scale")
    w, h = img.size
    if min(w, h) * scale < 8:
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="scale")
    return round(scale, 4)


def trace_options(p: dict, size: tuple[int, int], scale: float, quality_max_pixels: int) -> dict:
    """A trace job's settings from the request ``p``, checked, with their defaults.

    ``size`` is the stored image's and ``scale`` the one it is traced at. The
    quality check is left out (``quality_skipped``, reported as a warning) when
    the traced image has more than ``quality_max_pixels`` pixels.
    """
    named = check_bool(p, "named", False)
    # a number of curves (as demo makes them) instead of the fitting tolerance
    curves = check_number(p, "curves", None, 1, 50_000, optional=True)
    if curves is not None and not curves.is_integer():
        raise ApiError(HTTPStatus.BAD_REQUEST, "bad_params", field="curves")
    options = dict(
        tolerance=check_number(p, "tolerance", 1.0, 0.05, 20.0),
        curves=None if curves is None else int(curves),
        threshold=check_number(p, "threshold", None, 0.01, 0.99, optional=True),
        refine=check_bool(p, "refine", True),
        form=check_choice(p, "form", FORMS, "named" if named else "parametric"),
        shape_tolerance=check_number(p, "shape_tolerance", 0.5, 0.05, 10.0),
        upscale=check_choice(p, "upscale", ("auto", 1, 2), "auto"),
        faint=check_bool(p, "faint", True),
        denoise=check_number(p, "denoise", pipeline.DENOISE, 0.0, 100.0),
        faint_sensitivity=check_number(p, "faint_sensitivity", pipeline.FAINT_SENSITIVITY, 0.0, 100.0),
        quality=check_bool(p, "quality", False),
    )
    if options["quality"]:
        w, h = size
        if w * h * scale**2 > quality_max_pixels:
            options["quality"] = False  # reported as the "quality_skipped" warning
            options["quality_skipped"] = True
    return options


def scaled(rgb: np.ndarray, scale: float) -> np.ndarray:
    """``rgb`` resized by ``scale`` (at most 1) for tracing."""
    if scale >= 1.0:
        return rgb
    h, w = rgb.shape[:2]
    return pipeline.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))


def run_trace(rgb: np.ndarray, image_name: str, p: dict, step: Callable[[str], None], *,
              ink: np.ndarray | None = None, started: float) -> tuple[dict[str, bytes], dict]:
    """Trace ``rgb`` with the job settings ``p`` (``method``, ``scale`` and :func:`trace_options`).

    ``ink`` is a ready ink map for a line-art method other than "none";
    ``step(stage)`` is called before every stage (:data:`line2func.pipeline.STAGES`,
    then "export" and "quality") and may raise to stop; ``started`` is the job's
    start (``time.monotonic()``). Returns the files (``curves.json``, ``out.svg``,
    ``desmos.txt``, ``equations.tex``; ``lineart.png``, ``quality.json`` and
    ``quality.png`` when made) and the summary the page shows.
    """
    method = p["method"]
    curves, full_ink = pipeline.trace(
        rgb, lineart_method=method, fit_tolerance=p["tolerance"], threshold=p["threshold"],
        refine=p["refine"], upscale=p["upscale"], shape_tolerance=p["shape_tolerance"],
        faint_lines=p["faint"], ink=ink, progress=step, curve_count=p["curves"], denoise=p["denoise"],
        faint_sensitivity=p["faint_sensitivity"],
    )
    n_shapes = sum(c.shape is not None for c in curves)
    curves.meta.update(source=image_name, lineart=method, vectorizer="baseline", refined=p["refine"],
                       named=p["form"] == "named", form=p["form"], scale=p["scale"],
                       seconds=round(time.monotonic() - started, 3))
    if p["threshold"] is not None:
        curves.meta["threshold"] = p["threshold"]
    step("export")
    functions = attach(curves) if p["form"] == "function" else None
    if functions is not None:
        curves.meta["functions"] = functions
    files = {name: text.encode("utf-8") for name, text in output_texts(curves, form=p["form"]).items()}
    if method != "none":
        files["lineart.png"] = lineart_png(full_ink)
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
        report = finite(report)
        files["quality.json"] = json.dumps(report, indent=1, allow_nan=False).encode("utf-8")
        files["quality.png"] = png(quality.quality_map(maps))
        headline = {"kept": report["recall"]["line"]["within_2px"],
                    "faint": report["recall"]["with_faint"]["within_2px"],
                    "stray": report["flags"]["stray_curves"]}
    summary = {
        "curves": len(curves), "strokes": curves.num_strokes, "shapes": n_shapes, "form": p["form"],
        "equations": equations,
        "width": curves.width, "height": curves.height, "scale": p["scale"],
        "upscale": curves.meta.get("upscale", 1), "line_width": curves.meta.get("line_width"),
        "seconds": curves.meta["seconds"], "warnings": warnings, "quality": headline,
    }
    return files, summary

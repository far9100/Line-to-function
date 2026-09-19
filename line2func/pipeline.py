"""The whole tracing flow, shared by ``demo`` and the browser app.

image -> ink map -> (optional upscale) -> trace -> refine -> measure
      -> [outline] -> [residual] -> [fill] -> [count] -> [optimize] -> shapes

The bracketed stages are optional. ``outline`` turns solid areas such
as heavy eyelashes (:func:`line2func.baseline.solid_areas`, while tracing) and
thick or wedge-shaped strokes (:mod:`line2func.outline`) into filled outlines,
``residual`` traces the ink the first pass left uncovered
(:mod:`line2func.residual`) and ``fill`` adds rings inside the filled areas so
that they look filled in Desmos too (:mod:`line2func.fill`); these three are
on by default. ``count`` runs when a curve count is asked for (below), and
``optimize`` refines every curve by render-and-compare
(:mod:`line2func.optimize`; it needs PyTorch and is off unless asked for).

**Curve count.** ``curve_count=n`` asks for exactly ``n`` curves instead of a
fitting tolerance: the drawing is traced finely (:data:`COUNT_TOLERANCE`, or
:data:`FINE_TOLERANCE` if that gives fewer than ``n`` curves - chosen up front
when the drawing's lines are too short for ``n`` curves at the coarser one),
then the neighbouring pieces whose merge changes the drawing least are merged
until ``n`` remain (:mod:`line2func.budget`). At the same count this is as close to
the drawing as a plain tolerance, or closer; for small counts it keeps far
more of the lines. If even the finest tracing yields fewer than ``n`` curves,
that is the most the drawing supports and all of them are kept
(``meta["curve_count"]`` reports it).

**Upscaling.** Thin lines (about 1 px) leave the tracer too few pixels to
separate close lines and to place curves precisely. With ``upscale=2`` the
image is enlarged (Lanczos), traced, refined and measured at 2x, and the
curves are then mapped back to the original coordinates. The fitting tolerance
is scaled too, so it stays in *original* pixels and the curve count does not
double. ``upscale="auto"`` picks 2 when the estimated line width is below
:data:`AUTO_UPSCALE_BELOW` px, else 1.

**Ink threshold.** Without ``threshold``, line art (``lineart_method="none"``)
is traced at Otsu's threshold lowered to :data:`AUTO_THRESHOLD_MAX`, so that
faint strokes, such as light strands of hair, stay whole instead of breaking
into dashes. If what the lower threshold adds looks like paper rather than
lines (wide patches of shading, or much more ink), Otsu's threshold is kept
(:func:`trace_threshold`). Whether to upscale is still judged at Otsu's.

**Progress.** ``trace(..., progress=f)`` calls ``f(stage)`` before each stage
(:data:`STAGES`); ``f`` may raise, e.g. :class:`Cancelled`, to stop between
stages. The web app uses this for its progress display and its Cancel button.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from PIL import Image

from scipy import ndimage

from line2func import attributes, baseline, budget, lineart, shapes
from line2func.curves import CurveSet

AUTO_UPSCALE_BELOW = 1.75  # px: estimated line width under which "auto" upscales 2x
AUTO_THRESHOLD_MAX = 0.25  # the automatic ink threshold for line art goes no higher (see trace_threshold)
COUNT_TOLERANCE = 0.35  # px: fitting tolerance when a curve count is asked for (then merged down) ...
FINE_TOLERANCE = 0.25  # px: ... and when that gives too few curves (starting finer only costs time)
COUNT_PIECE_LENGTH = 10.0  # px of line per curve at COUNT_TOLERANCE (7-10.5 on real drawings)
SOLID_RATIO = 2.5  # line widths: ink at least this thick over some length is a solid area (outline=True)
FILL_SPACING = 1.5  # px: distance between the rings inside filled areas (fill=True)
FILL_TOLERANCE = 0.5  # px: fitting tolerance of those rings
STAGES = ("lineart", "upscale", "vectorize", "refine", "measure", "outline", "residual", "fill", "count",
          "optimize", "shapes")


class Cancelled(Exception):
    """Raised by a ``progress`` callback to stop :func:`trace` before its next stage."""


def resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 image resized to ``size`` = ``(width, height)`` (Lanczos)."""
    if (rgb.shape[1], rgb.shape[0]) == tuple(size):
        return rgb
    return np.asarray(Image.fromarray(np.ascontiguousarray(rgb)).resize(tuple(size), Image.LANCZOS))


def _resize_ink(ink: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    big = Image.fromarray(np.ascontiguousarray(ink, dtype=np.float32)).resize(tuple(size), Image.BICUBIC)
    return np.clip(np.asarray(big, dtype=np.float32), 0.0, 1.0)


def trace_threshold(ink: np.ndarray, scale: float = 1.0) -> float:
    """The ink threshold for line art when none is given: Otsu's, but at most :data:`AUTO_THRESHOLD_MAX`.

    Faint strokes darker than :data:`AUTO_THRESHOLD_MAX` are then traced as lines. Otsu's threshold is
    kept when the ink the lower threshold adds is not line-like: when it is more than twice the ink
    there already, or when over 1% of it lies in patches more than 5 px (x ``scale``, the upscaling)
    wide - paper shading or texture rather than strokes.
    """
    otsu = baseline.auto_threshold(ink)
    if otsu <= AUTO_THRESHOLD_MAX:
        return otsu
    strong = ink > otsu
    added = (ink > AUTO_THRESHOLD_MAX) & ~strong
    if added.sum() > 2.0 * strong.sum():
        return otsu
    wide = ndimage.distance_transform_edt(added) >= 2.5 * scale
    if wide.sum() > 0.01 * added.sum():
        return otsu
    return AUTO_THRESHOLD_MAX


def choose_upscale(ink: np.ndarray, upscale, threshold: float | None = None) -> int:
    if upscale == "auto":
        width = baseline.estimate_line_width(ink, threshold)
        return 2 if 0 < width < AUTO_UPSCALE_BELOW else 1
    factor = int(upscale)
    if not 1 <= factor <= 4:
        raise ValueError("upscale must be 1-4 or 'auto'")
    return factor


def trace(
    rgb: np.ndarray,
    lineart_method: str = "none",
    vectorize=None,
    fit_tolerance: float = 1.0,
    threshold: float | None = None,
    refine: bool = True,
    upscale=1,
    shape_tolerance: float = 0.5,
    faint_lines: bool = True,
    ink: np.ndarray | None = None,
    progress: Callable[[str], None] | None = None,
    residual: bool = True,
    outline: bool = True,
    fill: bool = True,
    optimize: bool = False,
    curve_count: int | None = None,
    decisions: str | None = "learned",
) -> tuple[CurveSet, np.ndarray]:
    """Trace an RGB image; returns ``(curves in original pixels, ink map at original size)``.

    ``vectorize(ink, fit_tolerance=...)`` replaces the baseline engine (e.g. the
    neural engine from :func:`line2func.model.infer.load_vectorizer`).
    ``ink`` is a ready ink map of ``rgb`` (for example a line-art preview the user
    has already seen): extraction is skipped, and when the drawing is traced at a
    higher resolution that ink map itself is enlarged, so exactly the previewed
    lines are traced. ``progress`` is described in the module docstring.
    ``curve_count`` asks for that many curves (see the module docstring); it
    replaces ``fit_tolerance`` unless that is finer than :data:`COUNT_TOLERANCE`.
    ``decisions`` picks who decides crossings, gap links, junction pairing and
    corners in the baseline engine: "learned" (default: the learned scorer
    bundled with line2func, :mod:`line2func.decision_model`), "rules" / ``None``
    (the angle rules), or a path to other learned weights. It applies to the
    first tracing pass; the second pass keeps the rules.
    """
    if curve_count is not None and int(curve_count) < 1:
        raise ValueError("curve_count must be at least 1")
    step = progress or (lambda stage: None)
    if ink is None:
        step("lineart")
        ink = lineart.extract(rgb, lineart_method)
        given = False
    else:
        ink = np.asarray(ink, dtype=np.float32)
        if ink.shape != rgb.shape[:2]:
            raise ValueError(f"ink map {ink.shape} does not match the image {rgb.shape[:2]}")
        given = True
    h, w = ink.shape
    factor = choose_upscale(ink, upscale, threshold)
    if factor > 1:
        step("upscale")
        big = resize(rgb, (w * factor, h * factor))
        work_rgb = big
        work_ink = _resize_ink(ink, (w * factor, h * factor)) if given else lineart.extract(big, lineart_method)
    else:
        work_rgb, work_ink = rgb, ink

    # the ink threshold to trace at: the given one, or for line art Otsu's lowered to keep faint strokes whole;
    # line widths and faint-stroke detection are still judged at Otsu's (reference_threshold)
    thr, reference = threshold, None
    if thr is None and lineart_method == "none":
        thr = trace_threshold(work_ink, factor)
        reference = baseline.auto_threshold(work_ink)

    if decisions == "learned":
        from line2func.decision_model import BUNDLED

        if not BUNDLED.is_file():
            warnings.warn("the learned decision scorer is missing from this installation; "
                          "the angle rules decide instead", stacklevel=2)
            decisions = None

    def traced(tolerance: float) -> CurveSet:
        tol = tolerance * factor  # keep the tolerance in original pixels
        step("vectorize")
        if vectorize is None:
            params = baseline.BaselineParams(fit_tolerance=tol, threshold=thr, faint_lines=faint_lines,
                                             very_faint_lines=faint_lines and lineart_method == "none",
                                             reference_threshold=reference, decisions=decisions,
                                             solid_ratio=SOLID_RATIO if outline else None)
            curves = baseline.vectorize(work_ink, params)
        else:
            curves = vectorize(work_ink, fit_tolerance=tol)
        if refine:
            step("refine")
            attributes.refine(curves, work_ink)
        step("measure")
        attributes.measure(curves, work_ink, work_rgb)
        if outline:
            # thick and wedge-shaped strokes become closed outlines (line2func.outline);
            # before the residual pass, so a thick stroke's edges are not re-traced as fragments
            from line2func.outline import outline_thick

            step("outline")
            outline_thick(curves, work_ink)
        if residual:
            # second pass over the ink the first one left uncovered (line2func.residual)
            from line2func.residual import residual_pass

            step("residual")
            extra = residual_pass(curves, work_ink, threshold=thr,
                                  params=baseline.BaselineParams(fit_tolerance=tol, faint_lines=faint_lines,
                                                                 reference_threshold=reference))
            if len(extra):
                if refine:
                    attributes.refine(extra, work_ink)
                attributes.measure(extra, work_ink, work_rgb)
                curves.curves.extend(extra.curves)
        if fill:
            # rings inside the filled areas, so that they look filled in Desmos too (line2func.fill)
            from line2func.fill import add_fill

            step("fill")
            add_fill(curves, spacing=FILL_SPACING * factor, tolerance=FILL_TOLERANCE * factor)
        return curves

    if curve_count is None:
        curves = traced(fit_tolerance)
    else:
        # trace finely, then merge down. COUNT_TOLERANCE leaves fewer pieces to merge; but when the drawing's
        # lines are too short to give curve_count curves at it, start at FINE_TOLERANCE right away, and trace
        # again at FINE_TOLERANCE if the first tracing still gives too few
        start = min(fit_tolerance, COUNT_TOLERANCE)
        if start > FINE_TOLERANCE and curve_count * COUNT_PIECE_LENGTH > baseline.ink_length(ink, thr):
            start = FINE_TOLERANCE
        curves = traced(start)
        if len(curves) < curve_count and start > FINE_TOLERANCE:
            curves = traced(FINE_TOLERANCE)
        step("count")
        report = budget.reduce_to(curves, int(curve_count), spacing=float(factor))
        if report["merged"] or report["dropped"]:
            attributes.measure(curves, work_ink, work_rgb)  # widths and colors of the merged pieces
        report["max_error"] = round(report["max_error"] / factor, 3)  # in original pixels
        curves.meta["curve_count"] = report
    if optimize:
        # render-and-compare refinement, on the GPU if there is one
        try:
            from line2func.optimize import optimize as render_optimize
        except ImportError as e:
            raise ImportError("optimize=True needs PyTorch (pip install -e .[train])") from e

        step("optimize")
        render_optimize(curves, work_ink)
        attributes.measure(curves, work_ink, work_rgb)  # widths and colors where the curves are now
    if factor > 1:
        curves = curves.scaled(1.0 / factor, w, h)
    step("shapes")
    shapes.recognize(curves, shape_tolerance)
    curves.meta.update(upscale=factor, faint_lines=faint_lines, refined=refine, residual=residual, outline=outline,
                       fill=fill, optimized=optimize)
    if thr is not None:
        curves.meta["ink_threshold"] = round(float(thr), 3)  # traced at (meta "threshold" is a given one)
    return curves, ink

"""Thick and wedge-shaped strokes as outlines.

A curve carries a single width, and Desmos draws every function as the same
thin line. Strokes whose width matters - a brush stroke much thicker than the
drawing's lines, or a wedge such as an eyelash that tapers from thick to thin -
therefore look wrong as one centerline. This module measures where the ink's
two edges are all along each curve and, for curves that are clearly thick or
strongly tapered, replaces the centerline by the stroke's closed outline
(tagged ``outline``). Nothing fills that outline - it is drawn as lines, like
everything else (:mod:`line2func.fill`) - except the quality rendering, which
fills it in order to measure it.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from line2func.attributes import _cross_sections, _own_peaks
from line2func.curves import Curve, CurveSet
from line2func.export import OUTLINE_WIDTH
from line2func.fit import fit_polyline
from line2func.geometry import arc_length


def edge_profile(ctrl: np.ndarray, ink: np.ndarray, reach: float, darkness: float, spacing: float = 1.0):
    """``(points, normals, left, right, valid)``: the ink's edges along the curve.

    ``left`` / ``right`` are offsets along the normal (px) where the ink
    cross-section crosses half the drawing's ink darkness, on each side of the
    curve's own ink bump.
    """
    _, pts, normals, offsets, prof = _cross_sections(ctrl, ink, reach, spacing)
    mask, _, valid = _own_peaks(prof, offsets)
    solid = mask & (prof >= 0.5 * darkness)
    has = solid.any(axis=1) & valid
    idx = np.arange(len(offsets))[None, :]
    first = np.where(solid, idx, len(offsets)).min(axis=1)
    last = np.where(solid, idx, -1).max(axis=1)
    left = np.where(has, offsets[np.clip(first, 0, len(offsets) - 1)] - 0.125, np.nan)
    right = np.where(has, offsets[np.clip(last, 0, len(offsets) - 1)] + 0.125, np.nan)
    return pts, normals, left, right, has


def _fill_gaps(v: np.ndarray) -> np.ndarray:
    ok = np.isfinite(v)
    if ok.all() or not ok.any():
        return np.nan_to_num(v)
    i = np.arange(len(v))
    return np.interp(i, i[ok], v[ok])


def outline_thick(
    curves: CurveSet,
    ink: np.ndarray,
    thick: float = 2.5,
    taper: float = 2.5,
    min_length: float = 6.0,
    tolerance: float = 0.5,
) -> int:
    """Replace thick or wedge-shaped curves by their outlines (in place); returns how many.

    A curve qualifies when its median width is at least ``thick`` x the
    drawing's line width, or when its width varies along it by at least
    ``taper`` x (90th / 10th percentile) and reaches ``thick`` x the line
    width somewhere. ``fill_outline`` curves are left alone.
    """
    ink = np.asarray(ink, dtype=np.float32)
    line_w = float(curves.meta.get("line_width") or 2.0)
    thr_mask = ink > 0.3
    darkness = float(np.percentile(ink[thr_mask], 90)) if thr_mask.any() else 1.0
    next_stroke = max((c.stroke for c in curves.curves), default=-1) + 1
    kept: list[Curve] = []
    added: list[Curve] = []
    replaced = 0
    for c in curves.curves:
        if "fill_outline" in c.tags or arc_length(c.ctrl) < min_length:
            kept.append(c)
            continue
        reach = max(3.0, 4.0 * (c.width or line_w))
        pts, normals, left, right, has = edge_profile(c.ctrl, ink, reach, darkness)
        if has.mean() < 0.8:
            kept.append(c)
            continue
        width = right - left
        w = ndimage.median_filter(_fill_gaps(width), size=5, mode="nearest")
        w_med, w_hi, w_lo = float(np.median(w)), float(np.percentile(w, 90)), float(np.percentile(w, 10))
        wedge = w_hi >= thick * line_w and w_hi / max(w_lo, 0.5) >= taper
        if not (w_med >= thick * line_w or wedge):
            kept.append(c)
            continue
        lo = ndimage.gaussian_filter1d(ndimage.median_filter(_fill_gaps(left), 5, mode="nearest"), 1.0, mode="nearest")
        hi = ndimage.gaussian_filter1d(ndimage.median_filter(_fill_gaps(right), 5, mode="nearest"), 1.0, mode="nearest")
        side_a = pts + normals * lo[:, None]
        side_b = pts + normals * hi[:, None]
        loop = np.vstack([side_a, side_b[::-1]])  # around the stroke: one edge out, the other back
        pieces = fit_polyline(loop, tolerance, closed=True)
        if not pieces:
            kept.append(c)
            continue
        for p in pieces:
            added.append(Curve(p, stroke=next_stroke, confidence=c.confidence, tags=tuple(c.tags) + ("outline",),
                               width=OUTLINE_WIDTH, color=c.color))
        next_stroke += 1
        replaced += 1
    curves.curves = kept + added
    return replaced

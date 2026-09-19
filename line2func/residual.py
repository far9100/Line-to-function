"""Second tracing pass over the ink the first pass left uncovered.

The quality report shows which ink no curve covers (``line2func.quality``).
This pass traces exactly that ink: everything within reach of an existing
curve is erased from the ink map, the rest is traced again with the same
threshold, and new curves are kept only if they are long enough, lie on the
uncovered ink and do not retrace an existing curve. They are added as new
strokes tagged ``residual``.

Typical sources of such ink: short branches pruned in dense detail, lines
merged into a neighbour at a junction, and faint strokes the first pass did
not connect.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from line2func import baseline
from line2func.curves import Curve, CurveSet
from line2func.metrics import sample_points
from line2func.render import filled_area, stroke_loops


def covered_mask(curves: CurveSet, ink: np.ndarray, margin: float = 1.5, floor: float = 0.02) -> np.ndarray:
    """Inked pixels within ``margin`` px of some curve's edge (its measured half width)."""
    line_w = float(curves.meta.get("line_width") or 2.0)
    out = np.zeros(ink.shape, dtype=bool)
    per_curve = [sample_points([c], 0.5) for c in curves.curves]
    if not per_curve:
        return out
    pts = np.vstack(per_curve)
    half = np.concatenate([np.full(len(p), 0.5 * (c.width or line_w)) for p, c in zip(per_curve, curves.curves)])
    ys, xs = np.nonzero(ink >= floor)  # paper pixels do not matter
    if not len(ys):
        return out
    d, j = cKDTree(pts).query(np.stack([xs + 0.5, ys + 0.5], axis=1), distance_upper_bound=float(half.max()) + margin)
    hit = np.isfinite(d)
    ok = np.zeros(len(d), dtype=bool)
    ok[hit] = d[hit] - half[j[hit]] <= margin
    out[ys[ok], xs[ok]] = True
    # the inside of filled outlines (thick / wedge strokes, fill areas) is covered too
    if stroke_loops(curves):
        inside = filled_area(curves, ink.shape[1], ink.shape[0]) > 0.0
        out |= ndimage.binary_dilation(inside, iterations=int(np.ceil(margin)))
    return out


def residual_pass(
    curves: CurveSet,
    ink: np.ndarray,
    threshold: float | None = None,
    params: baseline.BaselineParams | None = None,
    min_length: float | None = None,
    min_support: float = 0.7,
    max_overlap: float = 0.3,
    margin: float | None = None,
) -> CurveSet:
    """Trace the ink ``curves`` leave uncovered; returns the accepted new curves.

    The result shares ``curves``' size and meta; its stroke ids continue after
    ``curves``' last one, so it can be appended as is. ``curves`` should carry
    measured widths (``attributes.measure``) so the edges of wide lines count as
    covered. ``threshold`` must be the first pass's ink threshold (the residual
    alone would give a meaningless Otsu value). ``params`` are the first pass's
    settings; their fitting tolerance sets the default ``margin`` (how far
    beyond a curve's edge ink counts as covered, and how close a new curve may
    run along an old one): a curve may lie up to that tolerance off its line,
    and the ink it leaves beside it is not a line of its own.
    """
    ink = np.asarray(ink, dtype=np.float32)
    thr = threshold if threshold is not None else baseline.auto_threshold(ink)
    line_w = float(curves.meta.get("line_width") or 2.0)
    base = params or baseline.BaselineParams()
    if min_length is None:
        min_length = max(6.0, 3.0 * line_w) * base.denoise  # the noise filters' strength scales it too
    if margin is None:
        margin = max(1.5, base.fit_tolerance + 0.5)
    result = CurveSet(curves.width, curves.height, meta=dict(curves.meta))

    covered = covered_mask(curves, ink, margin=margin)
    rest = np.where(covered, 0.0, ink).astype(np.float32)
    if not (rest > thr).any():
        return result

    p = baseline.BaselineParams(**{**base.__dict__, "threshold": thr})
    found = baseline.vectorize(rest, p)
    if not len(found):
        return result

    # evidence tests: on uncovered ink, long enough, not a retrace of an existing curve
    rest_mask = rest > 0.5 * thr
    existing = sample_points(curves, 0.5)
    tree = cKDTree(existing) if len(existing) else None
    h, w = ink.shape
    groups: dict[int, list[Curve]] = {}
    for c in found.curves:
        groups.setdefault(c.stroke, []).append(c)
    next_stroke = max((c.stroke for c in curves.curves), default=-1) + 1
    for pieces in groups.values():
        pts = sample_points(pieces, 0.5)
        if len(pts) < 2:
            continue
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if length < min_length:
            continue
        xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
        if float(np.mean(rest_mask[yi, xi])) < min_support:
            continue
        if tree is not None and float(np.mean(tree.query(pts)[0] <= margin)) > max_overlap:
            continue
        for c in pieces:
            result.curves.append(Curve(c.ctrl, stroke=next_stroke, confidence=c.confidence,
                                       tags=tuple(c.tags) + ("residual",)))
        next_stroke += 1
    return result

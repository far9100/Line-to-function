"""Curves inside filled areas, so that they look filled in Desmos too.

Filled strokes - the outline of a solid area such as a heavy eyelash
(``fill_outline``, :func:`line2func.baseline.solid_areas`) or of a thick
stroke (``outline``, :mod:`line2func.outline`) - are filled by the SVG export
and the viewer. Desmos draws every pasted expression as a line of the same
width and fills none of them, so there they show as hollow outlines.
:func:`add_fill` adds rings inside every filled area, one every ``spacing`` px
from its edge (the contour lines of the distance to the edge), tagged
``fill``. A Desmos line is 2.5 screen pixels wide, so with the whole drawing on
screen (about one image pixel per screen pixel) the rings merge into a solid
area; zoomed in far, they show as rings. The SVG export leaves them out, since
it fills the areas itself.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from line2func import baseline
from line2func.curves import Curve, CurveSet
from line2func.render import filled_area

FILL_TAG = "fill"


def add_fill(curves: CurveSet, spacing: float = 1.5, tolerance: float = 0.5, max_rings: int = 8) -> int:
    """Add rings inside every filled area of ``curves`` (in place); returns how many curves were added.

    ``spacing`` and ``tolerance`` (the rings' fitting tolerance) are in the
    curves' pixels. An area gets at most ``max_rings`` rings: the middle of a
    large solid area stays hollow in Desmos rather than cost hundreds of curves.
    """
    inside = filled_area(curves, curves.width, curves.height) >= 0.5
    if not inside.any():
        return 0
    params = baseline.BaselineParams(fit_tolerance=tolerance)
    stroke = max((c.stroke for c in curves.curves), default=-1) + 1
    labels, _ = ndimage.label(inside, structure=baseline._EIGHT)
    added = []
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        # one area at a time, in its own box (one pixel of margin: the distance to its edge is exact)
        y0, x0 = max(0, box[0].start - 1), max(0, box[1].start - 1)
        area = labels[y0 : box[0].stop + 1, x0 : box[1].stop + 1] == k
        dist = ndimage.distance_transform_edt(area)
        for ring in range(1, max_rings + 1):
            level = dist > ring * spacing
            if not level.any():
                break
            edge = level & ~ndimage.binary_erosion(level, structure=baseline._EIGHT)
            for pts, closed in baseline._strokes_from_skeleton(baseline.thin(edge), 0.5, params, close_gaps=False):
                if not closed and np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < spacing:
                    continue  # a speck where the area is barely deeper than this ring
                ctrls = baseline._fit_stroke(pts + (x0, y0), closed, 0.5, params)
                added += [Curve(c, stroke=stroke, confidence=1.0, tags=(FILL_TAG,)) for c in ctrls]
                stroke += 1
    curves.curves.extend(added)
    return len(added)

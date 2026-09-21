"""Curves inside filled areas, so that they look filled - and as dark as they are - in Desmos too.

Filled strokes - the outline of a solid area such as a heavy eyelash
(``fill_outline``, :func:`line2func.baseline.solid_areas`) or of a thick
stroke (``outline``, :mod:`line2func.outline`) - are filled by the SVG export
and the viewer. Desmos draws every pasted expression as a line of the same
width and fills none of them, so there they show as hollow outlines, and the
only way to show how dark an area is, is how densely it is drawn.

:func:`add_fill` therefore fills every area at its own measured tone
(``Curve.tone``), one of two ways:

* An area as dark as the drawing's own dark ink gets **rings**, one every
  ``spacing`` px from its edge (the contour lines of the distance to the edge).
  A Desmos line is :data:`LINE_PX` screen pixels wide, so with the whole drawing
  on screen the rings merge into a solid area; zoomed in far, they show as rings.
* A lighter area - a shadow, a wash of pencil tone - gets **hatching**: parallel
  45 degree lines whose spacing is set so that the ink they cover matches the
  area's tone. Hatching rather than rings, because a ring at depth *k* x spacing
  only exists where the area is deeper than that: on lineArt (9) of the JPEG set,
  spacing the rings by tone left 23 of 59 areas (12% of the shaded pixels) with
  no ring at all, since a shadow along a jaw or a finger is only a few pixels
  deep. A hatch line crosses an area however thin it is.

The SVG export leaves all of these out, since it fills the areas itself.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from line2func import baseline
from line2func.boundary import contours
from line2func.curves import Curve, CurveSet
from line2func.render import filled_area

FILL_TAG = "fill"
FILL_SPACING = 1.5  # px: the closest the curves inside an area are ever drawn
LINE_PX = 2.5  # px: how wide Desmos draws a line, with the whole drawing on screen
COVER = 0.88  # curves tiled at their own width cover ~92% of an area, not 100%, because they curve:
# they have to overlap by this much for the darkest areas to come out solid (measured on lineArt (11))
MAX_SPACING = 12.0  # px: hatching wider apart than this reads as lines, not as tone


def _area_tones(curves: CurveSet, labels: np.ndarray) -> list[float | None]:
    """The measured tone of each labelled area, from the outline curves that run around it."""
    near = ndimage.grey_dilation(labels, size=3)  # an outline sits on the edge, not inside
    found: list[list[float]] = [[] for _ in range(int(labels.max()) + 1)]
    for c in curves.curves:
        if c.tone is None or not ("fill_outline" in c.tags or "outline" in c.tags):
            continue
        p = np.rint(c.ctrl).astype(int)
        under = near[np.clip(p[:, 1], 0, labels.shape[0] - 1), np.clip(p[:, 0], 0, labels.shape[1] - 1)]
        under = under[under > 0]
        if len(under):
            found[int(np.bincount(under).argmax())].append(c.tone)
    return [float(np.median(v)) if v else None for v in found]


def _hatch(area: np.ndarray, spacing: float, min_run: float) -> list[np.ndarray]:
    """Endpoints of 45 degree lines ``spacing`` px apart across ``area`` (in its own pixels)."""
    ys, xs = np.nonzero(area)
    if not len(xs):
        return []
    # x + y is constant along a 45 degree line and steps by 1 between neighbouring diagonals,
    # which are 1/sqrt(2) px apart: every sqrt(2) x spacing of them is one hatch line
    step = spacing * math.sqrt(2.0)
    diag = xs + ys
    wanted = np.rint(np.arange(float(diag.min()), float(diag.max()) + 1.0, step)).astype(int)
    on = np.isin(diag, wanted)
    if not on.any():
        return []
    dx, dy, dd = xs[on], ys[on], diag[on]
    order = np.lexsort((dx, dd))
    dx, dy, dd = dx[order], dy[order], dd[order]
    # a run is a stretch of neighbouring x on one diagonal
    cut = np.nonzero((np.diff(dd) != 0) | (np.diff(dx) != 1))[0] + 1
    runs = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(dx)]):
        if (dx[b - 1] - dx[a]) * math.sqrt(2.0) < min_run:
            continue
        runs.append(np.array([[dx[a], dy[a]], [dx[b - 1], dy[b - 1]]], dtype=np.float64))
    return runs


def _straight(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """A straight segment as a cubic, so it costs one Desmos expression like any other curve."""
    return np.array([p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1])


def spacing_for(tone: float | None, dark: float, closest: float = FILL_SPACING,
                widest: float = MAX_SPACING, line_px: float = LINE_PX, cover: float = COVER) -> float:
    """How far apart to draw the curves inside an area of darkness ``tone``.

    ``line_px / (tone / dark)``: a curve inks ``line_px`` of every ``spacing``,
    so the share of the area that comes out inked is the share of the drawing's
    dark ink that its own ink is. Continuous in ``tone`` - an area a little
    darker than another is drawn a little denser, with no step anywhere - and
    clipped to [``closest``, ``widest``]. An area with no measured tone is drawn
    at ``closest``, the way filled areas were before tones were measured.

    ``cover`` is why the spacing is a little tighter than that arithmetic says.
    Tiled curves only cover ``line_px`` of every ``spacing`` if they are straight
    and parallel; rings curve, and measured on lineArt (11) they cover 92% of an
    area at a spacing of their own width rather than 100%. Overlapping them by
    ``cover`` is what lets the darkest areas come out solid.

    Measured against the drawing's own dark ink rather than against black,
    because Desmos draws every line at one darkness whatever its ink: an area's
    tone is only readable next to the lines around it.
    """
    if tone is None or dark <= 0.0:
        return closest
    return float(np.clip(cover * line_px * dark / max(tone, 1e-3), closest, widest))


def add_fill(curves: CurveSet, spacing: float = FILL_SPACING, tolerance: float = 0.5, max_rings: int = 0,
             line_px: float = LINE_PX, max_spacing: float = MAX_SPACING, cover: float = COVER) -> int:
    """Draw every filled area of ``curves`` at its own tone (in place); returns how many curves were added.

    The spacing comes from :func:`spacing_for`, the same formula for every area.
    What is drawn at that spacing is chosen by the area's shape, not its tone:
    **rings** (contour lines of the distance to its edge) when the area is deeper
    than one spacing, and **45 degree hatching** when it is not. A ring only
    exists where the area is deeper than the ring's own depth, so on a shadow
    along a jaw or a finger, only a few pixels deep, rings alone leave most areas
    with nothing in them at all; a hatch line crosses an area however thin it is.

    ``max_rings`` caps how many rings one area gets; 0 is as many as it is deep,
    so its middle is drawn rather than left hollow. ``tolerance`` is the rings'
    fitting tolerance, in the curves' pixels.
    """
    inside = filled_area(curves, curves.width, curves.height) >= 0.5
    if not inside.any():
        return 0
    params = baseline.BaselineParams(fit_tolerance=tolerance)
    stroke = max((c.stroke for c in curves.curves), default=-1) + 1
    labels, _ = ndimage.label(inside, structure=baseline._EIGHT)
    dark = float(curves.meta.get("ink_dark") or 0.0)
    tones = _area_tones(curves, labels)
    added = []
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        # one area at a time, in its own box (one pixel of margin: the distance to its edge is exact)
        y0, x0 = max(0, box[0].start - 1), max(0, box[1].start - 1)
        area = labels[y0 : box[0].stop + 1, x0 : box[1].stop + 1] == k
        tone = tones[k] if k < len(tones) else None
        gap = spacing_for(tone, dark, spacing, max_spacing, line_px, cover)
        dist = ndimage.distance_transform_edt(area)
        if dist.max() <= gap:  # too shallow for even one ring: hatch across it instead
            for run in _hatch(area, gap, min_run=max(2.0, spacing)):
                added.append(Curve(_straight(run[0] + (x0, y0), run[1] + (x0, y0)),
                                   stroke=stroke, confidence=1.0, tags=(FILL_TAG,), tone=tone))
                stroke += 1
            continue
        rings = max_rings or int(dist.max() / gap) + 1  # 0: as many as the area is deep
        for ring in range(1, rings + 1):
            level = dist > ring * gap
            if not level.any():
                break
            # contours, for the same reason the area's own outline uses them: a ring is closed,
            # and thinning it and walking it as a skeleton drops and splits it into fragments
            for loop in contours(level):
                ctrls = baseline._fit_stroke(loop + (x0, y0), True, 0.5, params)
                added += [Curve(c, stroke=stroke, confidence=1.0, tags=(FILL_TAG,), tone=tone) for c in ctrls]
                stroke += 1
    curves.curves.extend(added)
    return len(added)

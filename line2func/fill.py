"""Curves inside filled areas, so that they look filled - and as dark as they are - everywhere.

A filled area - a solid one such as a heavy eyelash (``fill_outline``,
:func:`line2func.baseline.solid_areas`) or a thick stroke (``outline``,
:mod:`line2func.outline`) - arrives at Desmos as its closed boundary and
nothing else, because Desmos draws every pasted expression as a line of the
same width and fills none of them. The only way to show how dark an area is,
there, is how densely it is drawn.

:func:`add_fill` therefore draws every area at its own measured tone
(``Curve.tone``), spacing the curves by one formula for every area
(:func:`spacing_for`). *What* is drawn at that spacing is chosen by the area's
shape, not by its tone: **rings** (contour lines of the distance to its edge)
where the area is deeper than one spacing, and **45 degree hatching** where it
is not. A ring at depth *k* x spacing only exists where the area is deeper than
that, so on a shadow along a jaw or a finger, a few pixels deep, rings alone
leave most areas with nothing in them at all - on lineArt (9) of the JPEG set,
23 of 59 areas (12% of the shaded pixels) got no ring. A hatch line crosses an
area however thin it is.

Nothing anywhere fills these areas: the SVG export and the page draw the same
curves Desmos gets, so all three show the same drawing (:mod:`line2func.export`).
The one exception is the quality rendering, which fills an area at its tone in
order to measure it (:func:`line2func.render.fill_share`).
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


# ---------- the tone of a photo ----------
SHADE_TAG = "shade"
SHADE_OCTAVES = 3  # the hatching comes at this many spacings, each twice the one before
SHADE_SPACINGS = (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)  # px: the closest spacing is picked from these
SHADE_WIDTH = 1.0  # px: the width a hatch line is given, thinner than the lines it shades between
SHADE_DARK_MIN = 0.5  # a picture with nothing darker than this is not shaded as if its darkest were black


def shade_runs(tone: np.ndarray, closest: float, dark: float) -> tuple[np.ndarray, np.ndarray]:
    """Hatch segments for a tone map, ``closest`` px apart where it is as dark as ``dark``.

    The spacing a tone asks for is ``closest * dark / tone`` - half as dark, twice as far apart, the
    arithmetic of :func:`spacing_for` - and parallel straight lines cannot change their spacing
    gradually, so it is rounded to an octave: 45 degree diagonals ``closest`` apart, of which every
    other one carries on into areas half as dark, and every fourth into areas a quarter as dark
    (:data:`SHADE_OCTAVES`). Anything lighter than that stays paper. Because the wider sets are
    subsets of the closer ones, a line that crosses from a shadow into a mid-tone simply continues,
    and one segment does for both.

    The tone is smoothed over ``closest`` px first: finer than the hatching it is drawn with, a
    detail could only come out as a scatter of dashes. Returns the endpoints, ``(n, 2, 2)`` as
    ``[[x0, y0], [x1, y1]]``, and the smoothed tone at the middle of each segment.
    """
    t = ndimage.gaussian_filter(tone.astype(np.float32), closest)
    with np.errstate(divide="ignore"):
        need = np.rint(np.log2(max(dark, 1e-6) / np.maximum(t, 1e-6)))
    need = np.maximum(need, 0.0)  # darker than ``dark``: as dense as it gets
    h, w = t.shape
    step = closest * math.sqrt(2.0)  # x + y steps by this between hatch lines ``closest`` apart
    index = np.arange(int((h + w) / step) + 1)
    octave = np.zeros(len(index), dtype=np.intp)
    for k in range(1, SHADE_OCTAVES):
        octave[index % (1 << k) == 0] = k
    line_at = np.full(h + w, -1, dtype=np.intp)  # which hatch line, if any, a diagonal x + y is
    line_at[np.minimum(np.rint(index * step).astype(np.intp), h + w - 1)] = index
    ys, xs = np.mgrid[0:h, 0:w]
    line = line_at[xs + ys]
    on = (line >= 0) & (need < SHADE_OCTAVES) & (need <= octave[np.maximum(line, 0)])
    yy, xx = np.nonzero(on)
    if not len(xx):
        return np.zeros((0, 2, 2)), np.zeros(0)
    dd = xx + yy
    order = np.lexsort((xx, dd))
    xx, yy, dd = xx[order], yy[order], dd[order]
    # a run is a stretch of neighbouring x on one diagonal
    cut = np.nonzero((np.diff(dd) != 0) | (np.diff(xx) != 1))[0] + 1
    a, b = np.r_[0, cut], np.r_[cut, len(xx)] - 1
    keep = (xx[b] - xx[a]) * math.sqrt(2.0) >= max(6.0, 2.0 * closest)  # shorter reads as a speck
    a, b = a[keep], b[keep]
    ends = np.stack([np.stack([xx[a], yy[a]], axis=1), np.stack([xx[b], yy[b]], axis=1)], axis=1).astype(np.float64)
    middle = np.rint(ends.mean(axis=1)).astype(np.intp)
    return ends, t[middle[:, 1], middle[:, 0]]


def photo_tone(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """How dark a picture is at each pixel (0 white, 1 black), and the tone its darkest areas have."""
    tone = 1.0 - (rgb[..., :3].astype(np.float32) / 255.0) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return tone, max(float(np.percentile(tone, 99.0)), SHADE_DARK_MIN)


def shade_curves(rgb: np.ndarray, budget: int, first_stroke: int = 0) -> tuple[list[Curve], float]:
    """The tone of ``rgb`` as hatching of at most ``budget`` curves; returns them and the closest spacing.

    The spacing is the tightest of :data:`SHADE_SPACINGS` that fits the budget, so a picture with
    large dark areas is hatched more openly rather than running out of curves halfway down. Each
    curve is one straight segment tagged :data:`SHADE_TAG`, carrying the tone it stands for and that
    tone as its color, so the SVG and ``desmos.js`` draw it in that gray and it reads lighter than the
    lines it shades between.
    """
    from line2func.export import tone_color

    tone, dark = photo_tone(rgb)
    for closest in SHADE_SPACINGS:
        ends, tones = shade_runs(tone, closest, dark)
        if len(ends) <= budget:
            break
    else:
        keep = np.argsort(-np.hypot(*(ends[:, 1] - ends[:, 0]).T))[:budget]  # the longest ones
        ends, tones = ends[np.sort(keep)], tones[np.sort(keep)]
    curves = [Curve(_straight(p0 + 0.5, p1 + 0.5), stroke=first_stroke + i, confidence=1.0, tags=(SHADE_TAG,),
                    width=SHADE_WIDTH, color=tone_color(float(tn)), tone=round(float(tn), 3))
              for i, ((p0, p1), tn) in enumerate(zip(ends, tones))]
    return curves, closest

"""Solid areas (heavy eyelashes) become filled outlines, and rings inside filled areas make them look filled in
Desmos, where nothing can be filled."""

import numpy as np

from line2func import attributes, geometry as g, pipeline
from line2func.baseline import BaselineParams, vectorize
from line2func.curves import Curve, CurveSet
from line2func.export import to_desmos, to_svg
from line2func.fill import add_fill
from line2func.fit import fit_polyline
from line2func.render import fill_loops, filled_area, rasterize

W, H = 160, 110


def _wedge(x0, x1, y, w0, w1, n=60):
    """Closed polygon of a straight stroke tapering from width ``w0`` to ``w1``."""
    x = np.linspace(x0, x1, n)
    half = np.linspace(w0, w1, n) / 2
    return np.vstack([np.c_[x, y - half], np.c_[x[::-1], (y + half)[::-1]]])


def _square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _loop(poly, tag, stroke):
    return [Curve(p, stroke=stroke, tags=(tag,)) for p in fit_polyline(poly, 0.2, closed=True)]


def _points(curves, n=12):
    return np.vstack([g.evaluate(c.ctrl, np.linspace(0, 1, n)) for c in curves])


def _drawing():
    """Thin lines, a heavy black lash, and two thin lines drawn so close that their ink merges (lighter between)."""
    thin = [g.line([10, 12], [150, 12]), g.line([10, 100], [150, 100]), g.line([12, 20], [12, 95])]
    ink = rasterize(thin, W, H, line_width=3.0)
    ink = np.maximum(ink, fill_loops([_wedge(30, 140, 40, 13.0, 2.0)], W, H))
    band = fill_loops([_square(30, 65, 140, 75)], W, H)  # 10 px of ink ...
    ink = np.maximum(ink, 0.45 * band)  # ... light in the middle ...
    close = [g.line([30, 66.5], [140, 66.5]), g.line([30, 73.5], [140, 73.5])]
    return np.maximum(ink, 0.95 * rasterize(close, W, H, line_width=3.0))  # ... with a dark line on each side


def test_a_heavy_lash_is_a_solid_area_but_close_lines_are_not():
    ink = _drawing()
    cs = vectorize(ink, BaselineParams(threshold=0.3, solid_ratio=2.5))
    solid = [c for c in cs if "fill_outline" in c.tags]
    assert solid
    pts = _points(solid)
    assert np.all((pts[:, 0] > 25) & (pts[:, 0] < 145) & (pts[:, 1] > 30) & (pts[:, 1] < 50))  # the lash only
    lines = [c for c in cs if "fill_outline" not in c.tags]
    assert any(np.abs(_points([c])[:, 1] - 12).max() < 1.5 for c in lines)  # thin lines stay centerlines
    # off (the default of the baseline engine): the lash is traced as a line
    plain = vectorize(ink, BaselineParams(threshold=0.3))
    assert not any("fill_outline" in c.tags for c in plain)


def test_rings_make_a_solid_area_look_filled_in_desmos():
    wedge = _wedge(20, 140, 50, 12.0, 2.0)
    line = Curve(g.line([10, 95], [150, 95]), stroke=1)
    cs = CurveSet(W, H, _loop(wedge, "outline", 0) + [line], meta={"line_width": 2.0})
    n = add_fill(cs, spacing=1.5)
    rings = [c for c in cs if "fill" in c.tags]
    assert n == len(rings) > 0 and all(c.stroke >= 2 for c in rings)
    area = fill_loops([wedge], W, H) > 0.5
    pts = _points(rings)
    assert area[pts[:, 1].astype(int), pts[:, 0].astype(int)].mean() > 0.98  # the rings are inside the area
    # drawn the way Desmos draws them (every curve 2.5 px wide, the whole drawing on screen), the area is solid
    outline = [c for c in cs if c.stroke == 0]
    with_rings = rasterize(outline + rings, W, H, line_width=2.5) > 0.5
    hollow = rasterize(outline, W, H, line_width=2.5) > 0.5
    assert (with_rings & area).sum() / area.sum() > 0.97
    assert (hollow & area).sum() / area.sum() < 0.9
    # every output draws the same curves: Desmos one expression each, the SVG one path per stroke
    assert len(to_desmos(cs).splitlines()) == len(cs)
    assert to_svg(cs).count("<path") == cs.num_strokes
    assert "fill=" not in to_svg(cs).replace('fill="none"', "")
    attributes.measure(cs, fill_loops([wedge], W, H))
    assert all(c.width is None for c in rings)


def _dense(x0, y0, x1, y1, step=2.0):
    """A rectangle with points along its edges, so fitting it back gives the rectangle, not a blob."""
    def edge(a, b, n):
        return np.c_[np.linspace(a[0], b[0], n, endpoint=False), np.linspace(a[1], b[1], n, endpoint=False)]
    nx, ny = max(2, int((x1 - x0) / step)), max(2, int((y1 - y0) / step))
    return np.vstack([edge((x0, y0), (x1, y0), nx), edge((x1, y0), (x1, y1), ny),
                      edge((x1, y1), (x0, y1), nx), edge((x0, y1), (x0, y0), ny)])


def test_a_shadow_is_hatched_by_its_tone_and_a_solid_area_still_gets_rings():
    """In Desmos every line is equally dark, so a lighter area must be drawn less densely."""
    # thin enough that the 8 rings of a solid area reach its middle (see add_fill's max_rings)
    box = _dense(20, 30, 130, 52)
    area = fill_loops([box], W, H) > 0.5

    def inked(tone):
        cs = CurveSet(W, H, _loop(box, "fill_outline", 0), meta={"line_width": 2.0, "ink_dark": 1.0})
        for c in cs.curves:
            c.tone = tone
        assert add_fill(cs, spacing=1.5) > 0
        props = [c for c in cs if "fill" in c.tags]
        assert all(c.tone == tone for c in props)  # each prop knows which area it fills
        pts = _points(props)
        assert area[pts[:, 1].astype(int), pts[:, 0].astype(int)].mean() > 0.9  # inside the area
        # drawn the way Desmos draws it: every curve 2.5 px wide, the whole drawing on screen
        drawn = rasterize(props, W, H, line_width=2.5) > 0.5
        return float((drawn & area).sum() / area.sum())

    solid, half, light = inked(1.0), inked(0.5), inked(0.25)
    assert solid > 0.9  # as dark as the drawing's dark ink: rings, and the area reads solid
    assert 0.15 < light < 0.45  # a quarter as dark: hatched about a quarter as densely
    assert light < half < solid  # darker areas are drawn denser


def test_a_shadow_is_drawn_not_filled_in_every_line_color():
    """A shadow's outline is stroked in the gray measured inside it, and nothing is ever filled."""
    box = _dense(20, 20, 130, 85)
    cs = CurveSet(W, H, _loop(box, "fill_outline", 0), meta={"line_width": 2.0, "ink_dark": 1.0})
    for c in cs.curves:
        c.tone, c.color = 0.25, "#bfbfbf"
    for mode in ("measured", "bw", "palette", "random"):
        svg = to_svg(cs, color_mode=mode)
        assert "fill=" not in svg.replace('fill="none"', "") and 'stroke="none"' not in svg
    assert 'stroke="#bfbfbf"' in to_svg(cs, color_mode="measured")
    assert 'stroke="#bfbfbf"' not in to_svg(cs, color_mode="palette")  # a chosen color still wins


def test_no_rings_without_filled_areas():
    cs = CurveSet(W, H, [Curve(g.line([10, 20], [150, 20]))], meta={"line_width": 2.0})
    assert add_fill(cs) == 0 and len(cs) == 1


def test_filled_area_joins_overlapping_outlines_and_keeps_holes():
    a, b = _square(10, 10, 60, 60), _square(40, 40, 90, 90)
    cs = CurveSet(W, H, _loop(a, "outline", 0) + _loop(b, "outline", 1)
                  + _loop(_square(100, 10, 150, 60), "fill_outline", 2)
                  + _loop(_square(115, 25, 135, 45), "fill_outline", 3))
    cov = filled_area(cs, W, H)
    assert cov[50, 50] == 1.0  # two thick strokes overlap: still covered
    assert cov[20, 105] == 1.0 and cov[35, 125] == 0.0  # a fill area keeps its hole


def test_pipeline_fills_a_heavy_lash_for_desmos():
    rgb = np.repeat((255 * (1.0 - _drawing())).astype(np.uint8)[:, :, None], 3, axis=2)
    cs, _ = pipeline.trace(rgb, upscale=1, threshold=0.3)
    assert any("fill_outline" in c.tags for c in cs) and any("fill" in c.tags for c in cs)
    assert cs.meta["fill"] is True
    hollow, _ = pipeline.trace(rgb, upscale=1, threshold=0.3, fill=False)
    assert any("fill_outline" in c.tags for c in hollow) and not any("fill" in c.tags for c in hollow)
    lines, _ = pipeline.trace(rgb, upscale=1, threshold=0.3, outline=False, fill=False)
    assert not any("fill_outline" in c.tags or "outline" in c.tags for c in lines)

import xml.etree.ElementTree as ET

import numpy as np

from line2func import attributes, geometry as g, pipeline
from line2func.curves import Curve, CurveSet
from line2func.export import to_svg
from line2func.outline import outline_thick
from line2func.render import fill_loops, rasterize, render_coverage, render_lineart, stroke_loops

W, H = 160, 100


def _wedge(x0, x1, y, w0, w1, n=60):
    """Closed polygon of a straight stroke tapering from width ``w0`` to ``w1``."""
    x = np.linspace(x0, x1, n)
    half = np.linspace(w0, w1, n) / 2
    return np.vstack([np.c_[x, y - half], np.c_[x[::-1], (y + half)[::-1]]])


def _iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_fill_loops_is_even_odd():
    outer = np.array([[10, 10], [90, 10], [90, 70], [10, 70]], float)
    inner = np.array([[30, 30], [70, 30], [70, 50], [30, 50]], float)
    solid = fill_loops([outer], 100, 80)
    ring = fill_loops([outer, inner], 100, 80)
    assert abs(solid.sum() - 80 * 60) < 0.02 * 80 * 60
    assert abs(ring.sum() - (80 * 60 - 40 * 20)) < 0.02 * 80 * 60
    assert ring[40, 50] == 0 and ring[20, 50] == 1  # a hole inside, ink in the ring


def test_wedge_centerline_becomes_its_outline():
    wedge = _wedge(20, 130, 50, 9.0, 1.0)
    thin = g.line([10, 20], [150, 20])
    ink = np.maximum(fill_loops([wedge], W, H), rasterize([thin], W, H, line_width=2.0))
    cs = CurveSet(W, H, [Curve(g.line([20, 50], [130, 50]), stroke=0, color="#101010"),
                         Curve(thin, stroke=1, width=2.0)], meta={"line_width": 2.0})
    assert outline_thick(cs, ink) == 1
    outline = [c for c in cs if "outline" in c.tags]
    assert outline and {c.stroke for c in outline} == {2}  # a new stroke id
    assert all(c.color == "#101010" for c in outline)  # the replaced centerline's color
    assert [c.stroke for c in cs if "outline" not in c.tags] == [1]  # the thin line stays a centerline
    filled = fill_loops(stroke_loops(cs), W, H) > 0.5
    assert _iou(filled, fill_loops([wedge], W, H) > 0.5) > 0.85


def test_thick_stroke_becomes_outline_but_thin_lines_stay():
    lines = [g.line([10, 20], [150, 20]), g.line([20, 60], [140, 60])]
    ink = rasterize(lines, W, H, line_width=np.array([2.0, 8.0]))
    cs = CurveSet(W, H, [Curve(p, stroke=i) for i, p in enumerate(lines)], meta={"line_width": 2.0})
    assert outline_thick(cs, ink) == 1
    assert [c.stroke for c in cs if "outline" not in c.tags] == [0]
    covered = render_coverage(cs, W, H, line_width=2.0) > 0.5
    assert _iou(covered, ink > 0.5) > 0.85

    thin_only = rasterize(lines[:1], W, H, line_width=2.0)
    cs = CurveSet(W, H, [Curve(lines[0])], meta={"line_width": 2.0})
    assert outline_thick(cs, thin_only) == 0 and len(cs) == 1


def test_measure_leaves_outlines_alone():
    wedge = _wedge(20, 130, 50, 9.0, 1.0)
    ink = fill_loops([wedge], W, H)
    cs = CurveSet(W, H, [Curve(g.line([20, 50], [130, 50]), color="#101010")], meta={"line_width": 2.0})
    outline_thick(cs, ink)
    before = [(c.width, c.color) for c in cs]
    rgb = np.repeat(np.rint(255 * (1 - ink)).astype(np.uint8)[:, :, None], 3, axis=2)
    attributes.measure(cs, ink, rgb)
    assert [(c.width, c.color) for c in cs] == before


def test_svg_closes_outline_strokes_but_never_fills_them():
    loop = _wedge(20, 130, 50, 9.0, 1.0, n=4)
    pieces = [g.line(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
    cs = CurveSet(W, H, [Curve(p, stroke=0, tags=("outline",), width=1.0, color="#202020") for p in pieces]
                  + [Curve(g.line([10, 20], [150, 20]), stroke=1)])
    svg = to_svg(cs)
    root = ET.fromstring(svg.split("\n", 1)[1])
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    assert paths[0].get("d").endswith("Z")  # closed, so the curves inside it land in a shape
    assert paths[0].get("stroke") == "#202020"
    assert "fill=" not in svg.replace('fill="none"', "")  # only the group's own fill="none"


def test_pipeline_outline_stage():
    lines = [g.line([10, 20], [150, 20]), g.line([20, 60], [140, 60])]
    img = render_lineart(lines, W, H, line_width=2.0)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    seen = []
    cs, _ = pipeline.trace(rgb, upscale=1, outline=True, progress=seen.append)
    assert "outline" in seen and cs.meta["outline"] is True
    assert not any("outline" in c.tags for c in cs)  # nothing thick here

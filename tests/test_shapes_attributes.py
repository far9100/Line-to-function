import re

import numpy as np
import pytest

from line2func import attributes, geometry as g, shapes
from line2func.curves import Curve, CurveSet
from line2func.export import to_desmos, to_latex, to_svg
from line2func.render import rasterize

K = 0.5522847498


def quarter_arc(cx, cy, r, a0=0.0):
    a1 = a0 + np.pi / 2
    p0 = np.array([cx + r * np.cos(a0), cy + r * np.sin(a0)])
    p3 = np.array([cx + r * np.cos(a1), cy + r * np.sin(a1)])
    p1 = p0 + K * r * np.array([-np.sin(a0), np.cos(a0)])
    p2 = p3 - K * r * np.array([-np.sin(a1), np.cos(a1)])
    return np.array([p0, p1, p2, p3])


def test_classify_line_arc_and_free_curve():
    line = shapes.classify(g.line([10, 90], [90, 50]))
    assert line["type"] == "line"
    arc = shapes.classify(quarter_arc(50, 50, 40))
    assert arc["type"] == "arc"
    np.testing.assert_allclose(arc["center"], [50, 50], atol=0.05)
    assert arc["radius"] == pytest.approx(40, abs=0.05) and arc["sweep_deg"] == pytest.approx(90, abs=0.5)
    wiggle = np.array([[0, 0], [30, 40], [60, -40], [90, 0]], float)
    assert shapes.classify(wiggle) is None


def test_named_line_equation():
    shape = shapes.classify(g.line([10, 90], [90, 50]))  # math coords: (10, 10) -> (90, 50)
    assert shapes.named_latex(shape, 100) == r"y=0.5000x+5.00\left\{10.00\le x\le 90.00\right\}"
    steep = shapes.classify(g.line([20, 90], [30, 10]))  # math: (20, 10) -> (30, 90), steep
    eq = shapes.named_latex(steep, 100)
    assert eq.startswith("x=0.1250y+18.75") and r"10.00\le y\le 90.00" in eq


def test_named_arc_equation_selects_the_right_side():
    shape = shapes.classify(quarter_arc(50, 50, 40))  # image quarter from (90, 50) to (50, 90)
    eq = shapes.named_latex(shape, 100)
    assert eq.startswith(r"\left(x-50.00\right)^{2}+\left(y-50.00\right)^{2}=40.00^{2}")
    a, b, c = (float(v) for v in re.search(r"\\left\\\{(\S+?)x(\S+?)y(\S+?)\\ge 0", eq).groups())
    side = lambda x, y: a * x + b * y + c  # noqa: E731
    # arc midpoint (image (78.3, 78.3) -> math (78.3, 21.7)) is inside, the opposite point is not
    m = 50 + 40 / np.sqrt(2)
    assert side(m, 100 - m) > 0
    assert side(100 - m, m) < 0


def test_named_exports_and_rounding_error():
    cs = CurveSet(200, 200, [Curve(g.line([3, 190], [197, 11])), Curve(quarter_arc(100, 100, 60)),
                             Curve(np.array([[0, 0], [30, 40], [60, -40], [90, 0]], float) + 50)])
    shapes.recognize(cs)
    text = to_desmos(cs, named=True).splitlines()
    assert text[0].startswith("y=") and text[1].startswith(r"\left(x-") and text[2].startswith(r"\left(")
    assert "e" not in re.sub(r"\\left|\\right|\\le|\\ge", "", text[0])  # no scientific notation
    # the rounded slope/intercept reproduce the line within ~0.02 px
    m, c = (float(v) for v in re.match(r"y=(\S+?)x(\S+?)\\left", text[0]).groups())
    for x, y_img in ((3, 190), (197, 11)):
        assert abs(m * x + c - (200 - y_img)) < 0.02
    latex = to_latex(cs, named=True)
    assert r"\begin{align*}" in latex and "y=" in latex


def _scene(y=20.0, width=3.0, color=(200, 30, 30)):
    ctrl = g.line([10, y], [110, y])
    cov = rasterize([ctrl], 120, 40, line_width=width)
    rgb = np.full((40, 120, 3), 255.0)
    rgb = rgb * (1 - cov[:, :, None]) + np.array(color, float) * cov[:, :, None]
    return cov, rgb.astype(np.uint8)


def test_refine_snaps_to_the_ink_centerline():
    ink, _ = _scene(y=20.0)
    cs = CurveSet(120, 40, [Curve(g.line([10, 21.3], [110, 19.1]))], meta={"line_width": 3.0})
    attributes.refine(cs, ink)
    pts = g.evaluate(cs.curves[0].ctrl, np.linspace(0.1, 0.9, 20))
    assert np.abs(pts[:, 1] - 20.0).max() < 0.1


def test_refine_keeps_stroke_joints_together():
    ink, _ = _scene(y=20.0)
    a, b = g.line([10, 21], [60, 21]), g.line([60, 21], [110, 21])
    cs = CurveSet(120, 40, [Curve(a, 0), Curve(b, 0)], meta={"line_width": 3.0})
    attributes.refine(cs, ink)
    np.testing.assert_allclose(cs.curves[0].ctrl[3], cs.curves[1].ctrl[0])
    assert abs(cs.curves[0].ctrl[3][1] - 20.0) < 0.1


@pytest.mark.parametrize("width", [1.5, 3.0, 6.0])
def test_measure_width_and_color(width):
    # the line under test at y = 20 (on a pixel boundary: the hard case for thin
    # lines) plus a 4 px reference line from which the ink darkness is learned
    test_line, ref_line = g.line([10, 20], [110, 20]), g.line([10, 50], [110, 50])
    cov = rasterize([test_line, ref_line], 120, 70, line_width=[width, 4.0])
    rgb = (255.0 * (1 - cov[:, :, None]) + np.array([200.0, 30.0, 30.0]) * cov[:, :, None]).astype(np.uint8)
    cs = CurveSet(120, 70, [Curve(test_line, 0), Curve(ref_line, 1)], meta={"line_width": 3.0})
    attributes.measure(cs, cov, rgb)
    assert cs.curves[0].width == pytest.approx(width, abs=0.15)
    assert cs.curves[1].width == pytest.approx(4.0, abs=0.15)
    assert cs.curves[1].color == "#c81e1e"
    widths = [float(w) for w in re.findall(r'<path id="stroke-\d+" stroke-width="([\d.]+)"', to_svg(cs))]
    assert widths == pytest.approx([cs.curves[0].width, cs.curves[1].width], abs=0.006)


def test_measure_gray_ink():
    cov = rasterize([g.line([10, 20.3], [110, 20.3]), g.line([10, 50], [110, 50])], 120, 70, line_width=[2.5, 5.0])
    ink = 0.55 * cov  # pencil-gray ink
    cs = CurveSet(120, 70, [Curve(g.line([10, 20.3], [110, 20.3])), Curve(g.line([10, 50], [110, 50]), 1)],
                  meta={"line_width": 3.0})
    attributes.measure(cs, ink)
    assert cs.curves[0].width == pytest.approx(2.5, abs=0.2)
    assert cs.curves[1].width == pytest.approx(5.0, abs=0.2)


def test_measured_fields_round_trip(tmp_path):
    cs = CurveSet(10, 10, [Curve(g.line([1, 1], [9, 9]), width=2.5, color="#102030",
                                 shape={"type": "line", "p0": [1, 1], "p1": [9, 9]})])
    cs.save_json(tmp_path / "c.json")
    back = CurveSet.load_json(tmp_path / "c.json").curves[0]
    assert (back.width, back.color, back.shape["type"]) == (2.5, "#102030", "line")
    with pytest.raises(ValueError):
        Curve(g.line([0, 0], [1, 1]), color="red")

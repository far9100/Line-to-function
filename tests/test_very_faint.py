"""Very faint lines (fainter than the faint-stroke level, just above the paper's own noise): traced when they
are clearly lines, never for pencil texture or paper noise; and the quality tool counts them as ink."""

import numpy as np

from line2func import geometry as g, quality
from line2func.baseline import BaselineParams, paper_contrast_ceiling, vectorize
from line2func.curves import Curve, CurveSet
from line2func.metrics import f_score
from line2func.render import rasterize

W, H = 220, 170
K = 0.5522847498


def _circle(cx, cy, r):
    out = []
    for a in range(4):
        t0, t1 = a * np.pi / 2, (a + 1) * np.pi / 2
        p0 = np.array([cx + r * np.cos(t0), cy + r * np.sin(t0)])
        p3 = np.array([cx + r * np.cos(t1), cy + r * np.sin(t1)])
        out.append(np.array([p0, p0 + K * r * np.array([-np.sin(t0), np.cos(t0)]),
                             p3 - K * r * np.array([-np.sin(t1), np.cos(t1)]), p3]))
    return out


def _scene(seed=0):
    """Dark lines, a very faint circle (ink 0.07), a patch of equally faint pencil texture, paper noise."""
    rng = np.random.default_rng(seed)
    dark = [g.line([10, 12], [210, 12]), g.line([10, 158], [210, 158]), g.line([12, 20], [12, 150])]
    ink = rasterize(dark, W, H, line_width=3.0)
    circle = _circle(60, 85, 32)
    ink = np.maximum(ink, 0.07 * rasterize(circle, W, H, line_width=3.0))
    marks = []
    for _ in range(90):  # short criss-crossed strokes, 4-9 px, in a 50 x 50 patch
        p = rng.uniform([140, 60], [190, 110])
        a = rng.uniform(0, np.pi)
        marks.append(g.line(p, p + rng.uniform(4, 9) * np.array([np.cos(a), np.sin(a)])))
    ink = np.maximum(ink, 0.07 * rasterize(marks, W, H, line_width=2.0))
    ink = np.clip(ink + rng.normal(0.0, 0.004, ink.shape), 0, 1).astype(np.float32)
    return ink, circle


def _length_inside(cs, box):
    x0, y0, x1, y1 = box
    total = 0.0
    for c in cs.curves:
        p = g.flatten(c.ctrl, 0.3)
        inside = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total += float(seg[inside[1:] & inside[:-1]].sum())
    return total


def test_a_very_faint_outline_is_traced_but_texture_and_noise_are_not():
    ink, circle = _scene()
    # at threshold 0.35 the faint-stroke level is 0.105: the circle (contrast about 0.07) is below it
    on = vectorize(ink, BaselineParams(threshold=0.35, very_faint_lines=True))
    off = vectorize(ink, BaselineParams(threshold=0.35))
    assert f_score(on, circle, 2.0)["recall"] > 0.85  # the whole faint circle
    assert f_score(off, circle, 2.0)["recall"] < 0.3  # without: at most a few pieces of it
    assert _length_inside(on, (136, 56, 194, 114)) < 10.0  # the equally faint texture stays out
    assert _length_inside(on, (100, 20, 135, 150)) == 0.0  # and so does the blank paper


def test_paper_contrast_ceiling_comes_from_blank_paper():
    rng = np.random.default_rng(3)

    def paper(sigma):
        return np.clip(rng.normal(0.0, sigma, (120, 120)), 0, 1).astype(np.float32)

    quiet, grainy, rough = paper(0.002), paper(0.02), paper(0.08)
    assert paper_contrast_ceiling(quiet, quiet, 2.0) == 0.03  # never below 0.03
    assert paper_contrast_ceiling(grainy, grainy, 2.0) > 0.03  # grainier paper, higher ceiling
    assert paper_contrast_ceiling(rough, rough, 2.0) is None  # no blank paper left to measure it on


def test_quality_counts_curves_on_very_faint_lines_as_on_ink():
    ink, circle = _scene()
    cs = CurveSet(W, H, [Curve(c, stroke=0) for c in circle], meta={"line_width": 3.0, "faint_lines": True})
    report, _ = quality.assess(cs, ink)
    assert report["precision"]["including_faint"]["within_2px"] > 0.95
    assert not report["flags"]["stray_curves"] and report["paper_contrast_ceiling"] is not None
    stray = CurveSet(W, H, [Curve(g.line([110, 40], [110, 140]))], meta={"line_width": 3.0, "faint_lines": True})
    assert quality.assess(stray, ink)[0]["flags"]["stray_curves"]  # a curve on blank paper still is stray

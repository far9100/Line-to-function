import numpy as np
import pytest

from line2func import geometry as g, pipeline
from line2func.budget import reduce_to
from line2func.curves import Curve, CurveSet
from line2func.fit import fit_cubic, fit_polyline
from line2func.metrics import chamfer
from line2func.render import render_lineart

ARC = np.array([[10, 80], [40, 10], [110, 10], [140, 80]], float)


def _split(ctrl, k):
    """``ctrl`` cut into ``k`` exact pieces (same drawing, more curves)."""
    pieces, rest = [], ctrl
    for i in range(k - 1):
        a, rest = g.split(rest, 1.0 / (k - i))
        pieces.append(a)
    return pieces + [rest]


def test_fit_cubic_recovers_a_cubic_and_keeps_tangents():
    t = np.linspace(0, 1, 60)
    pts = g.evaluate(ARC, t)
    ctrl, err, _ = fit_cubic(pts)  # chord-length start: converges within its 40 steps
    assert err < 0.05 and np.allclose(ctrl[[0, 3]], ARC[[0, 3]])
    ctrl, err, _ = fit_cubic(pts, u=t, iterations=0)  # the right parameters: exact at once
    assert err < 1e-9 and np.allclose(ctrl, ARC)
    t1 = (ARC[1] - ARC[0]) / np.linalg.norm(ARC[1] - ARC[0])
    t2 = (ARC[2] - ARC[3]) / np.linalg.norm(ARC[2] - ARC[3])
    ctrl, err, _ = fit_cubic(pts, t1, t2)
    assert err < 0.05
    for end, handle, t in ((0, 1, t1), (3, 2, t2)):
        d = ctrl[handle] - ctrl[end]
        assert np.allclose(d / np.linalg.norm(d), t)  # the handle stays on the given tangent


def test_split_pieces_merge_back_exactly():
    cs = CurveSet(150, 90, [Curve(p, stroke=3, width=2.0, color="#123456") for p in _split(ARC, 6)])
    report = reduce_to(cs, 1)
    assert len(cs) == 1 and report == {"target": 1, "before": 6, "merged": 5, "dropped": 0,
                                        "max_error": report["max_error"], "after": 1}
    assert report["max_error"] < 1e-6  # exact splits are recognized and merged back exactly
    c = cs.curves[0]
    assert np.allclose(c.ctrl[[0, 3]], ARC[[0, 3]]) and c.stroke == 3 and c.width == 2.0 and c.color == "#123456"
    assert chamfer(cs, [ARC]) < 0.05


def test_count_is_exact_and_corners_merge_last():
    # an L (sharp corner) and a smooth arc, each cut into several pieces
    arm1, arm2 = g.line([10, 10], [10, 100]), g.line([10, 100], [100, 100])
    pieces = [Curve(p, stroke=0) for p in _split(arm1, 3) + _split(arm2, 3)]
    pieces += [Curve(p, stroke=1) for p in _split(ARC + [0, 120], 4)]
    cs = CurveSet(160, 220, pieces)
    reduce_to(cs, 3)
    assert len(cs) == 3
    corner = [c for c in cs if c.stroke == 0]
    assert len(corner) == 2  # the arms merged; the corner between them did not
    assert np.allclose(corner[0].ctrl[3], [10, 100]) and np.allclose(corner[1].ctrl[0], [10, 100])
    assert chamfer(corner, [arm1, arm2]) < 0.05


def test_joints_stay_attached_to_unmerged_neighbours():
    wave = fit_polyline(np.c_[np.linspace(0, 200, 400), 50 + 30 * np.sin(np.linspace(0, 6 * np.pi, 400))], 0.2)
    cs = CurveSet(210, 100, [Curve(p) for p in wave])
    n = len(cs) // 2
    reduce_to(cs, n)
    assert len(cs) == n
    for a, b in zip(cs.curves[:-1], cs.curves[1:]):
        assert np.allclose(a.ctrl[3], b.ctrl[0])  # still one connected stroke


def test_below_one_piece_per_run_the_shortest_are_dropped():
    lines = [g.line([10, 10 * i], [10 + 20 * i, 10 * i]) for i in range(1, 6)]  # lengths 20..100
    cs = CurveSet(120, 60, [Curve(p, stroke=i) for i, p in enumerate(lines)])
    report = reduce_to(cs, 2)
    assert len(cs) == 2 and report["dropped"] == 3
    assert [c.stroke for c in cs] == [3, 4]  # the two longest, in their original order


def test_nothing_happens_at_or_above_the_count():
    cs = CurveSet(150, 90, [Curve(p) for p in _split(ARC, 3)])
    before = [c.ctrl.copy() for c in cs]
    assert reduce_to(cs, 3)["after"] == 3 and reduce_to(cs, 10)["after"] == 3
    assert all(np.array_equal(a, c.ctrl) for a, c in zip(before, cs))
    with pytest.raises(ValueError):
        reduce_to(cs, 0)


def test_pipeline_curve_count():
    lines = [ARC, g.line([10, 20], [140, 20]), np.array([[20, 60], [60, 90], [100, 30], [140, 60]], float)]
    img = render_lineart(lines, 150, 90, line_width=2.0)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    coarse, _ = pipeline.trace(rgb, upscale=1, fit_tolerance=pipeline.COUNT_TOLERANCE)
    fine, _ = pipeline.trace(rgb, upscale=1, fit_tolerance=pipeline.FINE_TOLERANCE)
    assert len(fine) > len(coarse)
    seen = []
    cs, _ = pipeline.trace(rgb, upscale=1, curve_count=len(coarse) - 2, progress=seen.append)
    assert len(cs) == len(coarse) - 2 and "count" in seen and seen.count("vectorize") == 1
    assert cs.meta["curve_count"]["merged"] == 2
    # more than the first tracing gives: traced again, more finely. The trigger has to come from
    # the coarse tracing, which is what the first pass makes, not from the fine one
    seen = []
    more, _ = pipeline.trace(rgb, upscale=1, curve_count=len(coarse) + 1, progress=seen.append)
    assert len(more) == len(coarse) + 1 and seen.count("vectorize") == 2
    seen = []
    many, _ = pipeline.trace(rgb, upscale=1, curve_count=10_000, progress=seen.append)
    assert len(many) == len(fine) and many.meta["curve_count"]["after"] < 10_000
    assert seen.count("vectorize") == 1  # far more than the lines can give: traced finely right away
    with pytest.raises(ValueError):
        pipeline.trace(rgb, curve_count=0)

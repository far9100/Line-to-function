"""Function mode: every curve as pieces of y = f(x) / x = g(y) within the tolerance (line2func.functions)."""

import re

import numpy as np
import pytest
from scipy.spatial import cKDTree

from line2func import functions
from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.functions import FUNCTION_TOLERANCE, attach, curve_functions, function_pieces

H = 200
EQ = re.compile(r"(?P<v>[xy])=(?P<body>.+)\\left\\\{(?P<lo>-?\d+\.\d+)\\le (?P<u>[xy])\\le (?P<hi>-?\d+\.\d+)\\right\\\}")


def _parse(eq: str) -> tuple[str, float, float, callable]:
    """``(independent variable, lo, hi, f)`` read back from a printed equation, the way Desmos reads it."""
    m = EQ.fullmatch(eq)
    assert m, eq
    assert m["u"] != m["v"]
    expr = m["body"].replace(r"\left(", "(").replace(r"\right)", ")")
    expr = re.sub(r"\^\{(\d)\}", r"**\1", expr)
    expr = re.sub(r"(\d)(?=[(xy])", r"\1*", expr)  # implicit multiplication after a number
    assert re.fullmatch(r"[0-9.+\-*()xy]+", expr), expr
    return m["u"], float(m["lo"]), float(m["hi"]), lambda u: eval(expr, {"__builtins__": {}}, {m["u"]: u})


def _graph(eq: str, spacing: float = 0.01) -> np.ndarray:
    """Points along the printed function, in image pixels (y down)."""
    u_name, lo, hi, f = _parse(eq)
    u = np.linspace(lo, hi, int(np.ceil((hi - lo) / spacing)) + 1)
    v = np.broadcast_to(f(u), u.shape)
    x, y = (u, v) if u_name == "x" else (v, u)
    return np.stack([x, H - y], axis=1)


def _curve_points(ctrl, spacing: float = 0.01) -> np.ndarray:
    n = int(np.ceil(g.arc_length(ctrl) / spacing)) + 1
    return g.evaluate(ctrl, np.linspace(0.0, 1.0, max(n, 2)))


@pytest.mark.parametrize("ctrl, axis", [
    (g.line([10, 100], [190, 100]), "x"),   # horizontal: y = f(x)
    (g.line([100, 10], [100, 190]), "y"),   # vertical: x = g(y)
    (g.line([10, 190], [190, 10]), None),   # exactly 45 degrees, both ways: one piece, no slivers
    (g.line([10, 10], [190, 190]), None),
    (g.line([10, 20], [190, 30]), "x"),
])
def test_straight_lines_are_one_linear_piece(ctrl, axis):
    [piece] = function_pieces(ctrl, H)
    assert piece["degree"] == 1 and piece["max_error"] < 0.01
    if axis is not None:
        assert piece["axis"] == axis
    [eq] = curve_functions(ctrl, H)[0]
    assert re.fullmatch(r"[xy]=-?\d+\.\d+[xy][+-]\d+\.\d+\\left\\\{.*\\right\\\}", eq), eq  # y = m x + c


@pytest.mark.parametrize("name, ctrl, runs", [
    ("C", [[10, 10], [190, 10], [190, 190], [10, 190]], 3),       # turns back: flat, steep, flat
    ("S", [[10, 100], [100, -50], [100, 250], [190, 100]], None),
    ("loop", [[10, 190], [240, 0], [-40, 0], [190, 190]], None),
    ("cusp", [[10, 100], [190, 100], [190, 100], [10, 100]], 2),  # out and back along one line
])
def test_every_piece_is_a_function(name, ctrl, runs):
    p = g.flip_y(np.array(ctrl, float), H)
    intervals = functions._intervals(p)
    for t0, t1, axis in intervals:
        u = g.evaluate(p, np.linspace(t0, t1, 400))[:, axis]
        steps = np.diff(u)
        assert np.all(steps >= -1e-9) or np.all(steps <= 1e-9), (name, t0, t1)  # monotonic: a function exists
    if runs is not None:
        assert len(intervals) == runs
    assert len(intervals) <= len(function_pieces(ctrl, H)) <= 12  # halved where one polynomial does not fit


def test_a_cusp_is_not_merged_into_one_function():
    """Out and back along the same line: two pieces over the same domain, not one that doubles back."""
    eqs = curve_functions([[10, 100], [190, 100], [190, 100], [10, 100]], H)[0]
    assert len(eqs) == 2 and eqs[0] == eqs[1]


def test_functions_stay_within_the_tolerance_of_random_curves():
    rng = np.random.default_rng(3)
    worst_near, worst_cover = 0.0, 0.0
    for _ in range(200):
        ctrl = rng.uniform(0, H, (4, 2))
        eqs, error = curve_functions(ctrl, H)
        assert error <= FUNCTION_TOLERANCE
        graphs = [_graph(eq) for eq in eqs]
        curve = _curve_points(ctrl)
        # every printed function lies near the curve (dense samples: the distances only come out larger) ...
        near = cKDTree(curve).query(np.concatenate(graphs))[0].max()
        # ... and covers all of it: no gaps between pieces or curves
        cover = cKDTree(np.concatenate(graphs)).query(curve)[0].max()
        worst_near, worst_cover = max(worst_near, near), max(worst_cover, cover)
        # the ends of neighbouring pieces meet (up to rounding, domains rounded outward and left-out slivers)
        for a, b in zip(graphs[:-1], graphs[1:]):
            ends = [a[0], a[-1]], [b[0], b[-1]]
            assert min(np.hypot(*(p - q)) for p in ends[0] for q in ends[1]) <= 0.05
    # 0.02 px: rounding of the printed numbers, domains rounded outward, left-out slivers and this check's sampling
    assert worst_near <= FUNCTION_TOLERANCE + 0.02
    assert worst_cover <= FUNCTION_TOLERANCE + 0.02


def test_a_smaller_tolerance_gives_closer_and_more_pieces():
    ctrl = [[10, 190], [60, -40], [150, 240], [190, 20]]
    loose, fine = function_pieces(ctrl, H, 1.0), function_pieces(ctrl, H, 0.02)
    assert max(p["max_error"] for p in fine) <= 0.02 and max(p["max_error"] for p in loose) <= 1.0
    assert len(fine) > len(loose)
    with pytest.raises(ValueError):
        function_pieces(ctrl, H, 0.0)


def test_numbers_are_fixed_point():
    rng = np.random.default_rng(5)
    for _ in range(100):
        ctrl = rng.uniform(-50, 3000, (4, 2))
        for eq in curve_functions(ctrl, 3000)[0]:
            assert not re.search(r"\d[eE][+-]?\d", eq), eq  # no scientific notation
            assert not re.search(r"-0\.0+(?![0-9]*[1-9])", eq), eq  # no negative zero


def test_a_curve_of_zero_length_has_no_functions():
    assert curve_functions([[5, 5]] * 4, H) == ([], 0.0)
    assert function_pieces([[5, 5], [5, 5], [5.001, 5], [5.002, 5]], H) == []  # a sliver


def test_attach_sets_the_functions_of_every_curve():
    cs = CurveSet(H, H, [Curve(g.line([10, 10], [190, 10])), Curve([[10, 10], [90, 10], [90, 90], [10, 90]]),
                         Curve([[5, 5]] * 4)])
    report = attach(cs)
    assert [len(c.functions) for c in cs] == [1, 3, 0]
    assert report == {"count": 4, "max_error": report["max_error"], "tolerance": FUNCTION_TOLERANCE}
    assert 0 < report["max_error"] <= FUNCTION_TOLERANCE
    back = CurveSet.from_dict(cs.to_dict())
    assert [c.functions for c in back] == [c.functions for c in cs]
    assert "functions" not in CurveSet(H, H, [Curve(g.line([0, 0], [9, 9]))]).to_dict()["curves"][0]

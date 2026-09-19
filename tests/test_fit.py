import numpy as np
import pytest

from line2func import geometry as g
from line2func.fit import fit_polyline


def _max_dist(curves, pts):
    """Largest distance from any input point to the fitted chain."""
    return np.min(np.stack([g.distance_to_points(c, pts) for c in curves]), axis=0).max()


def test_straight_polyline_gives_one_curve():
    pts = np.stack([np.linspace(0, 100, 101), np.linspace(0, 50, 101)], axis=1)
    curves = fit_polyline(pts, tolerance=0.5)
    assert len(curves) == 1
    np.testing.assert_allclose(curves[0][0], pts[0])
    np.testing.assert_allclose(curves[0][3], pts[-1])


def test_single_cubic_is_recovered():
    # sampled every ~1 px of arc length, like the baseline's resampled strokes
    ctrl = np.array([[10, 80], [30, 0], [80, 120], [110, 20]], float)
    dense = g.evaluate(ctrl, np.linspace(0, 1, 5001))
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    s = np.linspace(0, cum[-1], int(cum[-1]) + 1)
    pts = np.stack([np.interp(s, cum, dense[:, 0]), np.interp(s, cum, dense[:, 1])], axis=1)
    curves = fit_polyline(pts, tolerance=0.2)
    assert len(curves) <= 2
    assert _max_dist(curves, pts) < 0.2


@pytest.mark.parametrize("tol", [0.25, 1.0])
def test_error_bound_and_continuity(tol):
    t = np.linspace(0, 4 * np.pi, 800)
    pts = np.stack([40 * t, 30 * np.sin(t) + 5 * np.sin(3.3 * t)], axis=1)
    curves = fit_polyline(pts, tolerance=tol)
    assert _max_dist(curves, pts) <= tol + 1e-6
    for a, b in zip(curves[:-1], curves[1:]):
        np.testing.assert_allclose(a[3], b[0])
        ta, tb = a[3] - a[2], b[1] - b[0]
        cos = ta @ tb / (np.linalg.norm(ta) * np.linalg.norm(tb))
        assert cos > 0.999  # smooth (G1) joins
    # tighter tolerance never needs fewer pieces
    assert len(fit_polyline(pts, tolerance=tol / 2)) >= len(curves)


def test_closed_circle():
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    pts = np.stack([50 + 40 * np.cos(th), 50 + 40 * np.sin(th)], axis=1)
    curves = fit_polyline(pts, tolerance=0.3, closed=True)
    assert 2 <= len(curves) <= 6
    np.testing.assert_allclose(curves[0][0], curves[-1][3])
    assert _max_dist(curves, pts) <= 0.3 + 1e-6


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_short_runs_far_from_origin_stay_local(n):
    """Regression: few points far from (0, 0) once gave control points pulled to the origin."""
    t = np.linspace(0, 1, n)
    pts = np.stack([300 + 3 * t, 383 + 1.5 * np.sin(np.pi * t)], axis=1)  # tiny bump at (300, 383)
    for ctrl in fit_polyline(pts, tolerance=1.0):
        assert np.abs(ctrl - [300, 383]).max() < 10
        assert g.arc_length(ctrl) < 3 * np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) + 2


def test_out_and_back_run_stays_local():
    # a run that goes out 30 px and comes back along almost the same line
    out = np.stack([np.linspace(900, 930, 31), np.full(31, 480.0)], axis=1)
    back = np.stack([np.linspace(930, 900.5, 30), np.full(30, 481.0)], axis=1)
    pts = np.vstack([out, back[1:]])
    for ctrl in fit_polyline(pts, tolerance=1.0):
        assert ctrl[:, 0].min() > 880 and ctrl[:, 0].max() < 950
        assert np.abs(ctrl[:, 1] - 480).max() < 20


def test_near_closed_run_stays_local():
    # a small loop whose ends almost meet, as left after a corner split
    th = np.linspace(0.2, 2 * np.pi - 0.2, 40)
    pts = np.stack([500 + 5 * np.cos(th), 400 + 5 * np.sin(th)], axis=1)
    for ctrl in fit_polyline(pts, tolerance=0.5):
        assert np.abs(ctrl - [500, 400]).max() < 20


def test_degenerate_inputs():
    assert fit_polyline([[1, 1]]) == []
    assert fit_polyline([[1, 1], [1, 1]]) == []
    two = fit_polyline([[0, 0], [3, 4]])
    assert len(two) == 1 and np.allclose(two[0][[0, 3]], [[0, 0], [3, 4]])
    with pytest.raises(ValueError):
        fit_polyline([[0, 0], [1, 1]], tolerance=0)

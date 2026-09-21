"""The decision metrics penalize wrong joins, unlike the older structure scores."""

import math

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.metrics import distance_base, f_score, f_sweep, stroke_length_scores, total_length

A0, A1 = np.array([20.0, 20.0]), np.array([180.0, 112.0])  # line A crosses line B at (100, 66), ~60 deg
B0, B1 = np.array([20.0, 112.0]), np.array([180.0, 20.0])
X = np.array([100.0, 66.0])


def _gt() -> CurveSet:
    curves = [
        Curve(g.line(A0, A1), stroke=0),
        Curve(g.line(B0, B1), stroke=1),
        Curve(g.line([100, 160], [100, 220]), stroke=2),  # T: this stem ends on the bar below
        Curve(g.line([20, 160], [180, 160]), stroke=3),
        Curve(g.line([220, 20], [220, 120]), stroke=4),  # L: a 90 deg corner inside one stroke
        Curve(g.line([220, 120], [320, 120]), stroke=4),
        Curve(g.line([250, 200], [330, 250]), stroke=5),
    ]
    return CurveSet(340, 260, curves, meta={"line_width": 2.0})



def _pred(curves) -> CurveSet:
    return CurveSet(340, 260, curves, meta={"line_width": 2.0})


def test_retracing_a_line_is_invisible_to_f_score_but_not_to_the_length():
    """A stroke drawn twice sits exactly on the ink: F_GT@2 stays perfect, the length doubles."""
    gt = _pred([Curve(g.line([20, 20], [180, 20]), stroke=0)])
    twice = _pred([Curve(g.line([20, 20], [180, 20]), stroke=0),
                   Curve(g.line([20, 20], [180, 20]), stroke=1)])
    assert f_score(twice, gt)["f"] == 1.0
    assert stroke_length_scores(gt, gt)["length_ratio"] == pytest.approx(1.0)
    assert stroke_length_scores(twice, gt)["length_ratio"] == pytest.approx(2.0)
    assert stroke_length_scores(twice, gt)["length_abs_diff"] == pytest.approx(1.0)


def test_a_missing_half_shows_in_the_length_too():
    gt = _pred([Curve(g.line([20, 20], [180, 20]), stroke=0)])
    half = _pred([Curve(g.line([20, 20], [100, 20]), stroke=0)])
    assert stroke_length_scores(half, gt)["length_ratio"] == pytest.approx(0.5, abs=0.01)


def test_the_length_of_an_empty_tracing_is_nan_against_empty_truth():
    empty = _pred([])
    assert math.isnan(stroke_length_scores(empty, empty)["length_ratio"])
    assert total_length(empty) == 0.0


def test_the_distance_base_is_a_thousandth_of_the_long_edge():
    gt = _gt()  # 340 x 260
    assert distance_base(gt) == pytest.approx(0.34)


def test_the_f_sweep_rises_with_the_tolerance_and_matches_f_score():
    gt = _gt()
    off = _pred([Curve(c.ctrl + np.array([1.5, 0.0]), stroke=c.stroke) for c in gt.curves])
    sweep = f_sweep(off, gt)
    values = [sweep[f"f_{d}"] for d in range(0, 40, 2)]
    assert values == sorted(values)  # never falls as the tolerance grows
    assert values[-1] == pytest.approx(1.0)
    # each step is f_score at that tolerance, counted in distance_base units
    base = distance_base(gt)
    assert sweep["f_6"] == pytest.approx(f_score(off, gt, threshold=6.0, base=base)["f"])

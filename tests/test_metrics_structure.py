"""The decision metrics penalize wrong joins, unlike the older structure scores."""

import math

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.metrics import (decision_counts, decision_scores, distance_base, f_score, f_sweep, find_touches,
                               gt_corners, joint_corners, stroke_length_scores, structure_scores, total_length)

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


def _scores(pred: CurveSet, gt: CurveSet) -> dict:
    return decision_scores(decision_counts(pred, gt))


def _pred(curves) -> CurveSet:
    return CurveSet(340, 260, curves, meta={"line_width": 2.0})


def test_ground_truth_against_itself_is_perfect():
    gt = _gt()
    s = _scores(gt, gt)
    assert s["bcubed_p"] == 1.0 and s["bcubed_r"] == 1.0 and s["bcubed_f"] == 1.0
    assert s["crossing_both_ok"] == 1.0 and s["crossing_false_turn"] == 0.0
    assert s["t_bar_ok"] == 1.0 and s["t_false_cont"] == 0.0
    assert s["joins_crossing_per100"] == s["joins_touch_per100"] == s["joins_other_per100"] == 0.0
    assert s["corner_p"] == 1.0 and s["corner_r"] == 1.0


def test_touches_and_corners_are_found():
    gt = _gt()
    touches = find_touches(gt)
    assert [(t["stem"], t["end"], t["bar"]) for t in touches] == [(2, 0, 3)]
    assert abs(touches[0]["s_bar"] - 80.0) < 1.0
    corners = gt_corners(gt)
    assert len(corners) == 1 and np.allclose(corners[0][0], [220, 120]) and abs(corners[0][1] - 90) < 1e-6


def test_merging_everything_is_caught():
    gt = _gt()
    one = _pred([Curve(c.ctrl, stroke=0) for c in gt])
    s = _scores(one, gt)
    assert s["bcubed_p"] < 0.5 and s["bcubed_r"] == 1.0
    assert s["joins_other_per100"] > 0
    # the older score cannot see it: "one stroke" continues through every crossing
    assert structure_scores(one, gt)["crossing_continuity"] == 1.0


def test_turning_at_a_crossing_is_caught():
    gt = _gt()
    bounce = [Curve(g.line(A0, X), stroke=0), Curve(g.line(X, B1), stroke=0),
              Curve(g.line(B0, X), stroke=1), Curve(g.line(X, A1), stroke=1)]
    s = _scores(_pred(bounce + [c for c in gt if c.stroke >= 2]), gt)
    assert s["crossing_false_turn"] == 1.0 and s["crossing_both_ok"] == 0.0
    assert s["joins_crossing_per100"] > 0 and s["bcubed_p"] < 1.0


def test_stem_continuing_into_the_bar_is_caught():
    gt = _gt()
    wrong = [Curve(g.line([100, 220], [100, 160]), stroke=2), Curve(g.line([100, 160], [180, 160]), stroke=2),
             Curve(g.line([20, 160], [100, 160]), stroke=3)]
    s = _scores(_pred([c for c in gt if c.stroke in (0, 1, 4, 5)] + wrong), gt)
    assert s["t_false_cont"] == 1.0 and s["t_bar_ok"] == 0.0
    assert s["joins_touch_per100"] > 0


def test_corner_scores():
    gt = _gt()
    # the two arms of the L as separate strokes: the corner is reached but not made
    split = [c for c in gt if c.stroke != 4] + [Curve(gt.curves[4].ctrl, stroke=4), Curve(gt.curves[5].ctrl, stroke=9)]
    s = _scores(_pred(split), gt)
    assert s["corner_r"] == 0.0
    # a smooth joint (G1) is not a corner
    arc = [np.array([[0, 0], [10, 0], [20, 5], [30, 10]], float), np.array([[30, 10], [40, 15], [50, 25], [60, 40]], float)]
    assert joint_corners(CurveSet(100, 100, [Curve(c, stroke=0) for c in arc]), 20.0) == []
    kink = [g.line([0, 0], [30, 0]), g.line([30, 0], [30 + 30 * math.cos(1.0), 30 * math.sin(1.0)])]
    turns = joint_corners(CurveSet(100, 100, [Curve(c, stroke=0) for c in kink]), 20.0)
    assert len(turns) == 1 and abs(turns[0][1] - math.degrees(1.0)) < 1e-6


# ---------------------------------------------------------------------------
# Total length: what the point-distance metrics cannot see
# ---------------------------------------------------------------------------


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

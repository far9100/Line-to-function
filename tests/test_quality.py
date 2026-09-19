"""The quality metrics must react correctly to known, controlled damage."""

import copy

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.quality import assess, quality_map, summary
from line2func.render import rasterize
from line2func.synth import random_scene


@pytest.fixture(scope="module")
def scene():
    gt, widths, img = random_scene(np.random.default_rng(3), 320, 320, n_strokes=10, line_width=(2.0, 3.0))
    gt.meta["line_width"] = 2.5
    for c, w in zip(gt.curves, widths):
        c.width = float(w)
    return gt, 1.0 - img / 255.0


def test_perfect_tracing_scores_perfectly(scene):
    gt, ink = scene
    r, _ = assess(gt, ink)
    assert r["recall"]["line"]["within_2px"] > 0.99
    assert r["precision"]["within_2px"] > 0.99
    assert r["distance"]["d_M"] < 0.5
    assert r["missed"]["share_of_ink"] < 0.02


def test_deleting_curves_lowers_recall_where_they_were(scene):
    gt, ink = scene
    cut = copy.deepcopy(gt)
    removed = cut.curves[::5]  # drop every 5th curve
    cut.curves = [c for i, c in enumerate(cut.curves) if i % 5]
    r_full, _ = assess(gt, ink)
    r_cut, maps = assess(cut, ink)
    drop = r_full["recall"]["line"]["within_2px"] - r_cut["recall"]["line"]["within_2px"]
    removed_len = sum(g.arc_length(c.ctrl) for c in removed) / sum(g.arc_length(c.ctrl) for c in gt.curves)
    assert drop == pytest.approx(removed_len, abs=0.08)
    assert r_cut["precision"]["within_2px"] > 0.99  # nothing invented
    # the missed ink sits on the removed curves
    missed = np.argwhere(maps["code"] > 0)[:, ::-1] + 0.5
    d = np.min(np.stack([g.distance_to_points(c.ctrl, missed) for c in removed]), axis=0)
    assert np.mean(d < 3.0) > 0.9
    assert max(r_cut["missed"]["by_cause"], key=r_cut["missed"]["by_cause"].get) == "untraced"


def test_a_stray_line_is_flagged(scene):
    gt, ink = scene
    stray = copy.deepcopy(gt)
    stray.curves.append(Curve(g.line([5, 315], [315, 5]), stroke=999))
    r, _ = assess(stray, ink)
    base, _ = assess(gt, ink)
    assert r["distance"]["curve_to_any_ink"]["max"] > 20.0
    assert base["distance"]["curve_to_any_ink"]["max"] < 3.0
    assert r["precision"]["within_2px"] < base["precision"]["within_2px"] - 0.05


def test_iou_is_harsh_on_small_shifts_but_recall_is_not(scene):
    gt, ink = scene
    shifted = copy.deepcopy(gt)
    for c in shifted.curves:
        c.ctrl = c.ctrl + [1.0, 0.0]
    base, _ = assess(gt, ink)
    r, _ = assess(shifted, ink)
    # a 1 px sideways shift moves a line at angle a by |sin a| across itself (2/pi = 0.64 px
    # on average); the ink centerline already sits ~0.4 px off the curves on the pixel grid,
    # and the two offsets combine partly, so d_M rises by ~0.35 px here
    assert r["distance"]["d_M"] > base["distance"]["d_M"] + 0.25
    assert r["recall"]["line"]["within_2px"] > 0.97  # still "kept" within 2 px
    assert r["raster"]["iou"] < base["raster"]["iou"] - 0.15  # IoU punishes a 1 px shift hard


def test_faint_ink_below_threshold_is_labelled():
    strong = g.line([10, 20], [150, 20])
    faint = g.line([10, 60], [150, 60])
    ink = rasterize([strong], 160, 80, line_width=3.0) + 0.15 * rasterize([faint], 160, 80, line_width=3.0)
    traced = CurveSet(160, 80, [Curve(strong, width=3.0)], meta={"line_width": 3.0})
    r, maps = assess(traced, ink, threshold=0.3)
    assert r["recall"]["by_band"]["dark"]["within_2px"] > 0.98
    assert r["recall"]["by_band"]["faint"]["within_2px"] < 0.05
    # counting faint strokes, only about half of the drawing's centerline is traced
    assert r["recall"]["line"]["within_2px"] > 0.98
    assert r["recall"]["with_faint"]["within_2px"] < 0.6
    assert r["missed"]["by_cause"]["below_threshold"] > 0.9
    assert r["missed"]["largest_regions"][0]["cause"] == "below_threshold"
    img = quality_map(maps)
    assert img.shape == (80, 160, 3)
    assert tuple(img[60, 80]) != (255, 255, 255)  # the faint line is highlighted
    assert len(summary(r)) == 5


def test_empty_result():
    ink = rasterize([g.line([5, 5], [50, 50])], 64, 64, line_width=2.0)
    r, _ = assess(CurveSet(64, 64), ink)
    assert r["recall"]["line"]["within_2px"] == 0.0
    assert r["structure"]["curves"] == 0

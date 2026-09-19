import time

import numpy as np
import pytest

from line2func import geometry as g
from line2func.baseline import BaselineParams, thin, vectorize
from line2func.curves import Curve, CurveSet
from line2func.metrics import f_score
from line2func.render import render_lineart
from line2func.synth import random_scene

K = 0.5522847498


def circle(cx, cy, r):
    out = []
    for a in range(4):
        t0, t1 = a * np.pi / 2, (a + 1) * np.pi / 2
        p0 = np.array([cx + r * np.cos(t0), cy + r * np.sin(t0)])
        p3 = np.array([cx + r * np.cos(t1), cy + r * np.sin(t1)])
        p1 = p0 + K * r * np.array([-np.sin(t0), np.cos(t0)])
        p2 = p3 - K * r * np.array([-np.sin(t1), np.cos(t1)])
        out.append(np.array([p0, p1, p2, p3]))
    return out


def trace(ctrls, width=2.0, size=128, edit=None):
    gt = CurveSet(size, size, [Curve(c, stroke=i) for i, c in enumerate(ctrls)])
    img = render_lineart(gt, size, size, line_width=width)
    if edit is not None:
        img = edit(img.copy())
    return vectorize(1.0 - img / 255.0), gt


def test_thin_gives_one_pixel_lines():
    mask = np.zeros((20, 40), bool)
    mask[8:13, 5:35] = True
    skel = thin(mask)
    assert skel[:, 10:30].sum(axis=0).tolist() == [1] * 20


def test_empty_image():
    cs = vectorize(np.zeros((32, 32)))
    assert len(cs) == 0 and cs.width == 32


def test_straight_line():
    cs, gt = trace([g.line([10, 20], [110, 90])])
    assert (len(cs), cs.num_strokes) == (1, 1)
    assert f_score(cs, gt)["f"] == 1.0
    assert cs.meta["line_width"] == pytest.approx(2.0, abs=0.3)
    assert cs.curves[0].confidence == 1.0


def test_circle_is_one_closed_stroke():
    cs, gt = trace(circle(64, 64, 40))
    assert cs.num_strokes == 1
    np.testing.assert_allclose(cs.curves[0].ctrl[0], cs.curves[-1].ctrl[3], atol=1e-9)
    assert f_score(cs, gt)["f"] == 1.0


@pytest.mark.parametrize("angle", [90, 30, 20])
def test_crossing_lines_stay_whole(angle):
    t = np.tan(np.radians(angle / 2)) * 54
    cs, gt = trace([g.line([10, 64 - t], [118, 64 + t]), g.line([10, 64 + t], [118, 64 - t])])
    assert (len(cs), cs.num_strokes) == (2, 2)
    assert f_score(cs, gt)["f"] == 1.0
    # each stroke really crosses (no "bounce" back to the same side)
    for c in cs:
        assert abs(c.ctrl[0, 1] - c.ctrl[3, 1]) > t


def test_t_junction_and_h_bar():
    cs, _ = trace([g.line([10, 30], [118, 30]), g.line([64, 30], [64, 118])])
    assert cs.num_strokes == 2
    # the short bar of an H is a real stroke, not a shallow crossing
    cs, _ = trace([g.line([50, 20], [50, 108]), g.line([60, 20], [60, 108]), g.line([50, 64], [60, 64])])
    assert cs.num_strokes == 3


def test_corner_is_kept_sharp():
    cs, _ = trace([g.line([20, 20], [20, 100]), g.line([20, 100], [100, 100])])
    assert (len(cs), cs.num_strokes) == (2, 1)
    joint = cs.curves[0].ctrl[3]
    assert np.linalg.norm(joint - [20, 100]) < 1.5


def test_small_gap_is_closed():
    def cut(img):
        img[:, 60:64] = 255
        return img

    cs, _ = trace([g.line([10, 64], [118, 64])], edit=cut)
    assert (len(cs), cs.num_strokes) == (1, 1)
    assert cs.curves[0].confidence < 1.0  # part of it bridges the gap


def test_specks_are_ignored():
    def specks(img):
        rng = np.random.default_rng(3)
        for _ in range(30):
            y, x = rng.integers(0, 126, 2)
            img[y : y + 2, x : x + 2] = 0
        return img

    cs, _ = trace([g.line([10, 64], [118, 64])], edit=specks)
    assert cs.num_strokes == 1


def test_filled_area_gives_outline():
    def blob(img):
        yy, xx = np.mgrid[:128, :128]
        img[(yy - 90) ** 2 + (xx - 90) ** 2 < 20**2] = 0
        return img

    cs, _ = trace([g.line([10, 20], [118, 20])], edit=blob)
    outline = [c for c in cs if "fill_outline" in c.tags]
    assert outline
    pts = np.vstack([g.evaluate(c.ctrl, np.linspace(0, 1, 20)) for c in outline])
    radius = np.linalg.norm(pts - [90, 90], axis=1)
    assert np.all(np.abs(radius - 20) < 2.0)


def test_thick_lines():
    cs, gt = trace([g.line([10, 20], [110, 90])] + circle(64, 64, 30), width=6.0)
    assert cs.num_strokes == 2
    assert f_score(cs, gt)["f"] > 0.99


def test_light_lines_on_dark_background_via_lineart():
    from line2func.lineart import extract

    gt = CurveSet(96, 96, [Curve(g.line([10, 48], [86, 48]))])
    img = 255 - render_lineart(gt, 96, 96)
    cs = vectorize(extract(img, "none"))
    assert cs.num_strokes == 1


def test_clean_random_scenes_meet_targets():
    """The baseline's targets (README, Evaluation): F_GT@2 >= 0.97 on clean drawings, <= 5 s CPU per MP."""
    rng = np.random.default_rng(7)
    scores, seconds, pixels = [], 0.0, 0
    for _ in range(6):
        gt, _, img = random_scene(rng, 384, 384, n_strokes=8)
        t = time.perf_counter()
        cs = vectorize(1.0 - img / 255.0)
        seconds += time.perf_counter() - t
        pixels += img.size
        scores.append(f_score(cs, gt)["f"])
    assert np.mean(scores) >= 0.97
    assert seconds / (pixels / 1e6) <= 5.0


def test_tolerance_parameter_controls_curve_count():
    rng = np.random.default_rng(11)
    _, _, img = random_scene(rng, 256, 256, n_strokes=6)
    ink = 1.0 - img / 255.0
    loose = vectorize(ink, BaselineParams(fit_tolerance=2.0))
    tight = vectorize(ink, BaselineParams(fit_tolerance=0.3))
    assert len(tight) >= len(loose)

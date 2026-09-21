from dataclasses import replace

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.metrics import find_crossings, structure_scores
from line2func.render import rasterize
from line2func.synth import (
    PRESETS,
    load_scene_dir,
    main,
    make_scene,
    random_single_curve,
    single_curve_sample,
)


def test_tapered_width_rendering():
    ctrl = g.line([4, 16], [60, 16])
    cov = rasterize([ctrl], 64, 32, line_width=[[1.0, 5.0]])
    thin, thick = cov[:, 8].sum(), cov[:, 56].sum()
    assert thin == pytest.approx(1.0 + 4.0 * (8 - 4) / 56, abs=0.2)
    assert thick == pytest.approx(1.0 + 4.0 * (56 - 4) / 56, abs=0.2)


@pytest.mark.parametrize("kind", ["clean", "hard"])
def test_scene_is_deterministic_and_valid(kind):
    a = make_scene(np.random.default_rng([3, 1]), 256, 256, kind)
    b = make_scene(np.random.default_rng([3, 1]), 256, 256, kind)
    np.testing.assert_array_equal(a.image, b.image)
    assert a.image.dtype == np.uint8 and a.image.shape == (256, 256)
    assert len(a.gt) > 0 and len(a.gt.meta["widths"]) == len(a.gt)
    lo, hi = PRESETS[kind].width
    w = np.array(a.gt.meta["widths"])
    assert w.min() >= 0.6 and w.max() <= hi * 1.25 + 1e-9


def test_clean_scene_has_no_noise():
    s = make_scene(np.random.default_rng(0), 256, 256, "clean")
    assert s.gaps == []
    cov = rasterize(s.gt, 256, 256, line_width=np.array(s.gt.meta["widths"]))
    ink = 1.0 - s.image / 255.0
    assert np.abs(ink - cov).max() < 0.01  # exactly the rendered strokes


def test_gaps_are_erased_but_kept_in_ground_truth():
    deg = replace(PRESETS["hard"], gaps_per_stroke=3.0, specks_per_mp=(0, 0), noise=(0, 0), blur=(0, 0), jpeg=0.0,
                  ink=(1.0, 1.0), paper_shading=0.0, shallow_crossings=(0, 0))
    s = make_scene(np.random.default_rng(1), 384, 384, deg)
    assert s.gaps
    for gap in s.gaps:
        x, y = (int(v) for v in gap["point"])
        assert s.image[y, x] > 200  # erased on the page
        d = g.distance_to_points(s.gt.curves[gap["curve"]].ctrl, [gap["point"]])[0]
        assert d < 0.01  # but the ground-truth curve passes through it


def test_single_curve_sample():
    rng = np.random.default_rng(5)
    for _ in range(20):
        img, ctrl, w = single_curve_sample(rng, 64, "hard")
        assert img.shape == (64, 64) and img.dtype == np.uint8
        pts = g.evaluate(ctrl, np.linspace(0, 1, 50))
        assert pts.min() >= 0 and pts.max() <= 64
        assert w.shape == (2,) and np.all(w > 0)
    ctrl = random_single_curve(np.random.default_rng(1), 64)
    assert np.linalg.norm(ctrl[3] - ctrl[0]) >= 0.3 * 64


def test_find_crossings():
    gt = CurveSet(100, 100, [Curve(g.line([10, 50], [90, 50]), 0), Curve(g.line([50, 10], [50, 90]), 1),
                             Curve(g.line([10, 20], [30, 20]), 2)])
    c = find_crossings(gt)
    assert len(c) == 1
    np.testing.assert_allclose(c[0]["point"], [50, 50], atol=1e-6)
    assert c[0]["angle"] == pytest.approx(90.0)


def test_structure_scores_perfect_and_broken():
    gt = CurveSet(128, 128, [Curve(g.line([10, 64], [118, 64]), 0), Curve(g.line([64, 10], [64, 118]), 1)],
                  meta={"gaps": [{"stroke": 0, "point": [30.0, 64.0], "length": 4.0}]})
    perfect = structure_scores(gt, gt)
    assert perfect["crossing_continuity"] == 1.0
    assert perfect["gap_closure"] == 1.0
    assert perfect["fragments_per_stroke"] == 1.0 and perfect["curve_ratio"] == 1.0

    # horizontal line split at the crossing and at the gap
    broken = CurveSet(128, 128, [
        Curve(g.line([10, 64], [28, 64]), 0), Curve(g.line([32, 64], [62, 64]), 1),
        Curve(g.line([66, 64], [118, 64]), 2), Curve(g.line([64, 10], [64, 118]), 3),
    ])
    s = structure_scores(broken, gt)
    assert s["crossing_continuity"] == 0.5  # vertical stroke fine, horizontal one broken
    assert s["gap_closure"] == 0.0
    assert s["fragments_per_stroke"] == 2.0  # (3 + 1) / 2
    assert s["curve_ratio"] == 2.0

    # a "bounce" (each prediction turns back at the crossing) is not continuity
    bounce = CurveSet(128, 128, [
        Curve(np.array([[10, 64], [60, 64], [64, 60], [64, 10]], float), 0),
        Curve(np.array([[118, 64], [68, 64], [64, 68], [64, 118]], float), 1),
    ])
    assert structure_scores(bounce, gt)["crossing_continuity"] == 0.0


def test_cli_scenes_and_valset(tmp_path):
    assert main(["scenes", "--kind", "hard", "--count", "2", "--size", "160", "--out", str(tmp_path / "s")]) == 0
    scenes = load_scene_dir(tmp_path / "s")
    assert len(scenes) == 2 and scenes[0][1].meta["kind"] == "hard"
    assert main(["preview", "--patches", "--out", str(tmp_path / "p.png")]) == 0
    assert (tmp_path / "p.png").is_file()

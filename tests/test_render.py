import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.render import rasterize, render_lineart, render_overlay, save_png


def test_horizontal_line_on_pixel_boundary():
    # y = 10 lies between rows 9 and 10; width 2 fills exactly those rows
    cov = rasterize([g.line([10, 10], [50, 10])], 64, 20, line_width=2.0)
    col = cov[:, 30]
    np.testing.assert_allclose(col[9:11], 1.0)
    assert col[:9].max() == 0.0 and col[11:].max() == 0.0


@pytest.mark.parametrize("width", [1.0, 1.5, 2.0, 3.3, 6.0])
@pytest.mark.parametrize("y", [10.0, 10.25, 10.5, 10.8])
def test_ink_per_column_equals_line_width(width, y):
    cov = rasterize([g.line([2, y], [60, y])], 64, 24, line_width=width)
    np.testing.assert_allclose(cov[:, 10:50].sum(axis=0), width, atol=1e-5)


def test_diagonal_ink_matches_area():
    # 45° line, length L, width w: total ink ≈ L * w (+ round caps, inside the image).
    # At 45° pixel centers sample the cross-section every 1/sqrt(2) px, where the
    # 1-px linear edge ramp is not summed exactly (≈1.7% low here), hence 3%.
    p0, p1, w = np.array([20.0, 20.0]), np.array([100.0, 100.0]), 3.0
    cov = rasterize([g.line(p0, p1)], 128, 128, line_width=w)
    length = np.linalg.norm(p1 - p0)
    expect = length * w + np.pi * (w / 2) ** 2
    assert cov.sum() == pytest.approx(expect, rel=0.03)


def test_curve_ink_matches_arc_length():
    k = 0.5522847498 * 40
    ctrl = np.array([[90, 50], [90, 50 + k], [50 + k, 90], [50, 90]], float)
    w = 2.0
    cov = rasterize([ctrl], 128, 128, line_width=w)
    expect = g.arc_length(ctrl) * w + np.pi * (w / 2) ** 2
    assert cov.sum() == pytest.approx(expect, rel=0.01)


def test_ink_lies_on_the_curve():
    ctrl = np.array([[10, 60], [30, 0], [70, 120], [110, 20]], float)
    cov = rasterize([ctrl], 128, 128, line_width=2.0)
    ys, xs = np.nonzero(cov > 0)
    d = g.distance_to_points(ctrl, np.stack([xs + 0.5, ys + 0.5], axis=1))
    assert d.max() <= 1.0 + 0.5 + 0.06  # half width + AA ramp + flatten tolerance
    # and every sample on the curve is fully inked
    pts = g.evaluate(ctrl, np.linspace(0, 1, 500))
    assert cov[pts[:, 1].astype(int), pts[:, 0].astype(int)].min() > 0.5


def test_overlap_takes_max_not_sum():
    a = g.line([0, 16], [32, 16])
    cov = rasterize([a, a], 32, 32, line_width=2.0)
    assert cov.max() <= 1.0
    np.testing.assert_allclose(cov, rasterize([a], 32, 32, line_width=2.0))


def test_per_curve_widths_and_offscreen_clipping():
    thin, thick = g.line([5, 5], [60, 5]), g.line([5, 20], [60, 20])
    cov = rasterize([thin, thick], 64, 32, line_width=[1.0, 4.0])
    assert cov[:, 30][:12].sum() == pytest.approx(1.0, abs=1e-5)
    assert cov[:, 30][12:].sum() == pytest.approx(4.0, abs=1e-5)
    off = g.line([-100, -100], [-50, -80])
    assert rasterize([off], 32, 32).sum() == 0.0
    # a line that leaves the image is simply clipped
    partial = rasterize([g.line([-20, 16], [50, 16])], 32, 32, line_width=2.0)
    np.testing.assert_allclose(partial[:, 0].sum(), 2.0, atol=1e-5)


def test_degenerate_curve_draws_a_dot():
    cov = rasterize([np.full((4, 2), 16.0)], 32, 32, line_width=4.0)
    assert cov.sum() == pytest.approx(np.pi * 4, rel=0.1)


def test_empty_and_bad_widths():
    assert rasterize([], 8, 8).sum() == 0.0
    with pytest.raises(ValueError):
        rasterize([g.line([0, 0], [5, 5])], 8, 8, line_width=[1.0, 2.0])
    with pytest.raises(ValueError):
        rasterize([g.line([0, 0], [5, 5])], 8, 8, line_width=0.0)


def test_lineart_and_overlay(tmp_path):
    cs = CurveSet(
        64,
        48,
        [
            Curve(g.line([4, 10], [60, 10]), stroke=0),
            Curve(g.line([4, 30], [60, 30]), stroke=1),
        ],
    )
    art = render_lineart(cs, 64, 48)
    assert art.dtype == np.uint8 and art.shape == (48, 64)
    assert art[9, 30] == 0 and art[0, 0] == 255

    over = render_overlay(art, cs, line_width=2.0, fade=0.5)
    assert over.shape == (48, 64, 3)
    assert tuple(over[9, 30]) == (230, 25, 75)  # stroke 0 color
    assert tuple(over[29, 30]) == (0, 130, 200)  # stroke 1 color
    assert tuple(over[0, 0]) == (255, 255, 255)

    save_png(tmp_path / "overlay.png", over)
    save_png(tmp_path / "art.png", art)
    assert (tmp_path / "overlay.png").stat().st_size > 0

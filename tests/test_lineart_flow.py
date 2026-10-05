"""The flow line extractor (lineart.flow): what it finds in a photo, and what it refuses to find."""

import numpy as np
import pytest
from scipy import ndimage

from line2func import baseline, lineart
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart


def _strokes(size: int = 224, count: int = 6, width: float = 2.0, seed: int = 3) -> np.ndarray:
    """A small drawing, as grayscale in [0, 1] (1 = white paper)."""
    rng = np.random.default_rng(seed)
    curves = [Curve(rng.uniform(15, size - 15, (4, 2)), stroke=i) for i in range(count)]
    lines = render_lineart(CurveSet(size, size, curves), size, size, line_width=width)
    return lineart.to_gray(lines)


def _mixed(size: int = 224, seed: int = 3, paper: float = 0.65) -> np.ndarray:
    """Strong strokes and faint ones together, the way a photo has both hard edges and soft detail."""
    rng = np.random.default_rng(seed)

    def layer(count: int, width: float) -> np.ndarray:
        curves = [Curve(rng.uniform(15, size - 15, (4, 2)), stroke=i) for i in range(count)]
        return lineart.to_gray(render_lineart(CurveSet(size, size, curves), size, size, line_width=width))

    faint = layer(9, 1.6)
    return np.minimum(layer(5, 2.4), paper + (1.0 - paper) * faint).astype(np.float32)


def _photoish(gray: np.ndarray, noise: float, seed: int = 7) -> np.ndarray:
    """``gray`` as a camera would have given it: softened, unevenly lit and noisy."""
    soft = ndimage.gaussian_filter(gray.astype(np.float64), 1.2)
    h, w = soft.shape
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 0.75 + 0.25 * (xx / max(w - 1, 1)) - 0.15 * (yy / max(h - 1, 1))
    rng = np.random.default_rng(seed)
    return np.clip(soft * shade + rng.normal(0.0, noise, soft.shape), 0.0, 1.0).astype(np.float32)


def _skeleton(ink: np.ndarray) -> np.ndarray:
    return baseline.thin(ink > baseline.auto_threshold(ink))


def _pieces_per_kpx(ink: np.ndarray) -> float:
    """Connected pieces per 1,000 skeleton pixels: how broken up the lines are."""
    skeleton = _skeleton(ink)
    drawn = int(skeleton.sum())
    if drawn == 0:
        return 0.0
    return 1000.0 * ndimage.label(skeleton, structure=np.ones((3, 3)))[1] / drawn


def test_a_flat_image_has_no_lines_in_it():
    for value in (0.1, 0.5, 0.9):
        assert lineart.flow(np.full((64, 64), value, np.float32)).max() == 0.0


def test_a_smooth_gradient_has_no_lines_in_it():
    """A sky or a cheek is not an edge. This is what keeps a photo off the curve budget."""
    yy, xx = np.mgrid[0:96, 0:96]
    for ramp in (xx / 95.0, yy / 95.0, (xx + yy) / 190.0):
        assert lineart.flow((0.1 + 0.8 * ramp).astype(np.float32)).max() == 0.0


def test_a_line_under_photo_noise_stays_one_piece_not_a_row_of_dashes():
    """The whole point of the flow: noise has no line to agree with it, so it is averaged away.

    A difference of Gaussians taken on its own shatters under the same noise, and every fragment
    that survives the tracer's speck filter costs a curve.
    """
    gray = _photoish(_strokes(), noise=0.06)
    flow_broken = _pieces_per_kpx(lineart.flow(gray))
    xdog_broken = _pieces_per_kpx(lineart.xdog(gray))
    assert flow_broken < xdog_broken / 4.0, (flow_broken, xdog_broken)


def test_noise_breaks_up_the_flow_lines_far_less_than_it_breaks_up_a_plain_difference():
    """Both get worse when the noise comes up; what matters is how much worse."""
    clean, noisy = _photoish(_strokes(), 0.01), _photoish(_strokes(), 0.06)
    flow_cost = _pieces_per_kpx(lineart.flow(noisy)) / max(_pieces_per_kpx(lineart.flow(clean)), 1e-9)
    xdog_cost = _pieces_per_kpx(lineart.xdog(noisy)) / max(_pieces_per_kpx(lineart.xdog(clean)), 1e-9)
    assert flow_cost * 3.0 < xdog_cost, (flow_cost, xdog_cost)


def test_more_detail_finds_more_line_and_never_less():
    """Turning the slider up must never take line away, or the control is lying about what it does."""
    gray = _photoish(_mixed(), noise=0.02)
    drawn = [int(_skeleton(lineart.flow(gray, detail)).sum()) for detail in (0, 25, 50, 75, 100)]
    assert drawn == sorted(drawn), drawn
    assert drawn[-1] > 1.5 * drawn[0], drawn  # and the slider has to be worth having


def test_the_lines_come_out_wide_enough_that_the_tracer_does_not_upscale():
    """Every detail setting draws lines thicker than pipeline.AUTO_UPSCALE_BELOW, which is 2x the work."""
    from line2func.pipeline import AUTO_UPSCALE_BELOW

    gray = _photoish(_strokes(), noise=0.03)
    for detail in (0, 50, 100):
        ink = lineart.flow(gray, detail)
        assert baseline.estimate_line_width(ink, baseline.auto_threshold(ink)) >= AUTO_UPSCALE_BELOW


def test_a_boundary_between_two_colours_of_one_brightness_is_a_line():
    """Gray cannot see it at all, and a painted picture is full of them: hair against sky, cloth on cloth."""
    rgb = np.empty((96, 96, 3), np.uint8)
    rgb[:, :48], rgb[:, 48:] = (200, 80, 80), (60, 128, 200)  # both 116 in luminance
    assert lineart.flow(lineart.to_gray(rgb)).max() == 0.0
    ink = lineart.flow(rgb)
    drawn = ink[10:-10] >= 0.5
    assert drawn.any(axis=1).all()  # a line the whole way down
    assert drawn[:, 40:56].sum() == drawn.sum()  # and nothing away from the boundary
    assert np.array_equal(lineart.extract(rgb, "flow"), ink)  # extract reads the colours too


def test_a_gray_picture_reads_the_same_with_one_channel_or_three():
    gray = _photoish(_strokes(size=96, count=3), noise=0.02)
    as_rgb = np.repeat((255 * gray).astype(np.uint8)[:, :, None], 3, axis=2)
    assert np.array_equal(lineart.flow(as_rgb), lineart.flow(as_rgb[..., 0].astype(np.float32) / 255.0))
    channels = [as_rgb[..., i].astype(np.float32) / 255.0 for i in range(3)]
    tx, ty, axis = lineart.flow_field(channels)
    one = lineart.flow_dog(channels[0], *lineart.edge_tangent_flow(channels[0]))
    assert np.abs(lineart.flow_dog(channels, tx, ty, axis=axis) - one).max() < 1e-3


def test_the_noise_in_a_picture_is_read_off_the_picture():
    assert lineart.noise_level(np.full((64, 64), 0.5, np.float32)) == 0.0
    yy, xx = np.mgrid[0:96, 0:96]
    assert lineart.noise_level((0.1 + 0.8 * xx / 95.0).astype(np.float32)) < 1e-6  # a slope is not noise
    for noise in (0.01, 0.03, 0.06):
        assert lineart.noise_level(_photoish(_strokes(), noise)) == pytest.approx(noise, rel=0.1)


def test_noise_alone_draws_next_to_nothing():
    """The bar rises with the picture's own noise, so grain is not traced where there is no line."""
    rng = np.random.default_rng(11)
    for noise in (0.02, 0.06, 0.12):
        grain = np.clip(0.5 + rng.normal(0.0, noise, (160, 160)), 0.0, 1.0).astype(np.float32)
        assert (lineart.flow(grain) >= 0.5).mean() < 0.002, noise


def test_a_faint_line_is_found_on_light_and_on_dark_alike():
    """One bar everywhere: the same step down is as much of a line on a face as in a night sky."""
    found = []
    for paper in (0.2, 0.9):
        gray = np.full((80, 80), paper, np.float32)
        gray[:, 39:41] -= 0.08
        found.append(float((lineart.flow(ndimage.gaussian_filter(gray, 0.7))[10:-10] >= 0.5).sum()))
    assert min(found) > 0 and max(found) / min(found) < 1.05, found


def test_the_flow_follows_the_stroke_and_not_the_gradient():
    gray = np.ones((128, 128), np.float32)
    rows = np.arange(20, 108)
    gray[rows, rows] = gray[rows, rows - 1] = gray[rows, rows + 1] = 0.0  # a 3 px wide diagonal
    tx, ty = lineart.edge_tangent_flow(ndimage.gaussian_filter(gray, 1.0))
    on_line = (rows[10:-10], rows[10:-10])
    along = np.abs(tx[on_line] + ty[on_line]) / np.sqrt(2.0)  # the stroke runs along (1, 1)/sqrt(2)
    assert along.min() > 0.95, float(along.min())


def test_which_way_round_the_flow_points_does_not_change_the_lines():
    """The flow is a direction without a sign, and it flips along a curve; reading it must not care."""
    gray = _photoish(_strokes(size=128, count=4), noise=0.03)
    tx, ty = lineart.edge_tangent_flow(gray)
    flip = np.where(np.random.default_rng(5).random(tx.shape) < 0.5, -1.0, 1.0)
    assert np.array_equal(lineart.flow_dog(gray, tx, ty), lineart.flow_dog(gray, tx * flip, ty * flip))


def test_the_detail_settings_are_the_tuned_ones_at_fifty_and_move_both_ways():
    assert lineart.detail_params() == lineart.detail_params(lineart.DETAIL) == lineart.DETAIL_TUNED
    assert lineart.detail_params(0) == lineart.DETAIL_COARSE
    assert lineart.detail_params(100) == pytest.approx(lineart.DETAIL_FINE)
    for name in lineart.DETAIL_TUNED:
        run = [lineart.detail_params(d)[name] for d in range(0, 101, 5)]
        assert run == sorted(run) or run == sorted(run, reverse=True), name


def test_a_detail_outside_zero_to_a_hundred_is_refused():
    for bad in (-1.0, 101.0):
        with pytest.raises(ValueError):
            lineart.detail_params(bad)
    with pytest.raises(ValueError):
        lineart.extract(np.full((16, 16), 0.5, np.float32), "flow", detail=200.0)


def test_only_the_flow_method_reads_the_detail():
    photo = (255 * _photoish(_strokes(size=96, count=3), noise=0.02)).astype(np.uint8)
    for method in ("none", "canny", "xdog"):
        assert np.array_equal(lineart.extract(photo, method), lineart.extract(photo, method, detail=0.0))
    assert not np.array_equal(lineart.extract(photo, "flow"), lineart.extract(photo, "flow", detail=0.0))


def test_the_extractor_needs_nothing_but_numpy_scipy_and_pillow():
    """It has to run in Pyodide, where there is no PyTorch to fall back on."""
    import subprocess
    import sys

    code = ("import sys, numpy as np\n"
            "from line2func import lineart\n"
            "lineart.extract(np.full((48, 48), 0.5, np.float32), 'flow')\n"
            "print(sorted(m for m in ('torch', 'cv2', 'skimage') if m in sys.modules))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"

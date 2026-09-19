import numpy as np
import pytest

from line2func import geometry as g

# The arch from the README, in y-up math coordinates:
#   x(t) = -40 t^3 + 60 t^2 + 60 t + 10
#   y(t) =  20 t^3 - 270 t^2 + 250 t + 10
ARCH_POWER = np.array([[-40.0, 20.0], [60.0, -270.0], [60.0, 250.0], [10.0, 10.0]])

RNG = np.random.default_rng(0)
RANDOM_CURVES = [RNG.uniform(-50, 150, size=(4, 2)) for _ in range(20)]


def test_arch_example_from_plan():
    ctrl = g.from_power(ARCH_POWER)
    np.testing.assert_allclose(g.evaluate(ctrl, 0.0), [10, 10])
    np.testing.assert_allclose(g.evaluate(ctrl, 0.5), [50, 70])
    np.testing.assert_allclose(g.evaluate(ctrl, 1.0), [90, 10])
    np.testing.assert_allclose(g.to_power(ctrl), ARCH_POWER, atol=1e-12)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES)
def test_power_form_matches_bernstein(ctrl):
    coeffs = g.to_power(ctrl)
    t = np.linspace(0, 1, 11)
    poly = (
        coeffs[0] * t[:, None] ** 3 + coeffs[1] * t[:, None] ** 2 + coeffs[2] * t[:, None] + coeffs[3]
    )
    np.testing.assert_allclose(g.evaluate(ctrl, t), poly, atol=1e-9)
    np.testing.assert_allclose(g.from_power(coeffs), ctrl, atol=1e-9)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES[:5])
def test_derivatives_match_finite_differences(ctrl):
    t = np.linspace(0.05, 0.95, 7)
    h = 1e-6
    fd1 = (g.evaluate(ctrl, t + h) - g.evaluate(ctrl, t - h)) / (2 * h)
    fd2 = (g.derivative(ctrl, t + h) - g.derivative(ctrl, t - h)) / (2 * h)
    np.testing.assert_allclose(g.derivative(ctrl, t), fd1, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(g.second_derivative(ctrl, t), fd2, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES[:5])
def test_split_reproduces_both_halves(ctrl):
    t = 0.3
    left, right = g.split(ctrl, t)
    s = np.linspace(0, 1, 9)
    np.testing.assert_allclose(g.evaluate(left, s), g.evaluate(ctrl, s * t), atol=1e-9)
    np.testing.assert_allclose(g.evaluate(right, s), g.evaluate(ctrl, t + s * (1 - t)), atol=1e-9)


def test_subcurve():
    ctrl = RANDOM_CURVES[0]
    piece = g.subcurve(ctrl, 0.2, 0.7)
    s = np.linspace(0, 1, 9)
    np.testing.assert_allclose(g.evaluate(piece, s), g.evaluate(ctrl, 0.2 + 0.5 * s), atol=1e-9)


def test_reverse_and_affine():
    ctrl = RANDOM_CURVES[1]
    s = np.linspace(0, 1, 5)
    np.testing.assert_allclose(g.evaluate(g.reverse(ctrl), s), g.evaluate(ctrl, 1 - s), atol=1e-9)
    m = np.array([[0.0, -2.0, 5.0], [1.5, 0.5, -3.0]])
    moved = g.affine(ctrl, m)
    expect = g.evaluate(ctrl, s) @ m[:, :2].T + m[:, 2]
    np.testing.assert_allclose(g.evaluate(moved, s), expect, atol=1e-9)


def test_flip_y_is_an_involution():
    ctrl = RANDOM_CURVES[2]
    flipped = g.flip_y(ctrl, 100)
    np.testing.assert_allclose(flipped[:, 0], ctrl[:, 0])
    np.testing.assert_allclose(flipped[:, 1], 100 - ctrl[:, 1])
    np.testing.assert_allclose(g.flip_y(flipped, 100), ctrl)


def test_line_is_straight_with_uniform_speed():
    ctrl = g.line([0, 0], [30, 40])
    np.testing.assert_allclose(g.evaluate(ctrl, 0.5), [15, 20])
    np.testing.assert_allclose(np.linalg.norm(g.derivative(ctrl, [0, 0.4, 1]), axis=1), 50)
    assert g.arc_length(ctrl) == pytest.approx(50.0, rel=1e-12)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES)
def test_bounding_box_is_tight(ctrl):
    box = g.bounding_box(ctrl)
    pts = g.evaluate(ctrl, np.linspace(0, 1, 20001))
    np.testing.assert_allclose(box[:2], pts.min(axis=0), atol=1e-6)
    np.testing.assert_allclose(box[2:], pts.max(axis=0), atol=1e-6)


def test_arc_length_of_quarter_circle():
    # standard cubic approximation of a unit quarter circle (radius 100)
    k = 0.5522847498 * 100
    ctrl = [[100, 0], [100, k], [k, 100], [0, 100]]
    assert g.arc_length(ctrl) == pytest.approx(np.pi * 50, rel=1e-3)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES)
def test_flatten_respects_tolerance(ctrl):
    tol = 0.1
    poly = g.flatten(ctrl, tol)
    np.testing.assert_array_equal(poly[0], ctrl[0])
    np.testing.assert_array_equal(poly[-1], ctrl[3])
    dense = g.evaluate(ctrl, np.linspace(0, 1, 4001))
    dist = np.min(g.point_segment_distance(dense[:, None], poly[None, :-1], poly[None, 1:]), axis=1)
    assert dist.max() <= tol + 1e-9


def test_distance_to_points():
    ctrl = g.line([0, 0], [10, 0])
    d = g.distance_to_points(ctrl, [[5, 3], [-4, 3], [12, 0], [7, 0]])
    np.testing.assert_allclose(d, [3, 5, 2, 0], atol=1e-9)


@pytest.mark.parametrize("ctrl", RANDOM_CURVES[:8])
def test_closest_point_agrees_with_dense_search(ctrl):
    q = np.array([50.0, 50.0])
    t, d = g.closest_point(ctrl, q)
    dense = np.linalg.norm(g.evaluate(ctrl, np.linspace(0, 1, 200001)) - q, axis=1)
    assert d == pytest.approx(dense.min(), abs=1e-4)
    assert d == pytest.approx(np.linalg.norm(g.evaluate(ctrl, t) - q), abs=1e-9)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        g.as_ctrl(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        g.as_ctrl([[0, 0], [1, np.nan], [2, 2], [3, 3]])
    with pytest.raises(ValueError):
        g.split(RANDOM_CURVES[0], 1.5)

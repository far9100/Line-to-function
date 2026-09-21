"""Tracing the ink's outline into closed loops (line2func.boundary)."""

import numpy as np

from line2func import boundary
from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart

SIZE = 200


def _mask(curves, width: float) -> np.ndarray:
    cs = CurveSet(SIZE, SIZE, [Curve(c, stroke=i) for i, c in enumerate(curves)])
    return (1.0 - render_lineart(cs, SIZE, SIZE, line_width=width) / 255.0) > 0.5


def test_one_loop_per_piece_of_ink_and_one_more_per_hole():
    two = _mask([g.line([20, 40], [180, 40]), g.line([20, 160], [180, 160])], 7.0)
    assert len(boundary.contours(two)) == 2
    ring = np.zeros((SIZE, SIZE), dtype=bool)
    ring[40:160, 40:160] = True
    ring[70:130, 70:130] = False
    assert len(boundary.contours(ring)) == 2  # the outside and the hole
    assert not boundary.contours(np.zeros((20, 20), dtype=bool))


def test_a_thin_line_is_one_loop_up_one_side_and_down_the_other():
    """A 2 px line has no inside, so a skeleton walk cannot trace its outline; this one can."""
    loops = boundary.contours(_mask([g.line([20, 100], [180, 100])], 2.0))
    assert len(loops) == 1
    assert len(loops[0]) > 2 * 150  # both sides of a 160 px line


def test_every_loop_comes_back_closed():
    """The reason this module exists: an open chain gets closed by a chord, inventing area."""
    ring = np.zeros((SIZE, SIZE), dtype=bool)
    ring[40:160, 40:160] = True
    ring[70:130, 70:130] = False
    for loop in boundary.contours(ring) + boundary.contours(_mask(
            [g.line([20, 30], [180, 170]), g.line([20, 170], [180, 30])], 7.0)):
        # consecutive points are neighbours, and so are the last and the first
        step = np.abs(np.diff(np.vstack([loop, loop[:1]]), axis=0)).max(axis=1)
        assert step.max() <= 1.0

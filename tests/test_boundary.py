"""The ink's outline judges the skeleton without ever looking at an angle (line2func.boundary)."""

import itertools

import numpy as np
import pytest

from line2func import baseline, boundary
from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart

SIZE = 200


def _mask(curves, width: float) -> np.ndarray:
    cs = CurveSet(SIZE, SIZE, [Curve(c, stroke=i) for i, c in enumerate(curves)])
    return (1.0 - render_lineart(cs, SIZE, SIZE, line_width=width) / 255.0) > 0.5


def _edges(mask: np.ndarray):
    """Points of every skeleton edge, longest first."""
    graph = baseline._trace_graph(baseline.thin(mask))
    return sorted((e.pts for e in graph.edges if e.alive), key=len, reverse=True)


def _split_edges(mask: np.ndarray):
    """``(arms, middles)``: skeleton edges with a loose end, and those between two junctions."""
    graph = baseline._trace_graph(baseline.thin(mask))
    degree = [len(x) for x in graph.incidence()]
    arms, middles = [], []
    for e in graph.edges:
        if not e.alive or len(e.pts) < 5:
            continue
        (middles if min(degree[e.a], degree[e.b]) >= 3 else arms).append(e.pts)
    return arms, middles


def _arms(mask: np.ndarray, width: float):
    """At the first junction: each arm's ``(direction, left chains, right chains)``."""
    bnd = boundary.outline(mask)
    graph = baseline._trace_graph(baseline.thin(mask))
    inc = graph.incidence()
    node = next(v for v, lst in enumerate(inc) if len(lst) >= 3)
    out = []
    for eid, end in inc[node]:
        arm = graph.edges[eid].from_end(end)
        window = arm[int(width): int(width) + 10]
        if len(window) < 3:
            window = arm[:3]
        r = boundary.trust(bnd, window)
        d = window[-1] - window[0]
        out.append((d / max(float(np.linalg.norm(d)), 1e-9),
                    set(r["chain_left"].tolist()) - {-1}, set(r["chain_right"].tolist()) - {-1}))
    return out


def _continues(a, b) -> bool:
    """Do two arms leaving one node share an outline chain, as one line through it would?

    They point away from each other, so the side that is one arm's left is the
    other's right.
    """
    _, la, ra = a
    _, lb, rb = b
    return bool((la & rb) or (ra & lb))


# ---------------------------------------------------------------------------
# Tracing the outline
# ---------------------------------------------------------------------------


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


def test_the_outline_of_a_crossing_is_cut_at_its_notches_and_caps():
    x = _mask([g.line([20, 30], [180, 170]), g.line([20, 170], [180, 30])], 7.0)
    bnd = boundary.outline(x)
    assert int(bnd.chain.max()) + 1 == 8  # four notches and four line ends


def test_a_single_pixel_step_does_not_cut_the_outline():
    """A round cap meeting a bar leaves a 1 px bump: an angle calls it a corner, a chord does not."""
    h = _mask([g.line([20, 60], [180, 60]), g.line([20, 140], [180, 140]), g.line([100, 60], [100, 140])], 7.0)
    bnd = boundary.outline(h)
    top = bnd.pts[:, 1] < 58  # the top edge of the upper bar, straight all the way across
    assert len(set(bnd.chain[top].tolist())) == 1


# ---------------------------------------------------------------------------
# Judging the skeleton
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2.0, 7.0, 14.0])
def test_a_straight_line_is_trusted_at_every_width(width):
    mask = _mask([g.line([20, 100], [180, 100])], width)
    bnd = boundary.outline(mask)
    d = boundary.trust(bnd, _edges(mask)[0])["d"]
    _, mean, bad = boundary.summarize(d)
    assert mean > 0.99 and bad < 0.1


def test_the_tails_of_a_shallow_crossing_are_trusted_least():
    """Two wide lines crossing at a shallow angle thin into two junctions with short tails
    hanging off them. The tails are what thinning invented, and they are the part the outline
    trusts least - it is asked no angle, which is the point, because here the angles are wrong.

    The piece *between* the two junctions is not caught this way: at a shallow crossing it runs
    nearly parallel to both edges of the ink, so it agrees with them. That case needs the
    crossing decision, not the outline.
    """
    mask = _mask([g.line([20, 90], [180, 110]), g.line([20, 110], [180, 80])], 14.0)
    bnd = boundary.outline(mask)
    runs = _edges(mask)
    bad = lambda pts: boundary.summarize(boundary.trust(bnd, pts)["d"])[2]  # noqa: E731
    arms = [bad(p) for p in runs[:2]]  # the two long arms
    tails = [bad(p) for p in runs[-2:]]  # the two short tails
    assert min(tails) > 1.5 * max(arms), (arms, tails)


def test_a_junction_the_thinning_shrank_to_a_stub_is_not_trusted_at_all():
    mask = _mask([g.line([20, 30], [180, 170]), g.line([20, 170], [180, 30])], 14.0)
    bnd = boundary.outline(mask)
    arm_runs, middle_runs = _split_edges(mask)
    assert not middle_runs  # the stub is shorter than 5 points
    assert max(boundary.summarize(boundary.trust(bnd, p)["d"])[2] for p in arm_runs) < 0.15


def test_two_sides_are_told_apart_only_when_the_line_is_wide_enough():
    thin = _mask([g.line([20, 100], [180, 100])], 2.0)
    wide = _mask([g.line([20, 100], [180, 100])], 7.0)
    for mask, expected in ((thin, True), (wide, False)):
        bnd = boundary.outline(mask)
        degenerate = boundary.trust(bnd, _edges(mask)[0])["degenerate"]
        assert degenerate.mean() > 0.9 if expected else degenerate.mean() < 0.1


@pytest.mark.parametrize("curves,label", [
    ([g.line([20, 100], [180, 100]), g.line([100, 100], [100, 180])], "T"),
    ([g.line([20, 60], [180, 60]), g.line([20, 140], [180, 140]), g.line([100, 60], [100, 140])], "H"),
])
def test_only_the_line_that_really_continues_shares_an_outline(curves, label):
    """The two halves of the bar share one outline chain over the junction; the stem shares
    none with either. No angle is measured, which is the point: at a wide junction the
    skeleton's angles are wrong, and the outline is not."""
    arms = _arms(_mask(curves, 7.0), 7.0)
    shared = {(i, j): _continues(a, b)
              for (i, a), (j, b) in itertools.combinations(list(enumerate(arms)), 2)}
    straight = [pair for pair, ok in shared.items()
                if ok and np.degrees(np.arccos(np.clip(arms[pair[0]][0] @ -arms[pair[1]][0], -1, 1))) < 20]
    assert len(straight) == 1, (label, shared)
    assert sum(shared.values()) == 1, (label, shared)


def test_an_empty_mask_gives_an_empty_boundary_and_no_error():
    bnd = boundary.outline(np.zeros((20, 20), dtype=bool))
    assert len(bnd) == 0
    r = boundary.trust(bnd, np.array([[5.0, 5.0], [6.0, 6.0]]))
    assert len(r["d"]) == 0
    assert boundary.summarize(np.zeros(0)) == (0.0, 0.0, 1.0)

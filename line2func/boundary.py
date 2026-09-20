"""The ink's outline, as a second opinion on the skeleton (numpy and scipy only).

Thinning distorts the skeleton wherever lines meet, and the thicker the lines the
worse it is: where two wide strokes cross, the skeleton's angles are simply wrong,
so every feature measured from them is wrong together. The *outline* of the ink is
exact at any width, though, so it can judge the skeleton.

For a run of skeleton points this module measures how well their own direction
agrees with the outline on either side of them
(:func:`trust`, ``D_skeleton`` of Zhang et al., CGF 2022), and which outline chain
each side belongs to. Two arms leaving one junction can then be asked whether
their outsides are the same outline chain - evidence that two lines continue each
other which does not look at angles at all, and so survives exactly where the
angles do not.

Unlike Zhang's method this is only evidence: line2func keeps its own skeleton and
its own line-width measurement, and the outline is asked how far to trust them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

NEIGHBOURS = 8  # boundary points looked at per query, to find one on each side
TRUSTED = 0.99  # a skeleton point agreeing with the outline this well is undistorted (Zhang 2022)
MIN_OFFSET = 1.5  # the outline is cut where it strays at least this far from its own chord, px
THIN = 2.0  # closer than this, the two sides of a line are the same pixels: no outline evidence
# the eight neighbours, clockwise from east, as (dy, dx)
_RING = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def contours(mask: np.ndarray) -> list[np.ndarray]:
    """Closed outline loops of an ink mask, each in order, the background always on the same side.

    Moore-neighbour following: from an outline pixel, the eight neighbours are
    tried clockwise from wherever the walk came in, and the first one on the ink
    is stepped to. It closes on itself, and unlike a skeleton walk it does not
    mind an outline that is only one pixel wide - both sides of a thin line are
    then stretches of one loop, which :func:`from_chains` cuts apart at the sharp
    turns around the line's ends.
    """
    mask = np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = mask.shape
    flat = mask.ravel()
    steps = np.array([dy * w + dx for dy, dx in _RING], dtype=np.int64)
    back = np.array([(k + 4) % 8 for k in range(8)])  # the way back, once a step has been taken
    # a loop starts at ink whose western neighbour is paper, so the walk knows where it came
    # from. In raster order that is each piece's leftmost ink, and the ink just right of a hole
    west_paper = np.zeros_like(mask)
    west_paper[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
    starts = np.nonzero(west_paper.ravel())[0]
    seen = np.zeros(h * w, dtype=bool)
    loops = []
    for start in starts.tolist():
        if seen[start]:
            continue
        # the walk enters from the west, which is background for the first outline pixel of a row
        p, entry, path = start, 4, [start]
        # the walk is decided by where it stands and where it came from, so it has closed the
        # loop as soon as that pair repeats. A thin line, whose two sides are one loop, is
        # walked up one side and down the other, and only then repeats.
        states = {p * 8 + entry}
        seen[start] = True
        while True:
            nxt = -1
            for k in range(1, 9):
                d = (entry + k) % 8
                q = p + steps[d]
                if flat[q]:
                    nxt, entry = q, back[d]
                    break
            if nxt < 0:
                break
            state = nxt * 8 + entry
            if state in states:
                break
            states.add(state)
            p = nxt
            seen[p] = True
            path.append(p)
        if len(path) >= 3:
            ys, xs = np.divmod(np.array(path, dtype=np.int64), w)
            loops.append(np.stack([xs - 1 + 0.5, ys - 1 + 0.5], axis=1).astype(np.float64))
    return loops


@dataclass
class Boundary:
    """The ink's outline: every outline pixel, in chain order, with a tangent.

    ``chain`` tells which chain a point belongs to, so two points can be asked
    whether they lie on the same one; ``arc`` is the distance along that chain,
    so how far apart they are along it can be measured.
    """

    pts: np.ndarray  # (n, 2) xy, pixel centres
    chain: np.ndarray  # (n,) chain index
    arc: np.ndarray  # (n,) arc length from the start of its chain
    tangent: np.ndarray  # (n, 2) unit tangent, in the chain's own direction
    tree: cKDTree

    def __len__(self) -> int:
        return len(self.pts)

    def sides(self, pts: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Indices of the nearest outline point on each side of ``pts`` (-1 where there is none).

        "Side" is which way the outline point lies across ``dirs``, so for a point
        on a line the two are its two edges. Returns ``(left, right)``.
        """
        n = len(pts)
        if n == 0 or len(self.pts) == 0:
            return np.full(n, -1), np.full(n, -1)
        k = min(NEIGHBOURS, len(self.pts))
        _, idx = self.tree.query(pts, k=k)
        idx = np.atleast_2d(idx.reshape(n, -1))
        v = self.pts[idx] - pts[:, None, :]  # (n, k, 2)
        cross = v[:, :, 0] * dirs[:, None, 1] - v[:, :, 1] * dirs[:, None, 0]
        out = []
        for want in (cross > 0, cross < 0):  # neighbours come back nearest first
            first = np.argmax(want, axis=1)
            out.append(np.where(want.any(axis=1), idx[np.arange(n), first], -1))
        return out[0], out[1]


def _tangents(pts: np.ndarray, span: int, closed: bool) -> np.ndarray:
    """Unit direction at every point of one chain, over ``+/- span`` points.

    The same measure the tracer uses along a stroke: a difference over a few
    points, not a derivative, so a pixel staircase does not show up as a wobble.
    """
    n = len(pts)
    if n == 1:
        return np.array([[1.0, 0.0]])
    i = np.arange(n)
    if closed:
        v = pts[(i + span) % n] - pts[(i - span) % n]
    else:
        v = pts[np.minimum(i + span, n - 1)] - pts[np.maximum(i - span, 0)]
    length = np.hypot(v[:, 0], v[:, 1])
    flat = length < 1e-9
    if flat.any():  # a point whose two neighbours coincide: fall back to the whole chain
        whole = pts[-1] - pts[0]
        fallback = whole / max(float(np.hypot(whole[0], whole[1])), 1e-9)
        v = np.where(flat[:, None], fallback, v)
        length = np.maximum(np.hypot(v[:, 0], v[:, 1]), 1e-9)
    return v / length[:, None]


def _cuts(pts: np.ndarray, span: int, min_offset: float) -> np.ndarray:
    """Where one outline loop bends away from its own chord: the ends of its smooth stretches.

    Zhang cuts the outline at its sharp turns, so that each piece is one line's
    own edge: the turns are where two lines meet (a concave notch) and where a
    line ends (a cap), which is exactly where the outline stops belonging to one
    stroke. The measure is how far the outline strays from the chord across it,
    in pixels, which an angle cannot do here: a single-pixel staircase step turns
    by 45 degrees or more and means nothing, while it never strays a pixel and a
    real corner strays several.
    """
    n = len(pts)
    if n < 2 * span + 2:
        return np.zeros(0, dtype=np.int64)
    i = np.arange(n)
    before, after = pts[(i - span) % n], pts[(i + span) % n]
    chord = after - before
    length = np.maximum(np.hypot(chord[:, 0], chord[:, 1]), 1e-12)
    v = pts[i] - before
    offset = np.abs(v[:, 0] * chord[:, 1] - v[:, 1] * chord[:, 0]) / length
    sharp = offset >= min_offset
    if not sharp.any():
        return np.zeros(0, dtype=np.int64)
    # one cut per run of sharp points, at the point that strays furthest, so a rounded
    # corner is cut once and at its tip
    starts = np.nonzero(sharp & ~np.roll(sharp, 1))[0]
    cuts = []
    for s in starts.tolist():
        run = [s]
        while sharp[(run[-1] + 1) % n] and len(run) < n:
            run.append((run[-1] + 1) % n)
        cuts.append(run[int(np.argmax(offset[run]))])
    return np.unique(cuts)


def from_chains(chains: list[np.ndarray], span: int = 4, min_offset: float = MIN_OFFSET) -> Boundary:
    """A :class:`Boundary` from closed outline loops, each cut into its smooth stretches."""
    pts, chain, arc, tangent = [], [], [], []
    k = 0
    for loop in chains:
        loop = np.asarray(loop, dtype=np.float64).reshape(-1, 2)
        if not len(loop):
            continue
        n = len(loop)
        cut = _cuts(loop, span, min_offset)
        if not len(cut):
            pieces = [(loop, True)]  # a smooth loop all the way round, such as a circle
        else:
            # start the loop at its first sharp turn, then cut at each of the others; a corner
            # point belongs to both of the stretches it joins
            rolled = np.roll(loop, -int(cut[0]), axis=0)
            at = [0] + sorted(int(c - cut[0]) % n for c in cut[1:]) + [n - 1]
            pieces = [(rolled[s: e + 1], False) for s, e in zip(at[:-1], at[1:])]
        for p, closed in pieces:
            if len(p) < 2:
                continue
            pts.append(p)
            chain.append(np.full(len(p), k, dtype=np.int64))
            arc.append(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]))
            tangent.append(_tangents(p, span, closed))
            k += 1
    if not pts:
        empty = np.zeros((0, 2))
        return Boundary(empty, np.zeros(0, np.int64), np.zeros(0), empty, cKDTree(np.zeros((1, 2))))
    pts = np.vstack(pts)
    return Boundary(pts, np.concatenate(chain), np.concatenate(arc), np.vstack(tangent), cKDTree(pts))


def outline(mask: np.ndarray, span: int = 4, min_offset: float = MIN_OFFSET) -> Boundary:
    """The outline of an ink mask, traced into loops and cut into smooth stretches."""
    return from_chains(contours(mask), span, min_offset)


def trust(bnd: Boundary, pts: np.ndarray, closed: bool = False, span: int = 3) -> dict:
    """How far the outline agrees with a run of skeleton points ``pts``.

    ``d`` is Zhang's ``D_skeleton`` per point: the cosine between the skeleton's
    own direction and the better-agreeing of the two outline tangents flanking
    it. It is 1 along a clean line and falls wherever thinning has bent the
    skeleton away from the ink it is supposed to run down the middle of.

    Also returns the flanking outline points and their chains, so two arms can be
    asked whether they share one outline (``left``/``right`` are indices into
    ``bnd``, -1 where no outline point lies on that side).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    empty = np.zeros(0, np.int64)
    if not len(pts) or not len(bnd):
        return {"d": np.zeros(0), "left": empty, "right": empty, "chain_left": empty, "chain_right": empty,
                "degenerate": np.zeros(0, bool)}
    dirs = _tangents(pts, span, closed)
    left, right = bnd.sides(pts, dirs)
    d = np.zeros(len(pts))
    for side in (left, right):
        ok = side >= 0
        if ok.any():
            cos = np.abs(np.sum(dirs[ok] * bnd.tangent[side[ok]], axis=1))
            d[ok] = np.maximum(d[ok], np.minimum(cos, 1.0))
    chain_of = lambda side: np.where(side >= 0, bnd.chain[np.maximum(side, 0)], -1)  # noqa: E731
    # a line thinner than THIN has one outline, not two: its "sides" are the same pixels,
    # so which chain each side is on says nothing
    both = (left >= 0) & (right >= 0)
    apart = np.zeros(len(pts))
    apart[both] = np.linalg.norm(bnd.pts[left[both]] - bnd.pts[right[both]], axis=1)
    return {"d": d, "left": left, "right": right,
            "chain_left": chain_of(left), "chain_right": chain_of(right),
            "degenerate": ~both | (apart < THIN)}


def summarize(d: np.ndarray) -> tuple[float, float, float]:
    """``(min, mean, share below TRUSTED)`` of a run of ``D_skeleton`` values.

    The share is the useful one on its own: it says how much of this stretch of
    skeleton is distorted, which is the same as saying how far the angles
    measured on it can be believed.
    """
    if not len(d):
        return 0.0, 0.0, 1.0
    return float(d.min()), float(d.mean()), float(np.mean(d < TRUSTED))

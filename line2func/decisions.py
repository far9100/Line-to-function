"""The decisions of the baseline tracer, as scores.

The baseline engine makes all the geometry: skeleton, stroke graph, curves. At
four places it must choose between candidates; the angle rules, or a learned
scorer (:mod:`line2func.decision_model`), decide:

* **crossing** - are two junctions joined by a short edge one shallow X
  crossing (merge them) or two T-junctions / the bar of an H (keep them)?
* **gap** - which pairs of stroke ends are one stroke with a break in it?
* **junction** - at a node with 3+ arms, which arms continue each other?
* **corner** - where does a stroke turn sharply enough to split the fit?

:class:`Scorer` states the rules as scores, and solvers turn the scores into
decisions: :func:`greedy_pairs`, :func:`exact_matching` and
:func:`pick_peaks`. Other scorers, a learned one or an oracle that knows the
ground truth, override the scores. The candidate generation and the solvers
stay the same.

With the rules the engine's output is pinned bit for bit by golden digests
(``tests/test_golden.py``).

Only numpy and scipy are used here: no PyTorch at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _direction_from(pts: np.ndarray, origin: np.ndarray, reach: float) -> np.ndarray:
    """Unit vector from ``origin`` toward the first point at least ``reach`` away."""
    d = np.linalg.norm(pts - origin, axis=1)
    far = np.nonzero(d >= reach)[0]
    q = pts[far[0]] if len(far) else pts[-1]
    v = q - origin
    n = np.hypot(v[0], v[1])
    if n < 1e-9:
        v = pts[-1] - pts[0]
        n = np.hypot(v[0], v[1])
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


# ---------------------------------------------------------------------------
# Context and candidate generation limits
# ---------------------------------------------------------------------------


@dataclass
class DecisionContext:
    """What a scorer may look at besides the skeleton graph (built only when a scorer needs it)."""

    ink: np.ndarray  # the ink map given to vectorize (1 = line)
    mask: np.ndarray  # final ink mask (faint strokes added, specks removed, pinholes filled)
    line_dist: np.ndarray  # distance transform of the thinned lines (filled areas removed): 2 * dist - 1 ~ width
    threshold: float
    line_w: float
    radius: float
    _ink_p90: float | None = field(default=None, repr=False)

    @property
    def ink_p90(self) -> float:
        """Typical darkness of the drawing's lines (90th percentile of the ink on the mask)."""
        if self._ink_p90 is None:
            on = self.ink[self.mask]
            self._ink_p90 = float(np.percentile(on, 90)) if on.size else 1.0
        return max(self._ink_p90, 1e-3)

    def px(self, arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """``arr`` at the pixels under ``pts`` (skeleton coordinates: pixel centres at +0.5)."""
        return arr[self.yx(pts)]

    def yx(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row and column indices of the pixels under ``pts`` (to read several maps at once)."""
        h, w = self.ink.shape
        x = np.clip((pts[:, 0] - 0.5).round().astype(np.int64), 0, w - 1)
        y = np.clip((pts[:, 1] - 0.5).round().astype(np.int64), 0, h - 1)
        return y, x

    def bilinear(self, arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        from scipy import ndimage

        return ndimage.map_coordinates(arr, [pts[:, 1] - 0.5, pts[:, 0] - 0.5], order=1, mode="nearest")


@dataclass(frozen=True)
class Limits:
    """How widely candidates are generated. The rules use the default limits (all 1.0), the learned scorer
    :data:`WIDE_LIMITS`."""

    gap_radius_scale: float = 1.0  # stroke ends are gap candidates within max_gap * scale
    gap_angle_gate: float | None = None  # hard angle gate for gap candidates (None: the rule's gap_angle)
    crossing_len_scale: float = 1.0  # junction pairs are shallow-crossing candidates up to (12r * scale + 6)
    min_corner_turn: float | None = None  # corner candidates turn at least this much (None: corner_angle)


RULE_LIMITS = Limits()
WIDE_LIMITS = Limits(gap_radius_scale=2.0, gap_angle_gate=80.0, crossing_len_scale=2.5, min_corner_turn=15.0)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class CrossingCand:
    """An edge between two degree-3 junctions: one shallow X crossing, or two T's?"""

    eid: int
    u: int  # the edge's nodes
    v: int
    length: float
    rule: bool  # the angle rule's answer (_looks_like_crossing)
    mid: np.ndarray  # the edge's points, u -> v
    arms_u: list = field(default_factory=list)  # [(edge key, points from u outward)] for the 2 other arms
    arms_v: list = field(default_factory=list)


@dataclass
class GapCand:
    """Two stroke ends (skeleton tips): one stroke with a break between them?"""

    i: int  # tip indices
    j: int
    key_i: tuple  # (edge id, end)
    key_j: tuple
    tip_i: np.ndarray
    tip_j: np.ndarray
    out_i: np.ndarray  # outward directions at the tips
    out_j: np.ndarray
    dist: float
    ai: float  # angle between each end's outward direction and the gap, degrees
    aj: float
    rule_ok: bool  # passes the rule's gates
    cost: float  # the rule's cost: dist * (1 + (ai + aj) / 90)
    arm_i: np.ndarray | None = None  # skeleton points from the tip inward
    arm_j: np.ndarray | None = None


@dataclass
class JunctionCand:
    """A node with 3+ arms: which pairs of arms continue each other?"""

    node: int
    keys: list  # [(edge id, end)] per arm
    center: np.ndarray
    spread: float
    reach: float  # look-ahead used for the rule's arm directions
    dirs: list  # the rule's arm directions (unit vectors)
    pairs: list  # [(i, j)], i < j, all pairs of arms
    bends: np.ndarray  # the rule's bend per pair: angle(dir_i, -dir_j), degrees
    arms: list = field(default_factory=list)  # points of each arm, from the node outward
    far_degree: list = field(default_factory=list)  # degree of the node at each arm's other end


@dataclass
class CornerCand:
    """A stroke's resampled points (1 px) and its turning profile."""

    pts: np.ndarray
    closed: bool
    index: np.ndarray  # sample indices that have a turn value
    turn: np.ndarray  # turning angle there, degrees (the rule's measure)
    skip: int
    k: int
    seams: list = field(default_factory=list)  # [(arc position, "junction" | "gap")] where edges were joined


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer:
    """The angle rules, stated as scores: the baseline engine's default scorer (``pipeline.trace`` uses the
    learned one, :mod:`line2func.decision_model`).

    Higher scores are better; ``-inf`` forbids a candidate.
    """

    limits = RULE_LIMITS
    needs_features = False
    junction_policy = "greedy"  # "greedy" (the rule) or "exact" (best matching per node)

    def begin(self, ctx: DecisionContext | None) -> None:
        """Called once per tracing, before any decision (``ctx`` is None unless ``needs_features``)."""

    def report(self) -> dict | None:
        """Anything worth recording in the result's ``meta["decisions"]`` (None: nothing)."""
        return None

    def crossing(self, cands: list[CrossingCand]) -> list[bool]:
        """Merge each candidate's two junctions into one crossing?"""
        return [c.rule for c in cands]

    def gap_scores(self, cands: list[GapCand]) -> np.ndarray:
        return np.array([-c.cost if c.rule_ok else -np.inf for c in cands], dtype=np.float64)

    def junction_scores(self, cands: list[JunctionCand], max_bend: float) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Per node: (score per arm pair, score for each arm to end here, or None)."""
        return [(np.where(c.bends <= max_bend, -c.bends, -np.inf), None) for c in cands]

    def corner_keys(self, cand: CornerCand, min_angle: float) -> tuple[np.ndarray, float]:
        """(ranking key per candidate sample, accept threshold) for :func:`pick_peaks`."""
        return cand.turn, min_angle


RULES = Scorer()


def load_scorer(name: str | None) -> Scorer:
    """The scorer called ``name``: ``None`` or "rules" (the angle rules), "learned" (the bundled
    weights) or the path to a ``decisions.npz`` (:mod:`line2func.decision_model`)."""
    if name in (None, "rules"):
        return RULES
    from line2func.decision_model import load

    return load(name)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def greedy_pairs(scores, pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Accept pairs in order of decreasing score (ties: smaller i, then j), each item at most once."""
    order = sorted((-float(s), i, j) for s, (i, j) in zip(scores, pairs) if s != -np.inf)
    used: set[int] = set()
    out = []
    for _, i, j in order:
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append((i, j))
    return out


def exact_matching(z, pairs: list[tuple[int, int]], e, k: int) -> list[tuple[int, int]]:
    """The matching of ``k`` arms with the largest total: paired arms score ``z``, unpaired ones ``e``.

    Exact (bitmask dynamic programming) up to 12 arms, greedy above. ``-inf``
    pairs are never chosen. On a tie an arm stays unpaired, and among pairings
    the partner with the lowest index wins.
    """
    if k > 12:
        return greedy_pairs(z, pairs)
    zmap = {pair: float(s) for pair, s in zip(pairs, z) if s != -np.inf}
    e = [0.0] * k if e is None else [float(x) for x in e]
    memo: dict[int, tuple[float, tuple]] = {0: (0.0, ())}

    def best(mask: int) -> tuple[float, tuple]:
        if mask in memo:
            return memo[mask]
        i = (mask & -mask).bit_length() - 1  # lowest remaining arm
        rest = mask & ~(1 << i)
        score, chosen = best(rest)
        top = (score + e[i], chosen)
        for j in range(i + 1, k):
            if rest >> j & 1 and (i, j) in zmap:
                s, c = best(rest & ~(1 << j))
                if s + zmap[(i, j)] > top[0]:
                    top = (s + zmap[(i, j)], ((i, j),) + c)
        memo[mask] = top
        return top

    return sorted(best((1 << k) - 1)[1])


def pick_peaks(index: np.ndarray, key: np.ndarray, threshold: float, n: int, closed: bool, reach: int) -> list[int]:
    """Samples with ``key >= threshold``, strongest first, at least ``reach`` samples apart (sorted)."""
    found: list[int] = []
    for j in np.argsort(-key):
        if key[j] < threshold:
            break
        idx = int(index[j])
        if all(min(abs(idx - c), n - abs(idx - c) if closed else n) > reach for c in found):
            found.append(idx)
    return sorted(found)

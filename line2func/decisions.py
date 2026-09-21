"""The decisions of the baseline tracer, as scores.

The baseline engine makes all the geometry: skeleton, stroke graph, curves. At
four places it must choose between candidates, and the angle rules decide:

* **crossing** - are two junctions joined by a short edge one shallow X
  crossing (merge them) or two T-junctions / the bar of an H (keep them)?
* **gap** - which pairs of stroke ends are one stroke with a break in it?
* **junction** - at a node with 3+ arms, which arms continue each other?
* **corner** - where does a stroke turn sharply enough to split the fit?

Each rule is stated as a score, and solvers turn the scores into decisions:
:func:`greedy_pairs` and :func:`pick_peaks`.

There was once a learned scorer here as well, and an abstraction for swapping
one in. Both are gone: it was measurably better and not visibly so, for a fifth
of the tracing time and a third of the browser download. ``docs/details.md``
keeps the measurements and the reasoning.

The engine's output is pinned bit for bit by golden digests
(``tests/test_golden.py``).

Only numpy is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

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
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class GapCand:
    """Two stroke ends (skeleton tips): one stroke with a break between them?"""

    i: int  # tip indices
    j: int
    cost: float  # the rule's cost: dist * (1 + (ai + aj) / 90)


@dataclass
class JunctionCand:
    """A node with 3+ arms: which pairs of arms continue each other?"""

    keys: list  # [(edge id, end)] per arm
    pairs: list  # [(i, j)], i < j, all pairs of arms
    bends: np.ndarray  # the rule's bend per pair: angle(dir_i, -dir_j), degrees


# ---------------------------------------------------------------------------
# The rules, as scores (higher is better; -inf forbids a candidate)
# ---------------------------------------------------------------------------


def gap_scores(cands: list[GapCand]) -> np.ndarray:
    """Score each gap candidate; the cheaper the break, the better. Every candidate passed the rule's gates."""
    return np.array([-c.cost for c in cands], dtype=np.float64)


def junction_scores(cands: list[JunctionCand], max_bend: float) -> list[np.ndarray]:
    """Per node, a score for every pair of arms: the straighter the pair, the better."""
    return [np.where(c.bends <= max_bend, -c.bends, -np.inf) for c in cands]


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

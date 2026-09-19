"""A chosen number of curves: merge the neighbouring pieces that cost the least.

The fitting tolerance sets the curve count only indirectly. For an exact count
``n``, ``pipeline.trace(curve_count=n)`` traces finely enough to get at least
``n`` curves; :func:`reduce_to` then merges, one pair at a time, the two
neighbouring pieces of a stroke whose single-cubic replacement deviates least
from what they drew. Each merge is fitted to the original samples of all the
pieces it replaces, so errors do not pile up. Intricate parts keep their
pieces and smooth parts merge first. Joints keep their position and tangent
direction, so strokes stay connected and corners stay sharp until the budget
forces them round. If ``n`` is below one curve per connected run of pieces
(roughly one per stroke), the shortest runs are dropped.
"""

from __future__ import annotations

import heapq
from itertools import count

import numpy as np

from line2func.curves import Curve, CurveSet
from line2func.fit import fit_cubic
from line2func.geometry import arc_length, evaluate


def _samples(ctrl: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    k = max(4, int(np.ceil(arc_length(ctrl) / spacing)) + 1)
    u = np.linspace(0.0, 1.0, k)
    return evaluate(ctrl, u), u


def _direction(p: np.ndarray, *toward: np.ndarray) -> np.ndarray | None:
    """Unit vector from ``p`` to the first of ``toward`` that is not on top of it."""
    for q in toward:
        v = q - p
        n = float(np.hypot(v[0], v[1]))
        if n > 1e-9:
            return v / n
    return None


class _Node:
    """A current piece: its curve, the original samples it covers (with their parameters), its neighbours."""

    __slots__ = ("curve", "pts", "u", "length", "order", "prev", "next", "alive")

    def __init__(self, curve: Curve, pts: np.ndarray, u: np.ndarray, length: float, order: int):
        self.curve, self.pts, self.u, self.length, self.order = curve, pts, u, length, order
        self.prev: _Node | None = None
        self.next: _Node | None = None
        self.alive = True


def _joined(a: Curve, b: Curve) -> bool:
    return bool(np.linalg.norm(a.ctrl[3] - b.ctrl[0]) < 1e-3)


def _split_param(a: _Node, b: _Node) -> float:
    """Where the joint falls on the merged curve's parameter.

    If ``a`` and ``b`` are the two parts of one cubic split at ``s``, the speed
    at the joint is ``3|A3 - A2| / s = 3|B1 - B0| / (1 - s)``, which gives ``s``
    exactly. Pieces fitted separately can have handles that say little, so the
    share of arc length is used when the two estimates disagree by over 2x.
    """
    by_length = a.length / (a.length + b.length) if a.length + b.length > 1e-9 else 0.5
    ha = float(np.linalg.norm(a.curve.ctrl[3] - a.curve.ctrl[2]))
    hb = float(np.linalg.norm(b.curve.ctrl[1] - b.curve.ctrl[0]))
    if ha + hb > 1e-9:
        by_speed = ha / (ha + hb)
        if 0.5 < by_speed / max(by_length, 1e-9) < 2.0:
            return float(np.clip(by_speed, 1e-3, 1.0 - 1e-3))
    return float(np.clip(by_length, 1e-3, 1.0 - 1e-3))


def _chord(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1]


def _merge_fit(a: _Node, b: _Node, iterations: int = 6) -> tuple[np.ndarray, float, np.ndarray]:
    """The single cubic replacing ``a`` and ``b``: ``(ctrl, max error, parameters of the samples)``.

    Two starting parameterizations are tried and the closer fit is kept: arc
    length, which suits pieces fitted separately (measured on drawing 1 it is
    clearly better than the other alone), and the pieces' own parameters joined
    at :func:`_split_param`, which is exact when ``a`` and ``b`` are parts of
    one cubic, as after earlier merges.
    """
    pts = np.vstack([a.pts, b.pts[1:]])
    if np.ptp(pts, axis=0).max() < 1e-9:  # two zero-length pieces
        return np.repeat(pts[:1], 4, axis=0), 0.0, np.linspace(0.0, 1.0, len(pts))
    # tangents stay as they are where a neighbour continues the stroke; open ends are free
    t1 = _direction(a.curve.ctrl[0], *a.curve.ctrl[1:]) if a.prev is not None else None
    t2 = _direction(b.curve.ctrl[3], *b.curve.ctrl[2::-1]) if b.next is not None else None
    s = _split_param(a, b)
    joined = fit_cubic(pts, t1, t2, u=np.concatenate([s * a.u, s + (1.0 - s) * b.u[1:]]), iterations=iterations)
    chord = fit_cubic(pts, t1, t2, u=_chord(pts), iterations=iterations)
    return joined if joined[1] <= chord[1] else chord


def reduce_to(curves: CurveSet, n: int, spacing: float = 1.0) -> dict:
    """Merge (and if needed drop) curves in place until there are at most ``n``; returns a report.

    ``spacing`` (px) is how densely the original pieces are sampled for fitting.
    """
    if n < 1:
        raise ValueError("the curve count must be at least 1")
    before = len(curves.curves)
    report = {"target": n, "before": before, "merged": 0, "dropped": 0, "max_error": 0.0}
    if before <= n:
        report["after"] = before
        return report

    nodes: list[_Node] = []
    for pieces in curves.strokes().values():
        prev = None
        for c in pieces:
            node = _Node(c, *_samples(c.ctrl, spacing), arc_length(c.ctrl), len(nodes))
            if prev is not None and _joined(prev.curve, c):
                prev.next, node.prev = node, prev
            nodes.append(node)
            prev = node

    tick = count()
    heap: list = []

    def push(a: _Node | None, b: _Node | None) -> None:
        if a is None or b is None:
            return
        ctrl, err, u = _merge_fit(a, b)
        heapq.heappush(heap, (err, next(tick), a, b, ctrl, u))

    for node in nodes:
        push(node, node.next)

    alive = before
    while alive > n and heap:
        err, _, a, b, ctrl, u = heapq.heappop(heap)
        if not (a.alive and b.alive and a.next is b):
            continue  # stale: one side was merged since
        ca, cb = a.curve, b.curve
        widths = [(c.width, length) for c, length in ((ca, a.length), (cb, b.length)) if c.width is not None]
        width = (sum(w * length for w, length in widths) / max(sum(length for _, length in widths), 1e-9)
                 if widths else None)
        merged = Curve(ctrl, stroke=ca.stroke, confidence=min(ca.confidence, cb.confidence), tags=ca.tags,
                       width=width, color=(ca if a.length >= b.length else cb).color)
        m = _Node(merged, np.vstack([a.pts, b.pts[1:]]), u, a.length + b.length, a.order)
        m.prev, m.next = a.prev, b.next
        if m.prev is not None:
            m.prev.next = m
        if m.next is not None:
            m.next.prev = m
        a.alive = b.alive = False
        nodes.append(m)
        alive -= 1
        report["merged"] += 1
        report["max_error"] = max(report["max_error"], float(err))
        push(m.prev, m)
        push(m, m.next)

    kept = sorted((node for node in nodes if node.alive), key=lambda node: node.order)
    if alive > n:
        # every connected run is one piece now and nothing can merge: drop the shortest
        drop = {id(node) for node in sorted(kept, key=lambda node: node.length)[: alive - n]}
        kept = [node for node in kept if id(node) not in drop]
        report["dropped"] = len(drop)
    curves.curves = [node.curve for node in kept]
    report["after"] = len(curves.curves)
    report["max_error"] = round(report["max_error"], 3)
    return report

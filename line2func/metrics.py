"""Accuracy metrics between traced curves and ground truth (docs/details.md, Evaluation).

* :func:`f_score` - F_GT@2: how much curve length matches within 2 px;
  :func:`chamfer` - mean distance both ways; :func:`f_sweep` - the same F1 over a
  range of tolerances, in :func:`distance_base` units so it is comparable across
  resolutions.
* :func:`stroke_length_scores` - total length drawn against the ground truth's:
  what the distance metrics above cannot see (retracing, doubled strokes).
* :func:`structure_scores` - crossing continuity, gap closure, fragments per
  stroke and curve count ratio: whether the *strokes* are right, not just the
  pixels.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from line2func.curves import Curve, CurveSet
from line2func.geometry import as_ctrl, flatten


def _ctrls(curves) -> list[np.ndarray]:
    if isinstance(curves, CurveSet):
        curves = curves.curves
    return [c.ctrl if isinstance(c, Curve) else as_ctrl(c) for c in curves]


def sample_points(curves, spacing: float = 0.5) -> np.ndarray:
    """Points every ``spacing`` px along all curves, shape ``(m, 2)``."""
    out = []
    for ctrl in _ctrls(curves):
        poly = flatten(ctrl, 0.05)
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(1, int(np.ceil(cum[-1] / spacing)))
        s = np.linspace(0.0, cum[-1], n + 1)
        out.append(np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], axis=1))
    return np.vstack(out) if out else np.zeros((0, 2))


def f_score(pred, gt, threshold: float = 2.0, spacing: float = 0.5, base: float = 1.0) -> dict[str, float]:
    """F_GT@threshold: precision, recall and F1 of curve length within ``threshold * base``.

    Precision is the fraction of predicted length within the tolerance of the
    ground truth; recall is the fraction of ground-truth length within the
    tolerance of the prediction. ``base`` is the unit the tolerance is counted
    in: 1 (the default) means pixels, :func:`distance_base` makes it relative
    to the drawing's size.
    """
    tol = threshold * base
    p = sample_points(pred, spacing)
    g = sample_points(gt, spacing)
    if len(p) == 0 or len(g) == 0:
        both_empty = len(p) == 0 and len(g) == 0
        v = 1.0 if both_empty else 0.0
        return {"precision": v, "recall": v, "f": v}
    precision = float(np.mean(cKDTree(g).query(p, distance_upper_bound=tol)[0] <= tol))
    recall = float(np.mean(cKDTree(p).query(g, distance_upper_bound=tol)[0] <= tol))
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f": f}


def distance_base(curves: CurveSet) -> float:
    """A thousandth of the drawing's long edge: the distance unit of the Rough Sketch Cleanup
    Benchmark, so tolerances mean the same thing at any resolution."""
    return max(curves.width, curves.height) * 0.001


def f_sweep(pred, gt, steps=range(0, 40, 2), spacing: float = 0.5) -> dict[str, float]:
    """F1 at every tolerance in ``steps``, in :func:`distance_base` units, as ``{"f_0": ..., "f_2": ...}``.

    One curve tells more than a single tolerance does: where the curve rises
    says whether a tracing is off by a little everywhere or by a lot in a few
    places. All the tolerances share one distance computation.
    """
    base = distance_base(gt)
    p = sample_points(pred, spacing)
    g = sample_points(gt, spacing)
    if len(p) == 0 or len(g) == 0:
        v = 1.0 if len(p) == 0 and len(g) == 0 else 0.0
        return {f"f_{d}": v for d in steps}
    d_p = cKDTree(g).query(p)[0]
    d_g = cKDTree(p).query(g)[0]
    out = {}
    for d in steps:
        tol = d * base
        precision = float(np.mean(d_p <= tol))
        recall = float(np.mean(d_g <= tol))
        out[f"f_{d}"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return out


def chamfer(pred, gt, spacing: float = 0.25) -> float:
    """Symmetric mean distance (px) between two curve sets; ``inf`` if either is empty."""
    p = sample_points(pred, spacing)
    g = sample_points(gt, spacing)
    if len(p) == 0 or len(g) == 0:
        return float("inf")
    return 0.5 * float(cKDTree(g).query(p)[0].mean() + cKDTree(p).query(g)[0].mean())


def total_length(curves) -> float:
    """Total length of all curves, px."""
    return float(sum(np.sum(np.linalg.norm(np.diff(flatten(ctrl, 0.05), axis=0), axis=1)) for ctrl in _ctrls(curves)))


def stroke_length_scores(pred, gt) -> dict[str, float]:
    """How much line the prediction draws against how much the ground truth has.

    Every other metric here measures distance point by point, and retracing,
    doubled strokes and curves that double back are all invisible to that: they
    lie on the ink, so they cost nothing. They show up only in the total length.
    ``length_ratio`` is 1 when the prediction draws exactly as much line as the
    ground truth, above 1 when it draws the same ink twice.
    """
    p, g = total_length(pred), total_length(gt)
    if g <= 0.0:
        return {"length_ratio": float("nan"), "length_abs_diff": float("nan")}
    return {"length_ratio": p / g, "length_abs_diff": abs(p - g) / g}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def _stroke_polylines(curves: CurveSet, spacing: float = 0.5) -> dict[int, np.ndarray]:
    """One polyline per stroke (its curves in order), sampled every ``spacing`` px."""
    return {stroke: sample_points(pieces, spacing) for stroke, pieces in curves.strokes().items()}


def _arc(poly: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])


def _at(poly: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    return np.array([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])])


def find_crossings(gt: CurveSet, end_margin: float = 10.0) -> list[dict]:
    """Points where two different ground-truth strokes cross.

    Returns ``[{"point", "strokes": (a, b), "s": (s_a, s_b)}]`` with ``s`` the
    arc-length position of the crossing along each stroke. Crossings closer
    than ``end_margin`` to a stroke end are skipped (those are touches, not
    crossings, and continuity is undefined there).
    """
    polys = _stroke_polylines(gt)
    ids = list(polys)
    cums = {k: _arc(v) for k, v in polys.items()}
    boxes = {k: (v.min(axis=0), v.max(axis=0)) for k, v in polys.items()}
    found = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            if np.any(alo > bhi) or np.any(blo > ahi):
                continue
            pa, pb = polys[a], polys[b]
            p, r = pa[:-1], pa[1:] - pa[:-1]  # (na, 2)
            q, s = pb[:-1], pb[1:] - pb[:-1]  # (nb, 2)
            rxs = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
            qp = q[None] - p[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / rxs
                u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / rxs
            hit = (np.abs(rxs) > 1e-12) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)
            for ia, ib in zip(*np.nonzero(hit)):
                sa = cums[a][ia] + t[ia, ib] * np.linalg.norm(r[ia])
                sb = cums[b][ib] + u[ia, ib] * np.linalg.norm(s[ib])
                if min(sa, cums[a][-1] - sa, sb, cums[b][-1] - sb) < end_margin:
                    continue
                point = p[ia] + t[ia, ib] * r[ia]
                if any(np.hypot(*(c["point"] - point)) < 1.0 for c in found):
                    continue
                cos = abs(float(r[ia] @ s[ib])) / (np.linalg.norm(r[ia]) * np.linalg.norm(s[ib]))
                angle = float(np.degrees(np.arccos(min(1.0, cos))))
                found.append({"point": point, "strokes": (a, b), "s": (float(sa), float(sb)), "angle": angle})
    return found


def _labelled_samples(curves: CurveSet, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    pts, ids = [], []
    for c in curves.curves:
        p = sample_points([c], spacing)
        pts.append(p)
        ids.append(np.full(len(p), c.stroke))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    return np.vstack(pts), np.concatenate(ids)


def structure_scores(
    pred: CurveSet,
    gt: CurveSet,
    gaps: list[dict] | None = None,
    match_radius: float = 3.0,
    probe: float = 8.0,
) -> dict[str, float]:
    """Stroke-structure metrics (crossing continuity, gap closure, fragments, curve count ratio).

    * ``crossing_continuity``: for each side of each ground-truth crossing, the
      stroke is probed ``probe`` px before and after the crossing; correct if
      both probes land on the same predicted stroke. ``nan`` if no crossings.
    * ``gap_closure``: for each erased gap, probes just outside both ends land
      on the same predicted stroke *and* the prediction passes through the gap.
      ``nan`` if no gaps.
    * ``fragments_per_stroke``: predicted strokes covering each ground-truth
      stroke (ideal 1), averaged over strokes that are found at all.
    * ``curve_ratio``: predicted curves / ground-truth curves (ideal 1).

    A probe "lands" on the nearest predicted point within ``match_radius`` px.
    """
    gaps = gt.meta.get("gaps", []) if gaps is None else gaps
    pts, ids = _labelled_samples(pred, 0.5)
    tree = cKDTree(pts) if len(pts) else None

    def owner(q: np.ndarray, radius: float = match_radius) -> int:
        if tree is None:
            return -1
        d, i = tree.query(q, distance_upper_bound=radius)
        return int(ids[i]) if np.isfinite(d) else -1

    polys = _stroke_polylines(gt)
    cums = {k: _arc(v) for k, v in polys.items()}

    def same_stroke_across(stroke: int, s: float, reach: float) -> bool:
        poly, cum = polys[stroke], cums[stroke]
        a = owner(_at(poly, cum, max(0.0, s - reach)))
        b = owner(_at(poly, cum, min(cum[-1], s + reach)))
        return a >= 0 and a == b

    crossings = find_crossings(gt, end_margin=probe + 2.0)
    checks = []
    for c in crossings:
        # probe far enough that the other stroke is > 2 match radii away
        sin = max(np.sin(np.radians(c["angle"])), 0.05)
        reach = min(max(probe, 2.0 * match_radius / sin), 40.0)
        for k, s in zip(c["strokes"], c["s"]):
            if s < reach or cums[k][-1] - s < reach:
                continue  # too close to an end to probe both sides
            checks.append(same_stroke_across(k, s, reach))
    continuity = float(np.mean(checks)) if checks else float("nan")

    closed = []
    for g in gaps:
        stroke = g["stroke"]
        if stroke not in polys:
            continue
        poly, cum = polys[stroke], cums[stroke]
        s = float(cum[np.argmin(np.linalg.norm(poly - np.asarray(g["point"]), axis=1))])
        # the prediction must pass through the middle of the gap, not just end near it
        bridged = owner(np.asarray(g["point"], dtype=np.float64), radius=1.5) >= 0
        closed.append(bridged and same_stroke_across(stroke, s, 0.5 * g["length"] + 4.0))
    closure = float(np.mean(closed)) if closed else float("nan")

    fragments = []
    for stroke, poly in polys.items():
        cum = cums[stroke]
        n = max(2, int(cum[-1] / 1.0))
        owners = [owner(_at(poly, cum, s)) for s in np.linspace(0, cum[-1], n)]
        counts: dict[int, int] = {}
        for o in owners:
            if o >= 0:
                counts[o] = counts.get(o, 0) + 1
        # ignore predicted strokes that only graze this one (e.g. at a crossing)
        real = [k for k, v in counts.items() if v >= max(4, 0.05 * n)]
        if real:
            fragments.append(len(real))
    return {
        "crossing_continuity": continuity,
        "gap_closure": closure,
        "fragments_per_stroke": float(np.mean(fragments)) if fragments else float("nan"),
        "curve_ratio": len(pred) / max(1, len(gt)),
        "crossing_checks": len(checks),
        "gap_checks": len(closed),
    }

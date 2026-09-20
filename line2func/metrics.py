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
* :func:`decision_counts` / :func:`decision_scores` - the tracer's decisions
  against the ground truth: BCubed stroke precision and recall, wrong joins,
  T-junctions, corners (for the decision scorers).
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


# ---------------------------------------------------------------------------
# Decisions: grouping, junctions, joins and corners
#
# The structure scores above only reward joining (putting everything in one
# stroke scores continuity 1.0). These also penalize wrong joins, so a tracer
# cannot win by merging strokes that should stay apart.
# ---------------------------------------------------------------------------


class _Owners:
    """Predicted samples with their stroke id and arc position along that stroke."""

    def __init__(self, pred: CurveSet, spacing: float = 0.5):
        pts, ids, arcs = [], [], []
        self.length: dict[int, float] = {}
        self.closed: dict[int, bool] = {}
        for stroke, pieces in pred.strokes().items():
            poly = sample_points(pieces, spacing)
            if not len(poly):
                continue
            cum = _arc(poly)
            pts.append(poly)
            ids.append(np.full(len(poly), stroke))
            arcs.append(cum)
            self.length[stroke] = float(cum[-1])
            self.closed[stroke] = bool(np.linalg.norm(poly[0] - poly[-1]) < 1e-3)
        self.tree = cKDTree(np.vstack(pts)) if pts else None
        self.ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        self.arcs = np.concatenate(arcs) if arcs else np.zeros(0)

    def owner(self, q: np.ndarray, radius: float) -> tuple[int, float] | None:
        if self.tree is None:
            return None
        d, i = self.tree.query(q, distance_upper_bound=radius)
        return (int(self.ids[i]), float(self.arcs[i])) if np.isfinite(d) else None

    def local(self, a, b, span: float) -> bool:
        """Both probes on one predicted stroke, and close along it (not joined far away)."""
        if a is None or b is None or a[0] != b[0]:
            return False
        ds = abs(a[1] - b[1])
        if self.closed.get(a[0]):
            ds = min(ds, self.length[a[0]] - ds)
        return ds <= 1.3 * span + 4.0


def _stroke_widths(gt: CurveSet, default: float = 2.0) -> dict[int, float]:
    widths = gt.meta.get("widths")
    out: dict[int, list[float]] = {}
    for i, c in enumerate(gt.curves):
        w = float(np.mean(widths[i])) if widths is not None and i < len(widths) else default
        out.setdefault(c.stroke, []).append(w)
    return {k: float(np.mean(v)) for k, v in out.items()}


def find_touches(gt: CurveSet, extra: float = 3.0) -> list[dict]:
    """T-junctions: a ground-truth stroke ending on (or just short of) another stroke's body.

    Returns ``[{"point", "stem", "end", "bar", "s_bar"}]``; ``end`` is 0 or 1.
    The touch point must lie at least two line widths from the bar's ends.
    """
    polys = _stroke_polylines(gt)
    cums = {k: _arc(v) for k, v in polys.items()}
    widths = _stroke_widths(gt)
    pts, ids, arcs = [], [], []
    for k, poly in polys.items():
        pts.append(poly)
        ids.append(np.full(len(poly), k))
        arcs.append(cums[k])
    if len(pts) < 2:
        return []
    tree = cKDTree(np.vstack(pts))
    ids_arr, arcs_arr = np.concatenate(ids), np.concatenate(arcs)
    w_max = max(widths.values())
    found = []
    for a, poly in polys.items():
        for end, p in ((0, poly[0]), (1, poly[-1])):
            best: dict[int, tuple[float, float]] = {}
            for i in tree.query_ball_point(p, 0.5 * (widths[a] + w_max) + extra):
                b = int(ids_arr[i])
                if b == a:
                    continue
                d = float(np.hypot(*(tree.data[i] - p)))
                if d <= 0.5 * (widths[a] + widths[b]) + extra and (b not in best or d < best[b][0]):
                    best[b] = (d, float(arcs_arr[i]))
            for b, (_, s) in best.items():
                if 2.0 * widths[b] <= s <= cums[b][-1] - 2.0 * widths[b]:
                    found.append({"point": p.copy(), "stem": a, "end": end, "bar": b, "s_bar": s})
    return found


def _tangents(ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents at the start and the end of a cubic (degenerate handles fall back to chords)."""

    def unit(*vs):
        for v in vs:
            n = float(np.hypot(*v))
            if n > 1e-9:
                return v / n
        return np.array([1.0, 0.0])

    return (unit(ctrl[1] - ctrl[0], ctrl[2] - ctrl[0], ctrl[3] - ctrl[0]),
            unit(ctrl[3] - ctrl[2], ctrl[3] - ctrl[1], ctrl[3] - ctrl[0]))


def joint_corners(curves: CurveSet, min_angle: float) -> list[tuple[np.ndarray, float]]:
    """``(point, turn in degrees)`` at every joint of a stroke whose tangent breaks by ``min_angle`` or more."""
    out = []
    for pieces in curves.strokes().values():
        joints = list(zip(pieces[:-1], pieces[1:]))
        if len(pieces) > 1 and np.linalg.norm(pieces[-1].ctrl[3] - pieces[0].ctrl[0]) < 1e-3:
            joints.append((pieces[-1], pieces[0]))  # the seam of a closed stroke
        for a, b in joints:
            if np.linalg.norm(a.ctrl[3] - b.ctrl[0]) > 1e-3:
                continue
            cos = float(_tangents(a.ctrl)[1] @ _tangents(b.ctrl)[0])
            turn = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            if turn >= min_angle:
                out.append((a.ctrl[3].copy(), turn))
    return out


def gt_corners(gt: CurveSet, min_angle: float = 30.0) -> list[tuple[np.ndarray, float]]:
    """Sharp corners of the ground truth: joints whose tangent breaks by ``min_angle`` or more."""
    return joint_corners(gt, min_angle)


def _runs(labels: list[int], min_len: int) -> list[tuple[int, int]]:
    """``(label, first index)`` of the runs of one label that are at least ``min_len`` long (label >= 0)."""
    runs, k = [], 0
    while k < len(labels):
        m = k
        while m + 1 < len(labels) and labels[m + 1] == labels[k]:
            m += 1
        if labels[k] >= 0 and m - k + 1 >= min_len:
            runs.append((labels[k], k))
        k = m + 1
    return runs


def decision_counts(
    pred: CurveSet,
    gt: CurveSet,
    match_radius: float = 3.0,
    probe: float = 8.0,
    exclusion: float = 6.0,
) -> dict[str, float]:
    """Raw counts behind :func:`decision_scores`; sum them over scenes to pool.

    * **BCubed** over ground-truth samples every 1 px, each matched to the
      nearest predicted stroke within ``match_radius``; samples within
      ``exclusion`` px of another ground-truth stroke are left out (their owner
      is ambiguous). Precision falls when a predicted stroke mixes ground-truth
      strokes (a wrong join), recall when one is split.
    * **Crossings**: both lines continue through, and neither turns into the other.
    * **T-junctions** (:func:`find_touches`): the bar stays whole, and the stem
      does not continue into it.
    * **Joins**: walking each predicted stroke, every change of ground-truth
      stroke (runs of >= 3 px, ambiguous samples skipped) is one join, by where
      it happens: at a crossing, at a touch, or elsewhere (e.g. a wrong gap link).
    * **Corners**: joints whose tangent breaks, >= 30 deg in the ground truth
      (only those the tracing reached) and >= 20 deg in the prediction, matched
      within max(4, 1.5 line widths).
    """
    owners = _Owners(pred)
    polys = _stroke_polylines(gt)
    cums = {k: _arc(v) for k, v in polys.items()}
    out: dict[str, float] = {}

    g_pts, g_ids = _labelled_samples(gt, 1.0)
    p_pts, p_ids = _labelled_samples(pred, 0.5)
    n_p = n_r = n_all = 0.0
    if len(g_pts) and len(p_pts):
        near = cKDTree(g_pts).query_ball_point(g_pts, exclusion)
        keep = np.array([np.all(g_ids[nb] == g_ids[i]) for i, nb in enumerate(near)])
        d, j = cKDTree(p_pts).query(g_pts[keep], distance_upper_bound=match_radius)
        hit = np.isfinite(d)
        g, p = g_ids[keep][hit], p_ids[j[hit]]
        if len(g):
            pairs, n = np.unique(np.stack([g, p], axis=1), axis=0, return_counts=True)
            per_g = dict(zip(*np.unique(g, return_counts=True)))
            per_p = dict(zip(*np.unique(p, return_counts=True)))
            n_p = float(sum(c * c / per_p[pi] for (gi, pi), c in zip(pairs, n)))
            n_r = float(sum(c * c / per_g[gi] for (gi, pi), c in zip(pairs, n)))
            n_all = float(len(g))
    out.update(bcubed_p_num=n_p, bcubed_r_num=n_r, bcubed_n=n_all)

    both = turns = checks = 0
    crossings = find_crossings(gt, end_margin=3.0)  # nearer a stroke end it is a touch (T), not a crossing
    for c in crossings:
        sin = max(np.sin(np.radians(c["angle"])), 0.05)
        reach = min(max(probe, 2.0 * match_radius / sin), 40.0)
        (a, b), (sa, sb) = c["strokes"], c["s"]
        if min(sa, sb) < reach or cums[a][-1] - sa < reach or cums[b][-1] - sb < reach:
            continue
        oa = [owners.owner(_at(polys[a], cums[a], sa + k * reach), match_radius) for k in (-1, 1)]
        ob = [owners.owner(_at(polys[b], cums[b], sb + k * reach), match_radius) for k in (-1, 1)]
        span = 2.0 * reach
        checks += 1
        both += owners.local(oa[0], oa[1], span) and owners.local(ob[0], ob[1], span)
        turns += any(owners.local(x, y, span) for x in oa for y in ob)
    out.update(crossing_checks=float(checks), crossing_both_ok_n=float(both), crossing_false_turn_n=float(turns))

    bar_ok = cont = t_checks = 0
    touches = find_touches(gt)
    for t in touches:
        stem, bar = t["stem"], t["bar"]
        if cums[stem][-1] < 2.0 * probe or t["s_bar"] < probe or cums[bar][-1] - t["s_bar"] < probe:
            continue
        s_stem = probe if t["end"] == 0 else cums[stem][-1] - probe
        o_s = owners.owner(_at(polys[stem], cums[stem], s_stem), match_radius)
        o_b = [owners.owner(_at(polys[bar], cums[bar], t["s_bar"] + k * probe), match_radius) for k in (-1, 1)]
        t_checks += 1
        bar_ok += owners.local(o_b[0], o_b[1], 2.0 * probe)
        cont += any(owners.local(o_s, y, 2.0 * probe) for y in o_b)
    out.update(t_checks=float(t_checks), t_bar_ok_n=float(bar_ok), t_false_cont_n=float(cont))

    joins = {"crossing": 0, "touch": 0, "other": 0}
    if len(g_pts):
        g_tree = cKDTree(g_pts)
        cross_pts = [(c["point"], set(c["strokes"])) for c in crossings]
        touch_pts = [(t["point"], {t["stem"], t["bar"]}) for t in touches]
        for pieces in pred.strokes().values():
            poly = sample_points(pieces, 1.0)
            dd, jj = g_tree.query(poly, k=min(6, len(g_pts)), distance_upper_bound=match_radius)
            labels = []
            for drow, jrow in zip(np.atleast_2d(dd), np.atleast_2d(jj)):
                found = {int(g_ids[j]) for d, j in zip(drow, jrow) if np.isfinite(d)}
                labels.append(found.pop() if len(found) == 1 else -1)  # -1: no ink, or ambiguous
            runs = _runs(labels, 3)
            for (la, _), (lb, start) in zip(runs[:-1], runs[1:]):
                if la == lb:
                    continue
                q = poly[start]
                kind = "other"
                if any(np.hypot(*(pt - q)) < 3.0 * probe and {la, lb} <= s for pt, s in touch_pts):
                    kind = "touch"
                elif any(np.hypot(*(pt - q)) < 3.0 * probe and {la, lb} <= s for pt, s in cross_pts):
                    kind = "crossing"
                joins[kind] += 1
    out.update({f"joins_{k}_n": float(v) for k, v in joins.items()})
    out["gt_strokes"] = float(gt.num_strokes)

    w = float(pred.meta.get("line_width") or 2.0)
    match = max(4.0, 1.5 * w)
    g_c = [p for p, _ in gt_corners(gt, 30.0)]
    if g_c and len(p_pts):
        dist_to_pred, _ = cKDTree(p_pts).query(np.array(g_c))
        g_c = [p for p, di in zip(g_c, dist_to_pred) if di <= 2.0]  # only corners the tracing reached
    p_c = [p for p, _ in joint_corners(pred, 20.0)]
    matched = 0
    if g_c and p_c:
        dist = np.linalg.norm(np.array(g_c)[:, None] - np.array(p_c)[None], axis=2)
        used_g, used_p = set(), set()
        for gi, pi in sorted(zip(*np.nonzero(dist <= match)), key=lambda ij: dist[ij]):
            if gi not in used_g and pi not in used_p:
                used_g.add(gi)
                used_p.add(pi)
                matched += 1
    out.update(corner_matched=float(matched), corner_pred=float(len(p_c)), corner_gt=float(len(g_c)))
    return out


def decision_scores(counts: dict[str, float]) -> dict[str, float]:
    """Rates from (pooled) :func:`decision_counts`; ``nan`` where there is nothing to measure."""

    def ratio(a: str, b: str) -> float:
        return float(counts.get(a, 0.0)) / counts[b] if counts.get(b) else float("nan")

    p, r = ratio("bcubed_p_num", "bcubed_n"), ratio("bcubed_r_num", "bcubed_n")
    cp, cr = ratio("corner_matched", "corner_pred"), ratio("corner_matched", "corner_gt")
    n_gt = counts.get("gt_strokes") or float("nan")
    return {
        "bcubed_p": p,
        "bcubed_r": r,
        "bcubed_f": 2 * p * r / (p + r) if p + r > 0 else float("nan"),
        "crossing_both_ok": ratio("crossing_both_ok_n", "crossing_checks"),
        "crossing_false_turn": ratio("crossing_false_turn_n", "crossing_checks"),
        "t_bar_ok": ratio("t_bar_ok_n", "t_checks"),
        "t_false_cont": ratio("t_false_cont_n", "t_checks"),
        "joins_crossing_per100": 100.0 * counts.get("joins_crossing_n", 0.0) / n_gt,
        "joins_touch_per100": 100.0 * counts.get("joins_touch_n", 0.0) / n_gt,
        "joins_other_per100": 100.0 * counts.get("joins_other_n", 0.0) / n_gt,
        "corner_p": cp,
        "corner_r": cr,
        "corner_f": 2 * cp * cr / (cp + cr) if cp + cr > 0 else float("nan"),
    }

"""Features of decision candidates, for learned decision scorers (numpy only).

Every candidate of :mod:`line2func.decisions` becomes a fixed-length row of
numbers. Lengths are divided by the line width ``w``, angles are in degrees,
ink is relative to the drawing's typical line darkness, and every value is
clipped to a finite range. Four common values end every row: ``w`` (px),
the ink threshold, the typical darkness and the radius.

An *arm window* is the stretch of an arm (skeleton points, from a node outward)
at a distance ``[r_in, r_in + max(6, 3w)]`` from the node, with
``r_in = (r + 1) + 2 * spread + 1``: just past the zone where thinning distorts
the skeleton and where the tracer trims it anyway.

The same functions serve the recorder (training data) and the learned scorer
(inference), so both always see identical features.
"""

from __future__ import annotations

import numpy as np

from line2func.decisions import (
    CornerCand,
    CrossingCand,
    DecisionContext,
    GapCand,
    JunctionCand,
    _angle,
    pick_peaks,
)

FEATURES_VERSION = 1
COMMON = ["w", "threshold", "ink_p90", "radius"]

JUNCTION = ["bend", "bend_near", "bend_far", "bend_fit", "lateral", "center_dev", "curv_min", "curv_max",
            "width_ratio", "ink_ratio", "len_min", "len_max", "terminal_n", "k", "spread", "node_width",
            "rank_min", "rank_max", "mutual_best", "alt_margin", "complement_bend", "resid_max"] + COMMON
JUNCTION_END = ["best_bend", "second_bend", "len", "terminal", "width_rel", "ink_rel", "curv", "aim",
                "others_straight", "k", "spread", "node_width"] + COMMON
GAP = ["dist", "dist_px", "a_min", "a_max", "bend", "lateral", "ink_mean", "ink_min", "ink_frac", "crosses_other",
       "width_ratio", "ink_ratio", "len_min", "len_max", "same_edge", "curv_max", "miss", "rank_min", "rank_max",
       "n_cand_max", "rule_ok"] + COMMON
CROSSING = ["len", "len_rel", "t_min", "t_max", "best_hyp", "other_hyp", "rule", "lateral_1", "lateral_2",
            "mid_width", "mid_ink_rel", "mid_vs_bisector", "cross_angle", "arm_len_min", "arm_width_ratio_min",
            "arm_width_rel_max"] + COMMON
CORNER = ["turn", "turn_k2", "turn_k8", "turn_skip2", "peak_excess", "resid_left", "resid_right", "seam_dist",
          "seam_junction", "seam_gap", "end_dist", "width_rel", "ink_rel", "closed", "stroke_len", "n_peaks_near"
          ] + COMMON
NAMES = {"junction": JUNCTION, "junction_end": JUNCTION_END, "gap": GAP, "crossing": CROSSING, "corner": CORNER}
CORNER_MIN_TURN = 15.0  # peaks at least this sharp are corner candidates (the rule's 60 deg ones are among them)


def _common(ctx: DecisionContext) -> list[float]:
    return [ctx.line_w, ctx.threshold, ctx.ink_p90, ctx.radius]


def _finish(rows, names: list[str]) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64).reshape(-1, len(names))
    x = np.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(x, -1e3, 1e3).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def _window(arm: np.ndarray, center: np.ndarray, r_in: float, span: float, d: np.ndarray | None = None) -> np.ndarray:
    """Arm points at distance [r_in, r_in + span] from ``center`` (at least two points when possible)."""
    if d is None:
        d = np.linalg.norm(arm - center, axis=1)
    sel = arm[(d >= r_in) & (d <= r_in + span)]
    if len(sel) >= 2:
        return sel
    beyond = arm[d >= r_in]
    if len(beyond) >= 2:
        return beyond[:2]
    return arm[-2:] if len(arm) >= 2 else np.vstack([arm, arm + 1e-6])


def _line_fit(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares line: (centroid, unit direction from the first point toward the last, rms residual).

    The principal axis of the 2x2 scatter matrix, in closed form (what an SVD
    of the centred points gives, without its per-call cost).
    """
    c = pts.mean(axis=0)
    if len(pts) < 2 or float(np.ptp(pts, axis=0).max()) < 1e-9:
        return c, np.array([1.0, 0.0]), 0.0
    x = pts - c
    a, b, cc = float(x[:, 0] @ x[:, 0]), float(x[:, 0] @ x[:, 1]), float(x[:, 1] @ x[:, 1])
    half = 0.5 * (a - cc)
    root = float(np.hypot(half, b))
    lam_max, lam_min = 0.5 * (a + cc) + root, max(0.5 * (a + cc) - root, 0.0)
    if abs(b) > 1e-12:
        d = np.array([lam_max - cc, b])
    else:
        d = np.array([1.0, 0.0]) if a >= cc else np.array([0.0, 1.0])
    d = d / float(np.hypot(d[0], d[1]))
    if d @ (pts[-1] - pts[0]) < 0:
        d = -d
    return c, d, float(np.sqrt(lam_min / len(pts)))


def _offset(point: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Distance from ``point`` to the line through ``c`` along ``d``."""
    v = point - c
    return abs(float(v[0] * d[1] - v[1] * d[0]))


def _width(ctx: DecisionContext, pts: np.ndarray) -> float:
    return float(np.median(np.maximum(2.0 * ctx.px(ctx.line_dist, pts) - 1.0, 0.5))) if len(pts) else ctx.line_w


def _ink(ctx: DecisionContext, pts: np.ndarray) -> float:
    return float(np.median(ctx.px(ctx.ink, pts))) / ctx.ink_p90 if len(pts) else 0.0


def _log_ratio(a: float, b: float) -> float:
    return abs(float(np.log(max(a, 1e-3) / max(b, 1e-3))))


class _Arm:
    """Measurements of one arm leaving a node (or a tip) that several features share."""

    def __init__(self, arm: np.ndarray, center: np.ndarray, r_in: float, ctx: DecisionContext, reach: float):
        w = ctx.line_w
        span = max(6.0, 3.0 * w)
        dist = np.linalg.norm(arm - center, axis=1)
        self.window = _window(arm, center, r_in, span, dist)
        self.c, self.d, self.resid = _line_fit(self.window)
        self.near, self.rule, self.far = _directions(arm, center, (r_in + 1.0, reach, reach + max(4.0, 2.0 * w)),
                                                     dist)
        # turning from the near to the far direction, in degrees
        self.curv = _angle(self.near, self.far)
        self.length = _length(arm)
        yx = ctx.yx(self.window)  # the same pixels for the width and the ink (as _width / _ink)
        self.width = float(np.median(np.maximum(2.0 * ctx.line_dist[yx] - 1.0, 0.5)))
        self.ink = float(np.median(ctx.ink[yx])) / ctx.ink_p90
        self.aim = _offset(center, self.c, self.d)  # how far the arm's line misses the node centre


def _directions(pts: np.ndarray, origin: np.ndarray, reaches, d: np.ndarray | None = None) -> list[np.ndarray]:
    """``_direction_from(pts, origin, r)`` for each ``r``, sharing one distance computation."""
    if d is None:
        d = np.linalg.norm(pts - origin, axis=1)
    out = []
    for reach in reaches:
        far = np.nonzero(d >= reach)[0]
        q = pts[far[0]] if len(far) else pts[-1]
        v = q - origin
        n = np.hypot(v[0], v[1])
        if n < 1e-9:
            v = pts[-1] - pts[0]
            n = np.hypot(v[0], v[1])
        out.append(v / n if n > 1e-9 else np.array([1.0, 0.0]))
    return out


def _r_in(radius: float, spread: float) -> float:
    """Where an arm window starts: past the zone where thinning distorts the skeleton.

    :mod:`line2func.labels` uses this too, so that the ground-truth labels are
    read off exactly the stretch of arm the features are measured on. It takes a
    radius rather than a context so both can call it; if the two ever drifted
    apart, the labels would shift with nothing failing.
    """
    return (radius + 1.0) + 2.0 * spread + 1.0


# ---------------------------------------------------------------------------
# Junctions: pairs of arms, and each arm ending here
# ---------------------------------------------------------------------------


def junction_features(c: JunctionCand, ctx: DecisionContext) -> tuple[np.ndarray, np.ndarray]:
    """``(pair rows (P, F), end rows (k, F_end))`` for one node, in the order of ``c.pairs`` / ``c.keys``."""
    w = ctx.line_w
    r_in = _r_in(ctx.radius, c.spread)
    arms = [_Arm(a, c.center, r_in, ctx, c.reach) for a in c.arms]
    k = len(arms)
    bends = np.asarray(c.bends, dtype=np.float64)
    node_width = _width(ctx, c.center[None]) / w
    widths = np.array([a.width for a in arms])
    med_w = float(np.median(widths)) if k else w
    inks = np.array([a.ink for a in arms])
    med_ink = float(np.median(inks)) if k else 1.0
    per_arm: list[list[float]] = [[] for _ in range(k)]
    for b, (i, j) in zip(bends, c.pairs):
        per_arm[i].append(float(b))
        per_arm[j].append(float(b))
    best = [min(v) if v else 180.0 for v in per_arm]
    second = [sorted(v)[1] if len(v) > 1 else 180.0 for v in per_arm]
    common = _common(ctx)

    pair_rows = []
    for b, (i, j) in zip(bends, c.pairs):
        ai, aj = arms[i], arms[j]
        rank_i = sorted(per_arm[i]).index(float(b))
        rank_j = sorted(per_arm[j]).index(float(b))
        alt_i = second[i] if best[i] == float(b) else best[i]
        alt_j = second[j] if best[j] == float(b) else best[j]
        others = [x for x in range(k) if x not in (i, j)]
        complement = 180.0
        if len(others) == 2:
            p = tuple(others)
            complement = float(bends[c.pairs.index(p)])
        mid = 0.5 * (ai.c + aj.c)
        through = _unit(aj.c - ai.c)
        pair_rows.append([
            float(b),
            _angle(ai.near, -aj.near),
            _angle(ai.far, -aj.far),
            _angle(ai.d, -aj.d),
            0.5 * (_offset(aj.c, ai.c, ai.d) + _offset(ai.c, aj.c, aj.d)) / w,
            _offset(c.center, mid, through) / w,
            min(ai.curv, aj.curv),
            max(ai.curv, aj.curv),
            _log_ratio(ai.width, aj.width),
            _log_ratio(ai.ink, aj.ink),
            min(ai.length, aj.length) / w,
            max(ai.length, aj.length) / w,
            float((c.far_degree[i] == 1) + (c.far_degree[j] == 1)) if c.far_degree else 0.0,
            float(k),
            c.spread / w,
            node_width,
            float(min(rank_i, rank_j)),
            float(max(rank_i, rank_j)),
            float(rank_i == 0 and rank_j == 0),
            min(alt_i, alt_j) - float(b),
            complement,
            max(ai.resid, aj.resid) / w,
            *common,
        ])
    end_rows = []
    for i, a in enumerate(arms):
        others = [float(b) for b, (p, q) in zip(bends, c.pairs) if i not in (p, q)]
        end_rows.append([
            best[i],
            second[i],
            a.length / w,
            float(c.far_degree[i] == 1) if c.far_degree else 0.0,
            a.width / max(med_w, 1e-3),
            a.ink / max(med_ink, 1e-3),
            a.curv,
            a.aim / w,
            min(others) if others else 180.0,
            float(k),
            c.spread / w,
            node_width,
            *common,
        ])
    return _finish(pair_rows, JUNCTION), _finish(end_rows, JUNCTION_END)


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def gap_features(cands: list[GapCand], ctx: DecisionContext) -> np.ndarray:
    """One row per gap candidate (pairs of stroke ends), in order."""
    if not cands:
        return _finish([], GAP)
    w = ctx.line_w
    common = _common(ctx)
    # competition: each tip's candidates, ranked by the rule's cost
    by_tip: dict[int, list[float]] = {}
    for g in cands:
        by_tip.setdefault(g.i, []).append(g.cost)
        by_tip.setdefault(g.j, []).append(g.cost)
    for v in by_tip.values():
        v.sort()
    rows = []
    arms: dict[int, _Arm] = {}

    def arm(tip: int, pts: np.ndarray, at: np.ndarray) -> _Arm:
        if tip not in arms:
            arms[tip] = _Arm(pts, at, 0.5 * w, ctx, 2.0 * ctx.radius + 3.0)
        return arms[tip]

    # every bridge (tip to tip, every 0.5 px), sampled in one go
    counts = [max(2, int(np.ceil(g.dist / 0.5)) + 1) for g in cands]
    bridges = np.vstack([g.tip_i + np.linspace(0.0, 1.0, n)[:, None] * (g.tip_j - g.tip_i)
                         for g, n in zip(cands, counts)])
    ink_all = ctx.bilinear(ctx.ink, bridges)
    mask_all = ctx.px(ctx.mask, bridges)
    ends = np.cumsum(counts)
    for g, stop, n in zip(cands, ends, counts):
        arm_i = arm(g.i, g.arm_i, g.tip_i)
        arm_j = arm(g.j, g.arm_j, g.tip_j)
        bridge, ink, on_mask = bridges[stop - n: stop], ink_all[stop - n: stop], mask_all[stop - n: stop]
        # bridge samples on ink away from both tips: the link would cross another line
        away = (np.linalg.norm(bridge - g.tip_i, axis=1) > 1.5 * w) & (np.linalg.norm(bridge - g.tip_j, axis=1) > 1.5 * w)
        crosses = float(np.mean(on_mask[away])) if away.any() else 0.0
        # extrapolate each end's own line across the gap: how far does it miss the other tip?
        miss = min(_offset(g.tip_j, g.tip_i, g.out_i), _offset(g.tip_i, g.tip_j, g.out_j)) if g.dist > 1e-6 else 0.0
        lateral = 0.5 * (_offset(g.tip_j, g.tip_i, g.out_i) + _offset(g.tip_i, g.tip_j, g.out_j))
        rank_i = by_tip[g.i].index(g.cost)
        rank_j = by_tip[g.j].index(g.cost)
        rows.append([
            g.dist / w,
            g.dist,
            min(g.ai, g.aj),
            max(g.ai, g.aj),
            _angle(g.out_i, -g.out_j),
            lateral / w,
            float(np.mean(ink)) / ctx.threshold,
            float(np.min(ink)) / ctx.threshold,
            float(np.mean(ink >= 0.5 * ctx.threshold)),
            crosses,
            _log_ratio(arm_i.width, arm_j.width),
            _log_ratio(arm_i.ink, arm_j.ink),
            min(arm_i.length, arm_j.length) / w,
            max(arm_i.length, arm_j.length) / w,
            float(g.key_i[0] == g.key_j[0]),
            max(arm_i.curv, arm_j.curv),
            miss / w,
            float(min(rank_i, rank_j)),
            float(max(rank_i, rank_j)),
            float(max(len(by_tip[g.i]), len(by_tip[g.j]))),
            float(g.rule_ok),
            *common,
        ])
    return _finish(rows, GAP)


# ---------------------------------------------------------------------------
# Shallow crossings
# ---------------------------------------------------------------------------


def crossing_features(cands: list[CrossingCand], ctx: DecisionContext) -> np.ndarray:
    """One row per candidate edge between two degree-3 junctions, in order."""
    if not cands:
        return _finish([], CROSSING)
    w = ctx.line_w
    common = _common(ctx)
    reach = 2.0 * ctx.radius + 3.0
    rows = []
    for cand in cands:
        cu, cv = cand.mid[0], cand.mid[-1]
        r_in = _r_in(ctx.radius, 0.0)
        au = [_Arm(pts, cu, r_in, ctx, reach) for _, pts in cand.arms_u]
        av = [_Arm(pts, cv, r_in, ctx, reach) for _, pts in cand.arms_v]
        if len(au) != 2 or len(av) != 2:
            rows.append([0.0] * (len(CROSSING) - len(COMMON)) + common)
            continue
        t_u = _angle(au[0].rule, -au[1].rule)
        t_v = _angle(av[0].rule, -av[1].rule)
        straight = max(_angle(au[0].rule, -av[0].rule), _angle(au[1].rule, -av[1].rule))
        swapped = max(_angle(au[0].rule, -av[1].rule), _angle(au[1].rule, -av[0].rule))
        pairing = [(au[0], av[0]), (au[1], av[1])] if straight <= swapped else [(au[0], av[1]), (au[1], av[0])]
        (p1, q1), (p2, q2) = pairing
        lat1 = 0.5 * (_offset(q1.c, p1.c, p1.d) + _offset(p1.c, q1.c, q1.d))
        lat2 = 0.5 * (_offset(q2.c, p2.c, p2.d) + _offset(p2.c, q2.c, q2.d))
        line1, line2 = _unit(q1.c - p1.c), _unit(q2.c - p2.c)
        cross = _angle(line1, line2)
        cross = min(cross, 180.0 - cross)
        bisector = _unit(line1 + (line2 if line1 @ line2 >= 0 else -line2))
        mid_dir = _unit(cv - cu)
        mid_vs = _angle(mid_dir, bisector)
        mid_vs = min(mid_vs, 180.0 - mid_vs)
        expected = 1.3 * w / max(np.tan(np.radians(max(cross, 1.0)) / 2.0), 1e-3)
        widths = [a.width for a in au + av]
        mid_pts = cand.mid[len(cand.mid) // 4: max(len(cand.mid) // 4 + 1, 3 * len(cand.mid) // 4)]
        mid_w = _width(ctx, mid_pts)
        rows.append([
            cand.length / w,
            cand.length / expected,
            min(t_u, t_v),
            max(t_u, t_v),
            min(straight, swapped),
            max(straight, swapped),
            float(cand.rule),
            lat1 / w,
            lat2 / w,
            mid_w / w,
            _ink(ctx, mid_pts),
            mid_vs,
            cross,
            min(a.length for a in au + av) / w,
            min(widths) / max(max(widths), 1e-3),
            mid_w / max(float(np.median(widths)), 1e-3),
            *common,
        ])
    return _finish(rows, CROSSING)


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------


def _turns(pts: np.ndarray, closed: bool, k: int, skip: int) -> np.ndarray:
    """Turning angle at every sample (0 where it cannot be measured), like baseline._turn_profile."""
    n = len(pts)
    reach = k + skip
    out = np.zeros(n)
    if n < 2 * reach + 1:
        return out
    i = np.arange(n) if closed else np.arange(reach, n - reach)
    v1 = pts[(i - skip) % n] - pts[(i - reach) % n]
    v2 = pts[(i + reach) % n] - pts[(i + skip) % n]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
    out[i] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def corner_peaks(c: CornerCand) -> list[int]:
    """Candidate corners: the turning peaks of at least :data:`CORNER_MIN_TURN` (a superset of the rule's)."""
    return pick_peaks(c.index, c.turn, CORNER_MIN_TURN, len(c.pts), c.closed, c.k + c.skip)


def corner_features(c: CornerCand, peaks: list[int], ctx: DecisionContext, only: list[int] | None = None) -> np.ndarray:
    """One row per candidate peak (sample index into ``c.pts``); ``only`` limits the rows to those peaks."""
    if not peaks or (only is not None and not only):
        return _finish([], CORNER)
    w = ctx.line_w
    common = _common(ctx)
    pts, n = c.pts, len(c.pts)
    reach = c.k + c.skip
    base = _turns(pts, c.closed, c.k, c.skip)
    t2 = _turns(pts, c.closed, 2, c.skip)
    t8 = _turns(pts, c.closed, 8, c.skip)
    ts2 = _turns(pts, c.closed, c.k, 2)
    total = _length(pts)
    all_peaks = np.asarray(peaks)
    idx = all_peaks if only is None else np.asarray(only)
    at = pts[idx]
    yx = ctx.yx(at)
    widths = np.maximum(2.0 * ctx.line_dist[yx] - 1.0, 0.5) / w
    inks = ctx.ink[yx] / ctx.ink_p90
    # the turn above the median turn around the peak (window clipped at the stroke's ends)
    win = idx[:, None] + np.arange(-reach, reach + 1)[None, :]
    valid = (win >= 0) & (win < n)
    around = np.sort(np.where(valid, base[np.clip(win, 0, n - 1)], np.inf), axis=1)  # invalid ones sort last
    m = valid.sum(axis=1)
    rows_ = np.arange(len(idx))
    excess = base[idx] - 0.5 * (around[rows_, (m - 1) // 2] + around[rows_, m // 2])  # the median of the valid ones
    # straightness of the two arms: rms residual of their line fits (as _line_fit)
    resid_left = _window_resid(pts, np.maximum(0, idx - reach), np.maximum(1, idx - c.skip + 1)) / w
    resid_right = _window_resid(pts, np.minimum(n - 1, idx + c.skip), np.minimum(n, idx + reach + 1)) / w
    # the nearest seam (where the stroke was joined across a junction or a gap)
    s = idx.astype(np.float64)  # samples are 1 px apart
    seam_d = np.full(len(idx), 1e3)
    seam_junction = np.zeros(len(idx))
    seam_gap = np.zeros(len(idx))
    if c.seams:
        s0 = np.array([sm[0] for sm in c.seams])
        s1 = np.array([sm[1] for sm in c.seams])
        dd = np.minimum(np.abs(s[:, None] - s0[None]), np.abs(s[:, None] - s1[None]))
        k = np.argmin(dd, axis=1)
        seam_d = np.minimum(dd[np.arange(len(idx)), k], 1e3)
        near = seam_d <= 3.0 * w
        kinds = np.array([sm[2] for sm in c.seams])[k]
        seam_junction = (near & (kinds == "junction")).astype(np.float64)
        seam_gap = (near & (kinds == "gap")).astype(np.float64)
    end_d = np.full(len(idx), 1e3) if c.closed else np.minimum(np.minimum(s, (n - 1) - s), 1e3)
    near_peaks = (np.abs(idx[:, None] - all_peaks[None, :]) <= 3 * reach).sum(axis=1) - 1
    cols = [base[idx], t2[idx], t8[idx], ts2[idx], excess, resid_left, resid_right, seam_d / w, seam_junction, seam_gap,
            end_d / w, widths, inks, np.full(len(idx), float(c.closed)), np.full(len(idx), total / w), near_peaks]
    cols += [np.full(len(idx), v) for v in common]
    return _finish(np.column_stack(cols), CORNER)


def _window_resid(pts: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """rms residual of a line fit to ``pts[start:stop]`` for each window (0 for fewer than two points)."""
    n = len(pts)
    size = int(max(1, (stops - starts).max()))
    j = starts[:, None] + np.arange(size)[None, :]
    inside = j < stops[:, None]
    p = pts[np.clip(j, 0, n - 1)] * inside[..., None]
    count = inside.sum(axis=1)
    mean = p.sum(axis=1) / np.maximum(count, 1)[:, None]
    x = (pts[np.clip(j, 0, n - 1)] - mean[:, None]) * inside[..., None]
    a, b, cc = (x[..., 0] ** 2).sum(1), (x[..., 0] * x[..., 1]).sum(1), (x[..., 1] ** 2).sum(1)
    lam_min = np.maximum(0.5 * (a + cc) - np.hypot(0.5 * (a - cc), b), 0.0)
    return np.where(count >= 2, np.sqrt(lam_min / np.maximum(count, 1)), 0.0)


# ---------------------------------------------------------------------------
# Recording (training data)
# ---------------------------------------------------------------------------


class Recorder:
    """Every decision candidate of one ``baseline.vectorize(..., recorder=...)`` run.

    Per kind (see ``NAMES``) it keeps, row by row: the features ``x``, the rule
    as ``(score, gate passed, chosen by the rule)``, whether the active scorer
    chose it, identifying ``ids`` and ``geom`` (the candidate objects, in memory
    only, for the ground-truth labeller).
    """

    def __init__(self, params):
        self.params = params
        self.ctx = None  # the context of the recorded run (set on the first decision)
        self.x = {k: [] for k in NAMES}
        self.rule = {k: [] for k in NAMES}
        self.chosen = {k: [] for k in NAMES}
        self.ids = {k: [] for k in NAMES}
        self.geom = {k: [] for k in NAMES}

    def _add(self, kind, x, rule, chosen, ids, geom) -> None:
        self.x[kind].append(x)
        self.rule[kind].append(rule)
        self.chosen[kind].append(bool(chosen))
        self.ids[kind].append(ids)
        self.geom[kind].append(geom)

    def crossings(self, cands, verdicts, ctx) -> None:
        self.ctx = ctx
        limit = 12.0 * ctx.radius + 6.0  # the rule never looks at longer edges (wider limits make more candidates)
        for c, v, x in zip(cands, verdicts, crossing_features(cands, ctx)):
            rule = float(c.rule and c.length < limit)
            self._add("crossing", x, (float(c.rule), rule, rule), v, (c.eid, c.u, c.v), c)

    def gaps(self, cands, chosen, ctx) -> None:
        self.ctx = ctx
        from line2func.decisions import greedy_pairs

        rule_scores = np.array([-c.cost if c.rule_ok else -np.inf for c in cands], dtype=np.float64)
        by_rule = set(greedy_pairs(rule_scores, [(c.i, c.j) for c in cands]))
        picked = set(chosen)
        for c, x in zip(cands, gap_features(cands, ctx)):
            self._add("gap", x, (-c.cost, float(c.rule_ok), float((c.i, c.j) in by_rule)), (c.i, c.j) in picked,
                      (c.i, c.j), c)

    def junctions(self, cands, decisions, ctx) -> None:
        self.ctx = ctx
        from line2func.decisions import greedy_pairs

        max_bend = self.params.continue_angle
        for c, chosen in zip(cands, decisions):
            pair_x, end_x = junction_features(c, ctx)
            rule_scores = np.where(c.bends <= max_bend, -c.bends, -np.inf)
            by_rule = set(greedy_pairs(rule_scores, c.pairs))
            picked = set(chosen)
            for (i, j), b, x in zip(c.pairs, c.bends, pair_x):
                self._add("junction", x, (-float(b), float(b <= max_bend), float((i, j) in by_rule)), (i, j) in picked,
                          (c.node, i, j), (c, i, j))
            paired_rule = {a for p in by_rule for a in p}
            paired = {a for p in picked for a in p}
            for i, x in enumerate(end_x):
                self._add("junction_end", x, (0.0, 1.0, float(i not in paired_rule)), i not in paired, (c.node, i, -1),
                          (c, i))

    def corners(self, cand, chosen, ctx, stroke) -> None:
        self.ctx = ctx
        peaks = corner_peaks(cand)
        if not peaks:
            return
        turn_at = dict(zip(cand.index.tolist(), cand.turn.tolist()))
        by_rule = set(pick_peaks(cand.index, cand.turn, self.params.corner_angle, len(cand.pts), cand.closed,
                                 cand.k + cand.skip))
        picked = set(chosen)
        for idx, x in zip(peaks, corner_features(cand, peaks, ctx)):
            t = turn_at.get(idx, 0.0)
            self._add("corner", x, (t, float(t >= self.params.corner_angle), float(idx in by_rule)), idx in picked,
                      (stroke, idx), (cand, idx, stroke))

    def table(self, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(features (n, F) float32, rule (n, 3) float32, chosen (n,) bool)`` for one kind."""
        n_feat = len(NAMES[kind])
        x = np.asarray(self.x[kind], dtype=np.float32).reshape(-1, n_feat)
        rule = np.asarray(self.rule[kind], dtype=np.float32).reshape(-1, 3)
        return x, rule, np.asarray(self.chosen[kind], dtype=bool)

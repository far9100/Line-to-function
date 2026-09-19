"""Fit cubic Bézier curves to polylines (Schneider, "An Algorithm for
Automatically Fitting Digitized Curves", Graphics Gems, 1990).

A polyline is fitted with one cubic by least squares; if the error is too
large the parameterization is refined with Newton steps, and if that is not
enough the polyline is split at the worst point (keeping a shared tangent
there, so the pieces join smoothly) and both halves are fitted recursively.
Tangents at interior split points are fixed; at the two outer ends of an open
polyline the inner control points are left free, which fits curved ends
exactly and is not thrown off by pixel staircases.
"""

from __future__ import annotations

import numpy as np

from line2func.geometry import derivative, evaluate, second_derivative

_REPARAM_ITERATIONS = 40  # Newton reparameterization converges linearly; stop early when it stalls


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-12 else None


def _end_tangent(pts: np.ndarray, reach: float = 3.0) -> np.ndarray:
    """Unit direction from ``pts[0]`` into the polyline, looked at ``reach`` px away."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.cumsum(seg)
    target = min(reach, 0.5 * cum[-1]) if len(cum) else 0.0
    k = int(np.searchsorted(cum, target)) + 1
    for idx in range(min(k, len(pts) - 1), len(pts)):
        t = _unit(pts[idx] - pts[0])
        if t is not None:
            return t
    return np.array([1.0, 0.0])


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0.0, 1.0, len(pts))


def _generate(
    pts: np.ndarray,
    u: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    free_start: bool = False,
    free_end: bool = False,
) -> np.ndarray:
    """Least-squares cubic with fixed end points.

    ``P1`` lies on the ray ``P0 + a t1`` unless ``free_start`` (then it is a
    free 2-D unknown); likewise ``P2`` on ``P3 + b t2`` unless ``free_end``.
    """
    if free_start or free_end:
        return _generate_free(pts, u, t1, t2, free_start, free_end)
    p0, p3 = pts[0], pts[-1]
    mu = 1.0 - u
    b0, b1, b2, b3 = mu**3, 3 * mu * mu * u, 3 * mu * u * u, u**3
    a1 = b1[:, None] * t1
    a2 = b2[:, None] * t2
    c00 = np.sum(a1 * a1)
    c01 = np.sum(a1 * a2)
    c11 = np.sum(a2 * a2)
    tmp = pts - (np.outer(b0 + b1, p0) + np.outer(b2 + b3, p3))
    x0 = np.sum(a1 * tmp)
    x1 = np.sum(a2 * tmp)
    det = c00 * c11 - c01 * c01
    seg_len = float(np.linalg.norm(p3 - p0))
    alpha_l = alpha_r = 0.0
    if abs(det) > 1e-12:
        alpha_l = (x0 * c11 - x1 * c01) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
    eps = 1e-6 * seg_len
    if alpha_l < eps or alpha_r < eps or not np.isfinite(alpha_l + alpha_r):
        alpha_l = alpha_r = seg_len / 3.0
    # a handle longer than the whole polyline means a degenerate fit (e.g. a run
    # that goes out and back): clamp it so the curve cannot fly off
    poly_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    alpha_l, alpha_r = min(alpha_l, poly_len), min(alpha_r, poly_len)
    return np.array([p0, p0 + alpha_l * t1, p3 + alpha_r * t2, p3])


def _generate_free(pts, u, t1, t2, free_start: bool, free_end: bool) -> np.ndarray:
    """General linear least squares for ``P1``/``P2`` (free or tangent-constrained).

    Unknowns are *offsets* from the end points (``P1 = P0 + d1``, ``P2 = P3 + d2``).
    With too few interior points the system is underdetermined, and least
    squares returns the minimum-norm solution: with offsets that keeps the
    control points next to the curve, whereas absolute coordinates would be
    pulled toward the image origin. Very short point runs use fixed tangents.
    """
    if len(pts) - 2 < 3:
        return _generate(pts, u, t1, t2)
    p0, p3 = pts[0], pts[-1]
    n = len(pts)
    mu = 1.0 - u
    b0, b1, b2, b3 = mu**3, 3 * mu * mu * u, 3 * mu * u * u, u**3
    # P1 = P0 + d1 and P2 = P3 + d2 in both cases; only d's parameterization differs
    rhs = (pts - np.outer(b0 + b1, p0) - np.outer(b2 + b3, p3)).T.ravel()  # [x..., y...]
    cols = []
    zeros = np.zeros(n)
    if free_start:
        cols += [np.concatenate([b1, zeros]), np.concatenate([zeros, b1])]
    else:
        cols.append(np.concatenate([b1 * t1[0], b1 * t1[1]]))
    if free_end:
        cols += [np.concatenate([b2, zeros]), np.concatenate([zeros, b2])]
    else:
        cols.append(np.concatenate([b2 * t2[0], b2 * t2[1]]))
    sol = np.linalg.lstsq(np.stack(cols, axis=1), rhs, rcond=None)[0]
    seg_len = float(np.linalg.norm(p3 - p0))
    k = 0
    if free_start:
        p1 = p0 + sol[0:2]
        k = 2
    else:
        a = sol[0] if sol[0] > 1e-6 * seg_len else seg_len / 3.0
        p1 = p0 + a * t1
        k = 1
    if free_end:
        p2 = p3 + sol[k : k + 2]
    else:
        b = sol[k] if sol[k] > 1e-6 * seg_len else seg_len / 3.0
        p2 = p3 + b * t2
    if not np.all(np.isfinite(np.concatenate([p1, p2]))):
        return _generate(pts, u, t1, t2)
    # guard: a fit much longer than the points it approximates is degenerate
    poly_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    ctrl_len = float(np.sum(np.linalg.norm(np.diff(np.array([p0, p1, p2, p3]), axis=0), axis=1)))
    if ctrl_len > 3.0 * poly_len + 2.0:
        return _generate(pts, u, t1, t2)
    return np.array([p0, p1, p2, p3])


def _max_error(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    d = np.linalg.norm(evaluate(bez, u) - pts, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _reparameterize(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton step on ``(B(u) - p) . B'(u) = 0`` for every point."""
    q = evaluate(bez, u) - pts
    q1 = derivative(bez, u)
    q2 = second_derivative(bez, u)
    num = np.sum(q * q1, axis=1)
    den = np.sum(q1 * q1, axis=1) + np.sum(q * q2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    return np.clip(u - step, 0.0, 1.0)


def _fit(
    pts: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tol: float,
    out: list,
    free_start: bool = False,
    free_end: bool = False,
) -> None:
    if len(pts) == 2:
        d = float(np.linalg.norm(pts[1] - pts[0])) / 3.0
        out.append(np.array([pts[0], pts[0] + d * t1, pts[1] + d * t2, pts[1]]))
        return
    u = _chord_params(pts)
    bez = _generate(pts, u, t1, t2, free_start, free_end)
    err, split = _max_error(pts, bez, u)
    if err <= tol:
        out.append(bez)
        return
    for _ in range(_REPARAM_ITERATIONS):
        u = _reparameterize(pts, bez, u)
        bez = _generate(pts, u, t1, t2, free_start, free_end)
        prev = err
        err, split = _max_error(pts, bez, u)
        if err <= tol:
            out.append(bez)
            return
        # stop if progress is too slow to reach the tolerance in the steps left
        if err > 0.95 * prev:
            break
    split = min(max(split, 1), len(pts) - 2)
    tc = _unit(pts[split - 1] - pts[split + 1])
    if tc is None:
        tc = _unit(pts[split - 1] - pts[split])
        if tc is None:
            tc = -t1
    # the split point is interior: both halves must share its tangent (G1)
    _fit(pts[: split + 1], t1, tc, tol, out, free_start, False)
    _fit(pts[split:], -tc, t2, tol, out, False, free_end)


def _dedupe(pts: np.ndarray) -> np.ndarray:
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    return pts[keep]


def fit_cubic(points, t1=None, t2=None, u=None, iterations: int = _REPARAM_ITERATIONS):
    """One cubic from ``points[0]`` to ``points[-1]`` fitted to all points: ``(ctrl, max error in px, u)``.

    ``t1`` is the unit tangent at the start, pointing into the curve, and ``t2``
    the one at the end, pointing back into it (the convention of
    :func:`fit_polyline`); ``None`` leaves that end's handle free. ``u`` are
    starting parameters for the points (default: chord length); good ones make
    the fit converge in a few steps. The returned ``u`` are the refined
    parameters. Used to merge neighbouring pieces into one
    (:mod:`line2func.budget`).
    """
    pts = np.asarray(points, dtype=np.float64)
    if u is None:
        pts = _dedupe(pts)
        u = _chord_params(pts) if len(pts) > 1 else np.zeros(len(pts))
    else:
        u = np.asarray(u, dtype=np.float64)
        if len(u) != len(pts):
            raise ValueError("u needs one parameter per point")
    if len(pts) < 2 or float(np.ptp(pts, axis=0).max()) <= 1e-9:
        raise ValueError("need at least two distinct points")
    free_start, free_end = t1 is None, t2 is None
    t1 = _end_tangent(pts) if t1 is None else np.asarray(t1, dtype=np.float64)
    t2 = _end_tangent(pts[::-1]) if t2 is None else np.asarray(t2, dtype=np.float64)
    if len(pts) == 2:
        d = float(np.linalg.norm(pts[1] - pts[0])) / 3.0
        return np.array([pts[0], pts[0] + d * t1, pts[1] + d * t2, pts[1]]), 0.0, np.array([0.0, 1.0])
    bez = _generate(pts, u, t1, t2, free_start, free_end)
    err, _ = _max_error(pts, bez, u)
    for _ in range(iterations):
        u_next = _reparameterize(pts, bez, u)
        cand = _generate(pts, u_next, t1, t2, free_start, free_end)
        e, _ = _max_error(pts, cand, u_next)
        if e >= err:
            break
        bez, err, u = cand, e, u_next
    return bez, err, u


def fit_polyline(points, tolerance: float = 1.0, closed: bool = False) -> list[np.ndarray]:
    """Fit a chain of cubics (each ``(4, 2)``) to ``points`` within ``tolerance`` px.

    Consecutive pieces share end points and tangent directions. For
    ``closed=True`` the polyline is treated as a loop (``points[0]`` need not be
    repeated at the end) and the chain starts and ends at ``points[0]``.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    pts = _dedupe(np.asarray(points, dtype=np.float64))
    if closed:
        if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= 1e-9:
            pts = pts[:-1]
        if len(pts) < 3:
            closed = False
    if len(pts) < 2:
        return []
    out: list[np.ndarray] = []
    if not closed:
        # the outer ends have no neighbour to stay smooth with, so their
        # tangents are left free in the least-squares fit
        _fit(pts, _end_tangent(pts), _end_tangent(pts[::-1]), tolerance, out, True, True)
        return out

    # Closed loop: split at the point farthest from the seam into two open
    # halves that share smooth tangents at the seam and at the split point.
    loop = np.vstack([pts, pts[:1]])
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    far = min(max(far, 1), len(pts) - 1)
    seam = _unit(pts[1] - pts[-1])
    if seam is None:
        seam = _end_tangent(loop)
    mid = _unit(loop[far - 1] - loop[far + 1])
    if mid is None:
        mid = -seam
    _fit(loop[: far + 1], seam, mid, tolerance, out)
    _fit(loop[far:], -mid, -seam, tolerance, out)
    return out

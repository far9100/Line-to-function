"""Geometry core: cubic Bézier curves and their polynomial (power) form.

Conventions
-----------
* A curve is a ``(4, 2)`` float64 array of control points ``P0..P3``.
* Coordinates are image pixels: origin at the top-left corner, x to the right,
  y **down**. Pixel ``(row i, col j)`` covers ``[j, j+1) x [i, i+1)`` and its
  center is ``(j + 0.5, i + 0.5)``. Exporters that want y up (Desmos, the
  equations shown to users) flip with :func:`flip_y`.
* The power form of a curve is a ``(4, 2)`` array ``[a, b, c, d]`` such that
  ``B(t) = a t^3 + b t^2 + c t + d`` for ``0 <= t <= 1`` (column 0 is x(t),
  column 1 is y(t)).

Every function accepts anything array-like of shape ``(4, 2)``.
"""

from __future__ import annotations

import numpy as np

# coeffs = BERNSTEIN_TO_POWER @ ctrl,  ctrl = POWER_TO_BERNSTEIN @ coeffs
BERNSTEIN_TO_POWER = np.array(
    [
        [-1.0, 3.0, -3.0, 1.0],
        [3.0, -6.0, 3.0, 0.0],
        [-3.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)
POWER_TO_BERNSTEIN = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0 / 3.0, 1.0],
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
)

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def as_ctrl(ctrl) -> np.ndarray:
    """Return ``ctrl`` as a finite ``(4, 2)`` float64 array, or raise ValueError."""
    arr = np.asarray(ctrl, dtype=np.float64)
    if arr.shape != (4, 2):
        raise ValueError(f"expected control points of shape (4, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("control points must be finite")
    return arr


# ---------------------------------------------------------------------------
# Representation changes
# ---------------------------------------------------------------------------


def to_power(ctrl) -> np.ndarray:
    """Control points -> power coefficients ``[a, b, c, d]`` (shape ``(4, 2)``)."""
    return BERNSTEIN_TO_POWER @ as_ctrl(ctrl)


def from_power(coeffs) -> np.ndarray:
    """Power coefficients ``[a, b, c, d]`` -> control points (shape ``(4, 2)``)."""
    return POWER_TO_BERNSTEIN @ as_ctrl(coeffs)


def line(p0, p1) -> np.ndarray:
    """Straight segment from ``p0`` to ``p1`` as a cubic with evenly spaced controls."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    return as_ctrl([p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(ctrl, t) -> np.ndarray:
    """Points ``B(t)``; ``t`` may be a scalar or array, result has shape ``t.shape + (2,)``."""
    p = as_ctrl(ctrl)
    t = np.asarray(t, dtype=np.float64)[..., None]
    mt = 1.0 - t
    return (
        mt * mt * mt * p[0]
        + 3.0 * mt * mt * t * p[1]
        + 3.0 * mt * t * t * p[2]
        + t * t * t * p[3]
    )


def derivative(ctrl, t) -> np.ndarray:
    """First derivative ``B'(t)``."""
    p = as_ctrl(ctrl)
    d = np.diff(p, axis=0)  # (3, 2)
    t = np.asarray(t, dtype=np.float64)[..., None]
    mt = 1.0 - t
    return 3.0 * (mt * mt * d[0] + 2.0 * mt * t * d[1] + t * t * d[2])


def second_derivative(ctrl, t) -> np.ndarray:
    """Second derivative ``B''(t)``."""
    p = as_ctrl(ctrl)
    dd = np.diff(p, n=2, axis=0)  # (2, 2)
    t = np.asarray(t, dtype=np.float64)[..., None]
    return 6.0 * ((1.0 - t) * dd[0] + t * dd[1])


# ---------------------------------------------------------------------------
# Subdivision and transforms
# ---------------------------------------------------------------------------


def split(ctrl, t: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Split at ``t`` with de Casteljau; returns ``(left, right)``, both over ``[0, 1]``."""
    p = as_ctrl(ctrl)
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    p01 = p[0] + t * (p[1] - p[0])
    p12 = p[1] + t * (p[2] - p[1])
    p23 = p[2] + t * (p[3] - p[2])
    p012 = p01 + t * (p12 - p01)
    p123 = p12 + t * (p23 - p12)
    mid = p012 + t * (p123 - p012)
    return np.array([p[0], p01, p012, mid]), np.array([mid, p123, p23, p[3]])


def subcurve(ctrl, t0: float, t1: float) -> np.ndarray:
    """The piece of the curve between ``t0`` and ``t1``, reparameterized to ``[0, 1]``."""
    if not 0.0 <= t0 <= t1 <= 1.0:
        raise ValueError("need 0 <= t0 <= t1 <= 1")
    _, right = split(ctrl, t0)
    if t0 >= 1.0:
        return right
    piece, _ = split(right, (t1 - t0) / (1.0 - t0))
    return piece


def reverse(ctrl) -> np.ndarray:
    """Same curve traversed from ``P3`` to ``P0``."""
    return as_ctrl(ctrl)[::-1].copy()


def affine(ctrl, matrix) -> np.ndarray:
    """Apply a ``2x3`` affine matrix ``[[a, b, tx], [c, d, ty]]``.

    Bézier curves are affine-invariant, so transforming the control points
    transforms the whole curve exactly.
    """
    p = as_ctrl(ctrl)
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (2, 3):
        raise ValueError(f"expected a (2, 3) affine matrix, got {m.shape}")
    return p @ m[:, :2].T + m[:, 2]


def flip_y(ctrl, height: float) -> np.ndarray:
    """Flip between y-down image coordinates and y-up math coordinates."""
    return affine(ctrl, [[1.0, 0.0, 0.0], [0.0, -1.0, float(height)]])


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _quadratic_roots_in_unit(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a t^2 + b t + c`` that lie strictly inside ``(0, 1)``."""
    scale = max(abs(a), abs(b), abs(c))
    if scale == 0.0:
        return []
    a, b, c = a / scale, b / scale, c / scale
    if abs(a) < 1e-12:
        roots = [] if abs(b) < 1e-12 else [-c / b]
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return []
        sq = np.sqrt(disc)
        # numerically stable form
        q = -0.5 * (b + np.copysign(sq, b))
        roots = [q / a]
        if q != 0.0:
            roots.append(c / q)
    return [float(r) for r in roots if 0.0 < r < 1.0]


def bounding_box(ctrl) -> np.ndarray:
    """Tight axis-aligned bounds ``[xmin, ymin, xmax, ymax]`` of the curve."""
    p = as_ctrl(ctrl)
    a, b, c, _ = to_power(p)
    ts = [0.0, 1.0]
    for axis in range(2):
        # derivative of a t^3 + b t^2 + c t + d
        ts += _quadratic_roots_in_unit(3.0 * a[axis], 2.0 * b[axis], c[axis])
    pts = evaluate(p, np.array(ts))
    return np.concatenate([pts.min(axis=0), pts.max(axis=0)])


def arc_length(ctrl, t0: float = 0.0, t1: float = 1.0, pieces: int = 8) -> float:
    """Arc length between ``t0`` and ``t1`` (composite 16-point Gauss-Legendre)."""
    p = as_ctrl(ctrl)
    edges = np.linspace(t0, t1, pieces + 1)
    half = 0.5 * (edges[1:] - edges[:-1])  # (pieces,)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ts = mid[:, None] + half[:, None] * _GL_NODES[None, :]  # (pieces, 16)
    speed = np.linalg.norm(derivative(p, ts), axis=-1)
    return float(np.sum(half[:, None] * _GL_WEIGHTS[None, :] * speed))


def max_second_derivative(ctrl) -> float:
    """``max |B''(t)|`` over ``[0, 1]``; ``B''`` is linear in t, so it peaks at an end."""
    dd = np.diff(as_ctrl(ctrl), n=2, axis=0)
    return 6.0 * float(np.max(np.linalg.norm(dd, axis=1)))


def flatten(ctrl, tolerance: float = 0.1) -> np.ndarray:
    """Polyline through ``B(t)`` at uniform ``t`` whose chords stay within ``tolerance`` px.

    For uniform steps ``h`` the chord error is at most ``max|B''| * h^2 / 8``,
    so ``n = ceil(sqrt(max|B''| / (8 * tolerance)))`` segments suffice.
    Returns an ``(n + 1, 2)`` array whose first and last rows are ``P0`` and ``P3``.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    p = as_ctrl(ctrl)
    n = max(1, int(np.ceil(np.sqrt(max_second_derivative(p) / (8.0 * tolerance)))))
    pts = evaluate(p, np.linspace(0.0, 1.0, n + 1))
    pts[0], pts[-1] = p[0], p[3]
    return pts


def point_segment_distance(points, a, b) -> np.ndarray:
    """Distance from each of ``points`` (``(..., 2)``) to segments ``a -> b`` (broadcast)."""
    points = np.asarray(points, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    ap = points - a
    denom = np.sum(ab * ab, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(denom > 0.0, np.sum(ap * ab, axis=-1) / denom, 0.0)
    u = np.clip(u, 0.0, 1.0)
    closest = a + u[..., None] * ab
    return np.linalg.norm(points - closest, axis=-1)


def distance_to_points(ctrl, points, tolerance: float = 0.01) -> np.ndarray:
    """Distance from each point to the curve, accurate to about ``tolerance`` px.

    ``points`` has shape ``(m, 2)``; the result has shape ``(m,)``.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    poly = flatten(ctrl, tolerance)
    best = np.full(len(pts), np.inf)
    # chunk over segments to bound memory at (m x chunk)
    chunk = max(1, 2_000_000 // max(1, len(pts)))
    n_seg = len(poly) - 1
    for s in range(0, n_seg, chunk):
        e = min(s + chunk, n_seg)
        a, b = poly[s:e], poly[s + 1 : e + 1]
        d = point_segment_distance(pts[:, None, :], a[None], b[None])
        best = np.minimum(best, d.min(axis=1))
    return best


def closest_point(ctrl, point, tolerance: float = 1e-9, iterations: int = 20) -> tuple[float, float]:
    """Parameter ``t`` of the curve point nearest ``point`` and the distance to it.

    A dense sample finds the basin, then Newton's method on
    ``(B(t) - q) . B'(t) = 0`` refines it (clamped to ``[0, 1]``).
    """
    p = as_ctrl(ctrl)
    q = np.asarray(point, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, 65)
    d2 = np.sum((evaluate(p, ts) - q) ** 2, axis=-1)
    t = float(ts[np.argmin(d2)])
    for _ in range(iterations):
        diff = evaluate(p, t) - q
        d1 = derivative(p, t)
        f = float(diff @ d1)
        fp = float(d1 @ d1 + diff @ second_derivative(p, t))
        if fp <= 0.0:
            break
        t_new = min(1.0, max(0.0, t - f / fp))
        if abs(t_new - t) < tolerance:
            t = t_new
            break
        t = t_new
    return t, float(np.linalg.norm(evaluate(p, t) - q))

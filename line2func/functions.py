"""Explicit functions: every curve as pieces of ``y = f(x)`` or ``x = g(y)``.

The Desmos and LaTeX exports write each traced cubic in parametric form
(:mod:`line2func.export`). With ``form="function"`` they write explicit
functions instead:

* **Cut where the slope is +1 or -1.** Between two such points a cubic is
  either flat (``|x'| >= |y'|``, written ``y = f(x)``) or steep (written
  ``x = g(y)``), and its independent variable changes monotonically, so the
  function exists. Neighbouring pieces are joined again when the slope only
  touched +-1 and the curve kept its direction (never across a cusp, where it
  turns back).
* **Fit a polynomial of degree 1 to 3** in the independent variable ``u``,
  through both end points (so pieces and neighbouring curves join exactly),
  least squares in between, the lowest degree that fits. Degree 1 is written
  like a named line, ``y = m x + c``; higher degrees are centered on the piece,
  ``v = a0 + a1 (u - c) + a2 (u - c)^2 + a3 (u - c)^3``, which keeps the
  coefficients small.
* **Check after rounding.** The error is the largest distance along the
  dependent axis between the printed function and the curve. With a slope of
  at most 1 this bounds the distance between the two from above (it is at most
  sqrt(2) times too cautious). A piece that strays more than ``tolerance`` px
  is halved, up to :data:`MAX_DEPTH` times.

Numbers are fixed-point, never scientific notation (Desmos would read ``1e-5``
as "1 x e - 5"); each coefficient gets the decimals that keep its rounding
within about 0.005 px. Pieces whose independent variable spans less than
:data:`MIN_EXTENT` px (slivers next to a cut) are left out. Like the other
Desmos exports, the functions are in math orientation (y up).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import polynomial as P

from line2func.curves import CurveSet
from line2func.geometry import _quadratic_roots_in_unit, arc_length, derivative, evaluate, flip_y, to_power
from line2func.shapes import _num, _signed, _slope_decimals

FUNCTION_TOLERANCE = 0.25  # px: the largest distance allowed between a function piece and its curve (default)
MIN_EXTENT = 0.01  # px: pieces whose independent variable spans less are left out
MAX_DEPTH = 6  # a piece that does not fit is halved at most this many times
MAX_DECIMALS = 12
FIT_SAMPLES = 65  # samples per piece for the least-squares fit (more check the error, 4 per px of length)
AXES = ("x", "y")


# ---------------------------------------------------------------------------
# Cutting a cubic into monotonic pieces
# ---------------------------------------------------------------------------


def _cuts(p: np.ndarray) -> list[float] | None:
    """Parameters in (0, 1) where the slope is +1 or -1 (``y' = x'`` or ``y' = -x'``); None for a point."""
    a, b, c, _ = to_power(p)
    q = np.array([3.0 * a, 2.0 * b, c])  # x'(t) and y'(t): coefficients of t^2, t, 1
    scale = float(np.abs(q).max())
    if scale == 0.0:
        return None
    ts: set[float] = set()
    for sign in (1.0, -1.0):
        poly = q[:, 1] - sign * q[:, 0]
        # all but zero: the slope is exactly +-1 everywhere (a 45 degree line); the roots would be noise
        if np.abs(poly).max() > 1e-9 * scale:
            ts.update(_quadratic_roots_in_unit(*poly))
    return sorted(ts)


def _intervals(p: np.ndarray) -> list[tuple[float, float, int]]:
    """``(t0, t1, axis)`` pieces of ``p`` (math coordinates) over which the independent variable
    ``axis`` (0: x, 1: y) changes monotonically, in order."""
    cuts = _cuts(p)
    if cuts is None:
        return []
    edges = [0.0]
    for t in cuts:
        if edges[-1] + 1e-12 < t < 1.0 - 1e-12:
            edges.append(t)
    edges.append(1.0)
    out: list[list] = []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        d = derivative(p, 0.5 * (t0 + t1))
        axis = 0 if abs(d[0]) >= abs(d[1]) else 1
        forward = bool(d[axis] > 0)
        if out and out[-1][2] == axis and out[-1][3] == forward:
            out[-1][1] = t1  # the slope only touched +-1 here: still one function
        else:
            out.append([t0, t1, axis, forward])
    return [(t0, t1, axis) for t0, t1, axis, _ in out]


# ---------------------------------------------------------------------------
# Fitting one piece
# ---------------------------------------------------------------------------


def _decimals(scale: float) -> int:
    """Decimals for a coefficient multiplied by ``scale``, so rounding it moves the curve by <= ~0.005 px."""
    return int(min(MAX_DECIMALS, max(2, math.ceil(math.log10(max(scale, 1e-300) / 0.01)))))


def _through_ends(s: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    """Coefficients (lowest first) of the polynomial of ``degree`` in ``s`` through the first and last
    samples that fits all of them best in the least-squares sense."""
    sa, sb, va, vb = s[0], s[-1], v[0], v[-1]
    slope = (vb - va) / (sb - sa)
    coef = np.zeros(degree + 1)
    coef[:2] = va - slope * sa, slope
    if degree >= 2:
        # corrections that vanish at both ends: (s - sa)(s - sb) s^j
        base = P.polyfromroots([sa, sb])
        basis = [P.polymulx(base) if j else base for j in range(degree - 1)]
        a = np.stack([P.polyval(s, b) for b in basis], axis=1)
        q = np.linalg.lstsq(a, v - P.polyval(s, coef), rcond=None)[0]
        for qj, b in zip(q, basis):
            coef[: len(b)] += qj * b
    return coef


def _outward(lo: float, hi: float) -> list[float]:
    """The domain rounded outward to 0.01, so neighbouring pieces leave no gap."""
    return [math.floor(lo * 100.0 + 1e-7) / 100.0, math.ceil(hi * 100.0 - 1e-7) / 100.0]


def _piece(u: np.ndarray, v: np.ndarray, axis: int, degree: int) -> dict:
    """The function of ``degree`` through the samples ``v(u)``, rounded as printed, with its error."""
    lo, hi = float(min(u[0], u[-1])), float(max(u[0], u[-1]))
    if degree == 1:
        # the chord, written y = m x + c like a named line (line2func.shapes)
        dm = _slope_decimals(max(abs(lo), abs(hi)))
        m = round(float((v[-1] - v[0]) / (u[-1] - u[0])), dm)
        c0 = round(float(v[0] - m * u[0]), 2)
        coeffs, decimals, center = [c0, m], [2, dm], None
        f = c0 + m * u
    else:
        center = round(0.5 * (lo + hi), 2)
        w = max(abs(lo - center), abs(hi - center))
        idx = np.unique(np.linspace(0, len(u) - 1, FIT_SAMPLES).round().astype(int))
        s = (u - center) / w
        coef = _through_ends(s[idx], v[idx], degree)
        decimals = [_decimals(w**k) for k in range(degree + 1)]
        coeffs = [round(float(coef[k] / w**k), decimals[k]) for k in range(degree + 1)]
        f = P.polyval(u - center, coeffs)
    return {"axis": AXES[axis], "domain": _outward(lo, hi), "degree": degree, "center": center,
            "coeffs": coeffs, "decimals": decimals, "max_error": float(np.abs(f - v).max())}


def _fit(p: np.ndarray, t0: float, t1: float, axis: int, tolerance: float, depth: int = 0) -> list[dict]:
    """Function pieces for ``p`` between ``t0`` and ``t1`` (independent variable ``axis``)."""
    n = int(min(1025, max(129, 4.0 * arc_length(p, t0, t1))))
    pts = evaluate(p, np.linspace(t0, t1, n))
    u, v = pts[:, axis], pts[:, 1 - axis]
    if abs(u[-1] - u[0]) < MIN_EXTENT:
        return []  # a sliver: its slope is at most 1, so it spans less than MIN_EXTENT both ways
    best = None
    for degree in (1, 2, 3):
        piece = _piece(u, v, axis, degree)
        if piece["max_error"] <= tolerance:
            return [piece]
        if best is None or piece["max_error"] < best["max_error"]:
            best = piece
    if depth < MAX_DEPTH:
        tm = 0.5 * (t0 + t1)
        return _fit(p, t0, tm, axis, tolerance, depth + 1) + _fit(p, tm, t1, axis, tolerance, depth + 1)
    return [best]  # still over the tolerance; its max_error says by how much


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def function_pieces(ctrl, height: float, tolerance: float = FUNCTION_TOLERANCE) -> list[dict]:
    """The curve ``ctrl`` (image pixels, y down) as explicit functions in math coordinates (y up).

    Each piece is ``{"axis": "x" | "y"`` (the independent variable), ``"domain": [lo, hi]``,
    ``"degree"``, ``"center"`` (``None`` for degree 1, written ``v = c0 + m u``), ``"coeffs"``
    (lowest first, rounded as printed), ``"decimals"`` and ``"max_error"`` (px)}.
    A curve of zero length gives no pieces.
    """
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    p = flip_y(ctrl, height)
    pieces: list[dict] = []
    for t0, t1, axis in _intervals(p):
        pieces += _fit(p, t0, t1, axis, tolerance)
    return pieces


def function_latex(piece: dict) -> str:
    """One piece as a Desmos / LaTeX equation with its domain, e.g.
    ``y=12.30+0.4100\\left(x-52.00\\right)^{2}\\left\\{40.00\\le x\\le 64.00\\right\\}``."""
    u = piece["axis"]
    v = "y" if u == "x" else "x"
    coeffs, decimals = piece["coeffs"], piece["decimals"]
    if piece["center"] is None:
        body = f"{_num(coeffs[1], decimals[1])}{u}{_signed(coeffs[0], 2)}"
    else:
        c = piece["center"]
        shift = u if c == 0.0 else rf"\left({u}{_signed(-c, 2)}\right)"
        body = _num(coeffs[0], decimals[0])
        for k in range(1, len(coeffs)):
            if float(_num(coeffs[k], decimals[k])) == 0.0:
                continue
            body += _signed(coeffs[k], decimals[k]) + shift + (f"^{{{k}}}" if k > 1 else "")
    lo, hi = piece["domain"]
    return rf"{v}={body}\left\{{{_num(lo, 2)}\le {u}\le {_num(hi, 2)}\right\}}"


def curve_functions(ctrl, height: float, tolerance: float = FUNCTION_TOLERANCE) -> tuple[list[str], float]:
    """The curve as Desmos / LaTeX function equations and the largest error of its pieces (px)."""
    pieces = function_pieces(ctrl, height, tolerance)
    return [function_latex(piece) for piece in pieces], max((piece["max_error"] for piece in pieces), default=0.0)


def attach(curves: CurveSet, tolerance: float = FUNCTION_TOLERANCE) -> dict:
    """Set ``functions`` on every curve (after all geometry steps); returns
    ``{"count": equations, "max_error": px, "tolerance": px}``."""
    count, worst = 0, 0.0
    for c in curves.curves:
        c.functions, error = curve_functions(c.ctrl, curves.height, tolerance)
        count += len(c.functions)
        worst = max(worst, error)
    return {"count": count, "max_error": round(worst, 3), "tolerance": tolerance}

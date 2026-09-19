"""Recognize straight lines and circular arcs, and write them as named equations.

A traced cubic that stays within ``tolerance`` px of its chord is a **line**;
one that stays within ``tolerance`` px of a circle is an **arc**. Recognized
shapes are stored on the curve (``Curve.shape``, image coordinates) and can be
exported instead of the parametric form:

* line:  ``y = m x + c`` for x in [x0, x1]  (or ``x = m y + c`` when steep)
* arc:   ``(x - h)^2 + (y - k)^2 = r^2`` restricted to the side of the chord
  A-B that holds the arc. That restriction is a single linear inequality and
  is exact for any arc, including arcs longer than a half circle.

As with the parametric export, numbers are fixed-point (never scientific
notation). Decimals are chosen so rounding moves the drawn shape by at most
about 0.01 px.
"""

from __future__ import annotations

import math

import numpy as np

from line2func.curves import CurveSet
from line2func.geometry import as_ctrl, evaluate, point_segment_distance


def _fit_circle(pts: np.ndarray, iterations: int = 10) -> tuple[np.ndarray, float] | None:
    """Least-squares circle: algebraic (Kasa) start, then Gauss-Newton on geometric distance."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    c = np.array([sol[0] / 2.0, sol[1] / 2.0])
    r2 = sol[2] + c @ c
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    for _ in range(iterations):
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        if np.any(dist < 1e-9):
            break
        u = d / dist[:, None]
        jac = np.column_stack([-u, -np.ones(len(pts))])  # d(dist - r) / d(cx, cy, r)
        res = dist - r
        step = np.linalg.lstsq(jac, -res, rcond=None)[0]
        c = c + step[:2]
        r = r + step[2]
        if np.abs(step).max() < 1e-9:
            break
    return (c, float(r)) if r > 0 and np.all(np.isfinite(c)) else None


def classify(ctrl, tolerance: float = 0.5, max_radius: float = 1e4) -> dict | None:
    """``{"type": "line", ...}``, ``{"type": "arc", ...}`` or ``None`` for a cubic (image coords)."""
    p = as_ctrl(ctrl)
    pts = evaluate(p, np.linspace(0.0, 1.0, 65))
    a, b = p[0], p[3]
    chord = float(np.linalg.norm(b - a))
    if chord < 1e-6:
        return None
    if point_segment_distance(pts, a, b).max() <= tolerance:
        return {"type": "line", "p0": [round(float(v), 4) for v in a], "p1": [round(float(v), 4) for v in b]}
    fit = _fit_circle(pts)
    if fit is None:
        return None
    center, r = fit
    if r > max_radius or np.abs(np.linalg.norm(pts - center, axis=1) - r).max() > tolerance:
        return None
    ang = np.unwrap(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    steps = np.diff(ang)
    if not (np.all(steps >= -1e-9) or np.all(steps <= 1e-9)):
        return None  # doubles back: not a simple arc
    mid = evaluate(p, 0.5)
    return {
        "type": "arc",
        "center": [round(float(v), 4) for v in center],
        "radius": round(r, 4),
        "p0": [round(float(v), 4) for v in a],
        "p1": [round(float(v), 4) for v in b],
        "through": [round(float(v), 4) for v in mid],
        "sweep_deg": round(float(np.degrees(abs(ang[-1] - ang[0]))), 2),
    }


def recognize(curves: CurveSet, tolerance: float = 0.5) -> int:
    """Set ``shape`` on every curve that is a line or an arc; returns how many were recognized.

    Each shape also carries its ready-made Desmos equation under ``"desmos"``
    (the viewer copies it as is).
    """
    n = 0
    for c in curves.curves:
        c.shape = classify(c.ctrl, tolerance)
        if c.shape is not None:
            c.shape["desmos"] = named_latex(c.shape, curves.height)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Named equations (math orientation: y up)
# ---------------------------------------------------------------------------


def _num(v: float, decimals: int) -> str:
    text = f"{v:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text


def _signed(v: float, decimals: int) -> str:
    """``+1.25`` / ``-1.25`` (for use after another term)."""
    text = _num(v, decimals)
    return text if text.startswith("-") else "+" + text


def _slope_decimals(extent: float) -> int:
    """Decimals for a slope so that rounding moves the line by <= ~0.01 px over ``extent``."""
    return int(min(8, max(2, math.ceil(math.log10(max(extent, 1.0) / 0.02)))))


def named_latex(shape: dict, height: float) -> str:
    """The shape as a LaTeX equation with its domain restriction (Desmos syntax)."""
    flip = lambda p: (float(p[0]), float(height) - float(p[1]))  # noqa: E731
    le = r"\le "
    if shape["type"] == "line":
        (x0, y0), (x1, y1) = flip(shape["p0"]), flip(shape["p1"])
        if abs(x1 - x0) >= abs(y1 - y0):  # y as a function of x
            m = (y1 - y0) / (x1 - x0)
            c = y0 - m * x0
            d = _slope_decimals(max(abs(x0), abs(x1)))
            lo, hi = sorted((x0, x1))
            return rf"y={_num(m, d)}x{_signed(c, 2)}\left\{{{_num(lo, 2)}{le}x{le}{_num(hi, 2)}\right\}}"
        m = (x1 - x0) / (y1 - y0)
        c = x0 - m * y0
        d = _slope_decimals(max(abs(y0), abs(y1)))
        lo, hi = sorted((y0, y1))
        return rf"x={_num(m, d)}y{_signed(c, 2)}\left\{{{_num(lo, 2)}{le}y{le}{_num(hi, 2)}\right\}}"
    if shape["type"] == "arc":
        h, k = flip(shape["center"])
        r = float(shape["radius"])
        a, b, m = (np.array(flip(shape[key])) for key in ("p0", "p1", "through"))
        chord = b - a
        n = np.array([-chord[1], chord[0]])
        n = n / max(np.linalg.norm(n), 1e-12)
        if n @ (m - a) < 0:
            n = -n  # point the normal toward the arc
        extent = max(abs(h) + r, abs(k) + r)
        dn = _slope_decimals(extent)
        cst = -float(n @ a)
        circle = rf"\left(x{_signed(-h, 2)}\right)^{{2}}+\left(y{_signed(-k, 2)}\right)^{{2}}={_num(r, 2)}^{{2}}"
        side = rf"{_num(n[0], dn)}x{_signed(n[1], dn)}y{_signed(cst, 2)}\ge 0"
        return circle + rf"\left\{{{side}\right\}}"
    raise ValueError(f"unknown shape type {shape['type']!r}")

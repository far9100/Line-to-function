"""Curve refinement and measured line width / color.

Both work from the ink's **cross-section**: at points along a curve the ink map
is sampled along the normal (bilinear interpolation), and only the ink peak
that the curve sits on is kept, so a neighbouring or crossing line does not
pull the result.

* :func:`refine` moves every point to the ink-weighted center of its
  cross-section and refits the control points by least squares. Ends shared
  by consecutive pieces of a stroke move together, so strokes stay connected.
* :func:`measure` sets ``Curve.width`` (the cross-section's ink integral divided
  by the drawing's ink darkness: exact for anti-aliased lines) and, given the
  original image, ``Curve.color`` (the median color at the line's core).
  Limitation: in a drawing made *only* of lines thinner than ~2 px, darkness
  and width cannot be separated, and widths read up to ~0.5 px too wide.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from line2func.curves import Curve, CurveSet
from line2func.geometry import arc_length, derivative, evaluate

_STEP = 0.25  # px between cross-section samples


def _sample(ink: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear ink at points ``(..., 2)`` (pixel centers at +0.5)."""
    flat = pts.reshape(-1, 2)
    vals = ndimage.map_coordinates(ink, [flat[:, 1] - 0.5, flat[:, 0] - 0.5], order=1, mode="constant")
    return vals.reshape(pts.shape[:-1])


def _cross_sections(ctrl: np.ndarray, ink: np.ndarray, reach: float, spacing: float = 1.0):
    """``(t, points, normals, offsets, profiles)`` for samples every ``spacing`` px."""
    n = max(8, int(np.ceil(arc_length(ctrl) / spacing)))
    t = np.linspace(0.0, 1.0, n + 1)
    pts = evaluate(ctrl, t)
    d = derivative(ctrl, t)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-d[:, 1], d[:, 0]], axis=1)
    offsets = np.arange(-reach, reach + 1e-9, _STEP)
    grid = pts[:, None, :] + offsets[None, :, None] * normals[:, None, :]
    return t, pts, normals, offsets, _sample(ink, grid)


def _own_peaks(prof: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every cross-section (row of ``prof``), the ink bump the curve sits on.

    The peak is the maximum within 1.5 px of the curve; the bump extends outward
    from it while the ink keeps decreasing (or stays level) and exceeds 0.02, so
    it stops at the first valley before a neighbouring line. Returns
    ``(mask (n, m) bool, peak (n,), valid (n,) bool)``; rows without ink
    (peak < 0.2) are invalid.
    """
    n, m = prof.shape
    center = int(np.argmin(np.abs(offsets)))
    window = max(1, int(round(1.5 / _STEP)))
    lo_w, hi_w = max(0, center - window), min(m, center + window + 1)
    k = lo_w + np.argmax(prof[:, lo_w:hi_w], axis=1)
    rows = np.arange(n)
    peak = prof[rows, k]
    j = np.arange(m)[None, :]
    kk = k[:, None]
    # moving left from the peak, column j stays in the bump if ink[j] <= ink[j + 1] and > 0.02
    left_ok = np.ones_like(prof, dtype=bool)
    left_ok[:, :-1] = (prof[:, :-1] <= prof[:, 1:] + 1e-6) & (prof[:, :-1] > 0.02)
    right_ok = np.ones_like(prof, dtype=bool)
    right_ok[:, 1:] = (prof[:, 1:] <= prof[:, :-1] + 1e-6) & (prof[:, 1:] > 0.02)
    a = np.where(~left_ok & (j < kk), j, -1).max(axis=1) + 1
    b = np.where(~right_ok & (j > kk), j, m).min(axis=1) - 1
    mask = (j >= a[:, None]) & (j <= b[:, None])
    return mask, peak, peak >= 0.2


def _line_width(curves: CurveSet) -> float:
    return float(curves.meta.get("line_width") or 2.0)


def refine(curves: CurveSet, ink: np.ndarray, iterations: int = 2) -> CurveSet:
    """Snap curves to the ink centerline (in place; returns ``curves``). ``fill_outline`` curves are skipped."""
    ink = np.asarray(ink, dtype=np.float32)
    reach = max(2.0, _line_width(curves))
    for _ in range(iterations):
        for pieces in curves.strokes().values():
            _refine_stroke([c for c in pieces if "fill_outline" not in c.tags], ink, reach)
    return curves


def _refine_stroke(pieces: list[Curve], ink: np.ndarray, reach: float) -> None:
    targets = []
    for c in pieces:
        t, pts, normals, offsets, prof = _cross_sections(c.ctrl, ink, reach)
        mask, _, valid = _own_peaks(prof, offsets)
        wt = np.where(mask, prof, 0.0)
        shift = (wt @ offsets) / np.maximum(wt.sum(axis=1), 1e-9)
        q = np.where(valid[:, None], pts + normals * shift[:, None], pts)
        w = np.where(valid, 1.0, 0.05)  # points without ink keep their place, weakly
        targets.append((t, q, w))

    # shared ends: pieces k and k+1 meet at the same point; move it once
    ends = [[tq[1][0].copy(), tq[1][-1].copy()] for tq in targets]
    for k in range(len(pieces) - 1):
        if np.linalg.norm(pieces[k].ctrl[3] - pieces[k + 1].ctrl[0]) < 1e-6:
            joint = 0.5 * (ends[k][1] + ends[k + 1][0])
            ends[k][1] = ends[k + 1][0] = joint
    for c, (t, q, w), (p0, p3) in zip(pieces, targets, ends):
        mt = 1.0 - t
        b0, b1, b2, b3 = mt**3, 3 * mt * mt * t, 3 * mt * t * t, t**3
        rhs = q - np.outer(b0, p0) - np.outer(b3, p3)
        a = np.stack([b1, b2], axis=1) * np.sqrt(w)[:, None]
        sol = np.linalg.lstsq(a, rhs * np.sqrt(w)[:, None], rcond=None)[0]
        new = np.array([p0, sol[0], sol[1], p3])
        if np.all(np.isfinite(new)) and np.abs(new - c.ctrl).max() <= 3.0 * reach:
            c.ctrl = new


def measure(curves: CurveSet, ink: np.ndarray, rgb: np.ndarray | None = None) -> CurveSet:
    """Set ``width`` and (with ``rgb``) ``color`` on every curve (in place; returns ``curves``).

    Curves tagged ``"outline"`` are left as they are: they trace the edge of a
    thick stroke and carry the width and color of the centerline they replaced
    (:func:`line2func.outline.outline_thick`); measured at the edge, they would
    get a gray, half-dark color. So are the rings inside filled areas (``"fill"``,
    :mod:`line2func.fill`), which only matter where nothing can be filled.
    """
    ink = np.asarray(ink, dtype=np.float32)
    reach = max(3.0, 2.0 * _line_width(curves))
    channels = None
    if rgb is not None:
        channels = [np.asarray(rgb[:, :, k], dtype=np.float32) for k in range(3)]

    # pass 1: cross-sections of every curve
    per_curve = []
    peaks_all = []
    todo = [c for c in curves.curves if "outline" not in c.tags and "fill" not in c.tags]
    for c in todo:
        _, pts, _, offsets, prof = _cross_sections(c.ctrl, ink, reach, spacing=2.0)
        mask, peak, valid = _own_peaks(prof, offsets)
        integral = np.where(mask, prof, 0.0).sum(axis=1) * _STEP
        if valid.any():
            peaks_all.extend(peak[valid].tolist())
        else:
            # a very faint stroke (line2func.baseline.very_faint_line_mask): its ink never reaches 0.2.
            # Measured all the same, it gets the thin width of its little ink instead of none
            valid = peak >= 0.03
        per_curve.append(list(zip(pts[valid], integral[valid], peak[valid])))

    # Ink darkness: a thin line straddling two pixels peaks below full ink, so its
    # own peak cannot tell "thin and black" from "wider and gray". The typical
    # peak of the whole drawing (its thicker lines) sets the darkness instead.
    darkness = float(np.percentile(peaks_all, 90)) if peaks_all else 1.0

    # pass 2: width = ink integral / darkness, color at the line core
    for c, samples in zip(todo, per_curve):
        widths = [integral / max(darkness, peak) for _, integral, peak in samples]
        core = [p for p, _, peak in samples if peak >= 0.5 * darkness]
        # the outline of a filled area sees ink on one side only: no meaningful width
        c.width = float(np.median(widths)) if widths and "fill_outline" not in c.tags else None
        if channels is not None and core:
            core_pts = np.array(core)
            col = [float(np.median(_sample(ch, core_pts))) for ch in channels]
            c.color = "#" + "".join(f"{int(round(min(255.0, max(0.0, v)))):02x}" for v in col)
    return curves

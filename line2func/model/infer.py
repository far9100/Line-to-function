"""Whole-image inference with the multi-curve model.

1. **Tile**: 64² patches every 32 px (for the default patch size). Each patch
   owns only its central 32² *core*; the cores tile the image exactly, so every
   piece of output was predicted with 16 px of context on each side.
2. **Predict** all patches in batches; keep curves whose confidence > 0.5 and
   cut away stretches that retrace a longer curve of the same patch.
3. **Clip** each curve to its patch's core (exact cut points by bisection).
4. **Stitch** piece ends across cores into strokes: ends that meet and continue
   straight are joined first; ends that touch almost exactly may form a corner.
   Strokes that mostly retrace a longer stroke are dropped.
5. **Refit** each stroke into a compact chain of cubics (same corner handling
   and Schneider fitting as the baseline engine).

The result is a :class:`CurveSet` in the same format as the baseline engine's.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from line2func.baseline import BaselineParams, _fit_stroke, _support
from line2func.curves import Curve, CurveSet
from line2func.decisions import _angle
from line2func.geometry import arc_length, derivative, evaluate, subcurve


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else None


def _clip(ctrl: np.ndarray, lo: np.ndarray, hi: np.ndarray, samples: int = 48) -> list[np.ndarray]:
    """Pieces of the cubic inside the box ``[lo, hi]`` (cut points refined by bisection)."""
    t = np.linspace(0.0, 1.0, samples + 1)
    inside = np.all((evaluate(ctrl, t) >= lo) & (evaluate(ctrl, t) <= hi), axis=1)
    if inside.all():
        return [ctrl]
    if not inside.any():
        return []

    def inside_at(u: float) -> bool:
        p = evaluate(ctrl, u)
        return bool(np.all((p >= lo) & (p <= hi)))

    def boundary(a: float, b: float) -> float:
        # a and b straddle the box edge; return the parameter where it is crossed
        ia = inside_at(a)
        for _ in range(20):
            m = 0.5 * (a + b)
            if inside_at(m) == ia:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    pieces = []
    i = 0
    n = len(t)
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and inside[j + 1]:
            j += 1
        t0 = t[i] if i == 0 else boundary(t[i], t[i - 1])
        t1 = t[j] if j == n - 1 else boundary(t[j], t[j + 1])
        if t1 - t0 > 1e-4:
            pieces.append(subcurve(ctrl, t0, t1))
        i = j + 1
    return pieces


def _trim_overlaps(
    ctrls: list[np.ndarray], radius: float = 0.75, max_angle: float = 8.0, min_length: float = 1.5
) -> list[tuple[int, np.ndarray]]:
    """Remove the parts of curves that retrace a longer curve of the same patch.

    Where one smooth line is split into pieces is partly arbitrary, so the model
    sometimes predicts two pieces that overlap on a stretch of line. Curves are
    taken longest first; stretches of a curve lying within ``radius`` px of an
    accepted curve *and running parallel to it* (within ``max_angle``) are cut
    away. Duplicates run within a few degrees of each other; genuine crossings
    (the hard set goes down to 10°) are steeper than ``max_angle``, so they are
    untouched. Returns ``(source index, piece)`` pairs.
    """
    order = np.argsort([-arc_length(c) for c in ctrls])
    kept: list[tuple[int, np.ndarray]] = []
    acc_pts: list[np.ndarray] = []
    acc_dir: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, 65)
    cos_limit = np.cos(np.radians(max_angle))
    for i in order:
        c = ctrls[i]
        pts = evaluate(c, t)
        d = derivative(c, t)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        covered = np.zeros(len(t), bool)
        if acc_pts:
            all_pts, all_dir = np.vstack(acc_pts), np.vstack(acc_dir)
            dist, j = cKDTree(all_pts).query(pts)
            covered = (dist <= radius) & (np.abs(np.sum(d * all_dir[j], axis=1)) >= cos_limit)
        runs, start = [], None
        for k, cov in enumerate(covered):
            if not cov and start is None:
                start = k
            elif cov and start is not None:
                runs.append((start, k - 1))
                start = None
        if start is not None:
            runs.append((start, len(t) - 1))
        for a, b in runs:
            piece = c if (a == 0 and b == len(t) - 1) else subcurve(c, t[a], t[b])
            if arc_length(piece) >= min_length:
                kept.append((int(i), piece))
        dense = evaluate(c, np.linspace(0.0, 1.0, 129))
        dd = derivative(c, np.linspace(0.0, 1.0, 129))
        acc_pts.append(dense)
        acc_dir.append(dd / np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-9))
    return kept


def _end_dir(ctrl: np.ndarray, end: int) -> np.ndarray:
    """Outward unit tangent at an end (0 = start, 1 = end)."""
    d = derivative(ctrl, 1.0) if end == 1 else -derivative(ctrl, 0.0)
    u = _unit(d)
    if u is None:
        u = _unit(ctrl[3] - ctrl[0]) if end == 1 else _unit(ctrl[0] - ctrl[3])
    return u if u is not None else np.array([1.0, 0.0])


def stitch(
    pieces: list[np.ndarray], join_dist: float = 4.5, max_bend: float = 30.0, corner_dist: float = 1.0,
    corner_bend: float = 100.0,
) -> list[tuple[list[tuple[int, int]], bool]]:
    """Group pieces into strokes by joining their ends.

    Two ends join if they are within ``join_dist`` px and continue each other
    within ``max_bend`` degrees, or within ``corner_dist`` px with a bend of at
    most ``corner_bend`` (a corner). Straighter, closer pairs win.
    Returns ``[(sequence of (piece, entry_end), closed)]``.
    """
    ends, dirs = [], []
    for i, c in enumerate(pieces):
        for e in (0, 1):
            ends.append((i, e))
            dirs.append(_end_dir(c, e))
    pos = np.array([pieces[i][3 if e else 0] for i, e in ends]) if ends else np.zeros((0, 2))
    cands = []
    if len(ends) > 1:
        for a, b in cKDTree(pos).query_pairs(join_dist):
            if ends[a][0] == ends[b][0] and arc_length(pieces[ends[a][0]]) < 3 * join_dist:
                continue
            d = float(np.hypot(*(pos[a] - pos[b])))
            bend = _angle(dirs[a], -dirs[b])
            if bend <= max_bend or (d <= corner_dist and bend <= corner_bend):
                cands.append((d / join_dist + bend / max_bend, a, b))
    links: dict[tuple[int, int], tuple[int, int]] = {}
    used: set[int] = set()
    for _, a, b in sorted(cands):
        if a in used or b in used:
            continue
        used.update((a, b))
        links[ends[a]] = ends[b]
        links[ends[b]] = ends[a]

    seen = [False] * len(pieces)
    strokes = []

    def walk(i: int, entry: int):
        seq = []
        while True:
            seen[i] = True
            seq.append((i, entry))
            nxt = links.get((i, 1 - entry))
            if nxt is None:
                return seq, False
            if seen[nxt[0]]:
                return seq, nxt == seq[0]
            i, entry = nxt

    for i in range(len(pieces)):
        for e in (0, 1):
            if not seen[i] and (i, e) not in links:
                strokes.append(walk(i, e))
    for i in range(len(pieces)):
        if not seen[i]:
            strokes.append(walk(i, 0))
    return strokes


def _stroke_points(pieces: list[np.ndarray], seq: list[tuple[int, int]], spacing: float = 0.5) -> np.ndarray:
    pts = []
    for i, entry in seq:
        c = pieces[i] if entry == 0 else pieces[i][::-1]
        n = max(2, int(np.ceil(arc_length(c) / spacing)))
        p = evaluate(c, np.linspace(0.0, 1.0, n + 1))
        pts.append(p if not pts else p[1:])
    return np.vstack(pts)


def _drop_redundant(strokes: list[tuple[np.ndarray, bool]], radius: float = 1.5, overlap: float = 0.7):
    """Drop strokes that mostly retrace a longer stroke (leftover duplicate strands)."""
    lengths = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p, _ in strokes]
    kept: list[tuple[np.ndarray, bool]] = []
    kept_pts: list[np.ndarray] = []
    for i in np.argsort(lengths)[::-1]:
        pts, closed = strokes[i]
        if kept_pts:
            near = cKDTree(np.vstack(kept_pts)).query(pts)[0] <= radius
            if np.mean(near) >= overlap:
                continue
        kept.append((pts, closed))
        kept_pts.append(pts)
    return kept


@torch.no_grad()
def predict_patches(model, ink: np.ndarray, device, batch: int = 256, threshold: float = 0.5):
    """Run the model on every tile. Yields ``(core_lo, core_hi, ctrls, probs, widths)`` in image px."""
    ps = int(model.hparams["patch_size"])
    stride = ps // 2
    margin = (ps - stride) // 2
    h, w = ink.shape
    ny, nx = int(np.ceil(h / stride)), int(np.ceil(w / stride))
    padded = np.zeros((ny * stride + 2 * margin, nx * stride + 2 * margin), np.float32)
    padded[margin : margin + h, margin : margin + w] = ink
    origins = [(ix * stride, iy * stride) for iy in range(ny) for ix in range(nx)]
    amp = device.type == "cuda"
    for s in range(0, len(origins), batch):
        chunk = origins[s : s + batch]
        x = torch.from_numpy(np.stack([padded[oy : oy + ps, ox : ox + ps] for ox, oy in chunk]))[:, None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x.to(device))
        prob = torch.sigmoid(out["logit"]).float().cpu().numpy()
        ctrl = out["ctrl"].float().cpu().numpy().astype(np.float64) * ps
        width = out["width"].float().cpu().numpy().mean(axis=-1)
        for k, (ox, oy) in enumerate(chunk):
            keep = prob[k] > threshold
            # patch pixel (0, 0) sits at image (ox - margin, oy - margin)
            shift = np.array([ox - margin, oy - margin], dtype=np.float64)
            lo = np.array([ox, oy], dtype=np.float64)
            yield lo, lo + stride, ctrl[k][keep] + shift, prob[k][keep], width[k][keep]


def vectorize_model(
    ink: np.ndarray,
    model,
    device: str | torch.device | None = None,
    threshold: float = 0.5,
    fit_tolerance: float = 1.0,
) -> CurveSet:
    """Trace an ink map (``(H, W)``, 1 = line) with a multi-curve model."""
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    ink = np.asarray(ink, dtype=np.float32)
    h, w = ink.shape
    pieces: list[np.ndarray] = []
    widths: list[float] = []
    pad = 0.25  # cores overlap by a hair so clipped ends from neighbours meet
    for lo, hi, ctrls, probs, wids in predict_patches(model, ink, dev, threshold=threshold):
        if not len(ctrls):
            continue
        for i, curve in _trim_overlaps(list(ctrls)):
            for piece in _clip(curve, lo - pad, np.minimum(hi, [w, h]) + pad):
                if arc_length(piece) >= 0.5:
                    pieces.append(piece)
                    widths.append(float(wids[i]))

    result = CurveSet(w, h, meta={"engine": "model"})
    line_w = float(np.median(widths)) if widths else 0.0
    result.meta["line_width"] = round(line_w, 2)
    params = BaselineParams(fit_tolerance=fit_tolerance, smooth_sigma=0.0)
    radius = max(0.5, 0.5 * line_w)
    support = (ink > 0.25).astype(np.float32)
    from scipy import ndimage

    support = ndimage.binary_dilation(support, structure=np.ones((3, 3), bool)).astype(np.float32)
    stroke_id = 0
    strokes = [(_stroke_points(pieces, seq), closed) for seq, closed in stitch(pieces)]
    for pts, closed in _drop_redundant(strokes):
        if np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) < max(2.0, line_w):
            continue
        ctrls = _fit_stroke(pts[:-1] if closed else pts, closed, radius, params)
        if not ctrls:
            continue
        for c in ctrls:
            result.curves.append(Curve(c, stroke=stroke_id, confidence=round(_support(c, support), 3)))
        stroke_id += 1
    return result


def load_vectorizer(ckpt: str, device: str | None = None):
    """Load a multi-curve checkpoint; returns a function ``ink -> CurveSet``."""
    from line2func.model.nets import load_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(ckpt, dev)
    if meta["task"] != "multi_curve":
        raise ValueError(f"{ckpt} holds a {meta['task']} model; whole-image tracing needs a multi_curve model")

    def run(ink: np.ndarray, **kwargs) -> CurveSet:
        return vectorize_model(ink, model, dev, **kwargs)

    return run

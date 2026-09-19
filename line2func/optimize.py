"""Render-and-compare refinement (needs PyTorch; uses the GPU when available).

The curves are drawn with a differentiable rasterizer and adjusted by gradient
descent until the drawing they render matches the ink map - the analysis-by-
synthesis idea behind Deep Vectorization's refinement stage (arXiv:2003.05471).
Unlike the cross-section refinement in :mod:`line2func.attributes`, which only
slides points sideways onto the ink, this sees the whole picture: where a line
is too thick or thin, bent the wrong way, or overshoots a corner.

Per curve the optimized parameters are the four control points, the width and
the ink darkness (so a faint stroke is not "explained" by making it thin).
Consecutive pieces of a stroke stay joined (their shared end is re-averaged
after every step), movement is kept small by a pull toward the start, and
filled outlines are kept as they are.

Rendering: each curve is sampled into short segments; every segment draws a
soft capsule (``sigmoid((r - d) / softness)``) into a small window of pixels,
and the curves combine by maximum, like ink on paper.
"""

from __future__ import annotations

import numpy as np
import torch

from line2func.curves import CurveSet
from line2func.geometry import arc_length
from line2func.render import FILLED_TAGS, filled_area, stroke_loops

_SIZES = (8, 12, 16, 24, 32, 48, 64)  # window sides (px) the segments are bucketed into


def _bernstein(t: torch.Tensor) -> torch.Tensor:
    mt = 1.0 - t
    return torch.stack([mt**3, 3 * mt * mt * t, 3 * mt * t * t, t**3], dim=-1)


class Renderer:
    """Differentiable renderer for a fixed set of curves on an ``h x w`` canvas.

    Each curve is sampled about every ``seg_len`` px of its initial length (the
    sampling stays fixed while the curves move a little); consecutive samples
    form segments. Every segment draws a soft capsule into a square window of
    pixels, and the windows are bucketed by size, so a few thick strokes do
    not make every window large.
    """

    def __init__(self, lengths: np.ndarray, h: int, w: int, device, softness: float = 0.35, seg_len: float = 3.0):
        self.h, self.w, self.device, self.softness = h, w, device, softness
        k = np.maximum(np.ceil(np.asarray(lengths, dtype=float) / seg_len).astype(int) + 1, 2)  # samples per curve
        first = np.cumsum(k) - k
        t = np.concatenate([np.linspace(0.0, 1.0, ki) for ki in k]) if len(k) else np.zeros(0)
        seg = np.concatenate([f + np.arange(ki - 1) for f, ki in zip(first, k)]) if len(k) else np.zeros(0)
        owner = np.repeat(np.arange(len(k)), k)
        self.owner = torch.from_numpy(owner).to(device)  # curve of each sample
        self.basis = _bernstein(torch.from_numpy(t.astype(np.float32)).to(device))  # (N, 4)
        self.seg = torch.from_numpy(seg.astype(np.int64)).to(device)  # segment = samples (seg, seg + 1)
        self.seg_owner = self.owner[self.seg]
        self.sizes = torch.tensor(_SIZES, dtype=torch.float32, device=device)

    def render(self, ctrl: torch.Tensor, half_width: torch.Tensor, alpha: torch.Tensor, margin: int = 2):
        """Coverage-times-darkness image ``(h, w)`` for ``ctrl`` ``(n, 4, 2)``, ``half_width``/``alpha`` ``(n,)``."""
        pts = torch.einsum("sj,sjd->sd", self.basis, ctrl[self.owner])  # (N, 2)
        a, b = pts[self.seg], pts[self.seg + 1]
        r, al = half_width[self.seg_owner], alpha[self.seg_owner]
        out = torch.zeros(self.h * self.w, device=self.device, dtype=pts.dtype)
        if not len(a):
            return out.view(self.h, self.w)
        with torch.no_grad():  # windows anchored at each segment's bounding box
            lo = torch.minimum(a, b) - r[:, None] - margin
            need = (torch.maximum(a, b) + r[:, None] + margin - lo).amax(dim=1) + 1.0
            bucket = torch.bucketize(need, self.sizes).clamp_max(len(_SIZES) - 1)
        for k in torch.unique(bucket).tolist():
            sel = torch.nonzero(bucket == k).squeeze(1)
            out = self._splat(out, a[sel], b[sel], r[sel], al[sel], lo[sel], _SIZES[k])
        return out.view(self.h, self.w)

    def _splat(self, out, a, b, r, al, lo, size):
        x0 = torch.floor(lo[:, 0]).long()
        y0 = torch.floor(lo[:, 1]).long()
        off = torch.arange(size, device=self.device)
        px = x0[:, None, None] + off[None, None, :]  # (S, 1, size)
        py = y0[:, None, None] + off[None, :, None]  # (S, size, 1)
        ab = b - a
        denom = (ab * ab).sum(-1).clamp_min(1e-9)
        apx = px.float() + 0.5 - a[:, 0, None, None]
        apy = py.float() + 0.5 - a[:, 1, None, None]
        u = ((apx * ab[:, 0, None, None] + apy * ab[:, 1, None, None]) / denom[:, None, None]).clamp(0.0, 1.0)
        dx = apx - u * ab[:, 0, None, None]
        dy = apy - u * ab[:, 1, None, None]
        dist = torch.sqrt(dx * dx + dy * dy + 1e-9)
        cov = torch.sigmoid((r[:, None, None] - dist) / self.softness) * al[:, None, None]
        pxb, pyb = px.expand_as(cov), py.expand_as(cov)
        valid = (pxb >= 0) & (pxb < self.w) & (pyb >= 0) & (pyb < self.h)
        idx = (pyb.clamp(0, self.h - 1) * self.w + pxb.clamp(0, self.w - 1)).reshape(-1)
        vals = torch.where(valid, cov, torch.zeros_like(cov)).reshape(-1)
        return out.scatter_reduce(0, idx, vals, reduce="amax", include_self=True)


def _joints(curves: CurveSet, movable: list[int]) -> list[tuple[int, int]]:
    """Pairs (i, j) of movable-curve indices where curve i's end is curve j's start."""
    pos = {ci: k for k, ci in enumerate(movable)}
    index = {id(c): i for i, c in enumerate(curves.curves)}  # by identity: Curve holds arrays
    pairs = []
    for pieces in curves.strokes().values():
        for p, q in zip(pieces[:-1], pieces[1:]):
            ip, iq = index[id(p)], index[id(q)]
            if ip in pos and iq in pos and np.linalg.norm(p.ctrl[3] - q.ctrl[0]) < 1e-3:
                pairs.append((pos[ip], pos[iq]))
    return pairs


def optimize(
    curves: CurveSet,
    ink: np.ndarray,
    steps: int = 150,
    lr: float = 0.05,
    anchor: float = 0.02,
    device: str | None = None,
) -> dict:
    """Refine ``curves`` in place so their rendering matches ``ink``; returns a small report."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ink = np.asarray(ink, dtype=np.float32)
    h, w = ink.shape
    line_w = float(curves.meta.get("line_width") or 2.0)
    # filled outlines and the rings inside them (line2func.fill) stay as they are
    movable = [i for i, c in enumerate(curves.curves) if not any(t in c.tags for t in FILLED_TAGS + ("fill",))]
    if not movable:
        return {"steps": 0}
    fixed = filled_area(curves, w, h) if stroke_loops(curves) else np.zeros_like(ink)

    ctrl0 = np.stack([curves.curves[i].ctrl for i in movable]).astype(np.float32)
    width0 = np.array([curves.curves[i].width or line_w for i in movable], dtype=np.float32)
    lum = []
    for i in movable:
        col = curves.curves[i].color
        lum.append(1.0 - (0.299 * int(col[1:3], 16) + 0.587 * int(col[3:5], 16) + 0.114 * int(col[5:7], 16)) / 255.0
                   if col else 0.9)
    alpha0 = np.clip(np.array(lum, dtype=np.float32), 0.05, 0.99)
    lengths = np.array([arc_length(c) for c in ctrl0])

    renderer = Renderer(lengths, h, w, dev)
    target = torch.from_numpy(ink).to(dev)
    fixed_t = torch.from_numpy(fixed.astype(np.float32)).to(dev) * float(np.percentile(ink[ink > 0.3], 90) if (ink > 0.3).any() else 1.0)
    ctrl_init = torch.from_numpy(ctrl0).to(dev)
    ctrl = ctrl_init.clone().requires_grad_(True)
    log_w = torch.log(torch.from_numpy(width0).to(dev) / 2.0).requires_grad_(True)  # half width, log space
    logit_a = torch.logit(torch.from_numpy(alpha0).to(dev)).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [ctrl], "lr": lr}, {"params": [log_w, logit_a], "lr": lr * 0.5}])
    pairs = _joints(curves, movable)
    ji = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
    jj = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)

    # compare only where there is ink or a curve (the rest is paper on both sides)
    with torch.no_grad():
        start = renderer.render(ctrl, torch.exp(log_w), torch.sigmoid(logit_a))
        roi = ((target > 0.02) | (start > 0.02)).float()
        roi = torch.nn.functional.max_pool2d(roi[None, None], 5, stride=1, padding=2)[0, 0]
        n_roi = roi.sum().clamp_min(1.0)

    def loss_fn():
        img = torch.maximum(renderer.render(ctrl, torch.exp(log_w), torch.sigmoid(logit_a)), fixed_t)
        data = (((img - target) ** 2) * roi).sum() / n_roi
        move = ((ctrl - ctrl_init) ** 2).sum(-1).mean()
        return data + anchor * move / (line_w**2), data

    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, data = loss_fn()
        loss.backward()
        opt.step()
        with torch.no_grad():
            if len(pairs):  # keep strokes joined: move each shared end to the average
                mid = 0.5 * (ctrl[ji, 3] + ctrl[jj, 0])
                ctrl[ji, 3] = mid
                ctrl[jj, 0] = mid
            log_w.clamp_(np.log(0.3), np.log(2.0 * line_w))
        value = data.item()
        if first is None:
            first = value
        last = value

    new_ctrl = ctrl.detach().cpu().numpy().astype(np.float64)
    new_w = (2.0 * torch.exp(log_w)).detach().cpu().numpy()
    for k, i in enumerate(movable):
        c = curves.curves[i]
        if np.all(np.isfinite(new_ctrl[k])):
            c.ctrl = new_ctrl[k]
            c.width = float(new_w[k])
    return {"steps": steps, "loss_start": first, "loss_end": last, "curves": len(movable), "device": str(dev)}

"""Anti-aliased rasterizer for cubic curves.

Each curve is flattened to a polyline, and every pixel gets the signed distance
``d - r`` from its center to the nearest stroke (``d`` = distance to the
centerline, ``r`` = half the line width). Coverage is
``clip(0.5 - (d - r), 0, 1)``: a 1-pixel linear ramp across the stroke edge,
so a straight line of width ``w`` puts about ``w`` units of ink in each pixel
column it crosses. Overlapping strokes take the maximum coverage, not the sum.

The renderer is used to draw ``overlay.png`` and to synthesize training images,
so it is exact about geometry (pixel centers at ``+0.5``, see
:mod:`line2func.geometry`) and supports a different width per curve.
:func:`filled_area` fills the outlines of filled strokes (``outline``,
``fill_outline``); the quality check, the second pass and render-and-compare
use it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from line2func.curves import Curve, CurveSet
from line2func.geometry import as_ctrl, flatten

# Long segments are cut to this length so every segment fits a small window.
_MAX_SEGMENT = 4.0
# Upper bound on window pixels evaluated at once (memory ~ 40 bytes each).
_CHUNK_PIXELS = 2_000_000

# Distinguishable stroke colors for overlays (RGB).
PALETTE = (
    (230, 25, 75),
    (0, 130, 200),
    (60, 180, 75),
    (245, 130, 48),
    (145, 30, 180),
    (0, 160, 160),
    (240, 50, 230),
    (128, 128, 0),
)


def _ctrl_list(curves) -> list[np.ndarray]:
    if isinstance(curves, CurveSet):
        curves = curves.curves
    if isinstance(curves, np.ndarray) and curves.shape == (4, 2):
        curves = [curves]
    return [c.ctrl if isinstance(c, Curve) else as_ctrl(c) for c in curves]


def _widths(line_width, n: int) -> np.ndarray:
    """Normalize ``line_width`` to an ``(n, 2)`` array of (start, end) widths per curve.

    Accepts a scalar, one width per curve ``(n,)``, or tapered widths ``(n, 2)``
    that change linearly in ``t`` from the curve's start to its end.
    """
    w = np.asarray(line_width, dtype=np.float64)
    if w.ndim == 0:
        w = np.full((n, 2), float(w))
    elif w.shape == (n,):
        w = np.repeat(w[:, None], 2, axis=1)
    if w.shape != (n, 2):
        raise ValueError(f"line_width must be a scalar, ({n},) or ({n}, 2)")
    if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("line widths must be positive and finite")
    return w


def _segments(ctrls: Sequence[np.ndarray], widths: np.ndarray, tolerance: float):
    """Polyline segments ``(a, b, half_width)`` of all curves, each at most ``_MAX_SEGMENT`` long."""
    starts, ends, t_start, t_end, w_start, w_end = [], [], [], [], [], []
    for ctrl, (w0, w1) in zip(ctrls, widths):
        poly = flatten(ctrl, tolerance)  # points at uniform t = k / n
        n = len(poly) - 1
        ts = np.arange(n + 1) / n
        starts.append(poly[:-1])
        ends.append(poly[1:])
        t_start.append(ts[:-1])
        t_end.append(ts[1:])
        w_start.append(np.full(n, w0))
        w_end.append(np.full(n, w1))
    if not starts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    a, b = np.concatenate(starts), np.concatenate(ends)
    ta, tb = np.concatenate(t_start), np.concatenate(t_end)
    wa, wb = np.concatenate(w_start), np.concatenate(w_end)

    # cut long segments into k equal pieces; the width is evaluated per piece,
    # so tapered strokes taper even where the curve is flattened coarsely (np.intp: repeat counts must be the
    # platform's index type, which is 32 bits in the browser's WebAssembly)
    k = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / _MAX_SEGMENT)).astype(np.intp)
    seg = np.repeat(np.arange(len(a)), k)
    first = np.repeat(np.cumsum(k) - k, k)
    j = (np.arange(len(seg)) - first).astype(np.float64)  # piece index within its segment
    kk = k[seg].astype(np.float64)
    d = b[seg] - a[seg]
    a, b = a[seg] + d * (j / kk)[:, None], a[seg] + d * ((j + 1) / kk)[:, None]
    t_mid = ta[seg] + (tb[seg] - ta[seg]) * (j + 0.5) / kk
    r = 0.5 * (wa[seg] + (wb[seg] - wa[seg]) * t_mid)
    return a, b, r


def signed_distance(curves, width: int, height: int, line_width=2.0, tolerance: float = 0.05) -> np.ndarray:
    """Per-pixel ``distance_to_centerline - half_width``, clamped above at 0.5.

    Pixels farther than half a pixel outside every stroke hold ``+inf``
    (they receive no ink). Shape ``(height, width)``, float64.
    """
    ctrls = _ctrl_list(curves)
    widths = _widths(line_width, len(ctrls))
    a, b, r = _segments(ctrls, widths, tolerance)
    sd = np.full(width * height, np.inf)
    if len(a) == 0:
        return sd.reshape(height, width)

    # Visit segments in chunks; each segment scans a K x K window around it.
    order = np.argsort(r, kind="stable")  # similar widths share a window size
    a, b, r = a[order], b[order], r[order]
    i = 0
    while i < len(a):
        k_win = int(np.ceil(_MAX_SEGMENT + 2.0 * r[min(len(r) - 1, i)] + 3.0))
        n = max(1, _CHUNK_PIXELS // (k_win * k_win))
        # the window must fit the widest segment of this chunk
        j = i + n
        k_win = int(np.ceil(_MAX_SEGMENT + 2.0 * r[min(len(r), j) - 1] + 3.0))
        ca, cb, cr = a[i:j], b[i:j], r[i:j]
        i = j

        x0 = np.floor(np.minimum(ca[:, 0], cb[:, 0]) - cr - 1.0).astype(np.int64)
        y0 = np.floor(np.minimum(ca[:, 1], cb[:, 1]) - cr - 1.0).astype(np.int64)
        off = np.arange(k_win)
        px = x0[:, None, None] + off[None, None, :]  # (S, 1, K)
        py = y0[:, None, None] + off[None, :, None]  # (S, K, 1)
        cx, cy = px + 0.5, py + 0.5

        abx, aby = (cb - ca)[:, 0], (cb - ca)[:, 1]
        denom = abx * abx + aby * aby
        safe = np.where(denom > 0.0, denom, 1.0)
        apx = cx - ca[:, 0, None, None]
        apy = cy - ca[:, 1, None, None]
        u = (apx * abx[:, None, None] + apy * aby[:, None, None]) / safe[:, None, None]
        u = np.clip(np.where(denom[:, None, None] > 0.0, u, 0.0), 0.0, 1.0)
        dx = apx - u * abx[:, None, None]
        dy = apy - u * aby[:, None, None]
        val = np.sqrt(dx * dx + dy * dy) - cr[:, None, None]

        px, py = np.broadcast_to(px, val.shape), np.broadcast_to(py, val.shape)
        keep = (val < 0.5) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        np.minimum.at(sd, (py[keep] * width + px[keep]), val[keep])
    return sd.reshape(height, width)


def rasterize(curves, width: int, height: int, line_width=2.0, tolerance: float = 0.05) -> np.ndarray:
    """Ink coverage in ``[0, 1]`` (1 = fully inked), shape ``(height, width)``, float32.

    ``curves`` may be a :class:`CurveSet`, a list of :class:`Curve` or of
    ``(4, 2)`` control-point arrays. ``line_width`` is in pixels: one value,
    one per curve, or ``(start, end)`` per curve for tapered strokes.
    ``tolerance`` is the flattening error in pixels.
    """
    sd = signed_distance(curves, width, height, line_width, tolerance)
    return np.clip(0.5 - sd, 0.0, 1.0).astype(np.float32)


def render_lineart(
    curves, width: int, height: int, line_width=2.0, ink: int = 0, paper: int = 255
) -> np.ndarray:
    """Grayscale uint8 drawing: ``ink`` lines on a ``paper`` background."""
    cov = rasterize(curves, width, height, line_width)
    img = paper + cov * (float(ink) - float(paper))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _to_rgb(image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as im:
            return np.asarray(im.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"unsupported image shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def render_overlay(
    image,
    curves,
    line_width: float = 1.5,
    fade: float = 0.5,
    colors: Iterable[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw ``curves`` over ``image`` (path or array); returns an RGB uint8 array.

    The background is blended toward white by ``fade`` (0 = unchanged) so the
    curves stand out. Each stroke gets a palette color, cycling by stroke id.
    """
    if not 0.0 <= fade <= 1.0:
        raise ValueError("fade must be in [0, 1]")
    rgb = _to_rgb(image).astype(np.float32)
    height, width = rgb.shape[:2]
    out = rgb + fade * (255.0 - rgb)

    palette = np.asarray(list(colors) if colors is not None else PALETTE, dtype=np.float32)
    if isinstance(curves, CurveSet):
        items = curves.curves
    else:
        items = [c if isinstance(c, Curve) else Curve(c, stroke=i) for i, c in enumerate(curves)]
    groups: dict[int, list[Curve]] = {}
    for c in items:
        groups.setdefault(c.stroke % len(palette), []).append(c)
    for color_idx, group in groups.items():
        cov = rasterize(group, width, height, line_width)[:, :, None]
        out = out * (1.0 - cov) + palette[color_idx] * cov
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


FILLED_TAGS = ("outline", "fill_outline")
SOLID_SHARE = 0.8  # of the drawing's dark ink (meta["ink_dark"]): a filled area this dark is solid


def is_solid(tone: float | None, dark: float) -> bool:
    """Whether a filled area of this tone counts as solid ink rather than as a shadow.

    Judged against the drawing's own dark ink, not against black: a light pencil
    drawing has no black in it, and its lines are still drawn at full darkness
    everywhere they are drawn. An area with no measured tone counts as solid, so
    anything traced before tones were measured looks the way it always did.
    """
    return tone is None or dark <= 0.0 or tone >= SOLID_SHARE * dark


def stroke_loops(curves, tags: tuple[str, ...] = FILLED_TAGS) -> list[np.ndarray]:
    """Closed polygons of the strokes whose curves carry one of ``tags`` (outlines)."""
    items = curves.curves if isinstance(curves, CurveSet) else list(curves)
    groups: dict[int, list[Curve]] = {}
    for c in items:
        if isinstance(c, Curve) and any(t in c.tags for t in tags):
            groups.setdefault(c.stroke, []).append(c)
    loops = []
    for pieces in groups.values():
        pts = [flatten(c.ctrl, 0.1) for c in pieces]
        poly = np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
        if len(poly) >= 3:
            loops.append(poly)
    return loops


def fill_loops(loops: Sequence[np.ndarray], width: int, height: int, supersample: int = 4,
               union: bool = False) -> np.ndarray:
    """Anti-aliased coverage (float32, ``(height, width)``) of closed polygons, even-odd rule.

    Each polygon is ``(n, 2)`` in image pixels; nested loops (holes) cancel out.
    With ``union=True`` the polygons are joined instead: overlaps stay covered.
    """
    from PIL import Image, ImageDraw

    out = np.zeros((height, width), dtype=np.float32)
    loops = [np.asarray(p, dtype=np.float64) for p in loops if len(p) >= 3]
    if not loops:
        return out
    allpts = np.vstack(loops)
    x0 = int(max(0, np.floor(allpts[:, 0].min())))
    y0 = int(max(0, np.floor(allpts[:, 1].min())))
    x1 = int(min(width, np.ceil(allpts[:, 0].max()) + 1))
    y1 = int(min(height, np.ceil(allpts[:, 1].max()) + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    s = supersample
    acc = np.zeros(((y1 - y0) * s, (x1 - x0) * s), dtype=bool)
    for poly in loops:
        # each loop is drawn in its own bounding box, then XOR-ed in (even-odd)
        bx0 = int(max(x0, np.floor(poly[:, 0].min())))
        by0 = int(max(y0, np.floor(poly[:, 1].min())))
        bx1 = int(min(x1, np.ceil(poly[:, 0].max()) + 1))
        by1 = int(min(y1, np.ceil(poly[:, 1].max()) + 1))
        if bx1 <= bx0 or by1 <= by0:
            continue
        img = Image.new("1", ((bx1 - bx0) * s, (by1 - by0) * s), 0)
        ImageDraw.Draw(img).polygon([((x - bx0) * s, (y - by0) * s) for x, y in poly], fill=1)
        window = acc[(by0 - y0) * s : (by1 - y0) * s, (bx0 - x0) * s : (bx1 - x0) * s]
        if union:
            window |= np.asarray(img, dtype=bool)
        else:
            window ^= np.asarray(img, dtype=bool)
    cov = acc.reshape(y1 - y0, s, x1 - x0, s).mean(axis=(1, 3))
    out[y0:y1, x0:x1] = cov
    return out


def filled_area(curves, width: int, height: int, supersample: int = 4) -> np.ndarray:
    """Coverage of every filled stroke: the fill areas with their holes, joined with thick strokes' outlines.

    The loops of fill areas (``fill_outline``) are the areas' edges and never
    cross each other, so they combine even-odd and a hole stays open. A thick
    stroke's outline (``outline``) is one loop of its own that may overlap
    others, so those are joined.
    """
    cov = fill_loops(stroke_loops(curves, ("fill_outline",)), width, height, supersample)
    thick = stroke_loops(curves, ("outline",))
    if thick:
        cov = np.maximum(cov, fill_loops(thick, width, height, supersample, union=True))
    return cov


def fill_share(curves, width: int, height: int, dark: float) -> np.ndarray:
    """How much of ``dark`` each filled area is drawn at: 0 outside one, ``tone / dark`` inside.

    A filled area is put on the paper at its own measured tone, not at the
    drawing's dark ink, so a shadow filled at its own gray has to be rendered -
    and judged - as that gray. Areas with no measured tone give 0, and the
    caller then leaves them at full coverage, the way they were drawn before
    tones were measured.
    """
    items = curves.curves if isinstance(curves, CurveSet) else list(curves)
    out = np.zeros((height, width), dtype=np.float32)
    if not (dark > 0.0):
        return out
    groups: dict[int, list[Curve]] = {}
    for c in items:
        if isinstance(c, Curve) and c.tone is not None and any(t in c.tags for t in FILLED_TAGS):
            groups.setdefault(c.stroke, []).append(c)
    for pieces in groups.values():
        loops = stroke_loops(pieces, FILLED_TAGS)
        if not loops:
            continue
        share = min(1.0, float(np.median([c.tone for c in pieces])) / dark)
        painted = fill_loops(loops, width, height, supersample=1, union=True) > 0.5
        np.maximum(out, share * painted, out=out)
    return out


def render_coverage(curves, width: int, height: int, line_width=2.0, fill_share=None) -> np.ndarray:
    """Coverage of a whole result: outline strokes filled, all other curves as lines.

    ``fill_share`` (see :func:`fill_share`) dims each filled area to the share of
    the drawing's dark ink it is really drawn at, so a shadow counts as the gray
    it is. Lines drawn over an area keep their own coverage.
    """
    items = curves.curves if isinstance(curves, CurveSet) else list(curves)
    widths = _widths(line_width, len(items)) if len(items) else np.zeros((0, 2))
    lines = [(c, w) for c, w in zip(items, widths)
             if not (isinstance(c, Curve) and any(t in c.tags for t in FILLED_TAGS))]
    cov = np.zeros((height, width), dtype=np.float32)
    if lines:
        cov = rasterize([c for c, _ in lines], width, height, line_width=np.array([w for _, w in lines]))
    if stroke_loops(items):
        area = filled_area(items, width, height)
        if fill_share is not None:
            area = area * np.where(fill_share > 0, fill_share, 1.0)
        cov = np.maximum(cov, area)
    return cov


def save_png(path: str | Path, image: np.ndarray) -> None:
    """Save a uint8 grayscale or RGB array as PNG."""
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(path)

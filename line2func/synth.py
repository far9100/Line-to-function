"""Synthetic line art with exact ground truth.

Everything the tracer is tested on is generated here, so no
dataset download or human labeling is needed.

* :func:`make_scene` draws a page of random smooth strokes and degrades it
  according to a :class:`Degradation` preset (``"clean"`` or ``"hard"``). The
  ground truth keeps every stroke whole, including the
  parts erased to make gaps, and records the gaps so gap closing can be scored.
* :func:`single_curve_sample` and :func:`multi_curve_sample` draw patches.
* ``python -m line2func.synth ...`` writes scene sets (the validation set
  ``--version 1``) and preview sheets.

Images are grayscale uint8, dark lines on light paper. Coordinates follow
:mod:`line2func.geometry` (pixels, y down).
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from line2func.curves import Curve, CurveSet
from line2func.geometry import arc_length, evaluate, subcurve
from line2func.render import rasterize, render_lineart, save_png


@dataclass(frozen=True)
class Degradation:
    """How a clean drawing is roughened. Ranges are ``(low, high)``, sampled uniformly."""

    width: tuple[float, float] = (1.5, 3.0)  # line width, px
    taper: float = 0.0  # probability that a stroke's width changes along it
    taper_ratio: tuple[float, float] = (0.4, 1.0)  # end width / start width when tapered
    ink: tuple[float, float] = (1.0, 1.0)  # ink darkness, 1 = black
    gaps_per_stroke: float = 0.0  # mean number of breaks per stroke
    gap_length: tuple[float, float] = (2.0, 6.0)  # px
    specks_per_mp: tuple[float, float] = (0.0, 0.0)  # noise specks per megapixel
    speck_radius: tuple[float, float] = (0.5, 1.8)  # px
    shallow_crossings: tuple[int, int] = (0, 0)  # extra stroke pairs crossing at 10-35°, per 512²
    blur: tuple[float, float] = (0.0, 0.0)  # Gaussian sigma, px
    noise: tuple[float, float] = (0.0, 0.0)  # Gaussian noise std, intensity 0-1
    paper_shading: float = 0.0  # max brightness drop across the page
    jpeg: float = 0.0  # probability of JPEG compression
    jpeg_quality: tuple[int, int] = (35, 90)
PRESETS: dict[str, Degradation] = {
    "clean": Degradation(),
    "hard": Degradation(
        width=(1.0, 4.5),
        taper=0.5,
        taper_ratio=(0.35, 1.0),
        ink=(0.55, 1.0),
        gaps_per_stroke=0.6,
        gap_length=(2.0, 7.0),
        specks_per_mp=(100.0, 800.0),
        speck_radius=(0.5, 2.0),
        shallow_crossings=(1, 3),
        blur=(0.0, 0.9),
        noise=(0.0, 0.06),
        paper_shading=0.15,
        jpeg=0.5,
    ),
}
def preset(kind: str | Degradation) -> Degradation:
    if isinstance(kind, Degradation):
        return kind
    try:
        return PRESETS[kind]
    except KeyError:
        raise ValueError(f"unknown preset {kind!r}; choose from {', '.join(PRESETS)}") from None


def _u(rng: np.random.Generator, bounds) -> float:
    lo, hi = bounds
    return float(lo) if hi <= lo else float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Strokes
# ---------------------------------------------------------------------------


def random_stroke(
    rng: np.random.Generator,
    width: int,
    height: int,
    pieces: tuple[int, int] = (1, 3),
    piece_length: tuple[float, float] = (25.0, 90.0),
    margin: float = 8.0,
) -> list[np.ndarray]:
    """A smooth (G1) chain of cubic pieces that stays inside the image."""
    lo = np.array([margin, margin])
    hi = np.array([width - margin, height - margin])
    p0 = rng.uniform(lo, hi)
    heading = rng.uniform(0.0, 2.0 * np.pi)
    ctrls = []
    for _ in range(int(rng.integers(pieces[0], pieces[1] + 1))):
        length = rng.uniform(*piece_length)
        chord = heading + rng.normal(0.0, 0.5)
        p3 = p0 + length * np.array([np.cos(chord), np.sin(chord)])
        if np.any(p3 < lo) or np.any(p3 > hi):
            # turn back toward the middle instead of leaving the image
            to_mid = 0.5 * (lo + hi) - p0
            chord = np.arctan2(to_mid[1], to_mid[0]) + rng.normal(0.0, 0.3)
            p3 = np.clip(p0 + length * np.array([np.cos(chord), np.sin(chord)]), lo, hi)
            heading = chord
        end_heading = chord + (chord - heading) + rng.normal(0.0, 0.2)
        a = length * rng.uniform(0.25, 0.4)
        b = length * rng.uniform(0.25, 0.4)
        p1 = p0 + a * np.array([np.cos(heading), np.sin(heading)])
        p2 = p3 - b * np.array([np.cos(end_heading), np.sin(end_heading)])
        ctrls.append(np.array([p0, p1, p2, p3]))
        p0, heading = p3, end_heading
    return ctrls


def crossing_pair(
    rng: np.random.Generator, width: int, height: int, angle: tuple[float, float] = (10.0, 35.0), margin: float = 8.0
) -> list[np.ndarray]:
    """Two gently bent strokes that cross each other at a shallow angle."""
    center = rng.uniform([margin + 40, margin + 40], [width - margin - 40, height - margin - 40])
    base = rng.uniform(0.0, np.pi)
    half = np.radians(rng.uniform(*angle)) / 2.0
    out = []
    for heading in (base - half, base + half):
        d = np.array([np.cos(heading), np.sin(heading)])
        n = np.array([-d[1], d[0]])
        length = rng.uniform(70.0, 160.0)
        # shrink until both ends are inside the image
        for _ in range(8):
            p0, p3 = center - 0.5 * length * d, center + 0.5 * length * d
            if np.all((p0 > margin) & (p0 < [width - margin, height - margin])) and np.all(
                (p3 > margin) & (p3 < [width - margin, height - margin])
            ):
                break
            length *= 0.75
        bend = rng.uniform(-3.0, 3.0)
        p1 = p0 + length / 3.0 * d + bend * n
        p2 = p3 - length / 3.0 * d + bend * n
        out.append(np.array([p0, p1, p2, p3]))
    return out


def random_single_curve(rng: np.random.Generator, size: int, margin: float = 4.0) -> np.ndarray:
    """One cubic without loops or cusps that fits inside a ``size``² patch."""
    lo, hi = margin, size - margin
    for _ in range(100):
        p0 = rng.uniform(lo, hi, 2)
        p3 = rng.uniform(lo, hi, 2)
        chord = p3 - p0
        length = float(np.hypot(*chord))
        if length < 0.3 * size:
            continue
        base = np.arctan2(chord[1], chord[0])
        a1 = base + rng.uniform(-1.3, 1.3)
        a2 = base + rng.uniform(-1.3, 1.3)
        p1 = p0 + length * rng.uniform(0.1, 0.55) * np.array([np.cos(a1), np.sin(a1)])
        p2 = p3 - length * rng.uniform(0.1, 0.55) * np.array([np.cos(a2), np.sin(a2)])
        ctrl = np.array([p0, p1, p2, p3])
        pts = evaluate(ctrl, np.linspace(0.0, 1.0, 64))
        if pts.min() >= lo and pts.max() <= hi:
            return ctrl
    return np.array([[lo, lo], [lo + 10, lo], [hi - 10, hi], [hi, hi]], dtype=np.float64)


def _stroke_widths(rng: np.random.Generator, n_pieces: int, deg: Degradation) -> np.ndarray:
    """``(n_pieces, 2)`` start/end widths; a tapered stroke changes width smoothly across pieces."""
    w = _u(rng, deg.width)
    ratio = _u(rng, deg.taper_ratio) if rng.random() < deg.taper else 1.0
    if rng.random() < 0.5:
        ratio = 1.0 / ratio if ratio > 0 else 1.0  # thickening as often as thinning
    knots = w * np.linspace(1.0, ratio, n_pieces + 1)
    lo, hi = deg.width
    knots = np.clip(knots, max(0.6, 0.8 * lo), hi * 1.25)
    return np.stack([knots[:-1], knots[1:]], axis=1)


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


def _degrade(ink_lines: np.ndarray, rng: np.random.Generator, deg: Degradation) -> np.ndarray:
    """Coverage of the (gapped) lines -> degraded grayscale uint8 image."""
    h, w = ink_lines.shape
    ink = ink_lines * _u(rng, deg.ink)

    n_specks = int(round(_u(rng, deg.specks_per_mp) * h * w / 1e6))
    if n_specks:
        centers = rng.uniform([0, 0], [w, h], size=(n_specks, 2))
        dots = [np.repeat(c[None], 4, axis=0) for c in centers]
        radii = rng.uniform(*deg.speck_radius, size=n_specks)
        specks = rasterize(dots, w, h, line_width=2.0 * radii)
        ink = np.maximum(ink, specks * _u(rng, deg.ink))

    paper = np.ones((h, w))
    if deg.paper_shading > 0:
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        ramp = xx * np.cos(angle) + yy * np.sin(angle)
        ramp = (ramp - ramp.min()) / max(np.ptp(ramp), 1e-9)
        paper = 1.0 - rng.uniform(0, deg.paper_shading) * ramp
    img = paper * (1.0 - ink)

    sigma = _u(rng, deg.blur)
    if sigma > 0.05:
        img = ndimage.gaussian_filter(img, sigma)
    std = _u(rng, deg.noise)
    if std > 0:
        img = img + rng.normal(0.0, std, img.shape)
    out = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)

    if deg.jpeg > 0 and rng.random() < deg.jpeg:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.integers(*deg.jpeg_quality)))
        buf.seek(0)
        with Image.open(buf) as im:
            out = np.asarray(im.convert("L")).copy()
    return out


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """A synthetic page: ``image`` plus ground truth ``gt``.

    ``gt.meta`` holds ``kind``, per-curve ``widths`` (start, end) and ``gaps``:
    ``[{"stroke", "curve", "t", "point", "length"}]`` for every erased break,
    where ``length`` is the break's visible length in px.
    """

    image: np.ndarray
    gt: CurveSet

    @property
    def gaps(self) -> list[dict]:
        return self.gt.meta.get("gaps", [])


def make_scene(
    rng: np.random.Generator,
    width: int = 512,
    height: int = 512,
    kind: str | Degradation = "clean",
    n_strokes: int | None = None,
    crossing_pairs: int | None = None,
) -> Scene:
    """Random page of strokes. ``crossing_pairs`` overrides the preset's shallow-crossing count."""
    deg = preset(kind)
    area_scale = width * height / 512.0**2
    if n_strokes is None:
        lo, hi = (6, 12) if kind == "clean" else (10, 20)
        n_strokes = max(1, int(round(rng.integers(lo, hi + 1) * area_scale)))

    strokes = [random_stroke(rng, width, height) for _ in range(n_strokes)]
    if crossing_pairs is not None:
        n_pairs = int(crossing_pairs)
    else:
        n_pairs = int(round(_u(rng, deg.shallow_crossings) * area_scale))
    for _ in range(n_pairs):
        if min(width, height) >= 120:
            strokes.extend([c] for c in crossing_pair(rng, width, height))

    page = deg
    curves: list[Curve] = []
    widths: list[np.ndarray] = []
    for s, pieces in enumerate(strokes):
        ws = _stroke_widths(rng, len(pieces), page)
        for ctrl, w in zip(pieces, ws):
            curves.append(Curve(ctrl, stroke=s))
            widths.append(w)
    width_arr = np.array(widths).reshape(-1, 2)
    coverage = rasterize(curves, width, height, line_width=width_arr)

    gaps = _cut_gaps(rng, curves, width_arr, deg)
    if gaps:
        erasers = [subcurve(curves[g["curve"]].ctrl, g["t0"], g["t1"]) for g in gaps]
        erase_w = np.array([width_arr[g["curve"]].max() + 3.0 for g in gaps])
        coverage = coverage * (1.0 - rasterize(erasers, width, height, line_width=erase_w))
        for g in gaps:
            del g["t0"], g["t1"]

    image = _degrade(coverage, rng, deg)
    name = kind if isinstance(kind, str) else "custom"
    meta = {"kind": name, "widths": width_arr.round(3).tolist(), "gaps": gaps}
    gt = CurveSet(width, height, curves, meta=meta)
    return Scene(image, gt)


def _cut_gaps(rng, curves: list[Curve], widths: np.ndarray, deg: Degradation) -> list[dict]:
    """Choose breaks along strokes, away from other strokes (so they stay unambiguous)."""
    if deg.gaps_per_stroke <= 0 or not curves:
        return []
    samples, owner = [], []
    for c in curves:
        pts = evaluate(c.ctrl, np.linspace(0, 1, max(8, int(arc_length(c.ctrl)))))
        samples.append(pts)
        owner.append(np.full(len(pts), c.stroke))
    tree = cKDTree(np.vstack(samples))
    owner_arr = np.concatenate(owner)
    by_stroke: dict[int, list[int]] = {}
    for i, c in enumerate(curves):
        by_stroke.setdefault(c.stroke, []).append(i)
    gaps: list[dict] = []
    for stroke, idx in by_stroke.items():
        for _ in range(int(rng.poisson(deg.gaps_per_stroke))):
            ci = int(rng.choice(idx))
            ctrl = curves[ci].ctrl
            length = arc_length(ctrl)
            g_len = _u(rng, deg.gap_length)
            if length < 4 * g_len + 10:
                continue
            t = rng.uniform(0.25, 0.75)
            dt = 0.5 * g_len / length
            point = evaluate(ctrl, t)
            clearance = g_len + 2 * float(widths.max()) + 4.0
            near = tree.query_ball_point(point, clearance)
            if np.any(owner_arr[near] != stroke):
                continue
            if any(np.hypot(*(np.array(g["point"]) - point)) < 2 * clearance for g in gaps):
                continue
            # the eraser has round caps as wide as the line + 3 px, so the break
            # visible on the page is longer than the erased stretch of centerline
            visible = g_len + float(widths[ci].max()) + 3.0
            gaps.append(
                {"stroke": stroke, "curve": ci, "t": round(float(t), 4), "point": [round(float(v), 3) for v in point],
                 "length": round(visible, 3), "t0": t - dt, "t1": t + dt}
            )
    return gaps


def random_scene(
    rng: np.random.Generator,
    width: int = 256,
    height: int = 256,
    n_strokes: int = 6,
    line_width: tuple[float, float] = (1.5, 3.0),
) -> tuple[CurveSet, np.ndarray, np.ndarray]:
    """Clean drawing with constant-width strokes: ``(ground_truth, per_curve_widths, image)``."""
    curves: list[Curve] = []
    widths: list[float] = []
    for s in range(n_strokes):
        w = float(rng.uniform(*line_width))
        for c in random_stroke(rng, width, height):
            curves.append(Curve(c, stroke=s))
            widths.append(w)
    gt = CurveSet(width, height, curves, meta={"source": "synth"})
    w_arr = np.array(widths)
    image = render_lineart(gt, width, height, line_width=w_arr) if curves else np.full((height, width), 255, np.uint8)
    return gt, w_arr, image


# ---------------------------------------------------------------------------
# Single-curve patches
# ---------------------------------------------------------------------------


def single_curve_sample(
    rng: np.random.Generator, size: int = 64, kind: str | Degradation = "hard"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One cubic in a ``size``² patch: ``(image uint8, ctrl (4, 2) px, widths (start, end) px)``."""
    deg = preset(kind)
    ctrl = random_single_curve(rng, size)
    widths = _stroke_widths(rng, 1, deg)[0]
    coverage = rasterize([ctrl], size, size, line_width=widths[None])
    if deg.gaps_per_stroke > 0 and rng.random() < min(1.0, deg.gaps_per_stroke):
        g_len = _u(rng, deg.gap_length)
        length = arc_length(ctrl)
        if length > 4 * g_len:
            t = rng.uniform(0.3, 0.7)
            dt = 0.5 * g_len / length
            eraser = subcurve(ctrl, t - dt, t + dt)
            coverage = coverage * (1.0 - rasterize([eraser], size, size, line_width=float(widths.max()) + 3.0))
    image = _degrade(coverage, rng, replace(deg, paper_shading=deg.paper_shading * 0.3))
    return image, ctrl, widths


def _inside_runs(points: np.ndarray, size: float) -> list[np.ndarray]:
    """Maximal runs of consecutive ``points`` inside the square ``[0, size]²``."""
    inside = np.all((points >= 0.0) & (points <= size), axis=1)
    runs, start = [], None
    for i, flag in enumerate(inside):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append(points[start:i])
            start = None
    if start is not None:
        runs.append(points[start:])
    return runs


def patch_targets(
    gt: CurveSet, x0: float, y0: float, size: int, widths: np.ndarray | None = None,
    tolerance: float = 0.4, min_length: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth curves inside the patch ``[x0, x0 + size) x [y0, y0 + size)``.

    Each stroke's part inside the patch is refitted into the shortest chain of
    cubics (Schneider, ``tolerance`` px). The generator's own piece boundaries
    are invisible in the image, so this canonical form is what a model can
    actually learn. Returns ``(ctrls (k, 4, 2), widths (k,))`` in patch pixels.
    """
    from line2func.fit import fit_polyline

    per_curve_w = None if widths is None else np.asarray(widths).reshape(len(gt), -1).mean(axis=1)
    index = {id(c): i for i, c in enumerate(gt.curves)}
    ctrls, ws = [], []
    offset = np.array([x0, y0], dtype=np.float64)
    for pieces in gt.strokes().values():
        pts = []
        for k, c in enumerate(pieces):
            n = max(8, int(np.ceil(arc_length(c.ctrl) / 0.5)))
            p = evaluate(c.ctrl, np.linspace(0.0, 1.0, n + 1)) - offset
            pts.append(p if k == 0 else p[1:])
        pts = np.vstack(pts)
        w = 2.0 if per_curve_w is None else float(np.mean([per_curve_w[index[id(c)]] for c in pieces]))
        for run in _inside_runs(pts, float(size)):
            if len(run) < 2 or np.sum(np.linalg.norm(np.diff(run, axis=0), axis=1)) < min_length:
                continue
            for ctrl in fit_polyline(run, tolerance):
                ctrls.append(ctrl)
                ws.append(w)
    if not ctrls:
        return np.zeros((0, 4, 2)), np.zeros(0)
    return np.array(ctrls), np.array(ws)


def multi_curve_sample(
    rng: np.random.Generator, size: int = 64, kind: str | Degradation = "hard", max_curves: int = 16,
    context: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ``size``² crop from the middle of a small scene, with every curve piece inside it.

    Strokes enter and leave the patch, cross each other (shallow crossings
    included for the hard preset) and may have gaps. Returns
    ``(image uint8, ctrls (k, 4, 2) px, widths (k,) px)`` with ``k <= max_curves``
    (the longest pieces are kept if there are more).
    """
    deg = preset(kind)
    canvas = size + 2 * context
    n_strokes = int(rng.integers(1, 5))
    pairs = 1 if deg.shallow_crossings[1] > 0 and rng.random() < 0.3 else 0
    scene = make_scene(rng, canvas, canvas, deg, n_strokes=n_strokes, crossing_pairs=pairs)
    image = scene.image[context : context + size, context : context + size].copy()
    ctrls, widths = patch_targets(scene.gt, context, context, size, np.array(scene.gt.meta["widths"]))
    if len(ctrls) > max_curves:
        lengths = np.array([arc_length(c) for c in ctrls])
        keep = np.sort(np.argsort(-lengths)[:max_curves])
        ctrls, widths = ctrls[keep], widths[keep]
    return image, ctrls, widths


# ---------------------------------------------------------------------------
# Scene sets on disk
# ---------------------------------------------------------------------------


def save_scene(scene: Scene, stem: Path) -> None:
    save_png(stem.with_suffix(".png"), scene.image)
    scene.gt.save_json(stem.with_suffix(".json"))


def load_scene_dir(folder: str | Path) -> list[tuple[Path, CurveSet]]:
    """``(image_path, ground_truth)`` for every ``NNNN.png`` / ``NNNN.json`` pair in ``folder``."""
    out = []
    for js in sorted(Path(folder).glob("*.json")):
        png = js.with_suffix(".png")
        if png.is_file():
            out.append((png, CurveSet.load_json(js)))
    return out


VALSETS = {"1": (("clean", 10_001), ("hard", 10_002))}


def write_scenes(out: Path, kind: str, count: int, size: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        rng = np.random.default_rng([seed, i])
        save_scene(make_scene(rng, size, size, kind), out / f"{i:04d}")


def _preview(kind: str, out: Path, seed: int, patches: bool) -> None:
    from PIL import Image

    if patches:
        tiles = [single_curve_sample(np.random.default_rng([seed, i]), 64, kind)[0] for i in range(64)]
        sheet = np.full((8 * 66, 8 * 66), 128, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, 8)
            sheet[r * 66 + 1 : r * 66 + 65, c * 66 + 1 : c * 66 + 65] = t
    else:
        tiles = [make_scene(np.random.default_rng([seed, i]), 384, 384, kind).image for i in range(4)]
        sheet = np.full((2 * 386, 2 * 386), 128, np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, 2)
            sheet[r * 386 + 1 : r * 386 + 385, c * 386 + 1 : c * 386 + 385] = t
    Image.fromarray(sheet).save(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.synth", description="Generate synthetic line art.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenes", help="write a set of scenes (PNG + ground-truth JSON)")
    s.add_argument("--kind", choices=sorted(PRESETS), default="clean")
    s.add_argument("--count", type=int, default=50)
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("valset", help="write the standard validation set (clean/ + hard/)")
    v.add_argument("--version", choices=("1",), default="1",
                   help="1: val_v1 (clean/ + hard/)")
    v.add_argument("--out", type=Path, default=None, help="default: data/val_v1")
    v.add_argument("--count", type=int, default=100)
    v.add_argument("--size", type=int, default=512)
    pv = sub.add_parser("preview", help="write a contact sheet of samples")
    pv.add_argument("--kind", choices=sorted(PRESETS), default="hard")
    pv.add_argument("--patches", action="store_true", help="single-curve patches instead of scenes")
    pv.add_argument("--seed", type=int, default=0)
    pv.add_argument("--out", type=Path, default=Path("preview.png"))
    args = p.parse_args(argv)

    if args.cmd == "scenes":
        write_scenes(args.out, args.kind, args.count, args.size, args.seed)
        print(f"wrote {args.count} {args.kind} scenes to {args.out}")
    elif args.cmd == "valset":
        # fixed seeds: the sets must be identical on every machine
        out = args.out or Path("data/val_v1")
        for kind, seed in VALSETS[args.version]:
            write_scenes(out / kind, kind, args.count, args.size, seed)
        kinds = " + ".join(kind for kind, _ in VALSETS[args.version])
        print(f"wrote {args.count} scenes each ({kinds}) to {out}")
    else:
        _preview(args.kind, args.out, args.seed, args.patches)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Judge traced curves against the drawing itself (no ground truth needed).

    python -m line2func.quality out/            # after: python -m line2func.demo drawing.png --out out/
    python -m line2func.demo drawing.png --quality --out out/

Everything is measured against the **ink map of the original image**, not
against the thresholded ink the tracer saw, so detail lost by thresholding is
visible too. Ink is split into darkness bands:

* ``dark``   ink >= 0.5
* ``mid``    threshold <= ink < 0.5
* ``faint``  0.08 <= ink < threshold (dropped by binarization before tracing)

Fainter strokes still count as ink where the report asks whether a curve lies
on ink (``including_faint``, the stray-curve flag): whatever stands out from the
paper by more than the paper's own noise does (local contrast above
:func:`line2func.baseline.paper_contrast_ceiling`, measured on this drawing's
blank paper), so curves on very light background strokes are not taken for
invented lines.

Report (``quality.json``):

* **recall**: share of the ink centerline (specks removed) within 1 / 2 px of
  a curve - for the lines the tracer could see (``line``, the headline "is any
  detail missing?" number) and including faint strokes (``with_faint``). Also
  the ink mass kept per darkness band (``by_band``), where a pixel counts as
  kept if it lies within 1 / 2 px of the nearest curve's *edge* (its measured
  width), so wide lines are not penalized. The centerline-length form matches
  ground-truth recall to 0.009 on average on hard synthetic scenes; weighting
  by ink mass would count noise as missing detail.
* **precision**: share of curve length within 1 / 2 px of ink ("invented lines").
* **distance**: ink-centerline <-> curve distances, mean (d_M, symmetric chamfer
  as in Deep Vectorization, arXiv:2003.05471), 95th percentile and max, both
  directions. The max catches single stray curves that means hide.
* **raster**: the curves rendered with their measured widths vs the ink map:
  PSNR, SSIM and IoU. IoU is harsh on lines thinner than ~2 px (a 1 px shift
  can halve it), so it is reported but not used as a headline.
* **structure**: curves, strokes, curves per 100 px of ink centerline, share of
  very short curves, and whether the result fits the Desmos budget.
* **missed**: connected regions of missed ink, largest first, each with its
  cause: ``below_threshold`` (fainter than the tracer's threshold),
  ``speck`` (removed by the speck filter), ``fill`` (inside a filled area) or
  ``untraced`` (a line the tracer saw but did not follow: dense detail merged,
  short branch pruned, ...).

``quality.png`` shows it on the image: dark gray = traced ink, red = missed
line ink, orange = missed faint ink (below threshold), blue = curve with no ink.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from line2func import lineart
from line2func.baseline import _remove_small, auto_threshold, local_contrast, paper_contrast_ceiling, thin
from line2func.curves import CurveSet
from line2func.export import DESMOS_CURVE_LIMIT
from line2func.metrics import sample_points
from line2func.render import filled_area, render_coverage, stroke_loops

FAINT = 0.08  # ink below this is paper texture / noise, not drawing
DARK = 0.5
TAUS = (1.0, 2.0)
MISS_TAU = 2.0  # ink farther than this from every curve counts as missed


def tracer_threshold(ink: np.ndarray, threshold: float | None = None) -> float:
    """The threshold to judge at: the given one, or Otsu's as ``baseline.vectorize`` picks it.

    For line art the pipeline traces at a lower threshold (at most 0.25,
    :func:`line2func.pipeline.trace_threshold`). Judging at Otsu's keeps
    "lines" meaning the drawing's clearly visible lines; lighter ink counts as
    faint strokes.
    """
    if threshold is not None:
        return float(threshold)
    return auto_threshold(ink)


def _ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """Mean structural similarity of two images in [0, 1] (Gaussian window)."""
    c1, c2 = 0.01**2, 0.03**2
    f = lambda x: ndimage.gaussian_filter(x, sigma)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    var_a = f(a * a) - mu_a**2
    var_b = f(b * b) - mu_b**2
    cov = f(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2))
    return float(s.mean())


def _stats(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"mean": round(float(d.mean()), 3), "p95": round(float(np.percentile(d, 95)), 3),
            "max": round(float(d.max()), 3)}


def assess(curves: CurveSet, ink: np.ndarray, threshold: float | None = None, top: int = 10):
    """Quality report (dict) and per-pixel maps for a traced drawing and its ink map."""
    ink = np.asarray(ink, dtype=np.float32)
    h, w = ink.shape
    thr = tracer_threshold(ink, threshold)
    line_w = float(curves.meta.get("line_width") or 2.0)
    mask = ink > thr  # the drawing's clearly visible lines

    # curve samples, each with the half width of its curve
    per_curve = [sample_points([c], 0.5) for c in curves.curves]
    pts = np.vstack(per_curve) if per_curve else np.zeros((0, 2))
    half = np.concatenate([np.full(len(p), 0.5 * (c.width or line_w)) for p, c in zip(per_curve, curves.curves)]) \
        if per_curve else np.zeros(0)
    tree = cKDTree(pts) if len(pts) else None

    # gap from every inked pixel to the nearest curve's *edge* (distance minus its half
    # width): the edge pixels of a wide line are covered, not missed
    ys, xs = np.nonzero(ink >= FAINT)
    d_pix = np.full(ink.shape, np.inf, dtype=np.float32)
    if tree is not None and len(ys):
        d, j = tree.query(np.stack([xs + 0.5, ys + 0.5], axis=1), distance_upper_bound=64.0)
        hit = np.isfinite(d)
        gap = np.full(len(d), np.inf)
        gap[hit] = np.maximum(0.0, d[hit] - half[j[hit]])
        d_pix[ys, xs] = gap
    # ink inside a filled outline (thick / wedge strokes, see line2func.outline) is covered
    inside = filled_area(curves, w, h) > 0.5 if stroke_loops(curves) else np.zeros(ink.shape, dtype=bool)
    d_pix[inside] = 0.0

    def to_curves(q: np.ndarray) -> np.ndarray:
        """Distance from points to the result: to the nearest curve, 0 inside filled outlines."""
        if tree is None:
            return np.full(len(q), np.inf)
        dq = tree.query(q)[0]
        xi = np.clip(q[:, 0].astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].astype(int), 0, h - 1)
        return np.where(inside[yi, xi], 0.0, dq)

    min_area = max(8.0, 2.0 * line_w * line_w)  # the baseline engine's speck filter

    def centerline_recall(region: np.ndarray) -> dict:
        """Share of the region's centerline length (specks removed) within tau of a curve.

        Length along the centerline, like ground-truth recall; validated on
        val_v1 hard scenes: mean |no-ref - GT| = 0.009 (ink-mass weighting: 0.063).
        """
        sk = np.argwhere(thin(_remove_small(region, min_area)))[:, ::-1] + 0.5
        if not len(sk):
            return {f"within_{t:g}px": None for t in TAUS} | {"centerline_px": 0}
        d = to_curves(sk)
        return {f"within_{t:g}px": round(float(np.mean(d <= t)), 4) for t in TAUS} | {"centerline_px": len(sk)}

    bands = {"dark": ink >= DARK, "mid": (ink >= thr) & (ink < DARK), "faint": (ink >= FAINT) & (ink < thr)}
    total_mass = float(ink[ink >= FAINT].sum()) or 1.0
    by_band = {}
    for name, m in bands.items():
        mass = float(ink[m].sum())
        entry = {"share_of_ink": round(mass / total_mass, 4)}
        for tau in TAUS:
            entry[f"within_{tau:g}px"] = round(float(ink[m & (d_pix <= tau)].sum()) / mass, 4) if mass else None
        by_band[name] = entry
    recall = {
        # headline: the lines the tracer could see (ink >= threshold)
        "line": centerline_recall(mask),
        # also counting faint strokes (down to half the threshold): shows detail lost to thresholding
        "with_faint": centerline_recall(ink >= max(FAINT, 0.5 * thr)),
        # diagnostic: ink mass kept per darkness band
        "by_band": by_band,
    }

    # precision: curve samples near ink. The validated definition counts ink >= half the
    # tracer threshold (a looser floor let synthetic noise count as ink: agreement with
    # ground truth fell from 0.012 to 0.033 and stray detection from 98% to 89%).
    # "including_faint" also accepts faint ink >= FAINT and fainter strokes that stand out from the paper
    # (visible, below), for tracings that follow faint strokes.
    def off_ink_at(floor: float) -> np.ndarray:
        if not len(pts):
            return np.zeros(0)
        edt = ndimage.distance_transform_edt(ink < floor)
        return edt[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)]

    # visible ink: >= FAINT, or standing out from the paper by more than the paper's own noise
    ridge = local_contrast(ink, line_w)
    ceiling = paper_contrast_ceiling(ink, ridge, line_w)
    visible = (ink >= FAINT) | (ridge >= ceiling) if ceiling is not None else ink >= FAINT

    off_ink = off_ink_at(0.5 * thr)
    off_faint = np.zeros(0)
    if len(pts):
        edt = ndimage.distance_transform_edt(~visible)
        off_faint = edt[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)]
    precision = {f"within_{t:g}px": round(float(np.mean(off_ink <= t)), 4) if off_ink.size else None for t in TAUS}
    precision["including_faint"] = {f"within_{t:g}px": round(float(np.mean(off_faint <= t)), 4)
                                    if off_faint.size else None for t in TAUS}

    # distances between the line centerline (specks removed, as for recall) and the curves
    skel = thin(_remove_small(mask, min_area))
    skel_pts = np.argwhere(skel)[:, ::-1] + 0.5
    faint_ink = _remove_small(ink >= max(FAINT, 0.5 * thr), min_area)
    faint_skel_pts = np.argwhere(thin(faint_ink))[:, ::-1] + 0.5
    # a curve on a very light stroke is on ink too (ink to curves: faint strokes down to half the threshold)
    seen_skel_pts = np.argwhere(thin(_remove_small(faint_ink | visible, min_area)))[:, ::-1] + 0.5
    if tree is not None and len(skel_pts):
        ink_to_curve = to_curves(skel_pts)
        curve_to_ink = cKDTree(skel_pts).query(pts)[0]
        d_m_faint = 0.5 * float(to_curves(faint_skel_pts).mean() + cKDTree(seen_skel_pts).query(pts)[0].mean())
    else:
        ink_to_curve = curve_to_ink = np.zeros(0)
        d_m_faint = None
    distance = {
        "d_M": round(0.5 * float(ink_to_curve.mean() + curve_to_ink.mean()), 3) if ink_to_curve.size else None,
        # the same between the curves and the centerline of all ink incl. faint strokes
        "d_M_including_faint": None if d_m_faint is None else round(d_m_faint, 3),
        "ink_to_curve": _stats(ink_to_curve),
        "curve_to_ink": _stats(curve_to_ink),
        # distance to *any* ink (specks included): large only for curves drawn where
        # there is no ink at all, i.e. stray lines (flag: max > 10 px)
        "curve_to_any_ink": _stats(np.asarray(off_ink, dtype=float)),
        # the same, also accepting faint ink (informational; a curve on a faint stroke scores ~0)
        "curve_to_faint_ink": _stats(np.asarray(off_faint, dtype=float)),
    }

    # raster fidelity: render with measured widths, scaled to the drawing's ink darkness
    widths = np.array([c.width if c.width else line_w for c in curves.curves]) if len(curves) else np.zeros(0)
    darkness = float(np.percentile(ink[mask], 90)) if mask.any() else 1.0
    # outline strokes (thick / wedge shapes) are filled, all other curves drawn as lines
    cov = render_coverage(curves, w, h, line_width=widths) if len(curves) else np.zeros_like(ink)
    render = cov * darkness
    mse = float(np.mean((render - ink) ** 2))
    drawn = render > 0.5 * darkness
    union = (drawn | mask).sum()
    raster = {
        "psnr_db": round(10 * np.log10(1.0 / max(mse, 1e-12)), 2),
        "ssim": round(_ssim(render.astype(np.float64), ink.astype(np.float64)), 4),
        "iou": round(float((drawn & mask).sum() / union), 4) if union else None,
        "iou_reliable": bool(line_w >= 2.0),
    }

    # structure and compactness
    lengths = np.array([float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in per_curve]) \
        if per_curve else np.zeros(0)
    skel_len = float(skel.sum())
    structure = {
        "curves": len(curves),
        "strokes": curves.num_strokes,
        "curves_per_100px_ink": round(100.0 * len(curves) / max(skel_len, 1.0), 3),
        "short_curves_under_5px": round(float(np.mean(lengths < 5.0)), 4) if lengths.size else 0.0,
        "fits_desmos": len(curves) <= DESMOS_CURVE_LIMIT,
    }

    # missed-detail inventory with causes
    missed = (ink >= FAINT) & (d_pix > MISS_TAU)
    speck_px = mask & ~_remove_small(mask, min_area)
    depth = ndimage.distance_transform_edt(mask)
    fill_px = ndimage.binary_dilation(depth >= max(4.0, 3.0 * line_w), iterations=int(3 * line_w) + 1) & mask
    cause_codes = {"below_threshold": 1, "speck": 2, "fill": 3, "untraced": 4}
    code = np.zeros(ink.shape, np.uint8)
    code[missed & ~mask] = cause_codes["below_threshold"]
    code[missed & mask] = cause_codes["untraced"]
    code[missed & fill_px] = cause_codes["fill"]
    code[missed & speck_px] = cause_codes["speck"]
    regions_lab, n_reg = ndimage.label(ndimage.binary_dilation(missed, iterations=1) & (ink >= FAINT),
                                       structure=np.ones((3, 3), bool))
    regions = []
    if n_reg:
        idx = np.arange(1, n_reg + 1)
        mass = ndimage.sum(ink * missed, regions_lab, idx)
        boxes = ndimage.find_objects(regions_lab)
        for r in np.argsort(-mass)[:top]:
            lab = int(idx[r])
            sl = boxes[lab - 1]
            region = (regions_lab[sl] == lab) & missed[sl]
            codes = code[sl][region]
            if codes.size == 0:
                continue
            main = int(np.bincount(codes, minlength=5)[1:].argmax()) + 1
            name = {v: k for k, v in cause_codes.items()}[main]
            regions.append({
                "cause": name,
                "ink_mass": round(float(mass[r]), 2),
                "pixels": int(region.sum()),
                "bbox_xywh": [int(sl[1].start), int(sl[0].start), int(sl[1].stop - sl[1].start),
                              int(sl[0].stop - sl[0].start)],
            })
    missed_mass = float(ink[missed].sum())
    by_cause = {}
    for name, c in cause_codes.items():
        by_cause[name] = round(float(ink[code == c].sum()) / max(missed_mass, 1e-9), 4) if missed_mass else 0.0

    # Stray curves: a curve point farther than 10 px from ink. When the tracer followed
    # faint strokes on purpose (meta "faint_lines"), faint ink counts as ink for this check.
    faint_mode = bool(curves.meta.get("faint_lines"))
    stray_d = distance["curve_to_faint_ink" if faint_mode else "curve_to_any_ink"]["max"]
    flags = {"stray_curves": bool(stray_d is not None and stray_d > 10.0), "judged_with_faint_ink": faint_mode}

    report = {
        "image": {"width": w, "height": h},
        "threshold": round(thr, 3),
        "paper_contrast_ceiling": None if ceiling is None else round(ceiling, 4),
        "flags": flags,
        "line_width": round(line_w, 2),
        "recall": recall,
        "precision": precision,
        "distance": distance,
        "raster": raster,
        "structure": structure,
        "missed": {"share_of_ink": round(missed_mass / total_mass, 4), "by_cause": by_cause, "largest_regions": regions},
    }
    # the map marks curve points far from the ink they are judged against, as the stray-curve flag does
    maps = {"ink": ink, "d_pix": d_pix, "code": code, "pts": pts, "off_ink": off_faint if faint_mode else off_ink,
            "thr": thr}
    return report, maps


def quality_map(maps: dict) -> np.ndarray:
    """RGB picture: traced ink dark gray, missed line ink red, missed faint ink orange, stray curve blue."""
    ink, d_pix, code, thr = maps["ink"], maps["d_pix"], maps["code"], maps["thr"]
    base = 255.0 - 60.0 * np.clip(ink, 0, 1)  # the drawing, very light
    rgb = np.repeat(base[:, :, None], 3, axis=2)
    traced = (ink >= thr) & (d_pix <= MISS_TAU)
    rgb[traced] = (255.0 - 200.0 * np.clip(ink[traced], 0, 1))[:, None]
    strength = np.clip(ink / max(thr, 1e-3), 0.35, 1.0)[:, :, None]
    for c, color in ((1, (255, 150, 0)), (2, (220, 30, 30)), (3, (220, 30, 30)), (4, (220, 30, 30))):
        m = code == c
        rgb[m] = 255.0 + (np.array(color, float) - 255.0) * strength[m]
    pts, off = maps["pts"], maps["off_ink"]
    if len(pts):
        stray = pts[off > MISS_TAU]
        h, w = ink.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                xi = np.clip(stray[:, 0].astype(int) + dx, 0, w - 1)
                yi = np.clip(stray[:, 1].astype(int) + dy, 0, h - 1)
                rgb[yi, xi] = (30, 90, 230)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def summary(report: dict) -> list[str]:
    """A few human-readable lines."""
    r, p, dist, s, m = report["recall"], report["precision"], report["distance"], report["structure"], report["missed"]

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"

    b = r["by_band"]
    lines = [
        f"detail kept (centerline within 2 px of a curve): lines {pct(r['line']['within_2px'])}, "
        f"with faint strokes {pct(r['with_faint']['within_2px'])}; ink by darkness: dark {pct(b['dark']['within_2px'])}, "
        f"mid {pct(b['mid']['within_2px'])}, faint {pct(b['faint']['within_2px'])} "
        f"(faint = below threshold {report['threshold']}, {pct(b['faint']['share_of_ink'])} of all ink)",
        f"invented lines: {pct(1 - p['within_2px']) if p['within_2px'] is not None else 'n/a'} of curve length "
        f"is > 2 px from ink ({pct(1 - p['including_faint']['within_2px']) if p['within_2px'] is not None else 'n/a'}"
        f" counting faint ink); farthest curve point from ink {dist['curve_to_any_ink']['max']} px"
        f" ({dist['curve_to_faint_ink']['max']} px counting faint ink)"
        + (" -- STRAY CURVES?" if report["flags"]["stray_curves"] else ""),
        f"accuracy: d_M {dist['d_M']} px ({dist['d_M_including_faint']} px incl. faint strokes), "
        f"ink->curve p95 {dist['ink_to_curve']['p95']} px (max {dist['ink_to_curve']['max']})",
        f"missed ink: {pct(m['share_of_ink'])} of all ink; by cause: "
        + ", ".join(f"{k} {pct(v)}" for k, v in m["by_cause"].items() if v),
        f"{s['curves']} curves / {s['strokes']} strokes, {s['curves_per_100px_ink']} per 100 px of ink"
        + ("" if s["fits_desmos"] else f" (over the ~{DESMOS_CURVE_LIMIT}-curve Desmos budget)"),
    ]
    return lines


def assess_folder(folder: str | Path, image: str | Path | None = None, threshold: float | None = None) -> dict:
    """Assess a ``demo`` output folder; writes ``quality.json`` and ``quality.png`` there."""
    from line2func.render import save_png

    folder = Path(folder)
    curves = CurveSet.load_json(folder / "curves.json")
    src = Path(image) if image else folder / "source.png"
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found; pass --image with the original drawing")
    method = curves.meta.get("lineart", "none")
    if threshold is None:
        threshold = curves.meta.get("threshold")
    report, maps = assess(curves, lineart.extract(src, method), threshold)
    (folder / "quality.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    save_png(folder / "quality.png", quality_map(maps))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.quality",
                                description="Judge traced curves against the original drawing.")
    p.add_argument("folder", type=Path, help="output folder written by line2func.demo")
    p.add_argument("--image", type=Path, default=None, help="original image (default: <folder>/source.png)")
    p.add_argument("--threshold", type=float, default=None,
                   help="ink threshold to judge at (default: the one given to demo, else Otsu's)")
    args = p.parse_args(argv)
    try:
        report = assess_folder(args.folder, args.image, args.threshold)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in summary(report):
        print(line)
    print(f"  -> {args.folder / 'quality.json'}, {args.folder / 'quality.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

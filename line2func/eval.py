"""Evaluate the tracer on synthetic ground truth.

    python -m line2func.synth valset --out data/val_v1           # once: write the standard set
    python -m line2func.eval --valset data/val_v1                    # the baseline engine
    python -m line2func.eval --realset path/to/drawings --json runs/real_a.json  # real drawings, no ground truth
    python -m line2func.eval --realset path/to/drawings --set local_trim=false --json runs/real_b.json
    python -m line2func.eval --compare runs/real_b.json runs/real_a.json         # paired, with intervals

Scene evaluation reports the metrics of docs/details.md's Evaluation section per
subset (clean, hard): F_GT@2, crossing continuity, gap closure, fragments per
stroke, curve count ratio, length ratio and CPU seconds per megapixel, plus an
F sweep over tolerances (in thousandths of the long edge, so it is comparable
across resolutions).

``--realset`` judges real drawings, which have no ground truth, against their
own ink (:mod:`line2func.quality`), traced at a fixed tolerance with no curve
budget so that the curve count is free to move. ``--compare`` puts two such
reports side by side: the same drawings, so each is its own control, reported
as the median per-drawing difference with a bootstrap interval. Its numbers are
differences between two settings and are only meant to be read that way -
``length_vs_skeleton`` in particular has a denominator that undercounts dense
drawings, and only cancels because both settings share it.

The length ratio is the only metric here that sees a line drawn twice: every
other one measures distance to the nearest curve, and a doubled stroke lies on
the ink. Read it against the ground truth's own habits, though - the synthetic
ground truth keeps erased gaps whole, so a faithful tracing scores a little
under 1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from line2func import baseline, lineart
from line2func.metrics import (
    f_score,
    f_sweep,
    structure_scores,
    total_length,
)
from line2func.synth import load_scene_dir

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")  # the drawings --realset reads
REAL_SEED = 2_000_011  # the bootstrap's resampling, so a comparison is reproducible


def _subsets(valset: Path) -> dict[str, Path]:
    if any(valset.glob("*.json")):
        return {valset.name: valset}
    subs = {p.name: p for p in sorted(valset.iterdir()) if p.is_dir() and any(p.glob("*.json"))}
    if not subs:
        raise FileNotFoundError(f"no scenes in {valset}; create them with: python -m line2func.synth valset --out {valset}")
    return subs


def eval_scenes(valset: str | Path, limit: int | None = None, repeat: int = 1) -> dict[str, dict]:
    """Evaluation metrics per subset of a scene set, as the baseline engine traces it.

    With ``repeat`` above 1 every scene is traced that many times and the
    subset's time is the median of the runs, with ``seconds_spread`` saying how
    far apart they were. One timing of a 100-scene subset is not a measurement:
    a gate whose margin is under a percent can be turned over by whatever else
    the machine is doing, which has happened here.
    """
    results = {}
    for name, folder in _subsets(Path(valset)).items():
        scenes = load_scene_dir(folder)[:limit]
        f, frag = [], []
        cont = [0.0, 0]
        clos = [0.0, 0]
        n_pred = n_gt = 0
        len_pred = len_gt = 0.0
        megapixels = 0.0
        runs = np.zeros(max(1, repeat))  # each pass over the subset timed on its own

        swept: dict[str, float] = {}
        for png, gt in scenes:
            ink = lineart.extract(png, "none")
            for k in range(len(runs)):
                t = time.perf_counter()
                pred = baseline.vectorize(ink)
                runs[k] += time.perf_counter() - t
            megapixels += gt.width * gt.height / 1e6
            f.append(f_score(pred, gt)["f"])
            len_pred += total_length(pred)
            len_gt += total_length(gt)
            for key, value in f_sweep(pred, gt).items():
                swept[key] = swept.get(key, 0.0) + value
            s = structure_scores(pred, gt)
            if s["crossing_checks"]:
                cont[0] += s["crossing_continuity"] * s["crossing_checks"]
                cont[1] += s["crossing_checks"]
            if s["gap_checks"]:
                clos[0] += s["gap_closure"] * s["gap_checks"]
                clos[1] += s["gap_checks"]
            if not np.isnan(s["fragments_per_stroke"]):
                frag.append(s["fragments_per_stroke"])
            n_pred += len(pred)
            n_gt += len(gt)
        results[name] = {
            "scenes": len(scenes),
            "f_gt2": float(np.mean(f)) if f else float("nan"),
            "crossing_continuity": cont[0] / cont[1] if cont[1] else float("nan"),
            "gap_closure": clos[0] / clos[1] if clos[1] else float("nan"),
            "fragments_per_stroke": float(np.mean(frag)) if frag else float("nan"),
            "curve_ratio": n_pred / max(1, n_gt),
            "length_ratio": len_pred / len_gt if len_gt > 0 else float("nan"),
            "seconds_per_mp": float(np.median(runs)) / max(megapixels, 1e-9),
            "seconds_spread": (float(np.ptp(runs) / max(np.median(runs), 1e-9)) if len(runs) > 1
                               else float("nan")),
            "repeat": len(runs),
            "crossing_checks": cont[1],
            "gap_checks": clos[1],
            "f_sweep": {k: v / len(scenes) for k, v in swept.items()} if scenes else {},
        }
    return results


# ---------------------------------------------------------------------------
# Real drawings: no ground truth, so the ink itself is the reference
# ---------------------------------------------------------------------------


def _real_images(folder: str | Path) -> list[Path]:
    """The drawings in a folder, sorted (:data:`IMAGE_TYPES`)."""
    folder = Path(folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES) if folder.is_dir() else []
    if not images:
        raise FileNotFoundError(f"no drawings ({', '.join(IMAGE_TYPES)}) in {folder}")
    return images


def eval_real(folder: str | Path, tolerance: float = 1.0, limit: int | None = None,
              options: dict | None = None) -> dict:
    """Trace real drawings and judge each against its own ink (:mod:`line2func.quality`).

    There is no ground truth here, so nothing is scored against a drawing that
    is known to be right; the ink is the reference, as ``demo --quality`` does.

    **Traced at a fixed tolerance, with no curve budget.** The budget would
    otherwise decide the curve count for four of the five test drawings, and the
    curve count at a fixed tolerance is the headline: a change that draws the
    same lines with fewer curves is what most geometric work is trying to do.

    ``length_vs_skeleton`` is the total curve length over the length of the ink's
    thinned centerline, standing in for
    :func:`line2func.metrics.stroke_length_scores` where there is no ground truth
    to divide by. **Read it only as a difference between two settings**, which is
    what :func:`compare_real` does: its denominator is identical in both arms and
    cancels there.

    Do not read the absolute value as "1 means every stretch of ink is drawn
    once", because the two sides count different ink. The denominator is one
    skeleton of the ink above Otsu's threshold; the tracer works below it
    (``pipeline.trace_threshold``) and adds faint strokes below that again, so it
    draws lines the denominator never counted. On the real test drawings the
    reading runs 1.01 to 1.64, and turning faint strokes off takes the densest
    from 1.64 to 1.29 - while 99.3% of that curve length is on ink and only 8.6%
    of it lies within 0.75 px of another stroke. It is not redundancy.
    """
    from line2func import pipeline, quality

    rows = []
    for path in _real_images(folder)[:limit]:
        rgb = lineart.load_rgb(path)
        t = time.perf_counter()
        curves, ink = pipeline.trace(rgb, fit_tolerance=tolerance, curve_count=None, upscale="auto",
                                     baseline_options=options)
        seconds = time.perf_counter() - t
        # the judge's threshold comes from the ink alone, so it is the same for every setting
        # compared on this drawing even when the tracer chooses a different one
        thr = quality.tracer_threshold(ink)
        report, _ = quality.assess(curves, ink, thr)
        rows.append({
            "image": path.name,
            "megapixels": round(ink.shape[0] * ink.shape[1] / 1e6, 3),
            "curves": report["structure"]["curves"],
            "strokes": report["structure"]["strokes"],
            "kept": report["recall"]["line"]["within_2px"],
            "kept_with_faint": report["recall"]["with_faint"]["within_2px"],
            "precision": report["precision"]["within_2px"],
            "missed_ink": report["missed"]["share_of_ink"],
            "d_M": report["distance"]["d_M"],
            "psnr_db": report["raster"]["psnr_db"],
            "ssim": report["raster"]["ssim"],
            "length_vs_skeleton": round(total_length(curves) / max(baseline.ink_length(ink, thr), 1.0), 4),
            "seconds": round(seconds, 2),
            "threshold": round(thr, 4),
        })
    return {"folder": str(folder), "tolerance": tolerance, "options": options or {},
            "images": rows, "median": _medians(rows)}


MEASURES = ("curves", "strokes", "kept", "kept_with_faint", "precision", "missed_ink", "d_M", "psnr_db", "ssim",
            "length_vs_skeleton", "seconds")


def _medians(rows: list[dict]) -> dict[str, float]:
    return {k: round(float(np.median([r[k] for r in rows])), 4) for k in MEASURES} if rows else {}


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, samples: int = 10_000,
                 seed: int = REAL_SEED) -> tuple[float, float]:
    """Percentile bootstrap interval for the median of ``values``.

    Five drawings is far too few for a normal approximation to mean anything,
    and the per-image spread is large, so the interval is what says whether a
    change is real. A wide interval that straddles zero is the honest answer,
    not a failure.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.median(values[rng.integers(0, len(values), (samples, len(values)))], axis=1)
    half = 0.5 * (1.0 - confidence)
    return (round(float(np.quantile(draws, half)), 4), round(float(np.quantile(draws, 1.0 - half)), 4))


def compare_real(before: dict, after: dict) -> dict:
    """Paired per-image differences between two :func:`eval_real` reports.

    The same drawings in both, so each one is its own control: the difference
    per drawing is what is measured, and the median of those differences with a
    bootstrap interval is what is reported. Means over a handful of drawings
    would be led by whichever one happens to be largest.

    Each measure also gets what the comparison could *resolve*, because an
    interval containing zero is not a finding. ``halfwidth`` is half the
    interval, so it is the size of a difference this many drawings can tell from
    nothing, and ``n_for`` is how many drawings a difference the size of the
    observed median would need before its interval cleared zero.

    The spread being estimated is not measurement noise - tracing is
    deterministic, and one setting compared with itself gives exactly zero
    everywhere. It is how much the *effect* varies from drawing to drawing, which
    is what actually limits what a handful of drawings can show.
    """
    a = {r["image"]: r for r in before["images"]}
    b = {r["image"]: r for r in after["images"]}
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two reports have no drawing in common")
    out = {}
    # a report saved by an older version may not have every measure; compare what both have
    for key in [k for k in MEASURES if all(k in a[i] and k in b[i] for i in shared)]:
        d = np.array([b[i][key] - a[i][key] for i in shared], dtype=np.float64)
        lo, hi = bootstrap_ci(d)
        median, sd = float(np.median(d)), float(np.std(d, ddof=1)) if len(d) > 1 else float("nan")
        out[key] = {"median": round(median, 4), "ci95": [lo, hi],
                    "sd": round(sd, 4) if np.isfinite(sd) else None,
                    "halfwidth": round(0.5 * (hi - lo), 4) if np.isfinite(hi - lo) else None,
                    "n_for": _n_for(median, sd),
                    "per_image": {i: round(float(b[i][key] - a[i][key]), 4) for i in shared}}
    return {"images": shared, "before": before.get("options", {}), "after": after.get("options", {}),
            "delta": out}


def _n_for(effect: float, sd: float) -> int | None:
    """Drawings needed before a difference of ``effect`` would clear zero, or None if it cannot say.

    The usual two-sided 95% / 80% power count, ``(2.8 sd / effect)^2``. It is a
    normal approximation, so it is a scale rather than a promise; with a
    deterministic tracer ``sd`` is effect variation between drawings, and an
    effect the same on every drawing needs only a couple.
    """
    if not np.isfinite(effect) or not np.isfinite(sd) or abs(effect) < 1e-12:
        return None
    if sd < 1e-12:
        return 2  # the same difference on every drawing
    return int(np.ceil((2.8 * sd / abs(effect)) ** 2))


def _print_real(report: dict) -> None:
    cols = [("curves", "curves"), ("kept", "kept"), ("missed_ink", "missed"), ("d_M", "d_M"),
            ("length_vs_skeleton", "len/skel"), ("psnr_db", "PSNR"), ("seconds", "s")]
    print(f"{'drawing':<28}" + "".join(f"{h:>10}" for _, h in cols))
    for r in report["images"]:
        print(f"{r['image'][:28]:<28}" + "".join(f"{r[k]:>10.4g}" for k, _ in cols))
    if report["median"]:
        print(f"{'median':<28}" + "".join(f"{report['median'][k]:>10.4g}" for k, _ in cols))


def _print_compare(result: dict) -> None:
    n = len(result["images"])
    print(f"paired over {n} drawings (median difference, 95% bootstrap interval):")
    print(f"  {'measure':<18}{'median':>11}   {'95% interval':>21}     {'resolves':>9}{'n needed':>10}")
    for key, d in result["delta"].items():
        lo, hi = d["ci95"]
        found = not (lo <= 0.0 <= hi)  # the interval misses zero
        need = "-" if d["n_for"] is None else d["n_for"]
        print(f"  {key:<18}{d['median']:>11.4g}   [{lo:>9.4g}, {hi:>9.4g}]{'  *' if found else '   '}"
              f"  {d['halfwidth'] if d['halfwidth'] is not None else '-':>9}{need:>10}")
    print(f"  * the interval misses zero: a difference was found. A row without one is NOT evidence of no\n"
          f"    effect - "
          f"'resolves' is the smallest difference these {n} drawings could tell from nothing, and 'n needed'\n"
          f"    is how many drawings a difference the size of that row's median would take.")


def _print_scenes(results: dict) -> None:
    cols = [("f_gt2", "F_GT@2"), ("crossing_continuity", "cross cont."), ("gap_closure", "gap closure"),
            ("fragments_per_stroke", "frag/stroke"), ("curve_ratio", "curve ratio"),
            ("length_ratio", "length ratio"), ("seconds_per_mp", "s/MP")]
    print(f"{'subset':<8}{'scenes':>7}" + "".join(f"{h:>13}" for _, h in cols))
    for name, r in results.items():
        print(f"{name:<8}{r['scenes']:>7}" + "".join(f"{r[k]:>13.3f}" for k, _ in cols))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.eval", description="Evaluate the tracer.")
    p.add_argument("--valset", type=Path, help="scene set (e.g. data/val_v1)")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most this many scenes per subset")
    p.add_argument("--realset", type=Path,
                   help="folder of real drawings: trace each at a fixed tolerance and judge it against its ink")
    p.add_argument("--tolerance", type=float, default=1.0, help="with --realset: fit tolerance, px")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", dest="options",
                   help="with --realset: a BaselineParams field to override (e.g. --set local_trim=false)")
    p.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                   help="paired per-drawing differences between two --realset reports, with bootstrap intervals")
    p.add_argument("--json", type=Path, help="also write the results to this JSON file")
    args = p.parse_args(argv)

    if args.compare is not None:
        before, after = (json.loads(p_.read_text(encoding="utf-8")) for p_ in args.compare)
        results = compare_real(before, after)
        _print_compare(results)
    elif args.realset is not None:
        results = eval_real(args.realset, args.tolerance, args.limit,
                            options=_options(p, args.options))
        _print_real(results)
    elif args.valset is not None:
        results = eval_scenes(args.valset, args.limit)
        _print_scenes(results)
    else:
        p.error("give --valset (baseline scenes), --realset (real drawings) or --compare")
    if args.json:
        from line2func.jobs import finite

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(finite(results), indent=1), encoding="utf-8", newline="\n")
    return 0


def _options(parser, pairs: list[str]) -> dict:
    """``["local_trim=false"]`` as ``{"local_trim": False}``, typed like the field it overrides."""
    import dataclasses

    fields = {f.name: f.type for f in dataclasses.fields(baseline.BaselineParams)}
    out = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in fields:
            parser.error(f"--set: {name!r} is not a BaselineParams field")
        kind = str(fields[name])
        if "bool" in kind:
            if value.lower() not in ("true", "false"):
                parser.error(f"--set {name}: expected true or false, got {value!r}")
            out[name] = value.lower() == "true"
        elif value.lower() == "none":
            out[name] = None
        elif "int" in kind and "float" not in kind:
            out[name] = int(value)
        elif "float" in kind:
            out[name] = float(value)
        else:
            out[name] = value
    return out


if __name__ == "__main__":
    raise SystemExit(main())

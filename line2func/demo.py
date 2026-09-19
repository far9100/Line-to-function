"""Trace an image and write every output format.

    python -m line2func.demo drawing.png --lineart none --vectorizer baseline --out out/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from line2func import lineart, pipeline
from line2func.export import DESMOS_CURVE_LIMIT, FORMS, write_outputs
from line2func.functions import FUNCTION_TOLERANCE, attach


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m line2func.demo", description=__doc__.strip().splitlines()[0])
    p.add_argument("image", type=Path, help="input image (PNG, JPEG, ...)")
    p.add_argument("--lineart", choices=lineart.METHODS, default="none",
                   help="line extraction: none (input is line art), canny, xdog, or the pretrained "
                        "informative / informative-coarse network (needs PyTorch and "
                        "'python -m line2func.weights fetch informative') (default: none)")
    p.add_argument("--vectorizer", choices=("baseline", "model"), default="baseline",
                   help="tracing engine (default: baseline)")
    p.add_argument("--ckpt", type=Path, help="checkpoint for --vectorizer model")
    p.add_argument("--out", type=Path, default=Path("out"), help="output folder (default: out/)")
    p.add_argument("--tolerance", type=float, default=None,
                   help="trace to this max curve fitting error in pixels (e.g. 1.0) instead of making a set "
                        "number of curves (default: off, see --curves)")
    p.add_argument("--threshold", type=float, default=None,
                   help="ink threshold in [0, 1] (default: automatic: Otsu's, but for line art at most 0.25, so "
                        "faint strokes stay whole, unless the paper itself would be traced)")
    p.add_argument("--no-refine", action="store_true",
                   help="skip snapping the curves to the ink centerline")
    p.add_argument("--form", choices=FORMS, default=None,
                   help="how desmos.txt / equations.tex write each curve: parametric (x(t), y(t)), named (straight "
                        "lines as y = mx + c and circular arcs as (x-h)^2 + (y-k)^2 = r^2, the others parametric) "
                        "or function (every curve cut into pieces of y = f(x) / x = g(y), each within "
                        "--function-tolerance) (default: parametric)")
    p.add_argument("--named", action="store_true", help="the same as --form named")
    p.add_argument("--function-tolerance", type=float, default=FUNCTION_TOLERANCE, metavar="PX",
                   help=f"with --form function: the largest distance (px) between a function and its curve "
                        f"(default: {FUNCTION_TOLERANCE})")
    p.add_argument("--shape-tolerance", type=float, default=0.5,
                   help="max deviation (px) for recognizing lines and arcs (default: 0.5)")
    p.add_argument("--upscale", default="auto", choices=("auto", "1", "2", "3", "4"),
                   help="trace at N x resolution, then map back (auto: 2 when lines are thinner than "
                        "~1.75 px, else 1; default: auto)")
    p.add_argument("--no-faint", action="store_true",
                   help="do not add faint strokes below the threshold, nor very faint lines (on by default; "
                        "conservative)")
    p.add_argument("--denoise", type=float, default=pipeline.DENOISE, metavar="0..100",
                   help="how strongly specks and short faint pieces are dropped as noise: 0 keeps them all (most "
                        "detail; on a noisy scan the noise is traced too), 100 is twice as strict (default: 50)")
    p.add_argument("--no-residual", action="store_true",
                   help="skip the second pass that traces ink the first pass left uncovered")
    p.add_argument("--no-outline", action="store_true",
                   help="keep solid areas (heavy eyelashes) and thick or wedge-shaped strokes as centerlines "
                        "instead of filled outlines")
    p.add_argument("--no-fill", action="store_true",
                   help="leave filled areas hollow in Desmos: no rings inside them (the SVG fills them anyway)")
    p.add_argument("--curves", type=int, default=None, metavar="N",
                   help=f"make N curves: trace finely, then merge the pieces whose merge changes the drawing "
                        f"least; a drawing that gives fewer keeps all of them. More curves follow the lines more "
                        f"closely (default: {DESMOS_CURVE_LIMIT})")
    p.add_argument("--decisions", default="learned", metavar="SCORER",
                   help="who decides crossings, gap links, junction pairing and corners: learned (default: "
                        "the learned scorer), rules (the angle rules), or a path to decision weights")
    p.add_argument("--optimize", action="store_true",
                   help="refine the curves by render-and-compare (needs PyTorch; a few seconds on a GPU)")
    p.add_argument("--quality", action="store_true",
                   help="also judge the result against the image: writes quality.json and quality.png "
                        "(missed detail map) and prints a summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 2
    if args.tolerance is not None and args.curves is not None:
        print("error: use either --curves (a number of curves) or --tolerance (a fitting error), not both",
              file=sys.stderr)
        return 2
    if args.tolerance is not None and args.tolerance <= 0:
        print("error: --tolerance must be positive", file=sys.stderr)
        return 2
    if args.curves is not None and args.curves < 1:
        print("error: --curves must be at least 1", file=sys.stderr)
        return 2
    if args.named and args.form not in (None, "named"):
        print("error: --named is short for --form named; give only one of them", file=sys.stderr)
        return 2
    if not args.function_tolerance > 0:
        print("error: --function-tolerance must be positive", file=sys.stderr)
        return 2
    if not 0 <= args.denoise <= 100:
        print("error: --denoise must be between 0 and 100", file=sys.stderr)
        return 2
    form = args.form or ("named" if args.named else "parametric")
    # a number of curves by default (the Desmos budget); a given tolerance traces to that instead
    if args.curves is not None:
        curve_count = args.curves
    else:
        curve_count = None if args.tolerance is not None else DESMOS_CURVE_LIMIT
    model_run = None
    if args.vectorizer == "model":
        if args.ckpt is None or not args.ckpt.is_file():
            print("error: --vectorizer model needs --ckpt pointing to a trained multi-curve checkpoint "
                  "(e.g. runs/m3/best.pt); the neural engine is experimental", file=sys.stderr)
            return 2
        try:
            from line2func.model.infer import load_vectorizer
        except ImportError:
            print("error: --vectorizer model needs PyTorch (pip install -e .[train])", file=sys.stderr)
            return 2
        model_run = load_vectorizer(str(args.ckpt))
    if args.optimize:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("error: --optimize needs PyTorch (pip install -e .[train])", file=sys.stderr)
            return 2

    t0 = time.perf_counter()
    rgb = lineart.load_rgb(args.image)
    curves, ink = pipeline.trace(
        rgb,
        lineart_method=args.lineart,
        vectorize=model_run if args.vectorizer == "model" else None,
        fit_tolerance=args.tolerance if args.tolerance is not None else 1.0,
        threshold=args.threshold,
        refine=not args.no_refine,
        upscale=args.upscale if args.upscale == "auto" else int(args.upscale),
        shape_tolerance=args.shape_tolerance,
        faint_lines=not args.no_faint and args.vectorizer == "baseline",
        residual=not args.no_residual,
        outline=not args.no_outline,
        fill=not args.no_fill,
        optimize=args.optimize,
        curve_count=curve_count,
        decisions=None if args.decisions == "rules" else args.decisions,
        denoise=args.denoise,
    )
    n_shapes = sum(c.shape is not None for c in curves)
    functions = attach(curves, args.function_tolerance) if form == "function" else None
    elapsed = time.perf_counter() - t0
    curves.meta.update(
        source=args.image.name,
        lineart=args.lineart,
        vectorizer=args.vectorizer,
        refined=not args.no_refine,
        named=form == "named",
        form=form,
        seconds=round(elapsed, 3),
    )
    if functions is not None:
        curves.meta["functions"] = functions
    if args.threshold is not None:
        curves.meta["threshold"] = args.threshold
    paths = write_outputs(curves, args.out, source_image=rgb, form=form, function_tolerance=args.function_tolerance)

    megapixels = curves.width * curves.height / 1e6
    extras = []
    if curves.meta.get("upscale", 1) > 1:
        extras.append(f"traced at {curves.meta['upscale']}x")
    if curves.meta.get("faint_lines"):
        extras.append("faint strokes on")
    if args.denoise != pipeline.DENOISE:
        extras.append(f"noise filter {args.denoise:g}")
    n_residual = sum("residual" in c.tags for c in curves)
    if n_residual:
        extras.append(f"{n_residual} curves from the second pass")
    n_outline = len({c.stroke for c in curves if "outline" in c.tags or "fill_outline" in c.tags})
    if n_outline:
        s = "s" if n_outline != 1 else ""
        extras.append(f"{n_outline} solid area{s} or thick stroke{s} as filled outline{s}")
    n_fill = sum("fill" in c.tags for c in curves)
    if n_fill:
        extras.append(f"{n_fill} curves filling them for Desmos")
    if curves.meta.get("optimized"):
        extras.append("render-and-compare refined")
    if curves.meta.get("decisions"):
        extras.append("learned decisions")
    count = curves.meta.get("curve_count")
    if count and count["merged"]:
        extras.append(f"merged from {count['before']} to {count['target']}")
    elif count and count["after"] < count["target"] and args.curves is None:
        extras.append(f"all curves of the fine tracing, fewer than the {count['target']} allowed")
    as_functions = ""
    if functions is not None:
        as_functions = (f"; {functions['count']} functions y = f(x) / x = g(y), "
                        f"max error {functions['max_error']:.3f} px")
    print(f"{args.image.name}: {curves.width}x{curves.height}, {len(curves)} curves in "
          f"{curves.num_strokes} strokes, {elapsed:.2f} s ({elapsed / max(megapixels, 1e-9):.2f} s/MP); "
          f"{n_shapes} recognized as lines or arcs{' (named in exports)' if form == 'named' else ''}{as_functions}"
          + (f" [{', '.join(extras)}]" if extras else ""))
    for key in ("curves", "svg", "desmos", "latex", "overlay"):
        print(f"  {paths[key]}")
    if count and count["after"] < count["target"] and args.curves is not None:
        print(f"note: traced finely, the drawing gives {count['after']} curves, fewer than {count['target']}; "
              "all of them are kept.", file=sys.stderr)
    if count and count["dropped"]:
        print(f"note: {count['dropped']} of the shortest pieces were dropped to get down to "
              f"{count['target']} curves.", file=sys.stderr)
    if len(curves) > DESMOS_CURVE_LIMIT:
        print(f"warning: {len(curves)} curves, more than the {DESMOS_CURVE_LIMIT} Desmos budget. Leave out "
              f"--tolerance and --curves (then line2func makes at most {DESMOS_CURVE_LIMIT}), or use a "
              f"simpler image.", file=sys.stderr)
    if functions is not None and functions["count"] > DESMOS_CURVE_LIMIT:
        # fewer, longer curves each give somewhat more functions (on the test drawings +2% with 7% fewer curves,
        # +5 to 10% with 37 to 44% fewer); the power keeps the suggestion within the budget
        suggested = max(1, int(len(curves) * (DESMOS_CURVE_LIMIT / functions["count"]) ** 1.4))
        print(f"warning: {functions['count']} functions, more than the {DESMOS_CURVE_LIMIT} Desmos budget "
              f"({functions['count'] / len(curves):.2f} per curve). For about {DESMOS_CURVE_LIMIT} functions, "
              f"try --curves {suggested}.", file=sys.stderr)
    if len(curves) == 0:
        print("warning: no lines found. If the drawing is light on dark or a photo, "
              "try --lineart canny or --lineart xdog.", file=sys.stderr)
    if args.quality:
        import json

        from line2func import quality
        from line2func.render import save_png

        report, maps = quality.assess(curves, ink, args.threshold)
        (args.out / "quality.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
        save_png(args.out / "quality.png", quality.quality_map(maps))
        print("quality:")
        for line in quality.summary(report):
            print(f"  {line}")
        print(f"  {args.out / 'quality.json'}, {args.out / 'quality.png'} (red = missed line, "
              "orange = missed faint ink, blue = curve without ink)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exporters: SVG, Desmos, LaTeX and the overlay image.

Desmos and LaTeX show the curves in math orientation (y up): every curve is
flipped with ``y' = height - y`` before its power coefficients are printed.
Numbers are printed with a fixed number of decimals and never in scientific
notation, because Desmos reads ``e`` as Euler's number, so ``1e-5`` would be
"1 × e − 5" and silently draw the wrong curve.

Each curve is written in one of three forms (:data:`FORMS`): parametric
(``x(t), y(t)``, the default), named (recognized lines and arcs as
``y = mx + c`` / ``(x-h)^2 + (y-k)^2 = r^2``, :mod:`line2func.shapes`) or as
explicit functions (pieces of ``y = f(x)`` / ``x = g(y)``,
:mod:`line2func.functions`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from line2func.curves import Curve, CurveSet
from line2func.functions import FUNCTION_TOLERANCE, attach, curve_functions
from line2func.geometry import flip_y, to_power
from line2func.render import FILLED_TAGS, render_overlay, save_png


DESMOS_CURVE_LIMIT = 5000  # the most curves line2func makes for Desmos by default (demo); more gets a warning

# ---------- line colors (must match line2func/viewer/viewer.js; tests/test_viewer_assets.py checks it) ----------

LINE_COLOR_MODES = ("measured", "bw", "palette", "random")
LINE_COLOR_DEFAULT = "measured"  # each stroke in the ink color measured from the drawing
# the viewer's eight hues, cycled by stroke number
PALETTE = ("#e6194b", "#0082c8", "#3cb44b", "#f58230", "#911eb4", "#00a0a0", "#f032e6", "#808000")
RANDOM_BUCKETS = 64  # random hues repeat every 64 strokes: a multiple of len(PALETTE), and few enough to draw fast
BW_COLOR = "#000"

# how thick the lines are drawn: each stroke's own measured width, or one width for all of them
LINE_WIDTH_MODES = ("measured", "uniform")
LINE_WIDTH_DEFAULT = "measured"


def stroke_color(mode: str, stroke: int, seed: int = 0) -> str:
    """The color for stroke number ``stroke`` in line color ``mode``.

    ``"measured"`` has no color of its own (the caller keeps the ink color);
    this returns :data:`BW_COLOR` for it. ``"random"`` hashes ``seed`` and the
    stroke's bucket into a hue, so the same seed always gives the same colors.
    """
    if mode not in LINE_COLOR_MODES:
        raise ValueError(f"unknown line color mode {mode!r}; choose from {', '.join(LINE_COLOR_MODES)}")
    if mode in ("measured", "bw"):
        return BW_COLOR
    if mode == "palette":
        return PALETTE[stroke % len(PALETTE)]
    return f"hsl({_hue(seed, stroke % RANDOM_BUCKETS)} 70% 45%)"


def _hue(seed: int, bucket: int) -> int:
    """A hue in 0..359 from ``seed`` and ``bucket`` (32-bit integer hash, mirrored in viewer.js)."""
    x = (seed + bucket * 0x9E3779B1) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x % 360


DECIMALS = 2
FORMS = ("parametric", "named", "function")  # how desmos.txt and equations.tex write each curve


def format_number(value: float, decimals: int = DECIMALS) -> str:
    """Fixed-point text, never scientific notation, and never ``-0.00``."""
    if not math.isfinite(value):
        raise ValueError(f"cannot export non-finite number {value!r}")
    text = f"{value:.{decimals}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{decimals}f}"
    return text


def _polynomial(coeffs, decimals: int) -> str:
    """LaTeX ``a t^3 + b t^2 + c t + d``, always all four terms, e.g. ``-40.00t^{3}+60.00t^{2}...``."""
    out = []
    for k, (c, power) in enumerate(zip(coeffs, ("t^{3}", "t^{2}", "t", ""))):
        term = format_number(float(c), decimals) + power
        out.append(term if k == 0 or term.startswith("-") else "+" + term)
    return "".join(out)


def math_coefficients(ctrl, height: float) -> np.ndarray:
    """Power coefficients ``[a, b, c, d]`` (``(4, 2)``) of the curve in y-up coordinates."""
    return to_power(flip_y(ctrl, height))


def desmos_line(ctrl, height: float, decimals: int = DECIMALS) -> str:
    """One Desmos parametric expression, e.g. ``\\left(x(t),\\ y(t)\\right)``."""
    c = math_coefficients(ctrl, height)
    x = _polynomial(c[:, 0], decimals)
    y = _polynomial(c[:, 1], decimals)
    return rf"\left({x},\ {y}\right)"


def _form(named: bool, form: str | None) -> str:
    """The form to write: ``form`` if given, else "named" when ``named``, else "parametric"."""
    form = form if form is not None else ("named" if named else "parametric")
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; expected one of: {', '.join(FORMS)}")
    return form


def _functions(c: Curve, height: float, tolerance: float) -> list[str]:
    """The curve's function equations: ``Curve.functions``, or made now with ``tolerance`` px."""
    return c.functions if c.functions is not None else curve_functions(c.ctrl, height, tolerance)[0]


def expressions(curves: CurveSet, decimals: int = DECIMALS, named: bool = False, form: str | None = None,
                function_tolerance: float = FUNCTION_TOLERANCE):
    """Yield ``(curve, latex)`` for every expression, in the ``form`` of :func:`to_desmos`.

    One pair per expression, not per curve: as functions a curve becomes one
    pair per piece of ``y = f(x)`` / ``x = g(y)``, all sharing the same curve.
    """
    from line2func.shapes import named_latex

    form = _form(named, form)
    for c in curves:
        if form == "function":
            for eq in _functions(c, curves.height, function_tolerance):
                yield c, eq
        elif form == "named" and c.shape is not None:
            yield c, named_latex(c.shape, curves.height)
        else:
            yield c, desmos_line(c.ctrl, curves.height, decimals)


def to_desmos(curves: CurveSet, decimals: int = DECIMALS, named: bool = False, form: str | None = None,
              function_tolerance: float = FUNCTION_TOLERANCE) -> str:
    """All curves, one Desmos expression per line.

    Plain LaTeX with no styling: pasted into the expression list, Desmos draws
    every one of them as a line of the same width and color. :func:`to_desmos_js`
    is the same expressions carrying the width and color measured from the drawing.

    ``form`` (:data:`FORMS`) picks how each curve is written: "parametric" (the
    default), "named" (the same as ``named=True``: curves recognized as lines or
    arcs, ``Curve.shape``, as ``y = mx + c`` / ``(x-h)^2 + (y-k)^2 = r^2`` with
    their domain) or "function" (every curve as pieces of ``y = f(x)`` /
    ``x = g(y)``: its ``Curve.functions``, or made within ``function_tolerance``
    px, see :mod:`line2func.functions`).
    """
    return "".join(eq + "\n" for _, eq in expressions(curves, decimals, named, form, function_tolerance))


def to_latex(curves: CurveSet, decimals: int = DECIMALS, named: bool = False, form: str | None = None,
             function_tolerance: float = FUNCTION_TOLERANCE) -> str:
    """A LaTeX ``align*`` block listing every curve, in the ``form`` of :func:`to_desmos`
    (as functions: one row per piece, numbered curve.piece)."""
    from line2func.shapes import named_latex

    form = _form(named, form)
    rows = []
    for i, c in enumerate(curves):
        if form == "function":
            for j, eq in enumerate(_functions(c, curves.height, function_tolerance)):
                eq = eq.replace(r"\left\{", r",\quad \left\{", 1)
                rows.append(rf"&\text{{{i}.{j}:}}\ {eq} &&")
        elif form == "named" and c.shape is not None:
            eq = named_latex(c.shape, curves.height).replace(r"\left\{", r",\quad \left\{", 1)
            rows.append(rf"&\text{{{i}:}}\ {eq} &&")
        else:
            k = math_coefficients(c.ctrl, curves.height)
            x = _polynomial(k[:, 0], decimals)
            y = _polynomial(k[:, 1], decimals)
            rows.append(rf"x_{{{i}}}(t) &= {x}, & y_{{{i}}}(t) &= {y}")
    head = f"% line2func: {len(curves)} curves, {curves.num_strokes} strokes, "
    if form == "function":
        head += f"{len(rows)} functions y = f(x) / x = g(y), image {curves.width}x{curves.height}, y axis up"
    else:
        head += f"image {curves.width}x{curves.height}, y axis up, 0 <= t <= 1"
    lines = [head, r"\begin{align*}"]
    lines += [row + (r" \\" if i < len(rows) - 1 else "") for i, row in enumerate(rows)]
    lines.append(r"\end{align*}")
    return "\n".join(lines) + "\n"


# ---------- styled Desmos output (desmos.js) ----------
#
# Desmos draws a pasted expression as a line of one fixed width and color, so desmos.txt
# can only show how dark an area is by how densely it is drawn (line2func.fill). The API,
# though, takes a color and a lineWidth per expression, so the same curves can carry the
# width and color measured from the drawing instead - a light shadow comes out light
# because it is drawn light, not because it is drawn sparsely.

DESMOS_JS_VAR = "LINE2FUNC"  # the name the styled output binds its expression list to
DESMOS_MIN_WIDTH = 0.5  # px: thinner than this a Desmos line all but disappears, and a faint
# stroke measured at 0.09 px would be dropped from the drawing rather than drawn faintly
OUTLINE_WIDTH = 1.0  # px: a filled area's outline is its edge, not a line anyone drew, so it is
# stroked just wide enough to close the area, and the curves inside it (line2func.fill) carry the
# tone. Falling back to the drawing's line width instead drew it half a line width outside the
# area on every side: on a heavy eyelash measured at a half width of 3.6 px that alone drew it
# 1.6x its own area. A dark area is spaced at the 1.5 px floor, so its first ring already reaches
# this seal; a light one is meant to have paper showing between its curves. A thick stroke's own
# outline (line2func.outline) has been written at this width all along; a filled area's had none.


def outline_width(c: Curve, line_width: float) -> float:
    """How wide the outline of a filled area is stroked.

    A filled area's outline has ink on one side only, so no width is measured
    along it (:mod:`line2func.attributes`) and every output used to fall back to
    the drawing's own line width - which is wider than most of its strokes when
    the ink threshold was lowered, and half of which lands outside the area.
    """
    if c.width is not None:
        return float(c.width)
    return OUTLINE_WIDTH if any(t in c.tags for t in FILLED_TAGS) else float(line_width)


def tone_color(tone: float) -> str:
    """The gray a filled area of darkness ``tone`` is drawn in (ink is ``1 - gray``)."""
    v = int(round(255.0 * (1.0 - min(max(float(tone), 0.0), 1.0))))
    return f"#{v:02x}{v:02x}{v:02x}"


def desmos_color(css: str) -> str:
    """``css`` as the ``#rrggbb`` Desmos wants: the API takes hex only, while the page
    and the SVG also take ``#rgb`` and the ``hsl()`` of the random color mode."""
    if css.startswith("#"):
        body = css[1:]
        return "#" + ("".join(ch * 2 for ch in body) if len(body) == 3 else body)
    if css.startswith("hsl("):
        h, s, lum = (float(p.rstrip("%")) for p in css[4:-1].replace(",", " ").split())
        s, lum = s / 100.0, lum / 100.0
        c = (1.0 - abs(2.0 * lum - 1.0)) * s
        x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
        rgb = [(c, x, 0.0), (x, c, 0.0), (0.0, c, x), (0.0, x, c), (x, 0.0, c), (c, 0.0, x)][int(h // 60) % 6]
        return "#" + "".join(f"{int(round(255.0 * (v + lum - c / 2.0))):02x}" for v in rgb)
    raise ValueError(f"cannot write {css!r} as a Desmos hex color")


def desmos_style(c: Curve, line_width: float, dark: float) -> tuple[str, float]:
    """The ``(color, lineWidth)`` that one curve is drawn with in the styled output.

    A curve inside a filled area (``fill``, :mod:`line2func.fill`) is drawn as wide
    as it is spaced, so the curves of an area meet instead of leaving paper between
    them, and the area comes out a solid patch of its own measured color. That is
    what lets the styled output show a tone exactly rather than approximate it by
    density: the spacing already says the tone once, and drawing the curves at their
    own spacing takes it back out, leaving the color to say it.

    Everything else keeps the width and color measured along it (``Curve.width`` /
    ``Curve.color``), falling back to the drawing's line width, and to the area tone
    where no color was measured - the outline of a filled area has ink on one side
    only, so it has no width of its own (:mod:`line2func.attributes`) and takes
    :func:`outline_width` instead.
    """
    from line2func.fill import FILL_TAG, spacing_for

    color = c.color
    if color is None:
        color = tone_color(c.tone) if c.tone is not None else BW_COLOR
    width = spacing_for(c.tone, dark) if FILL_TAG in c.tags else outline_width(c, line_width)
    return desmos_color(color), round(max(width, DESMOS_MIN_WIDTH), 2)


def to_desmos_js(curves: CurveSet, decimals: int = DECIMALS, named: bool = False, form: str | None = None,
                 function_tolerance: float = FUNCTION_TOLERANCE, line_width: float | None = None,
                 color_mode: str = LINE_COLOR_DEFAULT, seed: int = 0,
                 width_mode: str = LINE_WIDTH_DEFAULT) -> str:
    """The same expressions as :func:`to_desmos`, each carrying a color and a line width.

    A JavaScript file: it binds the expressions to :data:`DESMOS_JS_VAR` and hands them
    to ``setExpressions``, so it runs in a page that embeds the Desmos API, and in the
    browser console with a calculator open where the page exposes one. ``form`` and
    ``function_tolerance`` are as in :func:`to_desmos`; ``color_mode``, ``seed`` and
    ``width_mode`` override the measured color and width exactly as in :func:`to_svg`,
    so a downloaded file matches what the page is showing.
    """
    if color_mode not in LINE_COLOR_MODES:
        raise ValueError(f"unknown line color mode {color_mode!r}; choose from {', '.join(LINE_COLOR_MODES)}")
    if width_mode not in LINE_WIDTH_MODES:
        raise ValueError(f"unknown line width mode {width_mode!r}; choose from {', '.join(LINE_WIDTH_MODES)}")
    if line_width is None:
        line_width = float(curves.meta.get("line_width") or 2.0)
    dark = float(curves.meta.get("ink_dark") or 0.0)
    even = width_mode == "uniform"
    chosen = color_mode != "measured"
    rows = []
    for i, (c, eq) in enumerate(expressions(curves, decimals, named, form, function_tolerance)):
        color, width = desmos_style(c, line_width, dark)
        if chosen:
            color = desmos_color(stroke_color(color_mode, c.stroke, seed))
        if even:
            width = round(max(float(line_width), DESMOS_MIN_WIDTH), 2)
        rows.append(" " + json.dumps({"id": f"l2f{i}", "latex": eq, "color": color, "lineWidth": width},
                                     ensure_ascii=False, separators=(",", ":")))
    head = [
        f"// line2func: {len(rows)} expressions, {len(curves)} curves, {curves.num_strokes} strokes, "
        f"image {curves.width}x{curves.height}, y axis up, 0 <= t <= 1",
        "// Every expression carries the line width and color measured from the drawing, so a shadow",
        "// shows as the gray it is rather than as solid ink.",
        "//",
        "// In a page that embeds the Desmos API (https://www.desmos.com/api):",
        f"//     calculator.setExpressions({DESMOS_JS_VAR});",
        "// With a calculator open, pasting this whole file into the browser console does the same,",
        "// where the page exposes one. Widths are in pixels, so they read as measured with the",
        "// drawing at its own size on screen.",
    ]
    tail = [
        f'if (typeof Calc !== "undefined") Calc.setExpressions({DESMOS_JS_VAR});',
        f'else if (typeof calculator !== "undefined") calculator.setExpressions({DESMOS_JS_VAR});',
    ]
    body = f"var {DESMOS_JS_VAR} = [\n" + ",\n".join(rows) + "\n];"
    return "\n".join(head + [body] + tail) + "\n"


def _svg_num(v: float) -> str:
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def to_svg(curves: CurveSet, line_width: float | None = None, color: str = "#000", measured: bool = True,
           color_mode: str = LINE_COLOR_DEFAULT, seed: int = 0, width_mode: str = LINE_WIDTH_DEFAULT) -> str:
    """SVG in image coordinates; one ``<path>`` per stroke, joined pieces share a ``C`` run.

    With ``measured=True`` each stroke is drawn with its measured width and
    color (``Curve.width`` / ``Curve.color``, median over the stroke) where known;
    otherwise every stroke uses ``line_width`` and ``color``.

    Nothing is filled. A filled area (a heavy eyelash, a shadow) is its closed
    outline plus the rings or hatching across it (:mod:`line2func.fill`), both
    of them lines, so the SVG shows the same drawing the page and Desmos do.

    ``color_mode`` (:data:`LINE_COLOR_MODES`) overrides the color only: ``"measured"``
    keeps the ink colors above, while ``"bw"``, ``"palette"`` and ``"random"`` color
    every stroke with :func:`stroke_color` and leave the measured widths alone. The
    page picks the same colors, so a downloaded SVG matches what it shows.

    ``width_mode`` (:data:`LINE_WIDTH_MODES`) does the same for the thickness:
    ``"measured"`` keeps each stroke's own width, ``"uniform"`` gives every stroke
    the one ``line_width``, so the drawing has a single even line weight.
    """
    if line_width is None:
        line_width = float(curves.meta.get("line_width") or 2.0)
    if color_mode not in LINE_COLOR_MODES:
        raise ValueError(f"unknown line color mode {color_mode!r}; choose from {', '.join(LINE_COLOR_MODES)}")
    if width_mode not in LINE_WIDTH_MODES:
        raise ValueError(f"unknown line width mode {width_mode!r}; choose from {', '.join(LINE_WIDTH_MODES)}")
    even = width_mode == "uniform"
    chosen = color_mode != "measured"
    if chosen:
        color = BW_COLOR if color_mode == "bw" else color
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{curves.width}" height="{curves.height}" '
        f'viewBox="0 0 {curves.width} {curves.height}">',
        f"<g fill=\"none\" stroke={quoteattr(color)} stroke-width=\"{_svg_num(line_width)}\" "
        'stroke-linecap="round" stroke-linejoin="round">',
    ]
    # nothing is filled: a filled area is its closed outline plus the rings or hatching across it
    # (line2func.fill), all of them lines, so the SVG, the page and Desmos draw the same thing
    for stroke, pieces in curves.strokes().items():
        d = []
        prev_end = None
        for c in pieces:
            p = c.ctrl
            if prev_end is None or np.linalg.norm(p[0] - prev_end) > 1e-6:
                d.append(f"M{_svg_num(p[0, 0])} {_svg_num(p[0, 1])}")
            d.append("C" + " ".join(f"{_svg_num(x)} {_svg_num(y)}" for x, y in p[1:]))
            prev_end = p[3]
        tags = sorted({t for c in pieces for t in c.tags})
        attrs = f' class="{" ".join(tags)}"' if tags else ""
        picked = stroke_color(color_mode, stroke, seed) if chosen else None
        if measured:
            # a filled area's outline has no measured width: outline_width gives it the one that
            # closes the area rather than the drawing's line width (which the <g> would hand it)
            widths = [outline_width(c, line_width) for c in pieces
                      if c.width is not None or any(t in c.tags for t in FILLED_TAGS)]
            colors = [c.color for c in pieces if c.color is not None]
            if widths and not even:  # "uniform": the <g> above already carries the one width
                attrs += f' stroke-width="{_svg_num(float(np.median(widths)))}"'
            if colors and not chosen:  # a chosen color mode replaces the ink color, not the width
                attrs += f" stroke={quoteattr(max(set(colors), key=colors.count))}"
        if picked is not None:
            attrs += f" stroke={quoteattr(picked)}"
        if "outline" in tags or "fill_outline" in tags:
            d.append("Z")  # a filled area's outline is closed, but never filled

        parts.append(f'<path id="stroke-{stroke}"{attrs} d="{"".join(d)}"/>')
    parts += ["</g>", "</svg>"]
    return "\n".join(parts) + "\n"


def output_texts(curves: CurveSet, named: bool = False, form: str | None = None,
                 function_tolerance: float = FUNCTION_TOLERANCE) -> dict[str, str]:
    """The text outputs by file name: ``curves.json``, ``out.svg``, ``desmos.txt``,
    ``desmos.js`` and ``equations.tex``.

    There are two Desmos outputs, the same expressions written twice: ``desmos.txt``
    is plain LaTeX to paste into the expression list, and ``desmos.js`` carries the
    measured line width and color on each one (:func:`to_desmos_js`).

    ``named`` and ``form`` as in :func:`to_desmos`. As functions, curves without
    ``Curve.functions`` get them first (:func:`line2func.functions.attach`), so
    ``curves.json`` lists them too.
    """
    form = _form(named, form)
    if form == "function" and any(c.functions is None for c in curves):
        attach(curves, function_tolerance)
    return {
        "curves.json": json.dumps(curves.to_dict(), indent=1),
        "out.svg": to_svg(curves),
        "desmos.txt": to_desmos(curves, form=form, function_tolerance=function_tolerance),
        "desmos.js": to_desmos_js(curves, form=form, function_tolerance=function_tolerance),
        "equations.tex": to_latex(curves, form=form, function_tolerance=function_tolerance),
    }


def write_outputs(curves: CurveSet, out_dir: str | Path, source_image=None, named: bool = False,
                  form: str | None = None, function_tolerance: float = FUNCTION_TOLERANCE) -> dict[str, Path]:
    """Write ``curves.json``, ``out.svg``, ``desmos.txt``, ``desmos.js``, ``equations.tex``
    and, when ``source_image`` is given, ``overlay.png`` and ``source.png``.
    ``named`` and ``form`` pick how the equations are written (:func:`to_desmos`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = output_texts(curves, named=named, form=form, function_tolerance=function_tolerance)
    paths = {}
    for key, name in (("curves", "curves.json"), ("svg", "out.svg"), ("desmos", "desmos.txt"),
                      ("desmos_js", "desmos.js"), ("latex", "equations.tex")):
        paths[key] = out / name
        # newline="\n": identical bytes on every OS (no CRLF on Windows)
        paths[key].write_text(texts[name], encoding="utf-8", newline="\n")
    if source_image is not None:
        from line2func.lineart import load_rgb

        rgb = load_rgb(source_image)
        paths["source"] = out / "source.png"
        paths["overlay"] = out / "overlay.png"
        save_png(paths["source"], rgb)
        save_png(paths["overlay"], render_overlay(rgb, curves, line_width=1.5))
    return paths

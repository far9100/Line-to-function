"""Exporters: SVG, Desmos, LaTeX and the overlay image.

Desmos and LaTeX show the curves in math orientation (y up): every curve is
flipped with ``y' = height - y`` before its power coefficients are printed.
Numbers are printed with a fixed number of decimals and never in scientific
notation, because Desmos reads ``e`` as Euler's number, so ``1e-5`` would be
"1 × e − 5" and silently draw the wrong curve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from line2func.curves import CurveSet
from line2func.geometry import flip_y, to_power
from line2func.render import render_overlay, save_png

DESMOS_CURVE_LIMIT = 5000  # the most curves line2func makes for Desmos by default (demo); more gets a warning
DECIMALS = 2


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


def to_desmos(curves: CurveSet, decimals: int = DECIMALS, named: bool = False) -> str:
    """All curves, one Desmos expression per line.

    With ``named=True``, curves recognized as lines or arcs (``Curve.shape``) are
    written as ``y = mx + c`` / ``(x-h)^2 + (y-k)^2 = r^2`` with their domain.
    """
    from line2func.shapes import named_latex

    out = []
    for c in curves:
        if named and c.shape is not None:
            out.append(named_latex(c.shape, curves.height))
        else:
            out.append(desmos_line(c.ctrl, curves.height, decimals))
    return "".join(line + "\n" for line in out)


def to_latex(curves: CurveSet, decimals: int = DECIMALS, named: bool = False) -> str:
    """A LaTeX ``align*`` block listing every curve (parametric, or named when ``named``)."""
    from line2func.shapes import named_latex

    lines = [
        f"% line2func: {len(curves)} curves, {curves.num_strokes} strokes, "
        f"image {curves.width}x{curves.height}, y axis up, 0 <= t <= 1",
        r"\begin{align*}",
    ]
    for i, c in enumerate(curves):
        end = r" \\" if i < len(curves) - 1 else ""
        if named and c.shape is not None:
            eq = named_latex(c.shape, curves.height).replace(r"\left\{", r",\quad \left\{", 1)
            lines.append(rf"&\text{{{i}:}}\ {eq} &&{end}")
            continue
        k = math_coefficients(c.ctrl, curves.height)
        x = _polynomial(k[:, 0], decimals)
        y = _polynomial(k[:, 1], decimals)
        lines.append(rf"x_{{{i}}}(t) &= {x}, & y_{{{i}}}(t) &= {y}{end}")
    lines.append(r"\end{align*}")
    return "\n".join(lines) + "\n"


def _svg_num(v: float) -> str:
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def to_svg(curves: CurveSet, line_width: float | None = None, color: str = "#000", measured: bool = True) -> str:
    """SVG in image coordinates; one ``<path>`` per stroke, joined pieces share a ``C`` run.

    With ``measured=True`` each stroke is drawn with its measured width and
    color (``Curve.width`` / ``Curve.color``, median over the stroke) where known;
    otherwise every stroke uses ``line_width`` and ``color``.
    """
    if line_width is None:
        line_width = float(curves.meta.get("line_width") or 2.0)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{curves.width}" height="{curves.height}" '
        f'viewBox="0 0 {curves.width} {curves.height}">',
        f"<g fill=\"none\" stroke={quoteattr(color)} stroke-width=\"{_svg_num(line_width)}\" "
        'stroke-linecap="round" stroke-linejoin="round">',
    ]
    for stroke, pieces in curves.strokes().items():
        if any("fill" in c.tags for c in pieces):
            continue  # rings that make a filled area look filled in Desmos; here the area is filled itself
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
        if measured:
            widths = [c.width for c in pieces if c.width is not None]
            colors = [c.color for c in pieces if c.color is not None]
            if widths:
                attrs += f' stroke-width="{_svg_num(float(np.median(widths)))}"'
            if colors:
                attrs += f" stroke={quoteattr(max(set(colors), key=colors.count))}"
        if "outline" in tags or "fill_outline" in tags:
            # a closed outline of a thick or filled shape: fill it with its ink color
            fill_colors = [c.color for c in pieces if c.color] if measured else []
            fill = max(set(fill_colors), key=fill_colors.count) if fill_colors else color
            attrs += f" fill={quoteattr(fill)} fill-rule=\"evenodd\""
            d.append("Z")
        parts.append(f'<path id="stroke-{stroke}"{attrs} d="{"".join(d)}"/>')
    parts += ["</g>", "</svg>"]
    return "\n".join(parts) + "\n"


def output_texts(curves: CurveSet, named: bool = False) -> dict[str, str]:
    """The text outputs by file name: ``curves.json``, ``out.svg``, ``desmos.txt`` and ``equations.tex``."""
    return {
        "curves.json": json.dumps(curves.to_dict(), indent=1),
        "out.svg": to_svg(curves),
        "desmos.txt": to_desmos(curves, named=named),
        "equations.tex": to_latex(curves, named=named),
    }


def write_outputs(curves: CurveSet, out_dir: str | Path, source_image=None, named: bool = False) -> dict[str, Path]:
    """Write ``curves.json``, ``out.svg``, ``desmos.txt``, ``equations.tex`` and,
    when ``source_image`` is given, ``overlay.png`` and ``source.png``.
    ``named`` writes recognized lines and arcs as named equations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = output_texts(curves, named=named)
    paths = {}
    for key, name in (("curves", "curves.json"), ("svg", "out.svg"), ("desmos", "desmos.txt"), ("latex", "equations.tex")):
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

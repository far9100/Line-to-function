import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.export import (
    desmos_line,
    format_number,
    math_coefficients,
    output_texts,
    to_desmos,
    to_latex,
    to_svg,
    write_outputs,
)

ARCH_MATH = np.array([[-40.0, 20.0], [60.0, -270.0], [60.0, 250.0], [10.0, 10.0]])
PLAN_LINE = r"\left(-40.00t^{3}+60.00t^{2}+60.00t+10.00,\ 20.00t^{3}-270.00t^{2}+250.00t+10.00\right)"


def arch_set(height=100):
    # the README's arch is given in y-up math coordinates; store it y-down
    ctrl = g.flip_y(g.from_power(ARCH_MATH), height)
    return CurveSet(120, height, [Curve(ctrl, stroke=0)])


def test_desmos_matches_plan_example():
    cs = arch_set()
    assert desmos_line(cs.curves[0].ctrl, cs.height) == PLAN_LINE
    assert to_desmos(cs) == PLAN_LINE + "\n"


def test_math_coefficients_flip_y():
    cs = arch_set(height=300)
    np.testing.assert_allclose(math_coefficients(cs.curves[0].ctrl, 300), ARCH_MATH, atol=1e-9)


@pytest.mark.parametrize(
    "value, text",
    [(1e-5, "0.00"), (-1e-5, "0.00"), (-0.004, "0.00"), (0.005, "0.01"), (-2.5e-7, "0.00"),
     (123456.789, "123456.79"), (1e21, "1000000000000000000000.00"), (-3.14159, "-3.14")],
)
def test_format_number_never_scientific(value, text):
    assert format_number(value) == text
    assert "e" not in format_number(value).lower()


def test_format_number_rejects_nan():
    with pytest.raises(ValueError):
        format_number(float("nan"))


def test_rounding_error_is_small():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ctrl = rng.uniform(0, 1000, (4, 2))
        coeffs = math_coefficients(ctrl, 1000)
        rounded = np.round(coeffs, 2)
        t = np.linspace(0, 1, 101)[:, None]
        diff = sum((coeffs[k] - rounded[k]) * t ** (3 - k) for k in range(4))
        assert np.abs(diff).max() <= 0.02 + 1e-12


def test_latex_block():
    text = to_latex(arch_set())
    assert r"\begin{align*}" in text and r"\end{align*}" in text
    assert r"x_{0}(t) &= -40.00t^{3}+60.00t^{2}+60.00t+10.00" in text


def test_svg_is_valid_and_joins_pieces():
    cs = CurveSet(
        100,
        80,
        [
            Curve(g.line([10, 10], [50, 10]), stroke=0),
            Curve(g.line([50, 10], [50, 60]), stroke=0),
            Curve(g.line([70, 70], [90, 70]), stroke=1, tags=("fill_outline",)),
        ],
        meta={"line_width": 2.5},
    )
    svg = to_svg(cs)
    root = ET.fromstring(svg.split("\n", 1)[1])
    assert root.get("viewBox") == "0 0 100 80"
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    assert len(paths) == 2
    # the filled area is painted first, so the lines drawn over it stay visible
    assert paths[0].get("class") == "fill_outline"
    assert paths[1].get("d").count("M") == 1 and paths[1].get("d").count("C") == 2
    assert 'stroke-width="2.5"' in svg


def test_svg_fills_a_shadow_without_outlining_it():
    """A filled area lighter than the drawing's dark ink is a shadow: filled, not stroked."""
    cs = CurveSet(
        100, 80,
        [
            Curve(g.line([10, 10], [50, 10]), stroke=0),
            Curve(g.line([70, 70], [90, 70]), stroke=1, tags=("fill_outline",), tone=0.25, color="#bfbfbf"),
            Curve(g.line([20, 40], [40, 40]), stroke=2, tags=("fill_outline",), tone=0.80, color="#333333"),
        ],
        meta={"line_width": 2.5, "ink_dark": 0.8},
    )
    svg = to_svg(cs)
    root = ET.fromstring(svg.split("\n", 1)[1])
    by_stroke = {p.get("id"): p for p in root.findall(".//{http://www.w3.org/2000/svg}path")}
    shadow, solid = by_stroke["stroke-1"], by_stroke["stroke-2"]
    assert shadow.get("fill") == "#bfbfbf" and shadow.get("stroke") == "none"
    assert solid.get("fill") == "#333333" and solid.get("stroke") == "#333333"


def test_write_outputs(tmp_path):
    cs = arch_set()
    img = np.full((100, 120), 255, np.uint8)
    paths = write_outputs(cs, tmp_path / "out", source_image=img)
    for key in ("curves", "svg", "desmos", "latex", "overlay", "source"):
        assert paths[key].is_file()
    for key in ("curves", "svg", "desmos", "latex"):
        assert b"\r" not in paths[key].read_bytes()  # LF only, on every OS
    assert paths["desmos"].read_bytes() == (PLAN_LINE + "\n").encode()
    assert CurveSet.load_json(paths["curves"]).curves[0].ctrl.shape == (4, 2)


def test_output_texts_match_the_written_files(tmp_path):
    cs = arch_set()
    texts = output_texts(cs, named=True)
    paths = write_outputs(cs, tmp_path, named=True)
    names = {"curves": "curves.json", "svg": "out.svg", "desmos": "desmos.txt", "latex": "equations.tex"}
    assert set(texts) == set(names.values())
    for key, name in names.items():
        assert paths[key].read_bytes() == texts[name].encode("utf-8")


def test_the_arch_as_functions():
    """The README's arch rises steeply, turns flat at the top and falls: x = g(y), y = f(x), x = g(y)."""
    cs = arch_set()
    lines = to_desmos(cs, form="function").splitlines()
    assert [line[:2] for line in lines] == ["x=", "y=", "x="]
    assert all(line.endswith(r"\right\}") and not re.search(r"\d[eE][+-]?\d", line) for line in lines)
    latex = to_latex(cs, form="function")
    assert latex.startswith("% line2func: 1 curves, 1 strokes, 3 functions y = f(x) / x = g(y), image 120x100")
    assert r"&\text{0.0:}\ x=" in latex and r"&\text{0.2:}\ x=" in latex and r",\quad \left\{" in latex
    assert latex.count(r" \\") == 2  # rows end with \\ except the last


def test_forms_and_the_named_flag():
    cs = arch_set()
    assert to_desmos(cs, named=True) == to_desmos(cs, form="named")
    assert to_desmos(cs) == to_desmos(cs, form="parametric") == PLAN_LINE + "\n"
    assert to_latex(cs, form="parametric") == to_latex(cs)
    with pytest.raises(ValueError):
        to_desmos(cs, form="implicit")


def test_function_outputs_list_the_functions_in_curves_json():
    cs = arch_set()
    texts = output_texts(cs, form="function")
    doc = json.loads(texts["curves.json"])
    assert doc["curves"][0]["functions"] == texts["desmos.txt"].splitlines()
    assert CurveSet.from_dict(doc).curves[0].functions == cs.curves[0].functions
    # a tighter tolerance makes new functions only where none are attached yet
    assert to_desmos(cs, form="function", function_tolerance=0.01) == texts["desmos.txt"]
    cs.curves[0].functions = None
    assert len(to_desmos(cs, form="function", function_tolerance=0.01).splitlines()) > 3

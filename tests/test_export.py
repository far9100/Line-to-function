import json
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.export import (
    DESMOS_JS_VAR,
    DESMOS_MIN_WIDTH,
    desmos_color,
    desmos_line,
    format_number,
    math_coefficients,
    output_texts,
    to_desmos,
    to_desmos_js,
    to_latex,
    to_svg,
    write_outputs,
)
from line2func.fill import spacing_for

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
    assert paths[0].get("d").count("M") == 1 and paths[0].get("d").count("C") == 2
    assert paths[1].get("class") == "fill_outline"
    assert 'stroke-width="2.5"' in svg


def test_svg_never_fills_anything():
    """Every shaded part is lines, so the SVG, the page and Desmos draw the same thing."""
    cs = CurveSet(
        100, 80,
        [
            Curve(g.line([10, 10], [50, 10]), stroke=0),
            Curve(g.line([70, 70], [90, 70]), stroke=1, tags=("fill_outline",), tone=0.25, color="#bfbfbf"),
            Curve(g.line([20, 40], [40, 40]), stroke=2, tags=("fill_outline",), tone=0.80, color="#333333"),
        ],
        meta={"line_width": 2.5, "ink_dark": 0.8},
    )
    for mode in ("measured", "bw", "palette", "random"):
        svg = to_svg(cs, color_mode=mode)
        assert "fill=" not in svg.replace('fill="none"', "")  # only the <g>'s own fill="none"
        assert 'stroke="none"' not in svg
    root = ET.fromstring(to_svg(cs).split("\n", 1)[1])
    by_stroke = {p.get("id"): p for p in root.findall(".//{http://www.w3.org/2000/svg}path")}
    shadow, solid = by_stroke["stroke-1"], by_stroke["stroke-2"]
    # closed and stroked in the color measured inside the area, but never filled
    assert shadow.get("d").endswith("Z") and shadow.get("stroke") == "#bfbfbf"
    assert solid.get("d").endswith("Z") and solid.get("stroke") == "#333333"


# ---------- the styled Desmos output (desmos.js) ----------


def js_items(text):
    """The expression list of a desmos.js, as Python objects."""
    return json.loads("[" + text.split("= [\n", 1)[1].split("\n];", 1)[0] + "]")


def shaded_set():
    """A plain stroke, the outline of a light area and of a dark one, and a curve inside each."""
    return CurveSet(
        100, 80,
        [
            Curve(g.line([10, 10], [50, 10]), stroke=0, width=3.0, color="#222222"),
            Curve(g.line([70, 70], [90, 70]), stroke=1, tags=("fill_outline",), tone=0.40, color="#999999"),
            Curve(g.line([72, 72], [88, 72]), stroke=2, tags=("fill",), tone=0.40),
            Curve(g.line([20, 40], [40, 40]), stroke=3, tags=("fill_outline",), tone=0.80, color="#333333"),
            Curve(g.line([22, 42], [38, 42]), stroke=4, tags=("fill",), tone=0.80),
        ],
        meta={"line_width": 2.5, "ink_dark": 0.8},
    )


def test_desmos_js_writes_the_same_expressions_as_desmos_txt():
    """The two Desmos outputs differ only in the styling: the math in them is the same."""
    cs = shaded_set()
    items = js_items(to_desmos_js(cs))
    assert [i["latex"] for i in items] == to_desmos(cs).splitlines()
    assert [i["id"] for i in items] == [f"l2f{i}" for i in range(len(items))]
    text = to_desmos_js(cs)
    assert f"var {DESMOS_JS_VAR} = [" in text and f"Calc.setExpressions({DESMOS_JS_VAR});" in text


def test_desmos_js_carries_the_measured_width_and_color():
    """What desmos.txt can only say by drawing densely, desmos.js says with a color and a width."""
    by_stroke = {c.stroke: i for c, i in zip(shaded_set(), js_items(to_desmos_js(shaded_set())))}
    assert by_stroke[0]["color"] == "#222222" and by_stroke[0]["lineWidth"] == 3.0
    # an outline has ink on one side only, so no width of its own: the drawing's line width
    assert by_stroke[1]["color"] == "#999999" and by_stroke[1]["lineWidth"] == 2.5
    # a curve inside an area is drawn as wide as it is spaced, so the area comes out a solid
    # patch of its own gray instead of showing paper between the curves
    for stroke, tone in ((2, 0.40), (4, 0.80)):
        assert by_stroke[stroke]["lineWidth"] == round(spacing_for(tone, 0.8), 2)
        assert by_stroke[stroke]["color"] == "#" + f"{round(255 * (1 - tone)):02x}" * 3
    assert by_stroke[2]["color"] != by_stroke[4]["color"]  # the light area is lighter than the dark one


def test_desmos_js_only_ever_writes_hex_colors():
    """The API takes hex; the page's own modes also speak #rgb and hsl()."""
    cs = shaded_set()
    for mode in ("measured", "bw", "palette", "random"):
        colors = [i["color"] for i in js_items(to_desmos_js(cs, color_mode=mode, seed=7))]
        assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors), colors
    assert desmos_color("#000") == "#000000" and desmos_color("#e6194b") == "#e6194b"
    assert desmos_color("hsl(0 70% 45%)") == "#c32222" and desmos_color("hsl(120 70% 45%)") == "#22c322"
    with pytest.raises(ValueError):
        desmos_color("rebeccapurple")


def test_desmos_js_never_writes_a_line_too_thin_to_see():
    cs = CurveSet(100, 80, [Curve(g.line([10, 10], [50, 10]), stroke=0, width=0.09)],
                  meta={"line_width": 2.5})
    assert js_items(to_desmos_js(cs))[0]["lineWidth"] == DESMOS_MIN_WIDTH


def test_desmos_js_takes_the_pages_line_style():
    cs = shaded_set()
    uniform = js_items(to_desmos_js(cs, width_mode="uniform"))
    assert {i["lineWidth"] for i in uniform} == {2.5}
    assert {i["color"] for i in js_items(to_desmos_js(cs, color_mode="bw"))} == {"#000000"}
    with pytest.raises(ValueError):
        to_desmos_js(cs, color_mode="nope")
    with pytest.raises(ValueError):
        to_desmos_js(cs, width_mode="nope")


def test_write_outputs(tmp_path):
    cs = arch_set()
    img = np.full((100, 120), 255, np.uint8)
    paths = write_outputs(cs, tmp_path / "out", source_image=img)
    for key in ("curves", "svg", "desmos", "desmos_js", "latex", "overlay", "source"):
        assert paths[key].is_file()
    for key in ("curves", "svg", "desmos", "desmos_js", "latex"):
        assert b"\r" not in paths[key].read_bytes()  # LF only, on every OS
    assert paths["desmos"].read_bytes() == (PLAN_LINE + "\n").encode()
    assert CurveSet.load_json(paths["curves"]).curves[0].ctrl.shape == (4, 2)


def test_output_texts_match_the_written_files(tmp_path):
    cs = arch_set()
    texts = output_texts(cs, named=True)
    paths = write_outputs(cs, tmp_path, named=True)
    names = {"curves": "curves.json", "svg": "out.svg", "desmos": "desmos.txt",
             "desmos_js": "desmos.js", "latex": "equations.tex"}
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

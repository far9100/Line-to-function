"""The viewer's files: served by both servers with the right types, shipped in the package, and its
JavaScript equation code matches the Python exporter byte for byte."""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from line2func import app as app_mod
from line2func import pipeline, serve
from line2func.curves import Curve, CurveSet
from line2func.export import (DESMOS_CURVE_LIMIT, LINE_COLOR_MODES, LINE_WIDTH_MODES, PALETTE,
                              RANDOM_BUCKETS, desmos_line, stroke_color, to_svg)

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def test_the_asset_list_matches_the_folder_and_the_page():
    on_disk = {p.name for p in serve.VIEWER_DIR.iterdir() if p.is_file()} - {"index.html"}
    assert on_disk == set(serve.ASSETS)
    html = serve.VIEWER.read_text(encoding="utf-8")
    referenced = {src for src in re.findall(r'src="([^"]+)"', html) if not src.startswith("data:")}
    for source in [serve.VIEWER_DIR / name for name in serve.ASSETS if name.endswith(".js")]:
        referenced |= set(re.findall(r'from "\./([^"]+)"', source.read_text(encoding="utf-8")))
    assert referenced <= set(serve.ASSETS)


def test_the_page_knows_every_stage():
    """app.js maps each server stage to a progress step, with a weight, listed in the order the stages run."""
    source = (serve.VIEWER_DIR / "app.js").read_text(encoding="utf-8")

    def names(table: str) -> list[str]:
        return re.findall(r"(\w+):", re.search(rf"const {table} = \{{(.*?)\}};", source, re.S).group(1))

    stages = [*pipeline.STAGES, "load_engine", "resize", "load_model", "export", "quality"]
    assert set(stages) <= set(names("STEP_OF")) and set(stages) <= set(names("WEIGHTS"))
    ran = [s for s in names("WEIGHTS") if s in pipeline.STAGES]
    assert ran == [s for s in pipeline.STAGES if s in ran]  # the progress bar relies on this order


def test_the_viewer_and_the_exporter_share_the_desmos_limit():
    source = (serve.VIEWER_DIR / "viewer.js").read_text(encoding="utf-8")
    assert int(re.search(r"export const DESMOS_LIMIT = (\d+);", source).group(1)) == DESMOS_CURVE_LIMIT


def test_the_page_and_the_pipeline_share_the_default_noise_filter_strength():
    source = (serve.VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    assert float(re.search(r"const DENOISE = (\d+);", source).group(1)) == pipeline.DENOISE
    assert float(re.search(r"const FAINT = (\d+);", source).group(1)) == pipeline.FAINT_SENSITIVITY


def test_the_package_ships_every_asset():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'line2func\s*=\s*\[[^\]]*"viewer/\*"', pyproject), "package-data must include viewer/*"


@pytest.fixture(params=["app", "static"])
def base_url(request, tmp_path, monkeypatch):
    monkeypatch.setenv("LINE2FUNC_HOME", str(tmp_path / "home"))
    if request.param == "app":
        app = app_mod.make_app(port=0)
        server = app.server
    else:
        (tmp_path / "curves.json").write_text('{"format": "line2func.curves"}', encoding="utf-8")
        server = serve.make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_every_asset_is_served_with_its_type(base_url):
    for name, content_type in serve.ASSETS.items():
        with urllib.request.urlopen(f"{base_url}/{name}") as r:
            assert r.headers["Content-Type"] == content_type and r.headers["X-Content-Type-Options"] == "nosniff"
            assert r.read() == (serve.VIEWER_DIR / name).read_bytes()
    for bad in ("/viewer.js.bak", "/../pyproject.toml", "/%2e%2e/app.py", "/viewer/app.js"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base_url + bad)
        assert err.value.code == 404


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("name", [n for n in serve.ASSETS if n.endswith(".js")])
def test_javascript_parses(name):
    source = (serve.VIEWER_DIR / name).read_text(encoding="utf-8")
    out = subprocess.run([NODE, "--input-type=module", "--check", "-"], input=source, capture_output=True,
                         text=True, encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_equations_match_the_exporter():
    rng = np.random.default_rng(7)
    ctrls = rng.uniform(-50, 1100, (300, 4, 2))
    ctrls[:20] = np.round(ctrls[:20])  # whole pixels too
    height = 700
    script = (f"import {{ desmosLine }} from {json.dumps((serve.VIEWER_DIR / 'viewer.js').as_uri())};\n"
              f"for (const p of {json.dumps(ctrls.reshape(len(ctrls), 8).tolist())}) console.log(desmosLine(p, {height}));\n")
    out = subprocess.run([NODE, "--input-type=module", "-"], input=script, capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [desmos_line(c, height) for c in ctrls]


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_line_colors_match_the_exporter():
    """The page and the SVG must pick the same color, or a downloaded SVG would not match the screen."""
    cases = [(mode, seed, stroke)
             for mode in ("bw", "palette", "random")
             for seed in (0, 1, 12345, 0xFFFFFFFF)
             for stroke in (0, 1, 7, 8, 63, 64, 65, 999, 123456)]
    script = (f"import {{ strokeColor }} from {json.dumps((serve.VIEWER_DIR / 'viewer.js').as_uri())};\n"
              f"for (const [m, seed, s] of {json.dumps(cases)}) console.log(strokeColor(m, s, seed));\n")
    out = subprocess.run([NODE, "--input-type=module", "-"], input=script, capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [stroke_color(mode, stroke, seed) for mode, seed, stroke in cases]


def test_the_page_and_the_exporter_agree_on_the_color_modes():
    source = (serve.VIEWER_DIR / "viewer.js").read_text(encoding="utf-8")
    modes = json.loads(re.search(r"LINE_COLOR_MODES = (\[[^\]]*\])", source).group(1).replace("'", '"'))
    assert modes == [m for m in LINE_COLOR_MODES if m != "measured"]  # the page never draws the ink colors
    buckets = int(re.search(r"RANDOM_BUCKETS = (\d+)", source).group(1))
    assert buckets == RANDOM_BUCKETS
    assert buckets % len(PALETTE) == 0  # so a bucket's index picks the same palette hue as its stroke number
    widths = json.loads(re.search(r"LINE_WIDTH_MODES = (\[[^\]]*\])", source).group(1).replace("'", '"'))
    assert widths == list(LINE_WIDTH_MODES)


def test_uniform_width_gives_every_stroke_the_same_thickness():
    """The point of the uniform mode: one thickness everywhere, and it is the result's own line width."""
    rng = np.random.default_rng(3)
    curves = [Curve(ctrl=rng.uniform(0, 200, (4, 2)), stroke=i // 2, width=0.2 + 0.1 * i) for i in range(12)]
    cs = CurveSet(curves=curves, width=200, height=200, meta={"line_width": 1.75})

    uniform = to_svg(cs, width_mode="uniform")
    assert set(re.findall(r'stroke-width="([^"]+)"', uniform)) == {"1.75"}

    measured = to_svg(cs, width_mode="measured")
    assert len(set(re.findall(r'stroke-width="([^"]+)"', measured))) > 1

    with pytest.raises(ValueError):
        to_svg(cs, width_mode="thick")


def test_the_engine_and_the_worker_agree_on_the_protocol():
    """They are served together; a mismatch must be a caught "page is out of date", never a silent hang."""
    def protocol(name: str) -> int:
        source = (serve.VIEWER_DIR / name).read_text(encoding="utf-8")
        return int(re.search(r"PROTOCOL = (\d+)", source).group(1))

    assert protocol("engine.js") == protocol("worker.js")


def test_the_worker_handles_every_message_the_engine_sends():
    engine = (serve.VIEWER_DIR / "engine.js").read_text(encoding="utf-8")
    worker = (serve.VIEWER_DIR / "worker.js").read_text(encoding="utf-8")
    # every {type: "..."} the engine builds, less "module" (the Worker's own option)
    sent = set(re.findall(r'type:\s*"(\w+)"', engine)) - {"module"}
    handled = set(re.findall(r'm\.type === "(\w+)"', worker))
    assert sent == {"init", "open", "trace", "svg"}  # if this changes, the pair below is what matters
    assert sent <= handled, sent - handled

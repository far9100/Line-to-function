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
from line2func.export import DESMOS_CURVE_LIMIT, desmos_line

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

    stages = [*pipeline.STAGES, "resize", "load_model", "export", "quality"]
    assert set(stages) <= set(names("STEP_OF")) and set(stages) <= set(names("WEIGHTS"))
    ran = [s for s in names("WEIGHTS") if s in pipeline.STAGES]
    assert ran == [s for s in pipeline.STAGES if s in ran]  # the progress bar relies on this order


def test_the_viewer_and_the_exporter_share_the_desmos_limit():
    source = (serve.VIEWER_DIR / "viewer.js").read_text(encoding="utf-8")
    assert int(re.search(r"export const DESMOS_LIMIT = (\d+);", source).group(1)) == DESMOS_CURVE_LIMIT


def test_the_page_and_the_pipeline_share_the_default_noise_filter_strength():
    source = (serve.VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    assert float(re.search(r"const DENOISE = (\d+);", source).group(1)) == pipeline.DENOISE


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

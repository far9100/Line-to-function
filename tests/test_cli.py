import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from line2func import demo, geometry as g, serve
from line2func.curves import Curve, CurveSet
from line2func.export import DESMOS_CURVE_LIMIT
from line2func.lineart import canny, extract, xdog
from line2func.render import render_lineart, save_png


@pytest.fixture
def drawing(tmp_path):
    cs = CurveSet(160, 120, [Curve(g.line([10, 20], [150, 100])), Curve(g.line([10, 100], [150, 20]), stroke=1)])
    path = tmp_path / "drawing.png"
    save_png(path, render_lineart(cs, 160, 120, line_width=2.5))
    return path


def test_demo_writes_all_outputs(drawing, tmp_path, capsys):
    out = tmp_path / "out"
    assert demo.main([str(drawing), "--lineart", "none", "--vectorizer", "baseline", "--out", str(out)]) == 0
    for name in ("curves.json", "out.svg", "desmos.txt", "equations.tex", "overlay.png", "source.png"):
        assert (out / name).is_file(), name
    doc = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    assert doc["meta"]["vectorizer"] == "baseline"
    assert len({c["stroke"] for c in doc["curves"]}) == 2
    assert "2 strokes" in capsys.readouterr().out


def test_demo_errors(drawing, tmp_path):
    assert demo.main([str(tmp_path / "missing.png")]) == 2
    # the model engine needs a checkpoint
    assert demo.main([str(drawing), "--vectorizer", "model", "--out", str(tmp_path / "o")]) == 2
    # a number of curves or a fitting tolerance, not both
    assert demo.main([str(drawing), "--curves", "10", "--tolerance", "1", "--out", str(tmp_path / "o")]) == 2
    # --named is --form named
    assert demo.main([str(drawing), "--named", "--form", "function", "--out", str(tmp_path / "o")]) == 2
    assert demo.main([str(drawing), "--form", "function", "--function-tolerance", "0", "--out", str(tmp_path / "o")]) == 2


def test_demo_writes_functions(drawing, tmp_path, capsys):
    out = tmp_path / "out"
    assert demo.main([str(drawing), "--form", "function", "--out", str(out)]) == 0
    lines = (out / "desmos.txt").read_text(encoding="utf-8").splitlines()
    assert lines and all(line[:2] in ("x=", "y=") for line in lines)
    doc = json.loads((out / "curves.json").read_text(encoding="utf-8"))
    assert doc["meta"]["form"] == "function" and doc["meta"]["functions"]["count"] == len(lines)
    assert [f for c in doc["curves"] for f in c["functions"]] == lines
    assert f"{len(lines)} functions y = f(x) / x = g(y)" in capsys.readouterr().out
    # --named still writes named equations
    assert demo.main([str(drawing), "--named", "--out", str(tmp_path / "named")]) == 0
    doc = json.loads((tmp_path / "named" / "curves.json").read_text(encoding="utf-8"))
    assert doc["meta"]["form"] == "named" and doc["meta"]["named"] is True


def test_demo_uses_the_desmos_budget_unless_a_tolerance_is_given(drawing, tmp_path):
    def meta(folder):
        return json.loads((folder / "curves.json").read_text(encoding="utf-8"))["meta"]

    assert demo.main([str(drawing), "--out", str(tmp_path / "a")]) == 0
    assert meta(tmp_path / "a")["curve_count"]["target"] == DESMOS_CURVE_LIMIT  # traced finely, then merged
    assert demo.main([str(drawing), "--tolerance", "1.0", "--out", str(tmp_path / "b")]) == 0
    assert "curve_count" not in meta(tmp_path / "b")


@pytest.mark.parametrize("method", ["canny", "xdog"])
def test_photo_methods_produce_ink(method, drawing, tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :160]
    photo = np.where((xx - 80) ** 2 + (yy - 60) ** 2 < 40**2, 60, 200).astype(np.uint8)
    photo = np.clip(photo + rng.normal(0, 3, photo.shape), 0, 255).astype(np.uint8)
    ink = extract(photo, method)
    assert ink.shape == photo.shape and 0.0 <= ink.min() and ink.max() <= 1.0
    ring = np.abs(np.hypot(xx - 80, yy - 60) - 40) < 3
    assert ink[ring].mean() > 5 * ink[~ring].mean()
    path = tmp_path / "photo.png"
    save_png(path, photo)
    assert demo.main([str(path), "--lineart", method, "--out", str(tmp_path / method)]) == 0


def test_canny_and_xdog_on_flat_image():
    flat = np.full((32, 32), 0.5, np.float32)
    assert canny(flat).sum() == 0
    assert xdog(flat).max() < 0.5


def test_viewer_server(drawing, tmp_path):
    out = tmp_path / "out"
    demo.main([str(drawing), "--out", str(out)])
    (out / "secret.txt").write_text("nope")
    server = serve.make_server(out, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/") as r:
            assert b"line2func viewer" in r.read()
        with urllib.request.urlopen(base + "/data/curves.json") as r:
            assert r.headers["Content-Type"].startswith("application/json")
            assert json.loads(r.read())["format"] == "line2func.curves"
        with urllib.request.urlopen(base + "/data/out.svg?download") as r:
            assert "attachment" in r.headers["Content-Disposition"]
        with urllib.request.urlopen(base + "/api/info") as r:  # tells the page it is the plain viewer
            info = json.loads(r.read())
        assert info["mode"] == "static" and "curves.json" in info["files"] and "secret.txt" not in info["files"]
        for bad in ("/data/secret.txt", "/data/../drawing.png", "/data/%2e%2e/drawing.png", "/etc/passwd"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + bad)
            assert err.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_serve_requires_curves(tmp_path):
    assert serve.main([str(tmp_path), "--no-browser"]) == 2

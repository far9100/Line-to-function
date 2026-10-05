"""The in-browser engine (line2func.web): the same answers and files as the local server, within its limits."""

import io
import json
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from line2func import app as app_mod
from line2func import geometry as g
from line2func import lineart, web
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _drawing() -> bytes:
    cs = CurveSet(160, 120, [Curve(g.line([10, 20], [150, 100])), Curve(g.line([10, 100], [150, 20]), stroke=1)])
    return _png(render_lineart(cs, 160, 120, line_width=2.5))


def _photo() -> bytes:
    """Something suggest_mode calls a photo: a soft shape, uneven light and grain."""
    yy, xx = np.mgrid[0:120, 0:160]
    shape = np.where((xx - 80) ** 2 + (yy - 60) ** 2 < 42**2, 70.0, 205.0)
    lit = shape * (0.75 + 0.3 * xx / 159.0) + np.random.default_rng(2).normal(0, 6, shape.shape)
    return _png(np.repeat(np.clip(lit, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2))


class JsBuffer:
    """Stands in for the Uint8Array the worker passes (Pyodide's JsBuffer)."""

    def __init__(self, data: bytes):
        self.data = data

    def to_bytes(self) -> bytes:
        return bytes(self.data)


PAGE = {"kind": "trace", "method": "none", "scale": "auto", "form": "function", "curves": 5000, "quality": True,
        "denoise": 50, "faint_sensitivity": 50}  # what the page asks for


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(web, "_last", None)


@pytest.fixture
def local_app(tmp_path, monkeypatch):
    monkeypatch.setenv("LINE2FUNC_HOME", str(tmp_path / "home"))
    app = app_mod.make_app(port=0)
    yield app
    app.stop("test")
    app.server.server_close()


def _without_seconds(doc: dict) -> dict:
    return {**doc, "meta": {k: v for k, v in doc["meta"].items() if k != "seconds"}}


@pytest.mark.parametrize("faint", [50, 85])
def test_the_same_image_info_and_files_as_the_local_server(local_app, faint):
    page = {**PAGE, "faint_sensitivity": faint}
    data = _drawing()
    img = local_app.add_image(data, "drawing.png")
    opened = web.open_image(JsBuffer(data), "drawing.png", key="k1")
    answer = json.loads(opened["answer"])
    assert answer["image"] == {**img.info(), "image_id": "k1"} and answer["preview_type"] == img.preview_type
    assert opened["preview"] == img.preview

    job = local_app.make_job({**page, "image_id": img.id})
    job.started = time.monotonic()
    local_app.run_job(job)
    stages = []
    result = web.trace(data, "drawing.png", json.dumps(page), stages.append, key="k1")
    answer = json.loads(result["answer"])
    assert answer["params"] == job.params and answer["params"]["faint_sensitivity"] == faint
    assert {**answer["summary"], "seconds": 0} == {**job.summary, "seconds": 0}
    assert sorted(result["files"]) == sorted(job.files)
    for name, body in job.files.items():
        if name == "curves.json":
            assert _without_seconds(json.loads(result["files"][name])) == _without_seconds(json.loads(body))
        else:
            assert result["files"][name] == body, name
    assert stages[0] == "resize" and stages[1] == "lineart" and stages[-2:] == ["export", "quality"]
    with zipfile.ZipFile(io.BytesIO(result["zip"])) as zf:
        assert sorted(zf.namelist()) == sorted(result["files"])


def test_the_image_decoded_last_is_reused_by_key(monkeypatch):
    data = _drawing()
    web.open_image(data, "drawing.png", key="a")
    first = web._last[1]
    decoded = []
    real = web.jobs.read_image
    monkeypatch.setattr(web.jobs, "read_image", lambda *a, **kw: decoded.append(1) or real(*a, **kw))
    web.trace(data, "drawing.png", json.dumps({"kind": "trace"}), key="a")
    assert decoded == [] and web._last[1] is first
    web.trace(data, "drawing.png", json.dumps({"kind": "trace"}), key="b")
    assert decoded == [1] and web._last[0] == "b"


def test_errors_are_answers():
    def error(result):
        return json.loads(result["answer"])["error"]

    assert error(web.open_image(b"not an image", "x.png", key="x"))["code"] == "not_an_image"
    data = _drawing()
    for params, code, field in (("{broken", "bad_json", None), ("[1]", "bad_params", None),
                                (json.dumps({"kind": "lineart"}), "bad_params", "kind"),
                                (json.dumps({"method": "informative"}), "bad_params", "method"),
                                (json.dumps({"method": "nope"}), "bad_params", "method"),
                                (json.dumps({"lineart_detail": 101}), "bad_params", "lineart_detail"),
                                (json.dumps({"form": "implicit"}), "bad_params", "form"),
                                (json.dumps({"scale": 2}), "bad_params", "scale")):
        result = web.trace(data, "d.png", params, key="d")
        assert (error(result)["code"], error(result)["field"]) == (code, field), params
        assert result["files"] == {} and result["zip"] is None

    for params, code, field in ((json.dumps({"kind": "trace"}), "bad_params", "kind"),
                                (json.dumps({"kind": "lineart"}), "bad_params", "method"),
                                (json.dumps({"kind": "lineart", "method": "informative"}), "bad_params", "method")):
        result = web.lineart_preview(data, "d.png", params, key="d")
        assert (error(result)["code"], error(result)["field"]) == (code, field), params
        assert result["files"] == {}


def test_the_browser_extracts_the_same_line_art_as_the_local_server(local_app):
    """One code path, two transports: the page must get the very same PNG either way."""
    data = _photo()
    for method, detail in (("canny", 50), ("xdog", 50), ("flow", 20), ("flow", 80)):
        params = {"kind": "lineart", "method": method, "lineart_detail": detail, "scale": "auto"}
        img = local_app.add_image(data, "photo.png")
        job = local_app.make_job({**params, "image_id": img.id})
        local_app.run_job(job)
        mine = web.lineart_preview(data, "photo.png", json.dumps(params), key=f"{method}{detail}")
        assert list(mine["files"]) == ["lineart.png"] == list(job.files), method
        assert mine["files"]["lineart.png"] == job.files["lineart.png"], (method, detail)
        answer = json.loads(mine["answer"])
        assert answer["params"]["method"] == method and answer["params"]["lineart_detail"] == detail
        assert {k: answer["summary"][k] for k in job.summary} == job.summary


def test_the_previewed_line_art_is_the_one_that_gets_traced():
    data = _photo()
    preview = {"kind": "lineart", "method": "flow", "lineart_detail": 30, "scale": 1.0}
    assert json.loads(web.lineart_preview(data, "p.png", json.dumps(preview), key="p")["answer"])["summary"]
    stages = []
    traced = {**PAGE, "method": "flow", "lineart_detail": 30, "scale": 1.0, "quality": False}
    web.trace(data, "p.png", json.dumps(traced), stages.append, key="p")
    assert "lineart" not in stages, stages  # it was already made, for the preview the user confirmed
    stages.clear()
    web.trace(data, "p.png", json.dumps({**traced, "lineart_detail": 90}), stages.append, key="p")
    assert "lineart" in stages  # a different setting is a different picture


def test_the_limits(monkeypatch, local_app):
    assert web.MAX_BYTES <= app_mod.MAX_UPLOAD
    for name in ("max_pixels", "store_side", "auto_side", "max_work_pixels", "quality_max_pixels"):
        assert getattr(web.LIMITS, name) <= getattr(app_mod._limits(), name), name
    info = web.info()
    assert info["mode"] == "web" and info["limits"]["max_bytes"] == web.MAX_BYTES
    assert set(info["methods"]) == set(local_app.info()["methods"]) == set(lineart.METHODS)
    assert [m for m, s in info["methods"].items() if s["available"]] == list(lineart.PURE_METHODS)
    assert all(info["methods"][m]["reason"] == "torch_missing" for m in lineart.MODEL_METHODS)
    assert set(info["limits"]) == set(local_app.info()["limits"])  # the page reads the same keys
    assert info["desmos_limit"] == local_app.info()["desmos_limit"]
    monkeypatch.setattr(web, "MAX_BYTES", 100)
    assert json.loads(web.open_image(_drawing(), "d.png")["answer"])["error"]["code"] == "too_large"
    monkeypatch.setattr(web, "MAX_BYTES", 32 << 20)
    monkeypatch.setattr(web, "LIMITS", web.jobs.Limits(**{**web.LIMITS.__dict__, "quality_max_pixels": 1000}))
    answer = json.loads(web.trace(_drawing(), "d.png", json.dumps(PAGE), key="q")["answer"])
    assert answer["summary"]["warnings"] == ["quality_skipped"] and answer["params"]["quality"] is False


def test_the_engine_needs_no_server_code():
    code = ("import sys, line2func.web; "
            "print(sorted(m for m in ('line2func.app', 'line2func.serve', 'http.server', 'socketserver') "
            "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr


def test_export_svg_restyles_a_result_without_tracing_again():
    """The online page's downloads: the same SVG the local server's ?color=&width= sends."""
    result = web.trace(_drawing(), "d.png", json.dumps({"curves": 60}), key="svg")
    assert "error" not in json.loads(result["answer"]), result["answer"]
    curves_json = result["files"]["curves.json"]

    plain = web.export_svg(curves_json)
    assert json.loads(plain["answer"]) == {"ok": True}
    assert plain["svg"] == result["files"]["out.svg"]  # the default style is what the trace already wrote

    even = web.export_svg(curves_json, "bw", 0, "uniform")["svg"].decode()
    assert len(set(re.findall(r'stroke-width="([^"]+)"', even))) == 1
    assert set(re.findall(r'stroke="([^"]+)"', even)) == {"#000"}

    # the same curves, so only the style differs; a different seed gives different colors
    for seed, other in ((7, 8), (0, 1)):
        a = web.export_svg(curves_json, "random", seed, "measured")["svg"]
        b = web.export_svg(curves_json, "random", other, "measured")["svg"]
        assert a != b and web.export_svg(curves_json, "random", seed, "measured")["svg"] == a

    for args, field in ((("nope",), "color"), (("bw", 0, "thick"), "width"), (("random", "x"), "seed")):
        answer = json.loads(web.export_svg(curves_json, *args)["answer"])
        assert answer["error"]["field"] == field, args
        assert web.export_svg(curves_json, *args)["svg"] is None

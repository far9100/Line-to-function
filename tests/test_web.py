"""The in-browser engine (line2func.web): the same answers and files as the local server, within its limits."""

import io
import json
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

from line2func import app as app_mod
from line2func import geometry as g
from line2func import web
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _drawing() -> bytes:
    cs = CurveSet(160, 120, [Curve(g.line([10, 20], [150, 100])), Curve(g.line([10, 100], [150, 20]), stroke=1)])
    return _png(render_lineart(cs, 160, 120, line_width=2.5))


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
                                (json.dumps({"method": "canny"}), "bad_params", "method"),
                                (json.dumps({"form": "implicit"}), "bad_params", "form"),
                                (json.dumps({"scale": 2}), "bad_params", "scale")):
        result = web.trace(data, "d.png", params, key="d")
        assert (error(result)["code"], error(result)["field"]) == (code, field), params
        assert result["files"] == {} and result["zip"] is None


def test_the_limits(monkeypatch, local_app):
    assert web.MAX_BYTES <= app_mod.MAX_UPLOAD
    for name in ("max_pixels", "store_side", "auto_side", "max_work_pixels", "quality_max_pixels"):
        assert getattr(web.LIMITS, name) <= getattr(app_mod._limits(), name), name
    info = web.info()
    assert info["mode"] == "web" and info["limits"]["max_bytes"] == web.MAX_BYTES
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

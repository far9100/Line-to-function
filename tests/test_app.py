"""The app server (python -m line2func): upload, line-art preview, tracing, downloads, safety, lifetime."""

import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from urllib.parse import quote

import numpy as np
import pytest
from PIL import Image

from line2func import app as app_mod
from line2func import geometry as g
from line2func import lineart, pipeline
from line2func.curves import Curve, CurveSet
from line2func.export import to_desmos
from line2func.render import render_lineart
from line2func.serve import LocalServer


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _drawing() -> np.ndarray:
    cs = CurveSet(160, 120, [Curve(g.line([10, 20], [150, 100])), Curve(g.line([10, 100], [150, 20]), stroke=1)])
    return render_lineart(cs, 160, 120, line_width=2.5)


def _photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :160]
    photo = np.where((xx - 80) ** 2 + (yy - 60) ** 2 < 40**2, 60, 200).astype(np.float64)
    return np.clip(photo + rng.normal(0, 3, photo.shape), 0, 255).astype(np.uint8)


class Client:
    def __init__(self, app):
        self.app = app
        self.base = f"http://127.0.0.1:{app.port}"

    def call(self, method, path, data=None, headers=None, token=True):
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=dict(headers or {}))
        if method == "POST" and token:
            req.add_header("X-Line2func-Token", self.app.token)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def get_json(self, path):
        status, _, body = self.call("GET", path)
        return status, json.loads(body)

    def post_json(self, path, obj, **kw):
        status, _, body = self.call("POST", path, json.dumps(obj).encode(), {"Content-Type": "application/json"}, **kw)
        return status, json.loads(body)

    def upload(self, data: bytes, name="drawing.png"):
        status, _, body = self.call("POST", "/api/images", data, {"X-Filename": quote(name)})
        return status, json.loads(body)

    def run(self, **params):
        status, snap = self.post_json("/api/jobs", params)
        assert status == 202, snap
        return self.wait(snap["job_id"])

    def wait(self, job_id, timeout=60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _, snap = self.get_json(f"/api/jobs/{job_id}")
            if snap["state"] in ("done", "error", "cancelled"):
                return snap
            time.sleep(0.02)
        raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LINE2FUNC_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _start(**kw):
    app = app_mod.make_app(port=0, **kw)
    thread = threading.Thread(target=app.server.serve_forever, daemon=True)
    thread.start()
    return app, thread


@pytest.fixture
def client(home):
    app, thread = _start()
    yield Client(app)
    app.stop("test")
    thread.join(5)
    app.close()


def test_info_and_page(client):
    status, info = client.get_json("/api/info")
    assert status == 200 and info["app"] == "line2func" and info["mode"] == "app"
    assert info["token"] == client.app.token
    assert set(info["methods"]) == set(lineart.METHODS) and info["methods"]["canny"]["available"]
    status, headers, body = client.call("GET", "/")
    assert status == 200 and b"line2func viewer" in body and headers["Cache-Control"] == "no-store"
    assert "Access-Control-Allow-Origin" not in headers


def test_line_art_is_traced_like_the_pipeline(client):
    status, img = client.upload(_png(_drawing()))
    assert status == 201 and img["suggested"] == "lineart" and (img["width"], img["height"]) == (160, 120)
    status, headers, preview = client.call("GET", f"/api/images/{img['image_id']}/preview")
    assert status == 200 and headers["Content-Type"] == "image/png" and "immutable" in headers["Cache-Control"]
    snap = client.run(image_id=img["image_id"], kind="trace", method="none", scale=1.0)
    assert snap["state"] == "done", snap
    assert snap["summary"]["strokes"] == 2 and snap["summary"]["warnings"] == []
    assert snap["files"] == ["curves.json", "desmos.js", "desmos.txt", "equations.tex", "out.svg"]
    _, _, body = client.call("GET", f"/api/jobs/{snap['job_id']}/data/curves.json")
    doc = json.loads(body)
    expected, _ = pipeline.trace(np.repeat(_drawing()[:, :, None], 3, axis=2))
    assert doc["curves"] == expected.to_dict()["curves"]
    assert doc["meta"]["source"] == "drawing.png" and doc["meta"]["lineart"] == "none"
    _, _, desmos = client.call("GET", f"/api/jobs/{snap['job_id']}/data/desmos.txt")
    assert desmos.decode() == to_desmos(CurveSet.from_dict(doc))


def test_photo_preview_is_reused_for_tracing(client):
    status, img = client.upload(_png(_photo()), name="photo.png")
    assert status == 201 and img["suggested"] == "photo"
    iid = img["image_id"]
    snap = client.run(image_id=iid, kind="lineart", method="canny", scale="auto")
    assert snap["state"] == "done" and snap["files"] == ["lineart.png"] and snap["summary"]["scale"] == 1.0
    _, _, png = client.call("GET", f"/api/jobs/{snap['job_id']}/data/lineart.png")
    assert Image.open(io.BytesIO(png)).size == (160, 120)
    assert client.app.derived.get(("ink", iid, 1.0, "canny")) is not None
    stages = []
    real = client.app.board.set_stage
    client.app.board.set_stage = lambda job, stage: (stages.append(stage), real(job, stage))
    snap = client.run(image_id=iid, kind="trace", method="canny", scale=1.0)
    assert snap["state"] == "done" and "lineart.png" in snap["files"] and snap["summary"]["curves"] > 0
    assert "lineart" not in stages and "vectorize" in stages  # the previewed ink map was traced


def test_downloads_zip_and_unicode_names(client):
    _, img = client.upload(_png(_drawing()), name="測試 圖.png")
    snap = client.run(image_id=img["image_id"], kind="trace", method="none", scale=1.0, named=True)
    jid = snap["job_id"]
    status, headers, _ = client.call("GET", f"/api/jobs/{jid}/data/out.svg?download")
    assert status == 200 and headers["Content-Type"] == "image/svg+xml"
    disposition = headers["Content-Disposition"]
    assert disposition.startswith("attachment;") and "filename*=UTF-8''" + quote("測試 圖.svg", safe="") in disposition
    status, headers, body = client.call("GET", f"/api/jobs/{jid}/zip?download")
    assert status == 200 and headers["Content-Type"] == "application/zip"
    assert quote("測試 圖-line2func.zip", safe="") in headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        assert sorted(zf.namelist()) == ["curves.json", "desmos.js", "desmos.txt", "equations.tex", "out.svg"]
    assert client.call("GET", f"/api/jobs/{jid}/data/overlay.png")[0] == 404


def test_functions_from_a_number_of_curves_with_the_quality_check(client):
    """What the page asks for: traced like demo (a number of curves), written as functions, quality checked."""
    _, img = client.upload(_png(_drawing()))
    snap = client.run(image_id=img["image_id"], kind="trace", method="none", scale="auto", form="function",
                      curves=5000, quality=True)
    assert snap["state"] == "done", snap
    summary = snap["summary"]
    assert summary["form"] == "function" and summary["equations"] >= summary["curves"] > 0
    assert {"quality.json", "quality.png"} <= set(snap["files"])
    _, _, body = client.call("GET", f"/api/jobs/{snap['job_id']}/data/curves.json")
    doc = json.loads(body)
    assert doc["meta"]["form"] == "function" and doc["meta"]["curve_count"]["target"] == 5000
    _, _, desmos = client.call("GET", f"/api/jobs/{snap['job_id']}/data/desmos.txt")
    lines = desmos.decode().splitlines()
    assert len(lines) == summary["equations"] == doc["meta"]["functions"]["count"]
    assert lines == [f for c in doc["curves"] for f in c["functions"]]
    for bad in ({"form": "implicit"}, {"curves": 2.5}, {"curves": 0}, {"denoise": 101}, {"denoise": "50"},
                {"faint_sensitivity": -5}):
        status, err = client.post_json("/api/jobs", {"image_id": img["image_id"], "kind": "trace", **bad})
        assert status == 400 and err["error"]["field"] in ("form", "curves", "denoise", "faint_sensitivity"), bad
    # the noise filters' strength reaches the tracer
    snap = client.run(image_id=img["image_id"], kind="trace", method="none", scale=1.0, denoise=0,
                      faint_sensitivity=80)
    _, _, body = client.call("GET", f"/api/jobs/{snap['job_id']}/data/curves.json")
    meta = json.loads(body)["meta"]
    assert meta["denoise"] == 0 and meta["faint_sensitivity"] == 80


def test_empty_result_with_quality_check_is_strict_json(client):
    _, img = client.upload(_png(np.full((64, 64), 255, np.uint8)))
    snap = client.run(image_id=img["image_id"], kind="trace", method="none", scale=1.0, quality=True)
    assert snap["state"] == "done" and "no_lines" in snap["summary"]["warnings"]
    _, _, body = client.call("GET", f"/api/jobs/{snap['job_id']}/data/quality.json")

    def reject(name):
        raise ValueError(name)

    json.loads(body, parse_constant=reject)  # no NaN / Infinity


def test_errors_are_codes(client, monkeypatch):
    def code(result):
        return result[0], result[1]["error"]["code"]

    assert code(client.upload(b"not an image")) == (400, "not_an_image")
    status, _, body = client.call("POST", "/api/images", _png(_drawing()), token=False)
    assert (status, json.loads(body)["error"]["code"]) == (403, "bad_token")
    status, _, body = client.call("GET", "/api/info", headers={"Host": f"attacker.example:{client.app.port}"})
    assert (status, json.loads(body)["error"]["code"]) == (403, "forbidden")
    monkeypatch.setattr(app_mod, "MAX_PIXELS", 1000)
    assert code(client.upload(_png(_drawing()))) == (413, "too_many_pixels")
    monkeypatch.setattr(app_mod, "MAX_PIXELS", 50_000_000)

    _, img = client.upload(_png(_drawing()))
    iid = img["image_id"]
    assert code(client.post_json("/api/jobs", {"image_id": "0" * 16, "kind": "trace"})) == (404, "unknown_image")
    for bad in ({"tolerance": "1"}, {"refine": 1}, {"scale": 2}, {"upscale": True}, {"threshold": 1.5},
                {"kind": "draw"}, {"kind": "lineart", "method": "none"}):
        status, err = client.post_json("/api/jobs", {"image_id": iid, "kind": "trace", **bad})
        assert status == 400 and err["error"]["code"] in ("bad_params", "too_many_pixels"), (bad, err)
    status, err = client.post_json("/api/jobs", {"image_id": iid, "kind": "trace", "tolerance": "1"})
    assert err["error"]["field"] == "tolerance"
    assert client.call("GET", "/api/nothing")[0] == 404
    assert client.call("GET", f"/api/jobs/{'0' * 16}")[0] == 404
    assert client.call("POST", "/api/jobs", b"{broken", {"Content-Type": "application/json"})[0] == 400

    monkeypatch.setattr(app_mod, "MAX_UPLOAD", 100)
    assert code(client.upload(_png(_drawing()))) == (413, "too_large")


def test_model_methods_report_why_they_are_unavailable(client, monkeypatch, home):
    # an empty LINE2FUNC_HOME has no weights
    status = client.app.method_status("informative")
    assert status["available"] is False and status["reason"] in ("weights_missing", "torch_missing")
    monkeypatch.setattr(app_mod.importlib.util, "find_spec", lambda name: None)
    assert client.app.method_status("informative-coarse") == {"available": False, "reason": "torch_missing"}
    _, img = client.upload(_png(_photo()))
    status, err = client.post_json("/api/jobs", {"image_id": img["image_id"], "kind": "lineart", "method": "informative"})
    assert status == 409 and err["error"]["code"] == "torch_missing"


def test_newer_jobs_replace_queued_ones_and_cancel(client, monkeypatch):
    gate = threading.Event()

    def slow(self, job):
        gate.wait(10)
        if job.cancel.is_set():
            raise pipeline.Cancelled

    monkeypatch.setattr(app_mod.App, "run_job", slow)
    _, img = client.upload(_png(_drawing()))
    ids = []
    for _ in range(3):
        status, snap = client.post_json("/api/jobs", {"image_id": img["image_id"], "kind": "trace"})
        assert status == 202
        ids.append(snap["job_id"])
        time.sleep(0.05)  # let the worker pick up the first one
    status, snap = client.post_json(f"/api/jobs/{ids[0]}/cancel", {})
    assert status == 200
    gate.set()
    states = [client.wait(i)["state"] for i in ids]
    assert states == ["cancelled", "cancelled", "done"]


def _open_stream(app) -> socket.socket:
    s = socket.create_connection(("127.0.0.1", app.port), timeout=5)
    s.sendall(f"GET /api/events HTTP/1.1\r\nHost: 127.0.0.1:{app.port}\r\n\r\n".encode())
    _read_until(s, b"event: hello")
    return s


def _read_until(s: socket.socket, marker: bytes, timeout: float = 5.0) -> bytes:
    buf, deadline = b"", time.monotonic() + timeout
    while marker not in buf:
        assert time.monotonic() < deadline, buf
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def test_closing_the_last_window_stops_the_server(home):
    app, thread = _start(auto_exit=True, grace=0.3)
    try:
        time.sleep(0.5)
        assert thread.is_alive()  # no window has connected yet: never exit
        first = _open_stream(app)
        first.close()
        second = _open_stream(app)  # a reload reconnects within the grace period
        time.sleep(0.6)
        assert thread.is_alive() and app.lifecycle.streams == 1
        second.close()
        thread.join(3)
        assert not thread.is_alive() and app.stop_reason == "window_closed"
    finally:
        app.stop("test")
        app.close()


def test_quit_says_bye_to_open_pages(client):
    stream = _open_stream(client.app)
    status, body = client.post_json("/api/shutdown", {})
    assert status == 200 and body == {"ok": True}
    assert b"event: bye" in _read_until(stream, b"event: bye")
    stream.close()
    assert client.app.stopped.is_set() and client.app.stop_reason == "quit"


def test_events_carry_job_updates(client):
    stream = _open_stream(client.app)
    _, img = client.upload(_png(_drawing()))
    snap = client.run(image_id=img["image_id"], kind="trace")
    data = _read_until(stream, b'"state": "done"')
    assert f'"job_id": "{snap["job_id"]}"'.encode() in data and b'"stage": "vectorize"' in data
    stream.close()


def test_a_busy_port_falls_back_to_a_free_one(home):
    busy = LocalServer(("127.0.0.1", 0), app_mod.Handler)
    try:
        app = app_mod.make_app(port=busy.server_address[1])
        assert app.port != busy.server_address[1]
        app.stop("test")
        app.server.server_close()
    finally:
        busy.server_close()


def test_settings_are_saved(client, home):
    options = {"form": "function", "denoise": 30, "denoise_on": False, "faint_sensitivity": 70}
    status, body = client.post_json("/api/settings", {"lang": "zh-TW", "options": {**options, "junk": 1}})
    assert status == 200 and body["settings"] == {"lang": "zh-TW", "options": options}
    assert json.loads((home / "app-settings.json").read_text(encoding="utf-8"))["lang"] == "zh-TW"
    assert client.post_json("/api/settings", {"lang": "fr"})[0] == 400
    again = app_mod.make_app(port=0)
    try:
        assert again.info()["settings"]["lang"] == "zh-TW"
    finally:
        again.stop("test")
        again.server.server_close()


@pytest.mark.parametrize("argv, pref, auto_exit, message", [
    ([], "default", False, "Opened in the browser"),  # by default a tab, which keeps the server running
    (["--browser", "chrome"], "chrome", True, "Opened an app window"),  # closing an app window ends it
])
def test_main_opens_the_page_and_returns(home, monkeypatch, capsys, argv, pref, auto_exit, message):
    started = []

    def fake_launch(url, pref):
        started.append((url, pref))
        return pref

    real_make_app = app_mod.make_app

    def make_app(port=0, **kw):
        app = real_make_app(port, **kw)
        threading.Timer(0.3, app.stop, args=("test",)).start()
        started.append(app)
        return app

    monkeypatch.setattr(app_mod.browser, "launch", fake_launch)
    monkeypatch.setattr(app_mod, "make_app", make_app)
    assert app_mod.main(argv) == 0
    app = started[0]
    assert started[1] == (app.url, pref) and app.lifecycle.auto_exit is auto_exit
    out = capsys.readouterr().out
    assert app.url in out and message in out


def test_python_m_line2func_help():
    out = subprocess.run([sys.executable, "-m", "line2func", "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "--browser" in out.stdout and "--keep-running" in out.stdout


"""The online page's engine (viewer/engine.js), driven with a fake worker under Node.js (tests/engine_check.mjs)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from line2func import serve

NODE = shutil.which("node")


@pytest.fixture(scope="module")
def run():
    if NODE is None:
        pytest.skip("Node.js is not installed")
    script = Path(__file__).with_name("engine_check.mjs")
    out = subprocess.run([NODE, str(script), (serve.VIEWER_DIR / "engine.js").as_uri()], capture_output=True,
                         text=True, encoding="utf-8", timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_a_trace_gives_the_servers_snapshots_and_blob_urls(run):
    r = run["normal"]
    assert r["worker"].endswith("/worker.js?v=b1") and r["init"]["package"].endswith("/py/line2func.zip")
    assert r["init"]["pyodide"] == "https://cdn.example/pyodide/" and r["init"]["packages"] == ["numpy"]
    assert r["statuses"][:2] == ["loading", "loading"] and "ready" in r["statuses"]
    assert r["image"] == {"image_id": r["image"]["image_id"], "name": "a.png", "width": 10, "height": 8, "auto_scale": 1}
    assert r["preview"] is True
    assert r["first"] == {"state": "running", "stage": None}
    assert r["stages"] == ["resize", "vectorize", None]
    assert r["done"][0] == {"files": ["curves.json", "desmos.txt"], "summary": {"curves": 2},
                            "params": {"method": "none", "scale": 1}}
    assert r["urls"] == [True, True, True, False] and r["curves"] == '{"curves": []}'
    # the first trace left a large worker: it was replaced, so the second one waited for the engine
    assert r["workers"] == 2 and r["terminated"] is True
    assert r["second"] == {"state": "running", "stage": "load_engine"} and len(r["done"]) == 2


def test_cancel_stops_the_worker_and_starts_a_new_one(run):
    r = run["cancel"]
    assert r["states"][-1] == "cancelled" and r["workers"] == 2 and r["terminated"] is True
    assert r["status"] == "ready"  # the new worker


def test_a_crash_is_an_error_and_a_new_worker(run):
    r = run["crash"]
    assert r["last"]["state"] == "error" and r["last"]["error"] == {"code": "engine_crashed", "detail": "boom"}
    assert r["workers"] == 2 and r["unknown"] == "unknown_image"


def test_old_engine_files_say_the_page_is_out_of_date(run):
    r = run["protocol"]
    assert r["failed"] == "bad_token" and r["status"]["state"] == "failed" and r["status"]["code"] == "bad_token"

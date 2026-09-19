"""The online engine in real Pyodide (WebAssembly), under Node.js as the browser runs it.

Runs only with LINE2FUNC_PYODIDE=1, after ``npm ci --prefix tests/pyodide``; the first run
downloads Pyodide's numpy, SciPy and Pillow (about 20 MB) from the CDN.
"""

import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from line2func import geometry as g
from line2func import web, website
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart

HERE = Path(__file__).with_name("pyodide")
NODE = shutil.which("node")
PAGE = {"kind": "trace", "method": "none", "scale": "auto", "form": "function", "curves": 5000, "quality": True,
        "denoise": 50, "faint_sensitivity": 50}  # what the page asks for

pytestmark = pytest.mark.skipif(
    os.environ.get("LINE2FUNC_PYODIDE") != "1" or NODE is None or not (HERE / "node_modules" / "pyodide").is_dir(),
    reason="set LINE2FUNC_PYODIDE=1 after npm ci --prefix tests/pyodide")


def _drawing() -> bytes:
    """A small drawing with crossings, a curve and a circle, as lossless WebP (Pyodide's Pillow must read it)."""
    k = 0.5522847498
    circle = [np.array([[120 + 30 * np.cos(a), 60 + 30 * np.sin(a)],
                        [120 + 30 * np.cos(a) - k * 30 * np.sin(a), 60 + 30 * np.sin(a) + k * 30 * np.cos(a)],
                        [120 + 30 * np.cos(b) + k * 30 * np.sin(b), 60 + 30 * np.sin(b) - k * 30 * np.cos(b)],
                        [120 + 30 * np.cos(b), 60 + 30 * np.sin(b)]])
              for a, b in [(i * np.pi / 2, (i + 1) * np.pi / 2) for i in range(4)]]
    ctrls = [g.line([10, 20], [90, 110]), g.line([10, 110], [90, 20]),
             np.array([[20, 60], [50, 0], [70, 130], [100, 60]], float), *circle]
    cs = CurveSet(170, 130, [Curve(c, stroke=i) for i, c in enumerate(ctrls)])
    buf = io.BytesIO()
    Image.fromarray(render_lineart(cs, 170, 130, line_width=2.0)).save(buf, "WEBP", lossless=True)
    return buf.getvalue()


def test_the_engine_traces_in_webassembly(tmp_path):
    info = website.build(tmp_path / "site")
    image = tmp_path / "drawing.webp"
    image.write_bytes(_drawing())
    out = subprocess.run([NODE, str(HERE / "run.mjs"), str(tmp_path / "site" / info["engine"]["package"]), str(image)],
                         capture_output=True, text=True, encoding="utf-8", timeout=900, cwd=HERE)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout)
    assert r["pyodide"] == website.PYODIDE_VERSION and r["webp"] is True and r["standalone"] is True
    assert r["stages"][0] == "resize" and r["stages"][-2:] == ["export", "quality"]
    assert "error" not in r["answer"], r["answer"]
    summary = r["answer"]["summary"]
    assert sorted(r["files"]) == ["curves.json", "desmos.txt", "equations.tex", "out.svg", "quality.json", "quality.png"]
    # the same drawing in CPython: WebAssembly's floating point may move a curve or two, not the result
    native = json.loads(web.trace(image.read_bytes(), "drawing.webp", json.dumps(PAGE), key="n")["answer"])["summary"]
    assert summary["curves"] == pytest.approx(native["curves"], abs=max(2, 0.05 * native["curves"]))
    assert summary["strokes"] == pytest.approx(native["strokes"], abs=2)
    assert summary["quality"]["kept"] == pytest.approx(native["quality"]["kept"], abs=0.01)
    assert summary["quality"]["stray"] is False

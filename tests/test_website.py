"""The online page's build (python -m line2func.website): the page, its api/info and line2func as a ZIP."""

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from line2func import serve, web, website

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    return out, website.build(out)


def test_the_page_and_its_files(site):
    out, info = site
    for name in ["index.html", *serve.ASSETS]:
        assert (out / name).read_bytes() == (serve.VIEWER_DIR / name).read_bytes(), name
    assert json.loads((out / "api" / "info").read_text(encoding="utf-8")) == info
    assert info["mode"] == "web" and info["version"] == web.VERSION and info["limits"] == web.info()["limits"]
    engine = info["engine"]
    assert engine["pyodide"] == website.PYODIDE_URL == f"https://cdn.jsdelivr.net/pyodide/v{website.PYODIDE_VERSION}/full/"
    assert engine["packages"] == ["numpy", "scipy", "pillow"] and engine["build"] == info["build"]
    assert re.fullmatch(r"py/line2func-[\d.]+-[0-9a-f]{12}\.zip", engine["package"]) and (out / engine["package"]).is_file()


def test_api_info_has_everything_the_page_reads(site):
    _, info = site
    source = (serve.VIEWER_DIR / "app.js").read_text(encoding="utf-8")
    for key in set(re.findall(r"S\.info\??\.(\w+)", source)) - {"token", "weights_command", "settings"}:  # local only
        assert key in info, key
    for key in re.findall(r"S\.info\??\.limits\??\.(\w+)", source):
        assert key in info["limits"], key


def test_the_package_zip(site):
    out, info = site
    data = (out / info["engine"]["package"]).read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
    assert names == sorted(names) and all(n.startswith("line2func/") for n in names)
    assert {"line2func/__init__.py", "line2func/web.py", "line2func/jobs.py", "line2func/data/decisions_v1.npz"} <= set(names)
    assert not [n for n in names if "__pycache__" in n or "/viewer/" in n or not n.endswith((".py", ".npz"))]
    assert website.package_zip() == data  # the same files give the same bytes, and the same name


def test_a_build_replaces_only_an_earlier_build(tmp_path, site):
    _, info = site
    out = tmp_path / "again"
    assert website.build(out)["build"] == info["build"]
    assert website.build(out)["build"] == info["build"]  # an earlier build is replaced
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        website.build(other)
    assert (other / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_the_zip_is_the_whole_engine(site, tmp_path):
    """Unpacked on its own, as the browser does, line2func traces with nothing but numpy, SciPy and Pillow."""
    out, info = site
    with zipfile.ZipFile(out / info["engine"]["package"]) as zf:
        zf.extractall(tmp_path / "site-packages")
    code = """
import io, json, sys
import numpy as np
from PIL import Image
import line2func, line2func.web as web
assert line2func.__file__.startswith(sys.argv[1]), line2func.__file__
yy, xx = np.mgrid[:120, :160]
img = np.full((120, 160), 255, np.uint8)
img[np.abs(yy - 0.6 * xx) < 1.5] = 0
buf = io.BytesIO()
Image.fromarray(img).save(buf, "PNG")
answer = json.loads(web.trace(buf.getvalue(), "d.png", json.dumps({"curves": 50, "quality": True}), key="k")["answer"])
print(answer["summary"]["curves"], answer["summary"]["quality"] is not None)
"""
    run = subprocess.run([sys.executable, "-c", code, str(tmp_path / "site-packages")], capture_output=True, text=True,
                         timeout=300, cwd=tmp_path,
                         env={"PYTHONPATH": str(tmp_path / "site-packages"), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
    assert run.returncode == 0, run.stderr
    curves, checked = run.stdout.split()
    assert int(curves) > 0 and checked == "True"


def test_the_pyodide_version_is_the_tested_one():
    package = json.loads((ROOT / "tests" / "pyodide" / "package.json").read_text(encoding="utf-8"))
    assert package["dependencies"]["pyodide"] == website.PYODIDE_VERSION
    constraints = (ROOT / "tests" / "pyodide" / "constraints.txt").read_text(encoding="utf-8")
    assert f"Pyodide {website.PYODIDE_VERSION}" in constraints


def test_the_command(tmp_path, capsys):
    assert website.main(["--out", str(tmp_path / "s"), "--pyodide-url", "https://example.org/pyodide"]) == 0
    info = json.loads((tmp_path / "s" / "api" / "info").read_text(encoding="utf-8"))
    assert info["engine"]["pyodide"] == "https://example.org/pyodide/"
    assert "online page" in capsys.readouterr().out
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("x", encoding="utf-8")
    assert website.main(["--out", str(tmp_path / "busy")]) == 2

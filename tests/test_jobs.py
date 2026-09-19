"""The web page's jobs (line2func.jobs), shared by the local server and the in-browser version."""

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from line2func import geometry as g
from line2func import jobs, pipeline
from line2func.curves import Curve, CurveSet
from line2func.export import output_texts
from line2func.render import render_lineart

LIMITS = jobs.Limits(max_pixels=50_000_000, store_side=4096, auto_side=2048, max_work_pixels=12_000_000,
                     quality_max_pixels=4_000_000)


def _drawing(w=160, h=120) -> np.ndarray:
    cs = CurveSet(w, h, [Curve(g.line([10, 20], [w - 10, h - 20])), Curve(g.line([10, h - 20], [w - 10, 20]), stroke=1)])
    return render_lineart(cs, w, h, line_width=2.5)


def _encode(arr: np.ndarray, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


def _photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :160]
    photo = np.where((xx - 80) ** 2 + (yy - 60) ** 2 < 40**2, 60, 200).astype(np.float64)
    return np.clip(photo + rng.normal(0, 3, photo.shape), 0, 255).astype(np.uint8)


def _code(call) -> tuple[int, str, str | None]:
    with pytest.raises(jobs.ApiError) as err:
        call()
    return err.value.status, err.value.code, err.value.field


def test_read_image_keeps_the_upright_image_and_a_preview():
    img = jobs.read_image(_encode(_drawing()), "drawing.png", LIMITS, "abc")
    assert (img.id, img.name, img.size, img.suggested) == ("abc", "drawing.png", (160, 120), "lineart")
    assert img.preview_type == "image/png" and Image.open(io.BytesIO(img.preview)).size == (160, 120)
    info = img.info()
    assert info == {"image_id": "abc", "name": "drawing.png", "width": 160, "height": 120, "source_width": 160,
                    "source_height": 120, "suggested": "lineart", "auto_scale": 1.0, "max_scale": 1.0}

    photo = jobs.read_image(_encode(_photo()), "photo.png", LIMITS)
    assert photo.suggested == "photo" and photo.preview_type == "image/jpeg"

    # stored at most store_side, the source size is reported upright (EXIF orientation 6: turned 90 degrees)
    small = jobs.Limits(**{**LIMITS.__dict__, "store_side": 80})
    exif = Image.Exif()
    exif[0x0112] = 6
    turned = jobs.read_image(_encode(_drawing(), "JPEG", exif=exif), "turned.jpg", small)
    assert turned.size == (60, 80) and turned.info()["source_width"] == 120 and turned.info()["source_height"] == 160


def test_read_image_errors():
    assert _code(lambda: jobs.read_image(b"not an image", "x.png", LIMITS)) == (400, "not_an_image", None)
    few = jobs.Limits(**{**LIMITS.__dict__, "max_pixels": 1000})
    assert _code(lambda: jobs.read_image(_encode(_drawing()), "x.png", few)) == (413, "too_many_pixels", None)


def test_scales_follow_the_limits():
    img = jobs.read_image(_encode(_drawing(400, 300)), "big.png",
                          jobs.Limits(**{**LIMITS.__dict__, "auto_side": 200, "max_work_pixels": 30_000}))
    assert img.auto_scale() == 0.5 and img.max_scale() == pytest.approx(0.5)
    assert jobs.resolve_scale(img, "auto") == jobs.resolve_scale(img, None) == 0.5
    assert jobs.resolve_scale(img, 0.25) == 0.25
    assert _code(lambda: jobs.resolve_scale(img, 0.9)) == (400, "too_many_pixels", "scale")
    assert _code(lambda: jobs.resolve_scale(img, 0.01)) == (400, "bad_params", "scale")
    assert _code(lambda: jobs.resolve_scale(img, "big")) == (400, "bad_params", "scale")


def test_trace_options_defaults_and_checks():
    options = jobs.trace_options({}, (160, 120), 1.0, 4_000_000)
    assert options == {"tolerance": 1.0, "curves": None, "threshold": None, "refine": True, "form": "parametric",
                       "shape_tolerance": 0.5, "upscale": "auto", "faint": True, "denoise": pipeline.DENOISE,
                       "faint_sensitivity": pipeline.FAINT_SENSITIVITY, "quality": False}
    assert list(options) == ["tolerance", "curves", "threshold", "refine", "form", "shape_tolerance", "upscale",
                             "faint", "denoise", "faint_sensitivity", "quality"]
    assert jobs.trace_options({"named": True}, (160, 120), 1.0, 4_000_000)["form"] == "named"
    assert jobs.trace_options({"curves": 5000, "form": "function"}, (160, 120), 1.0, 4_000_000)["curves"] == 5000
    for bad, field in (({"curves": 2.5}, "curves"), ({"form": "implicit"}, "form"), ({"denoise": 101}, "denoise"),
                       ({"tolerance": "1"}, "tolerance"), ({"refine": 1}, "refine"), ({"upscale": True}, "upscale"),
                       ({"threshold": 1.5}, "threshold"), ({"named": "yes"}, "named"),
                       ({"faint_sensitivity": -1}, "faint_sensitivity")):
        assert _code(lambda bad=bad: jobs.trace_options(bad, (160, 120), 1.0, 4_000_000)) == (400, "bad_params", field)
    # the quality check is skipped, and says so, when the traced image is too large for it
    checked = jobs.trace_options({"quality": True}, (160, 120), 1.0, 160 * 120)
    assert checked["quality"] is True and "quality_skipped" not in checked
    skipped = jobs.trace_options({"quality": True}, (160, 120), 1.0, 160 * 120 - 1)
    assert skipped["quality"] is False and skipped["quality_skipped"] is True
    assert jobs.trace_options({"quality": True}, (160, 120), 0.5, 160 * 120 // 4)["quality"] is True


def test_scaled():
    rgb = np.repeat(_drawing()[:, :, None], 3, axis=2)
    assert jobs.scaled(rgb, 1.0) is rgb
    assert jobs.scaled(rgb, 0.5).shape == (60, 80, 3)


def test_run_trace_makes_the_files_and_summary():
    rgb = np.repeat(_drawing()[:, :, None], 3, axis=2)
    p = {"method": "none", "scale": 1.0,
         **jobs.trace_options({"form": "function", "curves": 5000, "quality": True}, (160, 120), 1.0, 4_000_000)}
    stages = []
    files, summary = jobs.run_trace(rgb, "drawing.png", p, stages.append, started=0.0)
    assert sorted(files) == ["curves.json", "desmos.txt", "equations.tex", "out.svg", "quality.json", "quality.png"]
    assert stages[0] == "lineart" and "vectorize" in stages and stages[-2:] == ["export", "quality"]
    doc = json.loads(files["curves.json"])
    assert doc["meta"]["source"] == "drawing.png" and doc["meta"]["form"] == "function"
    lines = files["desmos.txt"].decode().splitlines()
    assert len(lines) == summary["equations"] == doc["meta"]["functions"]["count"]
    assert summary["equations"] >= summary["curves"] > 0
    assert summary["strokes"] == 2 and summary["warnings"] == [] and summary["quality"]["kept"] > 0.9
    # the same curves as the pipeline gives
    expected, _ = pipeline.trace(rgb, upscale="auto", curve_count=5000)
    assert [c["ctrl"] for c in doc["curves"]] == [c["ctrl"] for c in expected.to_dict()["curves"]]
    assert files["out.svg"] == output_texts(expected)["out.svg"].encode()


def test_run_trace_warns_about_an_empty_result():
    rgb = np.full((64, 64, 3), 255, np.uint8)
    p = {"method": "none", "scale": 1.0, **jobs.trace_options({"quality": True}, (64, 64), 1.0, 4_000_000)}
    files, summary = jobs.run_trace(rgb, "blank.png", p, lambda stage: None, started=0.0)
    assert summary["warnings"] == ["no_lines"] and summary["curves"] == 0

    def reject(name):
        raise ValueError(name)

    json.loads(files["quality.json"], parse_constant=reject)  # no NaN / Infinity


def test_zip_files():
    files = {"b.txt": b"2", "a.txt": b"1"}
    with zipfile.ZipFile(io.BytesIO(jobs.zip_files(files))) as zf:
        assert zf.namelist() == ["a.txt", "b.txt"] and zf.read("b.txt") == b"2"


def test_names():
    assert jobs.clean_name("C:\\Users\\me\\draw\x00ing.png") == "drawing.png"
    assert jobs.clean_name("") == "image"
    assert jobs.stem("測試 圖.png") == "測試 圖" and jobs.stem(".png") == ".png" and jobs.stem("a.b.c") == "a.b"

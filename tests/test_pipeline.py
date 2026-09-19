import numpy as np
import pytest

from line2func import baseline, geometry as g, lineart, pipeline
from line2func.curves import Curve, CurveSet
from line2func.metrics import f_score
from line2func.render import rasterize, render_lineart


def _rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_scaled_curveset():
    cs = CurveSet(100, 80, [Curve(g.line([10, 20], [30, 40]), width=2.0, color="#101010",
                                  shape={"type": "line", "p0": [10, 20], "p1": [30, 40]})],
                  meta={"line_width": 2.0, "engine": "baseline"})
    half = cs.scaled(0.5, 50, 40)
    np.testing.assert_allclose(half.curves[0].ctrl[[0, 3]], [[5, 10], [15, 20]])
    assert half.curves[0].width == 1.0 and half.curves[0].color == "#101010"
    assert half.curves[0].shape is None and half.meta["line_width"] == 1.0
    assert (half.width, half.height) == (50, 40) and cs.curves[0].width == 2.0  # original untouched


def test_upscaled_tracing_maps_back_to_original_pixels():
    ctrl = np.array([[10, 60], [30, 10], [70, 110], [110, 40]], float)
    gt = CurveSet(128, 128, [Curve(ctrl)])
    img = render_lineart(gt, 128, 128, line_width=1.2)  # thin line: "auto" upscales
    cs, ink = pipeline.trace(_rgb(img), upscale="auto")
    assert cs.meta["upscale"] == 2 and (cs.width, cs.height) == (128, 128) and ink.shape == (128, 128)
    assert f_score(cs, gt, 1.0)["f"] > 0.98
    assert all(c.width < 2.5 for c in cs)  # widths are in original pixels too
    one, _ = pipeline.trace(_rgb(img), upscale=1)
    assert one.meta["upscale"] == 1
    # tolerance stays in original pixels: upscaling does not double the curve count
    assert len(cs) <= 2 * len(one) + 1


def test_auto_upscale_leaves_normal_lines_alone():
    gt = CurveSet(96, 96, [Curve(g.line([10, 48], [86, 48]))])
    img = render_lineart(gt, 96, 96, line_width=3.0)
    assert pipeline.choose_upscale(1.0 - img / 255.0, "auto") == 1
    with pytest.raises(ValueError):
        pipeline.choose_upscale(1.0 - img / 255.0, 7)


def _faint_scene(noise: float = 0.0, shading: float = 0.0, seed: int = 0):
    strong = g.line([10, 20], [150, 20])
    faint = np.array([[10, 70], [60, 50], [100, 90], [150, 70]], float)
    ink = rasterize([strong], 160, 100, line_width=2.0) + 0.14 * rasterize([faint], 160, 100, line_width=2.0)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:100, 0:160]
    ink = ink + shading * (xx / 160.0) + rng.normal(0.0, noise, ink.shape)
    return np.clip(ink, 0, 1).astype(np.float32), strong, faint


def test_faint_strokes_are_added():
    ink, strong, faint = _faint_scene()
    off = baseline.vectorize(ink, baseline.BaselineParams(faint_lines=False))
    on = baseline.vectorize(ink, baseline.BaselineParams(faint_lines=True))
    assert f_score(off, [faint], 2.0)["recall"] < 0.05
    assert f_score(on, [faint], 2.0)["recall"] > 0.9
    assert f_score(on, [strong, faint], 2.0)["precision"] > 0.98


@pytest.mark.parametrize("noise, shading", [(0.04, 0.0), (0.02, 0.2)])
def test_faint_mode_does_not_trace_noise_or_shading(noise, shading):
    rng = np.random.default_rng(1)
    strong = g.line([10, 20], [150, 20])
    yy, xx = np.mgrid[0:100, 0:160]
    ink = rasterize([strong], 160, 100, line_width=2.0) + shading * (xx / 160.0) + rng.normal(0, noise, (100, 160))
    ink = np.clip(ink, 0, 1).astype(np.float32)
    on = baseline.vectorize(ink, baseline.BaselineParams(faint_lines=True))
    assert f_score(on, [strong], 2.0)["precision"] > 0.97  # nothing traced but the real line


def _light_line_scene():
    """A black line and a light gray one (ink 0.3: under Otsu's threshold, over 0.25)."""
    strong = g.line([10, 20], [150, 20])
    light = np.array([[10, 70], [60, 50], [100, 90], [150, 70]], float)
    ink = np.maximum(rasterize([strong], 160, 100, line_width=2.5), 0.3 * rasterize([light], 160, 100, line_width=2.5))
    return _rgb((255 * (1.0 - ink)).astype(np.uint8)), strong, light


def test_line_art_is_traced_at_a_lower_threshold_so_light_lines_stay():
    rgb, strong, light = _light_line_scene()
    ink = lineart.extract(rgb, "none")
    assert pipeline.trace_threshold(ink) == pipeline.AUTO_THRESHOLD_MAX < baseline.otsu_threshold(ink)
    # faint_lines=False: only the ink threshold decides whether the light line is traced
    cs, _ = pipeline.trace(rgb, upscale=1, faint_lines=False)
    assert cs.meta["ink_threshold"] == pipeline.AUTO_THRESHOLD_MAX
    assert f_score(cs, [light], 2.0)["recall"] > 0.9 and f_score(cs, [strong, light], 2.0)["precision"] > 0.97
    given, _ = pipeline.trace(rgb, upscale=1, faint_lines=False, threshold=0.4)  # a given threshold is kept
    assert given.meta["ink_threshold"] == 0.4 and f_score(given, [light], 2.0)["recall"] < 0.1


def test_shaded_paper_keeps_otsus_threshold():
    lines = [g.line([10, y], [150, y]) for y in (15, 35, 55, 75, 90)]
    ink = rasterize(lines, 160, 100, line_width=4.0)
    ink[40:70, 100:160] = np.maximum(ink[40:70, 100:160], 0.3)  # a shadow on the paper, darker than 0.25
    rgb = _rgb((255 * (1.0 - ink)).astype(np.uint8))
    ink = lineart.extract(rgb, "none")
    assert pipeline.trace_threshold(ink) == baseline.otsu_threshold(ink) > 0.3  # Otsu's: the shadow stays paper
    cs, _ = pipeline.trace(rgb, upscale=1)
    assert f_score(cs, lines, 2.0)["recall"] > 0.95 and f_score(cs, lines, 2.0)["precision"] > 0.97


def _cross(width: float) -> tuple[np.ndarray, CurveSet]:
    gt = CurveSet(160, 120, [Curve(g.line([10, 20], [150, 100])), Curve(g.line([10, 100], [150, 20]), stroke=1)])
    return _rgb(render_lineart(gt, 160, 120, line_width=width)), gt


def test_progress_reports_each_stage_in_order():
    seen = []
    pipeline.trace(_cross(2.5)[0], upscale=1, progress=seen.append)
    assert seen == ["lineart", "vectorize", "refine", "measure", "outline", "residual", "fill", "shapes"]
    seen.clear()
    pipeline.trace(_cross(1.2)[0], upscale="auto", refine=False, residual=False, outline=False, fill=False,
                   progress=seen.append)  # thin: traced at 2x
    assert seen == ["lineart", "upscale", "vectorize", "measure", "shapes"]
    assert set(seen) <= set(pipeline.STAGES)


def test_progress_callback_can_cancel():
    def stop_at_refine(stage):
        if stage == "refine":
            raise pipeline.Cancelled

    with pytest.raises(pipeline.Cancelled):
        pipeline.trace(_cross(2.5)[0], progress=stop_at_refine)


def test_a_given_ink_map_is_traced_as_is():
    rgb, _ = _cross(2.5)
    ink = lineart.extract(rgb, "none")
    a, _ = pipeline.trace(rgb, upscale=1)
    b, _ = pipeline.trace(rgb, upscale=1, ink=ink)
    assert a.to_dict() == b.to_dict()
    # only the given ink counts, also when it is enlarged for thin lines (not re-extracted from the image)
    thin, gt = _cross(1.2)
    cs, _ = pipeline.trace(np.full_like(thin, 255), upscale="auto", ink=lineart.extract(thin, "none"))
    assert cs.meta["upscale"] == 2 and f_score(cs, gt, 1.5)["f"] > 0.95
    with pytest.raises(ValueError):
        pipeline.trace(rgb, ink=ink[:10])


def test_denoise_sets_how_many_specks_and_faint_pieces_go():
    gt = CurveSet(160, 120, [Curve(g.line([10, 60], [150, 60]))])
    img = render_lineart(gt, 160, 120, line_width=2.5)
    rng = np.random.default_rng(4)
    for _ in range(25):  # dark specks: short dashes, too small for the tuned speck filter
        y, x = rng.integers(0, 116, 2)
        img[y, x:x + 6] = 60
    counts = {d: len(pipeline.trace(_rgb(img), upscale=1, denoise=d)[0]) for d in (0, 50, 100)}
    assert counts[0] > counts[50] >= counts[100]  # off: the specks are traced too
    default, _ = pipeline.trace(_rgb(img), upscale=1)
    assert default.to_dict() == pipeline.trace(_rgb(img), upscale=1, denoise=pipeline.DENOISE)[0].to_dict()
    assert default.meta["denoise"] == pipeline.DENOISE
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            pipeline.trace(_rgb(img), denoise=bad)


def test_faint_params_are_the_tuned_values_at_the_default_and_loosen_above_it():
    tuned = pipeline.faint_params(pipeline.FAINT_SENSITIVITY)
    defaults = baseline.BaselineParams()
    assert tuned == pipeline.FAINT_TUNED and all(getattr(defaults, k) == v for k, v in tuned.items())
    assert pipeline.faint_params(100) == pipeline.FAINT_LOOSE
    assert pipeline.faint_params(0) == {"faint_lines": False, "very_faint_lines": False}
    levels = [pipeline.faint_params(s)["faint_contrast"] for s in (10, 30, 50, 70, 90)]
    assert levels == sorted(levels, reverse=True)
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            pipeline.faint_params(bad)


def test_faint_sensitivity_sets_how_light_a_traced_line_may_be():
    gt = CurveSet(200, 120, [Curve(g.line([10, 30], [190, 30]))])
    img = render_lineart(gt, 200, 120, line_width=2.5)
    light = render_lineart(CurveSet(200, 120, [Curve(g.line([10, 80], [190, 80]))]), 200, 120, line_width=2.0)
    img = np.minimum(img, 255 - (255 - light) * 0.2).astype(np.uint8)  # a light line, far below the threshold

    def near_light(s):
        cs, _ = pipeline.trace(_rgb(img), upscale=1, faint_sensitivity=s)
        return sum(abs(c.ctrl[:, 1].mean() - 80) < 3 for c in cs.curves), cs

    assert near_light(0)[0] == 0 and near_light(100)[0] > 0
    default = pipeline.trace(_rgb(img), upscale=1)[0]
    assert default.to_dict() == near_light(pipeline.FAINT_SENSITIVITY)[1].to_dict()
    assert default.meta["faint_sensitivity"] == pipeline.FAINT_SENSITIVITY
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            pipeline.trace(_rgb(img), faint_sensitivity=bad)

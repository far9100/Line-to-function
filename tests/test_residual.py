import numpy as np

from line2func import attributes, baseline, geometry as g, pipeline
from line2func.curves import Curve, CurveSet
from line2func.metrics import f_score
from line2func.render import rasterize, render_lineart
from line2func.residual import covered_mask, residual_pass


def _scene():
    a = g.line([10, 30], [150, 30])
    b = np.array([[20, 90], [60, 60], [100, 120], [150, 90]], float)
    ink = rasterize([a, b], 160, 140, line_width=2.0)
    return ink, a, b


def test_covered_mask_respects_measured_width():
    ink = rasterize([g.line([10, 30], [150, 30])], 160, 60, line_width=6.0)
    cs = CurveSet(160, 60, [Curve(g.line([10, 30], [150, 30]))], meta={"line_width": 2.0})
    thin_cover = covered_mask(cs, ink, margin=0.5)
    cs.curves[0].width = 6.0
    wide_cover = covered_mask(cs, ink, margin=0.5)
    assert (ink > 0.5)[thin_cover].size < (ink > 0.5)[wide_cover].size
    assert np.all(wide_cover[ink > 0.5])  # every inked pixel of the wide line is covered


def test_missing_stroke_is_added_back():
    ink, a, b = _scene()
    first = CurveSet(160, 140, [Curve(a, width=2.0)], meta={"line_width": 2.0})  # b was "missed"
    extra = residual_pass(first, ink)
    assert len(extra) >= 1 and all("residual" in c.tags for c in extra)
    assert min(c.stroke for c in extra) > 0  # new stroke ids
    assert f_score(extra, [b], 2.0)["recall"] > 0.9
    assert f_score(extra, [b], 2.0)["precision"] > 0.95


def test_nothing_is_added_when_everything_is_covered():
    ink, a, b = _scene()
    full = baseline.vectorize(ink)
    attributes.measure(full, ink)
    assert len(residual_pass(full, ink)) == 0


def test_noise_and_retraces_are_rejected():
    ink, a, b = _scene()
    rng = np.random.default_rng(0)
    noisy = np.clip(ink + rng.normal(0, 0.04, ink.shape), 0, 1).astype(np.float32)
    full = baseline.vectorize(noisy)
    attributes.measure(full, noisy)
    extra = residual_pass(full, noisy)
    assert len(extra) == 0 or f_score(extra, [a, b], 2.0)["precision"] > 0.9


def test_pipeline_residual_stage():
    ink, a, b = _scene()
    img = render_lineart([a, b], 160, 140, line_width=2.0)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    seen = []
    cs, _ = pipeline.trace(rgb, upscale=1, residual=True, progress=seen.append)
    assert "residual" in seen and cs.meta["residual"] is True
    assert f_score(cs, [a, b], 2.0)["f"] > 0.98


def test_coarse_fit_is_not_traced_twice():
    # fitted with a 2 px tolerance, a curve may run up to 2 px off its line; the ink it
    # leaves beside it is the same line, not a new one
    ink = rasterize([g.line([10, 30], [150, 30])], 160, 60, line_width=4.0)
    cs = CurveSet(160, 60, [Curve(g.line([10, 31.8], [150, 31.8]), width=2.0)], meta={"line_width": 2.0})
    assert len(residual_pass(cs, ink, margin=1.5)) >= 1  # a tight margin traces the far edge again
    assert len(residual_pass(cs, ink, params=baseline.BaselineParams(fit_tolerance=2.0))) == 0

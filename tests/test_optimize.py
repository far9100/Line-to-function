import numpy as np
import pytest

torch = pytest.importorskip("torch")

from line2func import geometry as g, pipeline  # noqa: E402
from line2func.curves import Curve, CurveSet  # noqa: E402
from line2func.optimize import Renderer, optimize  # noqa: E402
from line2func.render import rasterize, render_lineart  # noqa: E402

W, H = 120, 80


def _render(curves, half_width, alpha=1.0, grad=False):
    ctrl = torch.tensor(np.stack(curves), dtype=torch.float32, requires_grad=grad)
    n = len(curves)
    hw = torch.full((n,), float(half_width), requires_grad=grad)
    al = torch.full((n,), float(alpha), requires_grad=grad)
    lengths = np.array([g.arc_length(c) for c in curves])
    return Renderer(lengths, H, W, "cpu").render(ctrl, hw, al), (ctrl, hw, al)


def test_renderer_agrees_with_the_rasterizer():
    curves = [g.line([10, 20], [110, 20]), np.array([[10, 70], [40, 30], [80, 80], [110, 40]], float)]
    img, _ = _render(curves, 1.5)
    ref = rasterize(curves, W, H, line_width=3.0)
    img = img.numpy()
    assert abs(img.sum() - ref.sum()) < 0.05 * ref.sum()
    assert ((img > 0.5) == (ref > 0.5)).mean() > 0.995
    half, _ = _render(curves, 1.5, alpha=0.5)
    assert np.allclose(half.numpy(), 0.5 * img, atol=1e-6)  # alpha scales the ink


def test_renderer_gradients():
    img, (ctrl, hw, al) = _render([g.line([10, 20], [110, 20])], 1.5, grad=True)
    target = torch.from_numpy(rasterize([g.line([10, 22], [110, 22])], W, H, line_width=3.0))
    ((img - target) ** 2).sum().backward()
    for t in (ctrl, hw, al):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert ctrl.grad[0, :, 1].sum() < 0  # the loss falls when the line moves down (+y) onto the ink


def test_curves_move_onto_the_ink_and_stay_joined():
    ink = rasterize([g.line([10, 40], [60, 40]), g.line([60, 40], [110, 40])], W, H, line_width=2.0)
    a = g.line([10, 41.5], [60, 41.2])  # traced 1.2-1.5 px off, as two joined pieces of one stroke
    b = g.line([60, 41.2], [110, 41.5])
    cs = CurveSet(W, H, [Curve(a, stroke=0, width=2.0), Curve(b, stroke=0, width=2.0)], meta={"line_width": 2.0})
    report = optimize(cs, ink, steps=150, device="cpu")
    assert report["loss_end"] < 0.5 * report["loss_start"]
    for c in cs:
        ys = g.evaluate(c.ctrl, np.linspace(0, 1, 20))[:, 1]
        assert np.abs(ys - 40).max() < 0.4
        assert 1.4 < c.width < 2.8
    assert np.allclose(cs.curves[0].ctrl[3], cs.curves[1].ctrl[0])  # the joint stays shared


def test_filled_outlines_do_not_move():
    ink = rasterize([g.line([10, 40], [110, 40])], W, H, line_width=2.0)
    outline = Curve(g.line([20, 60], [100, 60]), stroke=1, tags=("outline",), width=1.0)
    line = Curve(g.line([10, 41.5], [110, 41.5]), stroke=0, width=2.0)
    cs = CurveSet(W, H, [line, outline], meta={"line_width": 2.0})
    before = outline.ctrl.copy()
    optimize(cs, ink, steps=50, device="cpu")
    assert np.array_equal(cs.curves[1].ctrl, before)
    assert not np.allclose(cs.curves[0].ctrl, g.line([10, 41.5], [110, 41.5]))


def test_pipeline_optimize_stage():
    lines = [g.line([10, 20], [110, 20]), np.array([[10, 70], [40, 30], [80, 80], [110, 40]], float)]
    img = render_lineart(lines, W, H, line_width=2.0)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    seen = []
    cs, _ = pipeline.trace(rgb, upscale=1, optimize=True, progress=seen.append)
    assert seen.index("optimize") < seen.index("shapes") and cs.meta["optimized"] is True
    from line2func.metrics import f_score

    assert f_score(cs, lines, 1.0)["f"] > 0.98

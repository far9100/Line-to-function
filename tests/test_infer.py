import json

import numpy as np
import pytest

from line2func import geometry as g

torch = pytest.importorskip("torch")

from line2func import blindtest, demo  # noqa: E402
from line2func.curves import Curve, CurveSet  # noqa: E402
from line2func.model import infer  # noqa: E402
from line2func.model.nets import CKPT_FORMAT, CKPT_VERSION, MultiCurveNet  # noqa: E402
from line2func.render import render_lineart, save_png  # noqa: E402

TINY = {"channels": [8, 16, 32, 32], "blocks": 1, "d_model": 32, "heads": 4, "enc_layers": 1, "dec_layers": 1,
        "queries": 4, "ff": 64}


def test_clip_to_box():
    c = g.line([-10, 5], [30, 5])
    pieces = infer._clip(c, np.array([0.0, 0.0]), np.array([20.0, 20.0]))
    assert len(pieces) == 1
    np.testing.assert_allclose(pieces[0][0], [0, 5], atol=1e-4)
    np.testing.assert_allclose(pieces[0][3], [20, 5], atol=1e-4)
    assert infer._clip(g.line([30, 30], [40, 40]), np.zeros(2), np.full(2, 20.0)) == []
    # a curve leaving and re-entering gives two pieces
    u = np.array([[2, 10], [40, -30], [40, 50], [2, 10.5]], float)
    assert len(infer._clip(u, np.zeros(2), np.full(2, 20.0))) == 2


def test_trim_overlaps_keeps_crossings_and_cuts_retraces():
    long = g.line([0, 10], [60, 10])
    retrace = g.line([40, 10.2], [70, 10.2])  # overlaps the long one on x in [40, 60]
    crossing = g.line([30, 0], [30, 20])
    kept = infer._trim_overlaps([long, retrace, crossing])
    by_src = {}
    for i, piece in kept:
        by_src.setdefault(i, []).append(piece)
    assert len(by_src[0]) == 1 and len(by_src[2]) == 1  # untouched
    rest = by_src[1][0]
    assert min(rest[0, 0], rest[3, 0]) >= 59.0  # only the part beyond the long line survives


def test_stitch_joins_straight_continuations_across_a_crossing():
    # a horizontal line cut into 3 pieces at x = 20 and 40, and a vertical line through (30, 10)
    h = [g.line([0, 10], [20, 10]), g.line([20.3, 10], [40, 10]), g.line([40, 10], [60, 10.2])]
    v = [g.line([30, -10], [30, 9.9]), g.line([30, 10.1], [30, 30])]
    strokes = infer.stitch(h + v)
    groups = sorted(sorted(i for i, _ in seq) for seq, _ in strokes)
    assert groups == [[0, 1, 2], [3, 4]]


def _fake_ckpt(tmp_path):
    torch.manual_seed(0)
    net = MultiCurveNet(64, **TINY)
    path = tmp_path / "m.pt"
    torch.save({"format": CKPT_FORMAT, "version": CKPT_VERSION, "task": "multi_curve", "hparams": net.hparams,
                "model": net.state_dict()}, path)
    return path


def test_vectorize_model_runs_end_to_end(tmp_path):
    run = infer.load_vectorizer(str(_fake_ckpt(tmp_path)), "cpu")
    ink = np.zeros((96, 130), np.float32)
    cs = run(ink)
    assert (cs.width, cs.height) == (130, 96) and cs.meta["engine"] == "model"


def test_demo_with_model_and_blindtest(tmp_path):
    ckpt = _fake_ckpt(tmp_path)
    cs = CurveSet(100, 80, [Curve(g.line([10, 40], [90, 40]))])
    img = tmp_path / "drawings"
    img.mkdir()
    save_png(img / "a.png", render_lineart(cs, 100, 80))
    save_png(img / "b.png", render_lineart(cs, 100, 80))
    out = tmp_path / "out"
    assert demo.main([str(img / "a.png"), "--vectorizer", "model", "--ckpt", str(ckpt), "--out", str(out)]) == 0
    assert json.loads((out / "curves.json").read_text(encoding="utf-8"))["meta"]["vectorizer"] == "model"

    kit = tmp_path / "kit"
    n = blindtest.make_kit(img, kit, infer.load_vectorizer(str(ckpt), "cpu"))
    assert n == 2 and (kit / "index.html").is_file()
    key = json.loads((kit / "key.json").read_text(encoding="utf-8"))
    # vote for whichever side is the model on the first drawing, tie on the second
    model_side = "A" if key["d000"]["A"] == "model" else "B"
    votes = tmp_path / "votes.json"
    votes.write_text(json.dumps({"votes": {"d000": model_side, "d001": "tie"}}), encoding="utf-8")
    r = blindtest.score(kit, [votes])
    assert r["votes"] == 2 and r["model"] == 1 and r["tie"] == 1 and r["pass"]
    page = (kit / "index.html").read_text(encoding="utf-8")
    assert '"model"' not in page and '"baseline"' not in page  # the page never names the engines

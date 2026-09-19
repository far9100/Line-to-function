import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.synth import multi_curve_sample, patch_targets

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from line2func import train as train_mod  # noqa: E402
from line2func.model.data import MultiCurveDataset, stack  # noqa: E402
from line2func.model.losses import hungarian_match, multi_curve_loss  # noqa: E402
from line2func.model.nets import MultiCurveNet, load_checkpoint  # noqa: E402

TINY = {"channels": [8, 16, 32, 32], "blocks": 1, "d_model": 32, "heads": 4, "enc_layers": 1, "dec_layers": 2,
        "queries": 6, "ff": 64}


def test_patch_targets_clip_and_merge():
    # one stroke made of two pieces joined smoothly, crossing a 40x40 patch at (30, 30)
    a = g.line([0, 50], [50, 50])
    b = g.line([50, 50], [100, 50])
    gt = CurveSet(120, 120, [Curve(a, 0), Curve(b, 0), Curve(g.line([0, 0], [10, 10]), 1)])
    ctrls, widths = patch_targets(gt, 30, 30, 40)
    assert len(ctrls) == 1  # the invisible piece joint is merged; stroke 1 is outside
    ends = sorted([ctrls[0][0, 0], ctrls[0][3, 0]])
    assert ends[0] == pytest.approx(0.0, abs=0.51) and ends[1] == pytest.approx(40.0, abs=0.51)
    np.testing.assert_allclose(ctrls[0][:, 1], 20.0, atol=1e-6)


def test_multi_curve_sample_targets_lie_on_ink():
    rng = np.random.default_rng(2)
    for _ in range(10):
        img, ctrls, widths = multi_curve_sample(rng, 64, "clean")
        assert img.shape == (64, 64) and 1 <= len(ctrls) <= 16 and len(widths) == len(ctrls)
        for c in ctrls:
            pts = g.evaluate(c, np.linspace(0.05, 0.95, 20))
            x = np.clip(pts[:, 0].astype(int), 0, 63)
            y = np.clip(pts[:, 1].astype(int), 0, 63)
            assert np.mean(img[y, x] < 200) > 0.9  # clean: the curve runs along dark ink


def test_network_output_shapes():
    net = MultiCurveNet(64, **TINY)
    out = net(torch.rand(2, 1, 64, 64))
    assert out["ctrl"].shape == (2, 6, 4, 2) and out["width"].shape == (2, 6, 2) and out["logit"].shape == (2, 6)
    assert len(out["aux"]) == 1


def test_hungarian_match_finds_the_right_pairs():
    gt = torch.rand(1, 3, 4, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    # queries: [noise, gt1 reversed, gt0]
    pred = torch.stack([torch.rand(4, 2) + 5, gt[0, 1].flip(0), gt[0, 0]])[None]
    out = {"ctrl": pred, "logit": torch.zeros(1, 3)}
    b, q, t, flipped = hungarian_match(out, gt, mask)
    pairs = sorted(zip(q.tolist(), t.tolist(), flipped.tolist()))
    assert pairs == [(1, 1, True), (2, 0, False)]
    width = torch.ones(1, 3, 2)
    out["width"] = torch.ones(1, 3, 2)
    out["logit"] = torch.tensor([[-20.0, 20.0, 20.0]])
    loss, parts = multi_curve_loss(out, gt, width, mask)
    assert parts["ctrl"] < 1e-6 and parts["pts"] < 1e-6 and parts["cls"] < 1e-6


def test_dataset_padding():
    ds = MultiCurveDataset(seed=1, length=4, patch_size=64, max_curves=16)
    ink, ctrl, width, mask = stack(ds, 4)
    assert ink.shape == (4, 1, 64, 64) and ctrl.shape == (4, 16, 4, 2) and mask.shape == (4, 16)
    assert torch.all(mask.sum(1) >= 1)


def test_multi_training_resume_and_checkpoint(tmp_path):
    cfg = train_mod.load_config(None, {
        "task": "multi_curve", "out_dir": str(tmp_path / "m"), "seed": 1,
        "data": {"patch_size": 64, "overfit_samples": 4}, "model": TINY,
        "train": {"steps": 3, "batch_size": 4, "num_workers": 0, "log_every": 1, "eval_every": 3,
                  "warmup_steps": 1, "max_wall_hours": 1e-9},
    })
    r = train_mod.train(cfg, device="cpu")
    assert r["status"] == "wall_cap" and r["step"] == 1
    cfg["train"]["max_wall_hours"] = 36
    r = train_mod.train(cfg, device="cpu", resume="auto")
    assert r["status"] == "done" and r["step"] == 3 and "f1" in r["metrics"]
    model, ckpt = load_checkpoint(tmp_path / "m" / "best.pt")
    assert ckpt["task"] == "multi_curve" and isinstance(model, MultiCurveNet)


def test_check_metrics():
    assert train_mod.check_metrics({"min_f1": 0.9, "max_mean_px": 1.0}, {"f1": 0.95, "mean_px": 0.5}) == []
    assert len(train_mod.check_metrics({"min_f1": 0.9}, {"f1": 0.5})) == 1

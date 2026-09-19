import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

from line2func import train as train_mod  # noqa: E402
from line2func.model.data import SingleCurveDataset, stack  # noqa: E402
from line2func.model.losses import orient, point_error_px, single_curve_loss  # noqa: E402
from line2func.model.nets import SingleCurveNet, load_checkpoint  # noqa: E402

TINY = {"channels": [8, 16], "blocks": 1, "hidden": 32}


def test_network_shapes_and_ranges():
    net = SingleCurveNet(64, **TINY)
    out = net(torch.rand(3, 1, 64, 64))
    assert out["ctrl"].shape == (3, 4, 2) and out["width"].shape == (3, 2)
    assert out["ctrl"].min() >= -0.25 and out["ctrl"].max() <= 1.25
    assert out["width"].min() > 0


def test_loss_ignores_curve_direction():
    ctrl = torch.rand(5, 4, 2)
    width = torch.rand(5, 2) + 1
    out = {"ctrl": ctrl.clone().requires_grad_(), "width": width.clone()}
    loss_same, _ = single_curve_loss(out, ctrl, width, 64)
    loss_rev, _ = single_curve_loss(out, ctrl.flip(1), width.flip(1), 64)
    assert loss_same.item() == pytest.approx(0.0, abs=1e-6)
    assert loss_rev.item() == pytest.approx(0.0, abs=1e-6)
    o, _ = orient(ctrl, ctrl.flip(1), width)
    torch.testing.assert_close(o, ctrl)
    assert point_error_px(ctrl, ctrl.flip(1), 64).max().item() < 1e-5


def test_dataset_is_deterministic_and_resumable():
    ds = SingleCurveDataset(seed=4, length=10, patch_size=32)
    a = ds[7]
    b = SingleCurveDataset(seed=4, length=5, patch_size=32, offset=5)[2]
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    ink, ctrl, width = stack(ds, 4)
    assert ink.shape == (4, 1, 32, 32) and ctrl.shape == (4, 4, 2) and width.shape == (4, 2)
    assert 0.0 <= ink.min() and ink.max() <= 1.0


def _cfg(tmp_path, **train):
    return train_mod.load_config(None, {
        "name": "t", "out_dir": str(tmp_path / "run"), "seed": 1,
        "data": {"patch_size": 32, "overfit_samples": 8},
        "model": TINY,
        "train": {"steps": 6, "batch_size": 8, "num_workers": 0, "log_every": 2, "eval_every": 3,
                  "warmup_steps": 2, **train},
    })


def test_training_checkpoint_and_resume(tmp_path):
    cfg = _cfg(tmp_path, steps=4)
    r = train_mod.train(cfg, device="cpu")
    assert r["status"] == "done" and r["step"] == 4
    run = tmp_path / "run"
    assert (run / "last.pt").is_file() and (run / "best.pt").is_file() and (run / "log.csv").is_file()
    model, ckpt = load_checkpoint(run / "last.pt")
    assert ckpt["step"] == 4 and ckpt["task"] == "single_curve"
    # extend the schedule and resume
    cfg["train"]["steps"] = 6
    r = train_mod.train(cfg, device="cpu", resume="auto")
    assert r["step"] == 6
    assert load_checkpoint(run / "last.pt")[1]["step"] == 6


def test_resume_matches_uninterrupted_run(tmp_path):
    """A run stopped by the wall-clock cap and resumed ends with the same weights."""
    full = train_mod.train(_cfg(tmp_path / "a"), device="cpu")
    cfg = _cfg(tmp_path / "b", max_wall_hours=1e-9)
    assert train_mod.train(cfg, device="cpu")["status"] == "wall_cap"
    cfg["train"]["max_wall_hours"] = 36
    resumed = train_mod.train(cfg, device="cpu", resume="auto")
    wa = torch.load(tmp_path / "a" / "run" / "last.pt", weights_only=False)["model"]
    wb = torch.load(tmp_path / "b" / "run" / "last.pt", weights_only=False)["model"]
    for k in wa:
        torch.testing.assert_close(wa[k], wb[k], rtol=1e-4, atol=1e-5)
    assert resumed["step"] == full["step"] == 6


def test_wall_clock_cap_stops_and_saves(tmp_path):
    cfg = _cfg(tmp_path, steps=1000, max_wall_hours=1e-9)
    r = train_mod.train(cfg, device="cpu")
    assert r["status"] == "wall_cap" and r["step"] == 1
    assert load_checkpoint(tmp_path / "run" / "last.pt")[1]["step"] == 1


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        train_mod.load_config(None, {"task": "bogus"})
    with pytest.raises(ValueError):
        train_mod.load_config(None, {"train": {"max_wall_hours": 0}})
    cfg = train_mod.load_config("configs/single_curve.yaml")
    assert cfg["train"]["max_wall_hours"] == 36 and cfg["out_dir"].replace("\\", "/") == "runs/single"

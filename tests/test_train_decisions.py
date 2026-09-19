"""Training the decision scorer (needs PyTorch): it learns, and the numpy export computes the same thing."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from line2func.decision_features import FEATURES_VERSION, NAMES  # noqa: E402
from line2func.decision_model import LearnedScorer  # noqa: E402
from line2func.train_decisions import export, train_kind  # noqa: E402


def _data(n=4000, f=len(NAMES["gap"]), seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, f)).astype(np.float32)
    y = ((x[:, 0] + 0.5 * x[:, 3]) > 0.2).astype(np.int8)
    group = (np.arange(n, dtype=np.int64) // 20) << 20  # 20 rows per "scene"
    rule = np.zeros((n, 3), np.float32)
    return x, y, group, rule


def test_learns_a_simple_rule():
    x, y, group, rule = _data()
    res = train_kind(x, y, group, rule, "cpu", epochs=30, verbose=False)
    assert res["report"]["val_acc"] > 0.95 and res["report"]["val_auc"] > 0.98


def test_numpy_export_matches_torch(tmp_path):
    x, y, group, rule = _data(n=1000)
    res = train_kind(x, y, group, rule, "cpu", epochs=3, verbose=False)
    path = tmp_path / "decisions.npz"
    export(path, {"gap": res}, NAMES, FEATURES_VERSION, {})
    scorer = LearnedScorer.from_file(path)
    p_np, trusted = scorer.models["gap"](x[:50])
    a = res["arrays"]
    h = (np.clip(x[:50], a["lo"], a["hi"]) - a["mu"]) / a["sd"]
    net = torch.nn.Sequential(torch.nn.Linear(h.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    with torch.no_grad():
        for layer, k in zip([net[0], net[2], net[4]], range(3)):
            layer.weight.copy_(torch.from_numpy(a[f"W{k}"].T))
            layer.bias.copy_(torch.from_numpy(a[f"b{k}"]))
        z = net(torch.from_numpy(h.astype(np.float32))).squeeze(1).double().numpy()
    p_torch = 1.0 / (1.0 + np.exp(-(z / float(a["T"]) + float(a["B"]))))
    assert trusted.all() and np.allclose(p_np, p_torch, atol=1e-5)

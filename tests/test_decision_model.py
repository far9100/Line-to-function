"""The learned decision scorer: numpy inference, the rules where it cannot be trusted, and safe loading."""

import subprocess
import sys

import numpy as np
import pytest

import golden
from line2func import lineart
from line2func.baseline import BaselineParams, vectorize
from line2func.decision_features import FEATURES_VERSION, NAMES
from line2func.decision_model import LearnedScorer


def _weights(path, tau=0.5, delta=0.0, seed=0, rename=None):
    rng = np.random.default_rng(seed)
    arrays = {"format": np.array("line2func.decisions"), "version": np.array(1),
              "features_version": np.array(FEATURES_VERSION), "kinds": np.array(sorted(NAMES)),
              "train_meta": np.array("{}")}
    for kind, names in NAMES.items():
        f = len(names)
        names = list(names)
        if rename and kind == rename:
            names[0] = "something_else"
        arrays.update({
            f"{kind}__names": np.array(names), f"{kind}__lo": np.full(f, -1e3, np.float32),
            f"{kind}__hi": np.full(f, 1e3, np.float32), f"{kind}__mu": np.zeros(f, np.float32),
            f"{kind}__sd": np.full(f, 50.0, np.float32),
            f"{kind}__W0": rng.normal(0, 0.5, (f, 64)).astype(np.float32), f"{kind}__b0": np.zeros(64, np.float32),
            f"{kind}__W1": rng.normal(0, 0.2, (64, 64)).astype(np.float32), f"{kind}__b1": np.zeros(64, np.float32),
            f"{kind}__W2": rng.normal(0, 0.2, (64, 1)).astype(np.float32), f"{kind}__b2": np.zeros(1, np.float32),
            f"{kind}__T": np.float32(1.0), f"{kind}__tau": np.float32(tau), f"{kind}__delta": np.float32(delta),
        })
    np.savez(path, **arrays)
    return path


def _inks():
    cases = golden.hand_cases()
    yield from (cases[k] for k in ("cross_30", "cross_15", "t_junction", "corner", "gap"))
    for i in range(3):
        yield lineart.extract(golden.val_scene("hard", i).image, "none")


def test_a_learned_scorer_traces_and_reports(tmp_path):
    scorer = LearnedScorer.from_file(_weights(tmp_path / "w.npz"))
    for ink in _inks():
        cs = vectorize(ink, scorer=scorer)
        assert len(cs) > 0 and all(np.all(np.isfinite(c.ctrl)) for c in cs)
    assert cs.meta["decisions"] and all(v["decisions"] >= v["fallback"] >= 0 for v in cs.meta["decisions"].values())


def test_unsure_everywhere_is_exactly_the_rules(tmp_path):
    # delta 1: every probability is "near" the threshold, so the rule decides everything
    scorer = LearnedScorer.from_file(_weights(tmp_path / "w.npz", delta=1.0))
    for ink in _inks():
        learned, rules = vectorize(ink, scorer=scorer), vectorize(ink)
        learned.meta.pop("decisions", None)
        assert golden.digest(learned) == golden.digest(rules)


def test_params_name_a_weights_file(tmp_path):
    path = _weights(tmp_path / "w.npz", delta=1.0)
    ink = golden.hand_cases()["cross_30"]
    cs = vectorize(ink, BaselineParams(decisions=str(path)))
    assert "decisions" in cs.meta


def test_wrong_features_are_refused(tmp_path):
    with pytest.raises(ValueError):
        LearnedScorer.from_file(_weights(tmp_path / "w.npz", rename="gap"))


def test_learned_tracing_never_imports_torch(tmp_path):
    path = _weights(tmp_path / "w.npz")
    code = ("import sys, numpy as np; from line2func.baseline import BaselineParams, vectorize; "
            f"vectorize(np.pad(np.ones((4, 60)), 20), BaselineParams(decisions=r'{path}')); "
            "print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_pipeline_uses_the_bundled_scorer_by_default(monkeypatch, tmp_path):
    from line2func import decision_model, pipeline

    ink = golden.hand_cases()["cross_30"]
    rgb = np.repeat(np.rint(255 * (1 - ink)).astype(np.uint8)[:, :, None], 3, axis=2)
    assert "decisions" in pipeline.trace(rgb, upscale=1)[0].meta  # the learned scorer reports its decisions
    assert "decisions" not in pipeline.trace(rgb, upscale=1, decisions="rules")[0].meta
    monkeypatch.setattr(decision_model, "BUNDLED", tmp_path / "missing.npz")
    with pytest.warns(UserWarning):
        cs, _ = pipeline.trace(rgb, upscale=1)  # no bundled weights: the rules decide, with a warning
    assert "decisions" not in cs.meta

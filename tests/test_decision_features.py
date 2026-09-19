"""Decision candidates are recorded with sane features, and recording changes nothing."""

import numpy as np

import golden
from line2func import lineart
from line2func.baseline import BaselineParams, vectorize
from line2func.decision_features import NAMES, Recorder, junction_features
from line2func.decisions import JunctionCand


def _hard_ink(i=0):
    return lineart.extract(golden.val_scene("hard", i).image, "none")


def test_recording_does_not_change_the_tracing():
    for ink in (_hard_ink(0), _hard_ink(1), golden.hand_cases()["cross_15"]):
        rec = Recorder(BaselineParams())
        assert golden.digest(vectorize(ink, recorder=rec)) == golden.digest(vectorize(ink))


def test_every_kind_is_recorded_with_finite_features():
    seen = {k: 0 for k in NAMES}
    for i in range(4):
        rec = Recorder(BaselineParams())
        vectorize(_hard_ink(i), recorder=rec)
        for kind, names in NAMES.items():
            x, rule, chosen = rec.table(kind)
            assert x.shape == (len(chosen), len(names)) and rule.shape == (len(chosen), 3)
            assert np.all(np.isfinite(x))
            # with the rules as the active scorer, what was chosen is exactly what the rule chose
            assert np.array_equal(chosen, rule[:, 2] > 0.5), kind
            seen[kind] += len(chosen)
    assert all(n > 0 for n in seen.values()), seen


def test_pair_features_do_not_depend_on_arm_order():
    rec = Recorder(BaselineParams())
    vectorize(_hard_ink(0), recorder=rec)
    ctx_nodes = [g for g in rec.geom["junction"]]
    cand, _, _ = ctx_nodes[0]
    # the same node with its arms listed in reverse
    order = list(range(len(cand.keys)))[::-1]
    pairs = [(i, j) for i in range(len(order)) for j in range(i + 1, len(order))]
    orig = {frozenset(p): b for p, b in zip(cand.pairs, cand.bends)}
    rev = JunctionCand(cand.node, [cand.keys[o] for o in order], cand.center, cand.spread, cand.reach,
                       [cand.dirs[o] for o in order], pairs,
                       np.array([orig[frozenset((order[i], order[j]))] for i, j in pairs]),
                       [cand.arms[o] for o in order], [cand.far_degree[o] for o in order])
    ctx = rec.ctx if hasattr(rec, "ctx") else None
    if ctx is None:  # rebuild the context the recorder saw
        from scipy import ndimage

        from line2func.decisions import DecisionContext
        ink = _hard_ink(0)
        thr = float(np.clip(lineart.otsu_threshold(ink), 0.2, 0.8))
        mask = ink > thr
        ctx = DecisionContext(ink, mask, ndimage.distance_transform_edt(mask), thr, 2.0, 1.0)
    a, a_end = junction_features(cand, ctx)
    b, b_end = junction_features(rev, ctx)
    row_of = {frozenset(p): r for p, r in zip(cand.pairs, a)}
    for (i, j), r in zip(pairs, b):
        assert np.allclose(r, row_of[frozenset((order[i], order[j]))], atol=1e-4)
    assert np.allclose(b_end, a_end[order], atol=1e-4)

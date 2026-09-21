"""The decision scorer interface: the rules are unchanged, the solvers are right, other scorers take effect."""

import subprocess
import sys

import numpy as np
import pytest

from line2func import geometry as g
from line2func.baseline import BaselineParams, vectorize
from line2func.curves import Curve, CurveSet
from line2func.decisions import (
    RULES,
    ImprovedRules,
    Scorer,
    WideGapRules,
    exact_matching,
    greedy_pairs,
    load_scorer,
    pick_peaks,
)
from line2func.metrics import joint_corners
from line2func.render import render_lineart


def _old_peaks(i, turn, min_angle, n, closed, reach):  # the loop _corners had before the refactor
    found = []
    for j in np.argsort(-turn):
        if turn[j] < min_angle:
            break
        idx = int(i[j])
        if all(min(abs(idx - c), n - abs(idx - c) if closed else n) > reach for c in found):
            found.append(idx)
    return sorted(found)


def _old_greedy(bends, pairs, max_bend):  # the loop _link_junctions had before the refactor
    cand = sorted((b, i, j) for b, (i, j) in zip(bends, pairs) if b <= max_bend)
    used, out = set(), []
    for _, i, j in cand:
        if i in used or j in used:
            continue
        used.update((i, j))
        out.append((i, j))
    return out


def test_pick_peaks_is_the_old_corner_loop():
    rng = np.random.default_rng(0)
    for trial in range(1000):
        n = int(rng.integers(10, 80))
        closed = bool(rng.random() < 0.3)
        reach = int(rng.integers(2, 8))
        turn = np.round(rng.uniform(0, 180, n), 0 if trial % 2 else 3)  # rounding makes ties
        i = np.arange(n) if closed else np.arange(n) + reach
        assert pick_peaks(i, turn, 60.0, n + 2 * reach, closed, reach) == _old_peaks(i, turn, 60.0, n + 2 * reach,
                                                                                       closed, reach)


def test_greedy_pairs_is_the_old_sort():
    rng = np.random.default_rng(1)
    for _ in range(500):
        k = int(rng.integers(3, 7))
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        bends = np.round(rng.uniform(0, 90, len(pairs)), 0)  # many ties
        scores = np.where(bends <= 45.0, -bends, -np.inf)
        assert greedy_pairs(scores, pairs) == _old_greedy(bends, pairs, 45.0)


def _brute(z, pairs, e, k):
    zmap = dict(zip(pairs, z))
    best, best_m = -np.inf, None

    def matchings(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        yield from matchings(rest)  # first stays unpaired
        for idx, other in enumerate(rest):
            for m in matchings(rest[:idx] + rest[idx + 1:]):
                yield [(first, other)] + m

    for m in matchings(list(range(k))):
        paired = {a for pair in m for a in pair}
        total = sum(zmap[p] for p in m) + sum(e[a] for a in range(k) if a not in paired)
        if total > best:
            best, best_m = total, m
    return best


def test_exact_matching_is_optimal():
    rng = np.random.default_rng(2)
    for _ in range(300):
        k = int(rng.integers(2, 8))
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        z = rng.normal(0, 2, len(pairs))
        z[rng.random(len(pairs)) < 0.2] = -np.inf
        e = rng.normal(0, 1, k)
        chosen = exact_matching(z, pairs, e, k)
        zmap = dict(zip(pairs, z))
        paired = {a for p in chosen for a in p}
        assert len(paired) == 2 * len(chosen)  # every arm at most once
        total = sum(zmap[p] for p in chosen) + sum(e[a] for a in range(k) if a not in paired)
        assert total == pytest.approx(_brute(z, pairs, e, k))


def _ink(ctrls, size=128, width=2.0):
    gt = CurveSet(size, size, [Curve(c, stroke=i) for i, c in enumerate(ctrls)])
    return 1.0 - render_lineart(gt, size, size, line_width=width) / 255.0


class _NoPairs(Scorer):
    def junction_scores(self, cands, max_bend):
        return [(np.full(len(c.pairs), -np.inf), None) for c in cands]


class _NoCorners(Scorer):
    def corner_keys(self, cand, min_angle):
        return cand.turn, 999.0


def test_other_scorers_take_effect():
    x = _ink([g.line([10, 34], [118, 94]), g.line([10, 94], [118, 34])])
    assert vectorize(x).num_strokes == 2  # the rule continues both lines through the crossing
    assert vectorize(x, scorer=_NoPairs()).num_strokes == 4  # nothing may continue: four arms
    corner = _ink([g.line([20, 20], [20, 100]), g.line([20, 100], [100, 100])])
    assert [round(t) for _, t in joint_corners(vectorize(corner), 20.0)] == [90]  # the rule splits at the corner
    assert joint_corners(vectorize(corner, scorer=_NoCorners()), 20.0) == []  # without: smooth (G1) joints only


def test_rules_are_the_engines_default_and_unknown_scorers_fail():
    x = _ink([g.line([10, 34], [118, 94]), g.line([10, 94], [118, 34])])
    assert vectorize(x).to_dict() == vectorize(x, scorer=RULES).to_dict()
    assert vectorize(x, BaselineParams(decisions="rules")).to_dict() == vectorize(x).to_dict()
    assert load_scorer(None) is RULES
    with pytest.raises(FileNotFoundError):
        vectorize(x, BaselineParams(decisions="no-such-scorer.npz"))


def test_tracing_never_imports_torch():
    code = ("import sys, numpy as np; from line2func.baseline import vectorize; "
            "vectorize(np.pad(np.ones((4, 60)), 20)); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_no_learning_rule_variants_are_reachable_by_name():
    """``r2`` / ``r2-gaps``: the rules with wider candidates, so the measurement can be redone.

    Both are measurably worse than the plain rules on real drawings (docs/details.md,
    "How strokes are joined"); they are kept so that result stays reproducible.
    """
    assert type(load_scorer("r2")) is ImprovedRules
    assert type(load_scorer("r2-gaps")) is WideGapRules
    # neither learns anything, so neither may need PyTorch or the bundled weights
    assert WideGapRules.needs_features is False
    for cls in (WideGapRules, ImprovedRules):
        assert cls.limits.gap_radius_scale == 1.5  # the widening they exist for
        assert cls.limits.gap_angle_gate is None  # ... at the rule's own angle gate


def test_a_wider_gap_radius_closes_a_break_the_rules_leave_open():
    """The rule's radius is 4 line widths; at 1.5x it reaches a break the rules give up on."""
    def broken(total_gap):
        h = total_gap // 2
        return _ink([g.line([20, 64], [64 - h, 64]), g.line([64 + h, 64], [108, 64])])

    for total_gap in (12, 14):  # measured: inside 1.5x the rule's radius, outside the rule's own
        ink = broken(total_gap)
        assert vectorize(ink).num_strokes == 2, total_gap  # the rules leave it broken
        assert vectorize(ink, scorer=WideGapRules()).num_strokes == 1, total_gap
        assert vectorize(ink, scorer=ImprovedRules()).num_strokes == 1, total_gap

    for total_gap in (10, 18):  # inside both radii, and outside both
        ink = broken(total_gap)
        assert vectorize(ink, scorer=WideGapRules()).num_strokes == vectorize(ink).num_strokes

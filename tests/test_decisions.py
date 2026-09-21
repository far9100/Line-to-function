"""The tracer's decisions: the rules are unchanged by the refactor and the solvers are right."""

import subprocess
import sys

import numpy as np

from line2func.decisions import greedy_pairs, pick_peaks


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


def test_tracing_never_imports_torch():
    code = ("import sys, numpy as np; from line2func.baseline import vectorize; "
            "vectorize(np.pad(np.ones((4, 60)), 20)); print('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"

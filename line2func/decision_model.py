"""The learned decision scorer: small MLPs evaluated with numpy (no PyTorch).

Weights come from :mod:`line2func.train_decisions` (``decisions.npz``); the
bundled ones are ``line2func/data/decisions_v1.npz``. One network per decision
kind returns a calibrated probability. Where a network's input lies far
outside what it was trained on, or it is unsure (within ``delta`` of its
threshold ``tau``; the bundled weights use ``delta`` 0), the rule decides
instead. With ``delta`` covering everything, the tracing is exactly the rules'
tracing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from line2func.decision_features import (
    FEATURES_VERSION,
    NAMES,
    corner_features,
    corner_peaks,
    crossing_features,
    gap_features,
    junction_features,
)
from line2func.decisions import WIDE_LIMITS, Scorer

FORMAT = "line2func.decisions"
VERSION = 1
BUNDLED = Path(__file__).parent / "data" / "decisions_v1.npz"
# Gentler turning peaks are never judged: the corner network called none of the
# ~340k peaks under 30 deg in its training data a corner, and they are 85% of all peaks.
CORNER_JUDGE_FROM = 30.0
_CACHE: dict[tuple[str, float], "LearnedScorer"] = {}


class _MLP:
    """One kind's network: clip, standardize, two ReLU layers, a logit, then Platt scaling (T, B)."""

    def __init__(self, data, kind: str):
        g = lambda key: data[f"{kind}__{key}"]  # noqa: E731
        self.names = [str(n) for n in g("names")]
        if self.names != NAMES[kind]:
            raise ValueError(f"decision weights for {kind!r} expect other features; retrain them")
        self.lo, self.hi, self.mu, self.sd = (np.asarray(g(k), dtype=np.float64) for k in ("lo", "hi", "mu", "sd"))
        self.layers = [(np.asarray(g(f"W{i}"), dtype=np.float64), np.asarray(g(f"b{i}"), dtype=np.float64))
                       for i in range(3)]
        self.T = float(g("T"))
        self.B = float(g("B")) if f"{kind}__B" in data.files else 0.0
        self.tau = float(g("tau"))
        self.delta = float(g("delta"))

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(probability, trusted)`` per row; untrusted rows lie far outside the training range."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, len(self.names))
        span = self.hi - self.lo
        trusted = np.all((x >= self.lo - span) & (x <= self.hi + span), axis=1)
        h = (np.clip(x, self.lo, self.hi) - self.mu) / self.sd
        for k, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if k < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        p = 1.0 / (1.0 + np.exp(-(h[:, 0] / self.T + self.B)))
        return p, trusted

    def sure(self, p: np.ndarray, trusted: np.ndarray) -> np.ndarray:
        return trusted & (np.abs(p - self.tau) >= self.delta)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


class LearnedScorer(Scorer):
    """Decisions by the learned networks; the rules decide where the networks cannot be trusted."""

    limits = WIDE_LIMITS
    needs_features = True
    junction_policy = "exact"

    def __init__(self, models: dict[str, _MLP]):
        self.models = models
        self.ctx = None
        self.counts: dict[str, list[int]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "LearnedScorer":
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"]) != FORMAT or int(data["version"]) != VERSION:
                raise ValueError(f"{path} is not a line2func decision weights file of version {VERSION}")
            if int(data["features_version"]) != FEATURES_VERSION:
                raise ValueError(f"{path} was trained on features version {int(data['features_version'])}, "
                                 f"this line2func computes version {FEATURES_VERSION}; retrain it")
            models = {str(k): _MLP(data, str(k)) for k in data["kinds"]}
        return cls(models)

    def begin(self, ctx) -> None:
        self.ctx = ctx
        self.counts = {}

    def report(self) -> dict | None:
        return {k: {"decisions": n, "fallback": f} for k, (n, f) in self.counts.items() if n}

    def _count(self, kind: str, n: int, fallback: int) -> None:
        c = self.counts.setdefault(kind, [0, 0])
        c[0] += n
        c[1] += fallback

    def _rule_crossing(self, c) -> bool:
        return bool(c.rule and c.length < 12.0 * self.ctx.radius + 6.0)  # the rules only look this far

    def crossing(self, cands):
        rule = [self._rule_crossing(c) for c in cands]
        m = self.models.get("crossing")
        if m is None or not cands:
            return rule
        p, trusted = m(crossing_features(cands, self.ctx))
        sure = m.sure(p, trusted)
        self._count("crossing", len(cands), int((~sure).sum()))
        return [bool(pi >= m.tau) if s else r for pi, s, r in zip(p, sure, rule)]

    def gap_scores(self, cands):
        rule = super().gap_scores(cands)
        m = self.models.get("gap")
        if m is None or not cands:
            return rule
        p, trusted = m(gap_features(cands, self.ctx))
        sure = m.sure(p, trusted)
        self._count("gap", len(cands), int((~sure).sum()))
        scores = np.full(len(cands), -np.inf)
        for n, c in enumerate(cands):
            if sure[n]:
                scores[n] = p[n] if p[n] >= m.tau else -np.inf
            elif c.rule_ok:
                scores[n] = -1.0 - c.cost / 1e4  # the rule's link, after every confident one, in the rule's order
        return scores

    def junction_scores(self, cands, max_bend):
        rule = super().junction_scores(cands, max_bend)
        mp, me = self.models.get("junction"), self.models.get("junction_end")
        if mp is None or me is None:
            return rule
        out = []
        for c, fallback in zip(cands, rule):
            xp, xe = junction_features(c, self.ctx)
            pp, tp = mp(xp)
            pe, te = me(xe)
            if not (mp.sure(pp, tp).all() and me.sure(pe, te).all()):
                self._count("junction", 1, 1)
                out.append(fallback)  # unsure somewhere at this node: the rule decides the whole node
            else:
                self._count("junction", 1, 0)
                out.append((_logit(pp), _logit(pe)))
        return out

    def corner_keys(self, cand, min_angle):
        key, threshold = super().corner_keys(cand, min_angle)
        m = self.models.get("corner")
        if m is None or not len(cand.index):
            return key, threshold
        peaks = corner_peaks(cand)
        at = {int(i): n for n, i in enumerate(cand.index)}
        out = np.full(len(cand.index), -1.0)
        judge = [i for i in peaks if cand.turn[at[i]] >= min(CORNER_JUDGE_FROM, min_angle)]
        if judge:
            p, trusted = m(corner_features(cand, peaks, self.ctx, only=judge))
            sure = m.sure(p, trusted)
            self._count("corner", len(judge), int((~sure).sum()))
            for idx, pi, s in zip(judge, p, sure):
                turn = float(cand.turn[at[idx]])
                if s:
                    out[at[idx]] = pi if pi >= m.tau else -1.0
                elif turn >= min_angle:
                    out[at[idx]] = m.tau + turn / 1e3  # the rule's corner, in the rule's order
        return out, m.tau


def load(name: str | Path) -> LearnedScorer:
    """``"learned"`` (the weights bundled with line2func) or a path to a ``decisions.npz``; cached."""
    path = BUNDLED if str(name) == "learned" else Path(name)
    if not path.is_file():
        if str(name) == "learned":
            raise FileNotFoundError("the bundled decision weights are missing from this installation; "
                                    "train them with python -m line2func.train_decisions and pass the path")
        raise FileNotFoundError(f"decision weights not found: {path}")
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _CACHE:
        _CACHE[key] = LearnedScorer.from_file(path)
    return _CACHE[key]

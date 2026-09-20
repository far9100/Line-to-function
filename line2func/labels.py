"""Ground-truth labels for the tracer's decisions, the oracle scorer, and the oracle experiment.

On a synthetic scene every stroke is known. Each decision candidate of
:mod:`line2func.decisions` can therefore be labelled positive (1), negative
(0) or ambiguous (-1) by mapping the skeleton it is built on to the ground
truth. Only points past the zone where thinning distorts the skeleton are used,
and anything that does not map cleanly is left ambiguous, never guessed.

* :class:`GTIndex` - all ground-truth strokes, sampled every 0.5 px with their
  stroke id, arc position and tangent; plus their sharp corners.
* ``*_label`` - one label per candidate (junction pair, arm end, gap,
  crossing, corner peak).
* :class:`OracleScorer` - decides every labelled candidate the way the ground
  truth says and leaves ambiguous ones to the rules. It measures how far better
  decisions alone can take the tracer, before any model is trained.
* :class:`ImprovedRules` ("R2") - the rules with wider candidates and one
  geometric check, no learning: the bar a learned scorer must clear.

Command line::

    python -m line2func.labels stats --valset data/val_v1        # label coverage, rule error rates
    python -m line2func.labels oracle --valset data/val_v1 --json runs/decisions/oracle_v1.json
    python -m line2func.labels audit --valset data/val_v1/hard --n 40 --out runs/decisions/audit
    python -m line2func.labels disagree drawing.png --decisions learned    # scorer vs rules on a real drawing
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from line2func import baseline, lineart
from line2func.curves import CurveSet
from line2func.decision_features import CROSSING, Recorder, _r_in, _window, corner_peaks, crossing_features
from line2func.decisions import RULE_LIMITS, WIDE_LIMITS, Limits, Scorer
from line2func.metrics import _arc, _stroke_polylines, _stroke_widths, gt_corners

POS, NEG, AMB = 1, 0, -1
NOISE = "noise"  # an arm that lies on no ground-truth stroke at all (a speck, a noise blob)
KINDS = ("junction", "gap", "crossing", "corner")
_COS35 = float(np.cos(np.radians(35.0)))


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


class GTIndex:
    """The ground-truth strokes of one scene, for point lookups."""

    def __init__(self, gt: CurveSet, spacing: float = 0.5):
        self.gt = gt
        self.polys = _stroke_polylines(gt, spacing)
        self.cums = {k: _arc(v) for k, v in self.polys.items()}
        self.length = {k: float(c[-1]) for k, c in self.cums.items()}
        self.widths = _stroke_widths(gt)
        pts, ids, arcs, tans = [], [], [], []
        for k, poly in self.polys.items():
            if len(poly) < 2:
                continue
            t = np.gradient(poly, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
            pts.append(poly)
            ids.append(np.full(len(poly), k))
            arcs.append(self.cums[k])
            tans.append(t)
        self.pts = np.vstack(pts) if pts else np.zeros((0, 2))
        self.ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        self.arcs = np.concatenate(arcs) if arcs else np.zeros(0)
        self.tans = np.vstack(tans) if tans else np.zeros((0, 2))
        self.tree = cKDTree(self.pts) if len(self.pts) else None
        self.w_max = max(self.widths.values()) if self.widths else 2.0
        # sharp corners (>= 15 deg, labels use 30 for "is a corner") with their stroke and arc position
        self.corners: list[tuple[int, float, np.ndarray, float]] = []
        for p, turn in gt_corners(gt, 15.0):
            _, i = self.tree.query(p)
            self.corners.append((int(self.ids[i]), float(self.arcs[i]), p, turn))

    def tol(self, stroke: int) -> float:
        return max(1.5, 0.5 * self.widths.get(stroke, 2.0) + 1.0)

    def match(self, pts: np.ndarray, dirs: np.ndarray | None = None) -> list:
        """Per point: ``(stroke, s)`` of the ground-truth line it lies on, ``None`` (none) or ``"amb"``.

        A point matches a stroke within max(1.5, w/2 + 1) px whose tangent agrees
        within 35 deg (when ``dirs`` are given); it is ambiguous if two strokes do.
        """
        if self.tree is None or not len(pts):
            return [None] * len(pts)
        radius = max(1.5, 0.5 * self.w_max + 1.0) + 1.0
        dd, ii = self.tree.query(pts, k=8, distance_upper_bound=radius)
        out = []
        for n, (drow, irow) in enumerate(zip(dd, ii)):
            best, strokes = None, set()
            for d, i in zip(drow, irow):
                if not np.isfinite(d):
                    break
                k = int(self.ids[i])
                if d > self.tol(k):
                    continue
                if dirs is not None and abs(float(dirs[n] @ self.tans[i])) < _COS35:
                    continue
                strokes.add(k)
                if best is None:
                    best = (k, float(self.arcs[i]))
            out.append(None if best is None else ("amb" if len(strokes) > 1 else best))
        return out


@dataclass
class ArmLabel:
    """The ground-truth stroke an arm lies on, and which way along it the arm runs (away from its node)."""

    stroke: int
    s_near: float
    sign: int  # +1: the arm runs toward increasing arc position
    near_pt: np.ndarray


def _tangents(pts: np.ndarray) -> np.ndarray:
    t = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[1.0, 0.0]])
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def label_arm(index: GTIndex, arm: np.ndarray, center: np.ndarray, r_in: float, w: float):
    """Label an arm (points from its node outward) from its window past ``r_in``.

    Returns an :class:`ArmLabel`, :data:`NOISE` when the arm is clearly on no
    ground-truth stroke, or ``None`` when it is unclear.
    """
    span = max(6.0, 3.0 * w)
    window = _window(arm, center, r_in, span)
    if len(window) < 3:
        d = np.linalg.norm(arm - center, axis=1)
        window = arm[d >= r_in][: max(3, int(3 * w))]
    if len(window) < 2 or index.tree is None:
        return None
    if float(np.median(index.tree.query(window)[0])) > max(3.0, w + 2.0):
        return NOISE
    label = _label_window(index, window)
    # where two strokes still run close together (a shallow crossing), look farther out
    dist = np.linalg.norm(arm - center, axis=1)
    for extra in (span, 2.0 * span):
        if label is not None:
            break
        farther = arm[(dist >= r_in + extra) & (dist <= r_in + extra + span)]
        if len(farther) >= 3:
            label = _label_window(index, farther)
    return label


def _label_window(index: GTIndex, window: np.ndarray) -> ArmLabel | None:
    m = index.match(window, _tangents(window))
    matched = [x for x in m if isinstance(x, tuple)]
    ambiguous = sum(1 for x in m if x == "amb")
    if len(matched) < min(3, len(window)) or ambiguous > 0.2 * len(m):
        return None
    strokes = [k for k, _ in matched]
    top = max(set(strokes), key=strokes.count)
    if strokes.count(top) < 0.8 * len(matched):
        return None
    s = [si for k, si in matched if k == top]
    if abs(s[-1] - s[0]) < 0.5:
        return None
    return ArmLabel(top, s[0], 1 if s[-1] > s[0] else -1, window[0])


def _through(a: ArmLabel, b: ArmLabel, w: float, slack: float) -> bool:
    """Does one ground-truth stroke run from arm ``a`` through the node into arm ``b``?"""
    if a.stroke != b.stroke or a.sign == b.sign:
        return False
    ds = b.s_near - a.s_near
    if np.sign(ds) != b.sign:
        return False
    chord = float(np.linalg.norm(b.near_pt - a.near_pt))
    return abs(ds) <= 1.3 * chord + 2.0 * w + slack


# ---------------------------------------------------------------------------
# Labels per decision
# ---------------------------------------------------------------------------


def junction_labels(index: GTIndex, cand, w: float, r: float) -> tuple[list[int], list[int]]:
    """(label per arm pair, label per arm "ends here") at one node."""
    r_in = _r_in(r, cand.spread)
    arms = [label_arm(index, a, cand.center, r_in, w) for a in cand.arms]
    pair = []
    for i, j in cand.pairs:
        a, b = arms[i], arms[j]
        if a is NOISE or b is NOISE:
            pair.append(NEG)  # a stroke never continues into noise
        elif a is None or b is None:
            pair.append(AMB)
        else:
            pair.append(POS if _through(a, b, w, 4.0) else NEG)
    partners = [0] * len(arms)
    for (i, j), lab in zip(cand.pairs, pair):
        if lab == POS:
            partners[i] += 1
            partners[j] += 1
    if any(n > 1 for n in partners):  # contradictory: the node is unclear
        return [AMB] * len(pair), [AMB] * len(arms)
    full = all(a is not None for a in arms)
    ends = [NEG if partners[i] else (POS if full else AMB) for i in range(len(arms))]
    return pair, ends


def gap_label(index: GTIndex, cand, w: float) -> int:
    a = label_arm(index, cand.arm_i, cand.tip_i, 0.5 * w, w)
    b = label_arm(index, cand.arm_j, cand.tip_j, 0.5 * w, w)
    if a is NOISE or b is NOISE:
        return NEG  # linking a line to a speck
    if a is None or b is None:
        return AMB
    chord = cand.dist
    if a.stroke != b.stroke:
        # two strokes meeting end to end in one line look exactly like a gap
        near_end = all(min(x.s_near, index.length[x.stroke] - x.s_near) <= 2.0 * w + 2.0 for x in (a, b))
        bend = float(np.degrees(np.arccos(np.clip(-(cand.out_i @ cand.out_j), -1.0, 1.0))))
        return AMB if near_end and bend < 25.0 else NEG
    if a.sign == b.sign:
        return NEG
    ds = b.s_near - a.s_near
    if np.sign(ds) == b.sign and abs(ds) <= 1.3 * chord + 2.0 * w + 3.0:
        return POS
    if abs(ds) > 3.0 * chord + 4.0 * w:
        return NEG  # the same stroke far along it: linking would close a U
    return AMB


def crossing_label(index: GTIndex, cand, w: float, r: float) -> int:
    r_in = _r_in(r, 0.0)
    cu, cv = cand.mid[0], cand.mid[-1]
    au = [label_arm(index, pts, cu, r_in, w) for _, pts in cand.arms_u]
    av = [label_arm(index, pts, cv, r_in, w) for _, pts in cand.arms_v]
    if len(au) != 2 or len(av) != 2:
        return AMB
    if any(x is NOISE for x in au + av):
        return NEG  # not two strokes crossing
    if any(x is None for x in au + av):
        return AMB
    if _through(au[0], au[1], w, 4.0) or _through(av[0], av[1], w, 4.0):
        return NEG  # a stroke runs straight through one of the junctions: a T or an H
    slack = cand.length + 4.0
    for (p1, q1), (p2, q2) in (((au[0], av[0]), (au[1], av[1])), ((au[0], av[1]), (au[1], av[0]))):
        t1, t2 = _through(p1, q1, w, slack), _through(p2, q2, w, slack)
        if t1 and t2 and p1.stroke != p2.stroke:
            return POS
        if (t1 and p2.stroke != q2.stroke) or (t2 and p1.stroke != q1.stroke):
            return NEG  # one line through, the other two arms are different strokes: offset T's
    return AMB


def corner_label(index: GTIndex, cand, idx: int, r: float) -> int:
    pts, n = cand.pts, len(cand.pts)
    reach = cand.k + cand.skip

    def stroke_at(i: int, half: int = 3):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = pts[lo:hi]
        m = [x for x in index.match(seg, _tangents(seg)) if isinstance(x, tuple)]
        if len(m) < 2:
            return None
        ks = [k for k, _ in m]
        top = max(set(ks), key=ks.count)
        if ks.count(top) < 0.8 * len(m):
            return None
        return top, float(np.median([s for k, s in m if k == top]))

    here = stroke_at(idx)
    if here is None:
        return AMB
    stroke, s = here
    near = [(cs, turn) for k, cs, _, turn in index.corners if k == stroke and abs(cs - s) <= 3.0 * reach + 4.0]
    if any(turn >= 30.0 and abs(cs - s) <= max(3.0, r + 2.0) for cs, turn in near):
        return POS
    if near:
        return AMB  # a weak corner, or one a little farther away
    for other in (idx - 2 * reach, idx + 2 * reach):
        if 0 <= other < n:
            there = stroke_at(other)
            if there is not None and there[0] != stroke:
                return AMB  # the traced stroke changes ground-truth stroke here (a wrong join)
    return NEG


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class OracleScorer(Scorer):
    """Decides every labelled candidate as the ground truth says; ambiguous ones follow the rules."""

    needs_features = True  # for the context (line width)

    def __init__(self, gt: CurveSet, kinds=KINDS, limits: Limits = RULE_LIMITS):
        self.index = GTIndex(gt)
        self.kinds = set(kinds)
        self.limits = limits
        self.ctx = None

    def begin(self, ctx) -> None:
        self.ctx = ctx

    @property
    def _wr(self) -> tuple[float, float]:
        return (self.ctx.line_w, self.ctx.radius) if self.ctx is not None else (2.0, 1.0)

    def crossing(self, cands):
        rule = super().crossing(cands)
        if "crossing" not in self.kinds:
            return rule
        w, r = self._wr
        out = []
        for c, v in zip(cands, rule):
            lab = crossing_label(self.index, c, w, r)
            out.append(v if lab == AMB else lab == POS)
        return out

    def gap_scores(self, cands):
        scores = super().gap_scores(cands)
        if "gap" not in self.kinds:
            return scores
        w, _ = self._wr
        scores = scores.copy()
        for n, c in enumerate(cands):
            lab = gap_label(self.index, c, w)
            if lab == POS:
                scores[n] = 1000.0 - c.cost
            elif lab == NEG:
                scores[n] = -np.inf
        return scores

    def junction_scores(self, cands, max_bend):
        base = super().junction_scores(cands, max_bend)
        if "junction" not in self.kinds:
            return base
        w, r = self._wr
        out = []
        for c, (z, _) in zip(cands, base):
            pair, _ = junction_labels(self.index, c, w, r)
            z = z.copy()
            for n, lab in enumerate(pair):
                if lab == POS:
                    z[n] = 1000.0 - c.bends[n]
                elif lab == NEG:
                    z[n] = -np.inf
            out.append((z, None))
        return out

    def corner_keys(self, cand, min_angle):
        key, threshold = super().corner_keys(cand, min_angle)
        if "corner" not in self.kinds or not len(cand.index):
            return key, threshold
        w, r = self._wr
        key = np.asarray(key, dtype=np.float64).copy()
        at = {int(i): p for p, i in enumerate(cand.index)}
        lowest = self.limits.min_corner_turn if self.limits.min_corner_turn is not None else min_angle
        reach = cand.k + cand.skip
        n = len(cand.pts)
        for idx in corner_peaks(cand):
            t = cand.turn[at[idx]]
            if t < lowest:
                continue
            lab = corner_label(self.index, cand, idx, r)
            if lab == POS:
                key[at[idx]] = 1000.0 + t
            elif lab == NEG:  # suppress the whole bump, not just its top
                for q in range(idx - reach, idx + reach + 1):
                    q = q % n if cand.closed else q
                    if q in at and key[at[q]] < 1000.0:
                        key[at[q]] = -1.0
        return key, threshold


class ImprovedRules(Scorer):
    """R2: the angle rules with wider candidates and one geometric check (no learning).

    Shallow crossings are looked for up to 2.5x farther apart, but two junctions
    whose paired arms are offset by more than 1.5 line widths are two T's, not
    one crossing. Gaps are searched 1.5x farther with the same 35 deg gate.
    """

    needs_features = True
    limits = Limits(gap_radius_scale=1.5, crossing_len_scale=2.5)

    def begin(self, ctx) -> None:
        self.ctx = ctx

    def crossing(self, cands):
        if not cands:
            return []
        x = crossing_features(cands, self.ctx)
        l1, l2 = CROSSING.index("lateral_1"), CROSSING.index("lateral_2")
        return [bool(c.rule and max(row[l1], row[l2]) <= 1.5) for c, row in zip(cands, x)]

    def gap_scores(self, cands):
        # the angle gate was applied when the candidates were made; the wider radius is the change
        return np.array([-c.cost for c in cands], dtype=np.float64)


def make_scorer(variant: str, gt: CurveSet | None = None) -> Scorer:
    """``r0`` (rules), ``r2``, ``o1-<kind>``, ``o1-all`` or ``o2-all`` (the oracles need ``gt``).

    Any other name is passed to :func:`line2func.decisions.load_scorer`, so
    ``learned`` or a path to weights works too.
    """
    from line2func.decisions import RULES

    if variant == "r0":
        return RULES
    if variant == "r2":
        return ImprovedRules()
    if variant.startswith(("o1-", "o2-")):
        kind = variant.split("-", 1)[1]
        kinds = KINDS if kind == "all" else (kind,)
        if kind != "all" and kind not in KINDS:
            raise ValueError(f"unknown decision kind {kind!r}")
        limits = RULE_LIMITS
        if variant.startswith("o2"):
            # wider candidates only for the decision under test, so the others keep the rules' candidates
            limits = WIDE_LIMITS if kind == "all" else _WIDE_ONLY[kind]
        return OracleScorer(gt, kinds, limits)
    # anything else names a learned scorer ("learned", or a path to weights)
    from line2func.decisions import load_scorer

    return load_scorer(variant)


_WIDE_ONLY = {
    "junction": RULE_LIMITS,  # junction candidates do not depend on the limits
    "gap": Limits(gap_radius_scale=WIDE_LIMITS.gap_radius_scale, gap_angle_gate=WIDE_LIMITS.gap_angle_gate),
    "crossing": Limits(crossing_len_scale=WIDE_LIMITS.crossing_len_scale),
    "corner": Limits(min_corner_turn=WIDE_LIMITS.min_corner_turn),
}


# ---------------------------------------------------------------------------
# Labelling recorded candidates
# ---------------------------------------------------------------------------


def label_recording(rec: Recorder, index: GTIndex) -> dict[str, np.ndarray]:
    """Labels (1 / 0 / -1) for every row a :class:`Recorder` collected, per kind."""
    ctx = rec.ctx
    w, r = (ctx.line_w, ctx.radius) if ctx is not None else (2.0, 1.0)
    out: dict[str, list[int]] = {k: [] for k in rec.geom}
    node_cache: dict[int, tuple[list[int], list[int]]] = {}
    for geom in rec.geom["junction"]:
        cand, i, j = geom
        if id(cand) not in node_cache:
            node_cache[id(cand)] = junction_labels(index, cand, w, r)
        out["junction"].append(node_cache[id(cand)][0][cand.pairs.index((i, j))])
    for cand, i in rec.geom["junction_end"]:
        if id(cand) not in node_cache:
            node_cache[id(cand)] = junction_labels(index, cand, w, r)
        out["junction_end"].append(node_cache[id(cand)][1][i])
    out["gap"] = [gap_label(index, c, w) for c in rec.geom["gap"]]
    out["crossing"] = [crossing_label(index, c, w, r) for c in rec.geom["crossing"]]
    out["corner"] = [corner_label(index, cand, idx, r) for cand, idx, _ in rec.geom["corner"]]
    return {k: np.asarray(v, dtype=np.int8) for k, v in out.items()}


def _scenes(valset: Path, limit: int | None):
    from line2func.eval import _subsets
    from line2func.synth import load_scene_dir

    for name, folder in _subsets(Path(valset)).items():
        yield name, load_scene_dir(folder)[:limit]


def stats(valset: Path, limit: int | None = None, variant: str = "r0") -> dict:
    """Per subset and decision kind: candidates, label shares, and how often each decider is right.

    Two deciders are tallied on the same candidates: the rule, whose verdict the
    recorder keeps for every candidate, and whichever scorer ``variant`` names,
    whose choice the recorder also keeps. Only *labelled* candidates count, which
    is what makes this the clean measure of decision headroom - unlike the oracle
    runs, where unlabelled candidates fall back to the rules and the result is a
    mixture of the two rather than a ceiling.

    So ``python -m line2func.labels stats --valset data/val_v2 --variant learned``
    answers "how much better could any scorer get at deciding" directly.
    """
    report = {}
    for name, scenes in _scenes(valset, limit):
        tally: dict[str, dict[str, float]] = {}
        for png, gt in scenes:
            ink = lineart.extract(png, "none")
            params = baseline.BaselineParams()
            rec = Recorder(params)
            baseline.vectorize(ink, params, scorer=make_scorer(variant, gt), recorder=rec)
            labels = label_recording(rec, GTIndex(gt))
            for kind, y in labels.items():
                _, rule, chosen = rec.table(kind)
                t = tally.setdefault(kind, {"n": 0, "pos": 0, "neg": 0, "amb": 0} |
                                     {f"{who}_{cell}": 0 for who in ("rule", "scorer")
                                      for cell in ("tp", "fp", "fn", "tn")})
                t["n"] += len(y)
                t["pos"] += int((y == POS).sum())
                t["neg"] += int((y == NEG).sum())
                t["amb"] += int((y == AMB).sum())
                for who, yes in (("rule", rule[:, 2] > 0.5), ("scorer", chosen)):
                    t[f"{who}_tp"] += int((yes & (y == POS)).sum())
                    t[f"{who}_fp"] += int((yes & (y == NEG)).sum())
                    t[f"{who}_fn"] += int((~yes & (y == POS)).sum())
                    t[f"{who}_tn"] += int((~yes & (y == NEG)).sum())
        for t in tally.values():
            lab = t["pos"] + t["neg"]
            t["labelled"] = lab / t["n"] if t["n"] else float("nan")
            for who in ("rule", "scorer"):
                tp, fp = t[f"{who}_tp"], t[f"{who}_fp"]
                t[f"{who}_accuracy"] = (tp + t[f"{who}_tn"]) / lab if lab else float("nan")
                t[f"{who}_recall"] = tp / t["pos"] if t["pos"] else float("nan")
                t[f"{who}_precision"] = tp / (tp + fp) if tp + fp else float("nan")
            # what any scorer could still gain on these candidates
            t["headroom"] = 1.0 - t["scorer_accuracy"] if lab else float("nan")
        report[name] = tally
    return report


def _print_stats(report: dict) -> None:
    print(f"{'subset':<7}{'kind':<14}{'cands':>8}{'labelled':>9}{'pos':>7}{'neg':>7}"
          f"{'rule acc':>9}{'this acc':>9}{'headroom':>9}{'this P':>8}{'this R':>8}")
    for name, tally in report.items():
        for kind, t in tally.items():
            print(f"{name:<7}{kind:<14}{t['n']:>8}{t['labelled']:>9.1%}{t['pos']:>7}{t['neg']:>7}"
                  f"{t['rule_accuracy']:>9.3f}{t['scorer_accuracy']:>9.3f}{t['headroom']:>9.3f}"
                  f"{t['scorer_precision']:>8.3f}{t['scorer_recall']:>8.3f}")
    print("  'this' is the scorer --variant names (r0 = the rules, so the two accuracy columns match).")
    print("  Labelled candidates only, so 'headroom' is what any scorer could still gain at deciding.")


VARIANTS = ("r0", "r2", "o1-junction", "o1-gap", "o1-crossing", "o1-corner", "o1-all", "o2-all")


def oracle(valset: Path, variants=VARIANTS, limit: int | None = None) -> dict:
    """Evaluate each variant on the scene set (evaluation and decision metrics per subset)."""
    from line2func.eval import eval_scenes

    results = {}
    for v in variants:
        t = time.perf_counter()

        def run(ink, gt, v=v):
            return baseline.vectorize(ink, scorer=make_scorer(v, gt))

        results[v] = eval_scenes(valset, limit, vectorize=run, pass_gt=True)
        print(f"  {v}: {time.perf_counter() - t:.0f} s", flush=True)
    return results


def _print_oracle(results: dict) -> None:
    cols = [("f_gt2", "F@2"), ("crossing_continuity", "cross"), ("gap_closure", "gaps"),
            ("fragments_per_stroke", "frag"), ("curve_ratio", "curves"), ("bcubed_p", "BC-P"),
            ("bcubed_r", "BC-R"), ("bcubed_f", "BC-F"), ("crossing_false_turn", "X turn"),
            ("t_false_cont", "T cont"), ("joins_other_per100", "joins"), ("corner_p", "cor-P"),
            ("corner_r", "cor-R")]
    subsets = sorted({s for r in results.values() for s in r})
    for sub in subsets:
        print(f"\n{sub}:")
        print(f"{'variant':<13}" + "".join(f"{h:>8}" for _, h in cols))
        for v, r in results.items():
            if sub in r:
                print(f"{v:<13}" + "".join(f"{r[sub][k]:>8.3f}" for k, _ in cols))


_COLORS = {POS: (0, 150, 0), NEG: (220, 0, 0), AMB: (200, 140, 0)}


def _tile(ink: np.ndarray, kind: str, geom, ring=None, size: int = 40, zoom: int = 4):
    """A zoomed crop of the ink around one candidate, with its arms drawn (blue / magenta)."""
    from PIL import Image, ImageDraw

    if kind in ("junction", "junction_end"):
        cand, i, j = geom if kind == "junction" else (geom[0], geom[1], None)
        center = cand.center
        lines = [(cand.arms[i], (0, 90, 255))] + ([(cand.arms[j], (255, 0, 200))] if j is not None else [])
        lines += [(a, (170, 170, 170)) for n, a in enumerate(cand.arms) if n not in (i, j)]
    elif kind == "gap":
        center = 0.5 * (geom.tip_i + geom.tip_j)
        lines = [(geom.arm_i, (0, 90, 255)), (geom.arm_j, (255, 0, 200))]
    elif kind == "crossing":
        center = 0.5 * (geom.mid[0] + geom.mid[-1])
        lines = [(p, (0, 90, 255)) for _, p in geom.arms_u] + [(p, (255, 0, 200)) for _, p in geom.arms_v]
        lines.append((geom.mid, (0, 0, 0)))
    else:
        cand, idx, _ = geom
        center, lines = cand.pts[idx], [(cand.pts, (0, 90, 255))]
    x0, y0 = int(center[0]) - size // 2, int(center[1]) - size // 2
    crop = np.zeros((size, size))
    h, w = ink.shape
    ys, xs = slice(max(0, y0), min(h, y0 + size)), slice(max(0, x0), min(w, x0 + size))
    crop[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0] = ink[ys, xs]
    tile = Image.fromarray(np.uint8(255 - 150 * np.clip(crop, 0, 1))).convert("RGB").resize(
        (size * zoom, size * zoom), Image.NEAREST)
    d = ImageDraw.Draw(tile)
    for pts, col in lines:
        q = [((pt[0] - x0) * zoom, (pt[1] - y0) * zoom) for pt in pts]
        if len(q) > 1:
            d.line(q, fill=col, width=2)
    if kind == "corner":
        cx, cy = (center[0] - x0) * zoom, (center[1] - y0) * zoom
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=ring or (0, 0, 0), width=3)
    return tile


def _sheet(tiles: list, path: Path, cols: int = 8) -> None:
    """Tiles ``(image, caption, color)`` in a grid."""
    from PIL import Image, ImageDraw

    if not tiles:
        return
    tw, th = tiles[0][0].size
    sheet = Image.new("RGB", (cols * (tw + 4), ((len(tiles) + cols - 1) // cols) * (th + 16)), "white")
    for k, (tile, caption, color) in enumerate(tiles):
        r_, c_ = divmod(k, cols)
        px, py = c_ * (tw + 4), r_ * (th + 16)
        sheet.paste(tile, (px, py + 14))
        ImageDraw.Draw(sheet).text((px + 2, py), caption, fill=color)
    sheet.save(path)


def audit(valset: Path, out: Path, n: int = 40, seed: int = 0) -> Path:
    """Contact sheets of labelled candidates (ink, arms, label) for checking the labeller by eye."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    picks: dict[str, list] = {k: [] for k in ("junction", "gap", "crossing", "corner")}
    for _, scenes in _scenes(valset, None):
        for png, gt in scenes[: max(5, n // 4)]:
            ink = lineart.extract(png, "none")
            params = baseline.BaselineParams()
            rec = Recorder(params)
            baseline.vectorize(ink, params, scorer=make_scorer("o2-all", gt), recorder=rec)
            labels = label_recording(rec, GTIndex(gt))
            for kind in picks:
                for geom, y in zip(rec.geom[kind], labels[kind]):
                    picks[kind].append((ink, geom, int(y)))
    names = {POS: "positive", NEG: "negative", AMB: "ambiguous"}
    for kind, items in picks.items():
        if items:
            chosen = [items[i] for i in rng.choice(len(items), size=min(n, len(items)), replace=False)]
            _sheet([(_tile(ink, kind, geom, _COLORS[y]), names[y], _COLORS[y]) for ink, geom, y in chosen],
                   out / f"audit_{kind}.png")
    return out


def disagree(image: Path, decisions: str, out: Path, n: int = 100, seed: int = 0) -> dict:
    """Where a decision scorer and the rules decide differently on a real drawing: contact sheets to review.

    The drawing is traced like ``pipeline.trace`` (2x for thin lines) with the
    scorer; the recorder keeps, for every candidate, both the scorer's choice
    and what the rule would have chosen. Captions say which one links (or
    splits); tiles show the arms in blue / magenta.
    """
    from line2func import pipeline
    from line2func.decisions import load_scorer

    rgb = lineart.load_rgb(image)
    ink = lineart.extract(rgb, "none")
    factor = pipeline.choose_upscale(ink, "auto")
    if factor > 1:
        h, w = ink.shape
        ink = lineart.extract(pipeline.resize(rgb, (w * factor, h * factor)), "none")
    params = baseline.BaselineParams(fit_tolerance=float(factor))
    rec = Recorder(params)
    baseline.vectorize(ink, params, scorer=load_scorer(decisions), recorder=rec)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"image": str(image), "upscale": factor, "decisions": str(decisions)}
    verbs = {"junction": "continue", "junction_end": "end here", "gap": "link", "crossing": "merge X",
             "corner": "corner"}
    for kind in ("junction", "gap", "crossing", "corner"):
        _, rule, chosen = rec.table(kind)
        differ = np.nonzero((rule[:, 2] > 0.5) != chosen)[0]
        summary[kind] = {"candidates": int(len(chosen)), "disagree": int(len(differ)),
                         "model_yes": int(chosen[differ].sum()), "rule_yes": int((rule[differ, 2] > 0.5).sum())}
        pick = rng.choice(differ, size=min(n, len(differ)), replace=False) if len(differ) else []
        tiles = []
        for k in sorted(int(i) for i in pick):
            yes = bool(chosen[k])
            caption = f"#{k} model: {verbs[kind]}" if yes else f"#{k} rule: {verbs[kind]}"
            tiles.append((_tile(ink, kind, rec.geom[kind][k], (0, 150, 0) if yes else (220, 0, 0)), caption,
                          (0, 110, 0) if yes else (190, 0, 0)))
        _sheet(tiles, out / f"disagree_{kind}.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.labels", description="Decision labels and the oracle.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats", help="label coverage and rule error rates")
    s.add_argument("--valset", type=Path, required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--variant", default="r0", help="who decides while recording, and whose accuracy is reported: r0 (the rules), r2, an oracle, learned, or a weights path")
    o = sub.add_parser("oracle", help="evaluate the rules, improved rules and oracles")
    o.add_argument("--valset", type=Path, required=True)
    o.add_argument("--variants", default=",".join(VARIANTS))
    o.add_argument("--limit", type=int, default=None)
    o.add_argument("--json", type=Path)
    a = sub.add_parser("audit", help="contact sheets of labelled candidates")
    a.add_argument("--valset", type=Path, required=True)
    a.add_argument("--n", type=int, default=40)
    a.add_argument("--out", type=Path, default=Path("runs/decisions/audit"))
    d = sub.add_parser("disagree", help="where a decision scorer and the rules differ on a real drawing")
    d.add_argument("image", type=Path)
    d.add_argument("--decisions", required=True, help="learned, or a path to decision weights")
    d.add_argument("--n", type=int, default=100, help="tiles per decision kind")
    d.add_argument("--out", type=Path, default=Path("runs/decisions/real_audit"))
    args = p.parse_args(argv)
    if args.cmd == "stats":
        report = stats(args.valset, args.limit, args.variant)
        _print_stats(report)
    elif args.cmd == "oracle":
        results = oracle(args.valset, [v.strip() for v in args.variants.split(",") if v.strip()], args.limit)
        _print_oracle(results)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    elif args.cmd == "disagree":
        print(json.dumps(disagree(args.image, args.decisions, args.out, args.n), indent=1))
    else:
        print(f"wrote {audit(args.valset, args.out, args.n)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

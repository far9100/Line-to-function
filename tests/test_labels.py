"""Ground-truth labels for decision candidates, and the oracle built on them (hand-made scenes)."""

import numpy as np
import pytest

from line2func import geometry as g
from line2func.baseline import BaselineParams, vectorize
from line2func.curves import Curve, CurveSet
from line2func.decision_features import Recorder
from line2func.labels import AMB, NEG, POS, GTIndex, label_recording, make_scorer
from line2func.render import render_lineart

SIZE = 128


def _scene(lines, edit=None, strokes=None):
    strokes = strokes or list(range(len(lines)))
    gt = CurveSet(SIZE, SIZE, [Curve(np.asarray(c, float), stroke=k) for c, k in zip(lines, strokes)],
                  meta={"widths": [[2.0, 2.0]] * len(lines)})
    img = render_lineart(gt, SIZE, SIZE, line_width=2.0)
    if edit is not None:
        img = edit(img.copy())
    return gt, 1.0 - img / 255.0


def _labels(gt, ink, variant="r0"):
    rec = Recorder(BaselineParams())
    vectorize(ink, scorer=make_scorer(variant, gt), recorder=rec)
    return rec, label_recording(rec, GTIndex(gt))


def _cross(angle):
    t = np.tan(np.radians(angle / 2)) * 54
    return [g.line([10, 64 - t], [118, 64 + t]), g.line([10, 64 + t], [118, 64 - t])]


def test_crossing_pairs_are_labelled_by_the_lines_they_belong_to():
    gt, ink = _scene(_cross(30))
    rec, y = _labels(gt, ink)
    # at the crossing node: two positive pairs (the lines), the rest negative
    node_rows = [(geom, lab) for geom, lab in zip(rec.geom["junction"], y["junction"]) if len(geom[0].keys) == 4]
    assert node_rows and sorted(lab for _, lab in node_rows) == [NEG, NEG, NEG, NEG, POS, POS]
    for (cand, i, j), lab in node_rows:
        # the lines go straight through (bend ~0); turning into the other line bends by 30 or 150 deg
        straight = cand.bends[cand.pairs.index((i, j))] < 15
        assert (lab == POS) == straight


@pytest.mark.parametrize("angle", [12, 15])
def test_the_oracle_keeps_shallow_crossings_whole(angle):
    gt, ink = _scene(_cross(angle))
    assert vectorize(ink, scorer=make_scorer("o2-all", gt)).num_strokes == 2


def test_t_junction_and_offset_double_t():
    gt, ink = _scene([g.line([10, 40], [118, 40]), g.line([64, 40], [64, 118])])
    rec, y = _labels(gt, ink)
    for (cand, i, j), lab in zip(rec.geom["junction"], y["junction"]):
        bar = all(abs(a[-1][1] - 40) < 3 for a in (cand.arms[i], cand.arms[j]))
        assert lab == (POS if bar else NEG)
    assert vectorize(ink, scorer=make_scorer("o1-all", gt)).num_strokes == 2
    # two stems on opposite sides of a bar, 6 px apart: not one crossing
    gt, ink = _scene([g.line([10, 64], [118, 64]), g.line([60, 64], [60, 16]), g.line([66, 64], [66, 112])])
    assert vectorize(ink, scorer=make_scorer("o2-all", gt)).num_strokes == 3


def test_gap_labels():
    def cut(img):
        img[:, 62:66] = 255
        return img

    gt, ink = _scene([g.line([10, 64], [118, 64])], edit=cut)
    _, y = _labels(gt, ink)
    assert list(y["gap"]) == [POS]
    # two different strokes end to end, in one line: looks exactly like a gap
    gt, ink = _scene([g.line([10, 64], [60, 64]), g.line([66, 64], [118, 64])])
    _, y = _labels(gt, ink)
    assert list(y["gap"]) == [AMB]

    # a dash that is not in the ground truth (noise), in line with a stroke end
    def dash(img):
        img[63:66, 72:80] = 0
        return img

    gt, ink = _scene([g.line([10, 64], [64, 64])], edit=dash)
    _, y = _labels(gt, ink, "o2-all")
    assert NEG in list(y["gap"]) and POS not in list(y["gap"])


def test_corner_labels():
    # an L: one stroke of two pieces with a sharp corner between them
    gt, ink = _scene([g.line([20, 20], [20, 100]), g.line([20, 100], [100, 100])], strokes=[0, 0])
    rec, y = _labels(gt, ink)
    rows = [(geom[0].pts[geom[1]], lab) for geom, lab in zip(rec.geom["corner"], y["corner"])]
    at_corner = [lab for p, lab in rows if np.linalg.norm(p - [20, 100]) < 4]
    assert at_corner == [POS]
    assert all(lab in (NEG, AMB) for p, lab in rows if np.linalg.norm(p - [20, 100]) >= 8)

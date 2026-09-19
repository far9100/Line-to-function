import json

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet


def make_set() -> CurveSet:
    return CurveSet(
        width=120,
        height=80,
        curves=[
            Curve(g.line([10, 10], [60, 10]), stroke=0),
            Curve([[60, 10], [80, 10], [90, 30], [90, 50]], stroke=0, confidence=0.8),
            Curve(g.line([5, 70], [100, 70]), stroke=1, tags=("fill_outline",)),
        ],
        meta={"engine": "baseline", "line_width": 2.1},
    )


def test_json_round_trip(tmp_path):
    cs = make_set()
    path = tmp_path / "curves.json"
    cs.save_json(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "line2func.curves"
    assert data["image"] == {"width": 120, "height": 80}
    assert [c["id"] for c in data["curves"]] == [0, 1, 2]

    back = CurveSet.load_json(path)
    assert (back.width, back.height, len(back)) == (120, 80, 3)
    assert back.meta == {"engine": "baseline", "line_width": 2.1}
    for a, b in zip(cs, back):
        np.testing.assert_array_equal(a.ctrl, b.ctrl)
        assert (a.stroke, a.confidence, a.tags) == (b.stroke, b.confidence, b.tags)


def test_strokes_grouping():
    cs = make_set()
    assert cs.num_strokes == 2
    groups = cs.strokes()
    assert list(groups) == [0, 1]
    assert len(groups[0]) == 2


def test_validation():
    with pytest.raises(ValueError):
        Curve(g.line([0, 0], [1, 1]), confidence=1.5)
    with pytest.raises(ValueError):
        CurveSet(width=0, height=10)
    with pytest.raises(ValueError):
        CurveSet.from_dict({"format": "something-else"})
    bad_version = make_set().to_dict() | {"version": 99}
    with pytest.raises(ValueError):
        CurveSet.from_dict(bad_version)

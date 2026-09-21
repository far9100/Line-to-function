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


def test_tone_survives_every_transform_a_result_goes_through():
    """A filled area's tone decides how it is filled, and two rebuilds used to drop fields silently."""
    from line2func.budget import reduce_to

    area = [Curve(p, stroke=0, tags=("fill_outline",), tone=0.3, color="#b3b3b3")
            for p in (g.line([10, 10], [60, 10]), g.line([60, 10], [60, 40]),
                      g.line([60, 40], [10, 40]), g.line([10, 40], [10, 10]))]
    cs = CurveSet(120, 80, area + [Curve(g.line([5, 70], [100, 70]), stroke=1)], meta={"line_width": 2.0})

    back = CurveSet.from_dict(json.loads(json.dumps(cs.to_dict())))
    assert [c.tone for c in back] == [0.3, 0.3, 0.3, 0.3, None]

    bigger = cs.scaled(2.0, 240, 160)  # the upscale path
    assert [c.tone for c in bigger] == [0.3, 0.3, 0.3, 0.3, None]

    reduce_to(cs, 3)  # the curve-budget path
    assert len(cs) == 3
    assert all(c.tone == 0.3 for c in cs if "fill_outline" in c.tags)

    with pytest.raises(ValueError):
        Curve(g.line([0, 0], [1, 1]), tone=1.5)


def test_a_curves_json_without_tone_still_loads():
    """Tone is optional, so results saved before it existed open unchanged."""
    data = make_set().to_dict()
    assert all("tone" not in c for c in data["curves"])
    assert [c.tone for c in CurveSet.from_dict(data)] == [None, None, None]


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

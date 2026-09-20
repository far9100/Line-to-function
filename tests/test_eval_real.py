"""Real drawings have no ground truth, so they are judged against their own ink, and two settings
are compared drawing by drawing (line2func.eval --realset / --compare)."""

import json

import numpy as np
import pytest

from line2func import geometry as g
from line2func.curves import Curve, CurveSet
from line2func.eval import (MEASURES, _options, _real_images, bootstrap_ci, compare_real, eval_real, main)
from line2func.render import render_lineart, save_png


def _drawing(path, seed: int = 0, size: int = 96) -> None:
    rng = np.random.default_rng(seed)
    curves = [Curve(g.line(rng.uniform(8, size - 8, 2), rng.uniform(8, size - 8, 2)), stroke=i) for i in range(6)]
    save_png(path, render_lineart(CurveSet(size, size, curves), size, size, line_width=2.0))


def _folder(tmp_path, n: int = 3):
    folder = tmp_path / "real"
    folder.mkdir()
    for i in range(n):
        _drawing(folder / f"d{i}.png", seed=i)
    return folder


# ---------------------------------------------------------------------------
# Finding the drawings
# ---------------------------------------------------------------------------


def test_the_drawings_are_found_in_order_and_an_empty_folder_is_an_error(tmp_path):
    folder = _folder(tmp_path)
    (folder / "notes.txt").write_text("not a drawing", encoding="utf-8")
    assert [p.name for p in _real_images(folder)] == ["d0.png", "d1.png", "d2.png"]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _real_images(empty)


# ---------------------------------------------------------------------------
# The paired comparison
# ---------------------------------------------------------------------------


def _report(values: dict[str, list]) -> dict:
    n = len(next(iter(values.values())))
    rows = []
    for i in range(n):
        row = {"image": f"d{i}.png"}
        row.update({k: 0.0 for k in MEASURES})
        row.update({k: v[i] for k, v in values.items()})
        rows.append(row)
    return {"images": rows, "options": {}}


def test_a_setting_compared_with_itself_is_zero_with_a_zero_width_interval():
    r = _report({"curves": [100, 200, 300, 400, 500]})
    out = compare_real(r, r)
    assert out["delta"]["curves"]["median"] == 0.0
    assert out["delta"]["curves"]["ci95"] == [0.0, 0.0]


def test_a_shift_on_every_drawing_is_reported_with_an_interval_that_misses_zero():
    before = _report({"curves": [100, 200, 300, 400, 500]})
    after = _report({"curves": [90, 190, 290, 390, 490]})
    d = compare_real(before, after)["delta"]["curves"]
    assert d["median"] == -10.0
    lo, hi = d["ci95"]
    assert hi < 0.0, (lo, hi)  # every drawing moved the same way, so the interval clears zero
    assert d["per_image"]["d2.png"] == -10.0


def test_a_change_that_is_not_in_the_same_direction_keeps_zero_inside_the_interval():
    """Four drawings down and one up is exactly the case a mean would hide."""
    before = _report({"curves": [100, 200, 300, 400, 500]})
    after = _report({"curves": [99, 196, 297, 398, 530]})
    lo, hi = compare_real(before, after)["delta"]["curves"]["ci95"]
    assert lo <= 0.0 <= hi, (lo, hi)


def test_reports_with_no_drawing_in_common_are_refused():
    a = _report({"curves": [1.0]})
    b = {"images": [{"image": "other.png", **{k: 0.0 for k in MEASURES}}], "options": {}}
    with pytest.raises(ValueError):
        compare_real(a, b)


def test_the_interval_is_reproducible_and_brackets_the_median():
    v = np.array([-3.0, -1.0, -1.0, 0.0, 5.0])
    lo, hi = bootstrap_ci(v)
    assert bootstrap_ci(v) == (lo, hi)  # seeded
    assert lo <= float(np.median(v)) <= hi
    assert np.isnan(bootstrap_ci(np.array([1.0]))).all()  # one drawing says nothing


# ---------------------------------------------------------------------------
# --set, and the whole command
# ---------------------------------------------------------------------------


def test_set_parses_against_the_real_field_types():
    p = pytest.importorskip("argparse").ArgumentParser()
    assert _options(p, ["local_trim=false"]) == {"local_trim": False}
    assert _options(p, ["fit_tolerance=0.5", "corner_angle=70"]) == {"fit_tolerance": 0.5, "corner_angle": 70.0}
    assert _options(p, ["max_gap=none"]) == {"max_gap": None}
    with pytest.raises(SystemExit):
        _options(p, ["no_such_field=1"])
    with pytest.raises(SystemExit):
        _options(p, ["local_trim=maybe"])


def test_the_command_writes_a_report_that_compares_with_itself_to_zero(tmp_path):
    folder = _folder(tmp_path, n=2)
    out = tmp_path / "a.json"
    assert main(["--realset", str(folder), "--tolerance", "2.0", "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["image"] for r in report["images"]] == ["d0.png", "d1.png"]
    assert all(k in report["images"][0] for k in MEASURES)
    assert report["median"]["curves"] > 0
    # every metric but the clock is settled by the drawing, so the same run twice is identical
    same = compare_real(report, report)["delta"]
    assert all(same[k]["median"] == 0.0 for k in MEASURES)


def test_a_traced_drawing_draws_about_as_much_line_as_there_is_ink(tmp_path):
    """length_per_ink is the no-ground-truth form of the length ratio: ~1 when each stretch of
    ink is drawn once, well above 1 when it is drawn twice."""
    report = eval_real(_folder(tmp_path, n=2), tolerance=2.0)
    for row in report["images"]:
        assert 0.8 < row["length_per_ink"] < 1.3, row


# ---------------------------------------------------------------------------
# G7 refuses a timing it cannot trust
# ---------------------------------------------------------------------------


def _subset(seconds_per_mp: float, spread: float) -> dict:
    """A gate-shaped subset result: only the timing matters for G7."""
    row = {k: 1.0 for k in ("bcubed_f", "bcubed_p", "f_gt2", "crossing_continuity", "gap_closure",
                            "joins_other_per100", "t_false_cont", "crossing_false_turn", "corner_p",
                            "corner_f", "curve_ratio")}
    row.update(seconds_per_mp=seconds_per_mp, seconds_spread=spread)
    return {"hard2": row}


def test_a_steady_timing_decides_g7_either_way():
    from line2func.eval import gate_decisions

    ref, cand = _subset(1.0, 0.01), _subset(1.10, 0.01)
    g7 = [c for c in gate_decisions(ref, cand) if c["check"].startswith("G7")]
    assert len(g7) == 1 and g7[0]["ok"] is True
    g7 = [c for c in gate_decisions(ref, _subset(1.30, 0.01)) if c["check"].startswith("G7")][0]
    assert g7["ok"] is False


def test_a_timing_whose_runs_disagree_is_reported_as_unmeasured_not_as_a_failure():
    """The margin is under a percent, and CPU contention has already produced an impossible
    result here, so a noisy run must not be allowed to read as a verdict."""
    from line2func.eval import TIMING_SPREAD, gate_decisions

    g7 = [c for c in gate_decisions(_subset(1.0, 0.01), _subset(1.30, TIMING_SPREAD + 0.01))
          if c["check"].startswith("G7")][0]
    assert g7["ok"] is None  # would have been a clear FAIL on the number alone
    assert g7["spread"] > TIMING_SPREAD


def test_one_timing_run_still_decides_g7_as_before():
    """repeat=1 leaves the spread unknown, and the gate behaves as it always has."""
    from line2func.eval import gate_decisions

    g7 = [c for c in gate_decisions(_subset(1.0, float("nan")), _subset(1.10, float("nan")))
          if c["check"].startswith("G7")][0]
    assert g7["ok"] is True


def test_repeating_a_timing_reports_how_far_the_runs_disagreed(tmp_path):
    from line2func.eval import eval_scenes
    from line2func.synth import write_scenes

    write_scenes(tmp_path / "s", "clean", 2, 64, seed=5)
    once = eval_scenes(tmp_path / "s", repeat=1)[(tmp_path / "s").name]
    twice = eval_scenes(tmp_path / "s", repeat=3)[(tmp_path / "s").name]
    assert once["repeat"] == 1 and np.isnan(once["seconds_spread"])
    assert twice["repeat"] == 3 and twice["seconds_spread"] >= 0.0
    assert once["f_gt2"] == twice["f_gt2"]  # repeating changes the clock, not the result

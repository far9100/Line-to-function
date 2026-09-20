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


def test_on_sparse_lines_the_curve_length_is_about_the_skeleton_length(tmp_path):
    report = eval_real(_folder(tmp_path, n=2), tolerance=2.0)
    for row in report["images"]:
        assert 0.8 < row["length_vs_skeleton"] < 1.3, row


def test_lighter_lines_push_it_above_one_because_the_two_sides_count_different_ink(tmp_path):
    """The denominator is one skeleton of the ink above Otsu's threshold; the tracer works below
    it and adds faint strokes below that again. So it draws lines the denominator never counted,
    and the ratio rises without anything being drawn twice.

    Pinned so the reading is not mistaken for redundancy a second time: on the real drawings it
    runs 1.01 to 1.64, and the densest falls to 1.29 with faint strokes turned off.
    """
    folder = tmp_path / "mixed"
    folder.mkdir()
    size = 128
    rows = [20, 40, 60], [80, 95, 110]
    dark, light = (render_lineart(
        CurveSet(size, size, [Curve(g.line([12, y], [size - 12, y]), stroke=i) for i, y in enumerate(ys)]),
        size, size, line_width=2.0).astype(np.float64) for ys in rows)
    # the second set drawn lightly: the tracer still traces it, Otsu's threshold does not see it
    save_png(folder / "mixed.png",
             np.minimum(dark, 255.0 - (255.0 - light) * 0.30).astype(np.uint8))
    row = eval_real(folder, tolerance=2.0)["images"][0]
    assert row["length_vs_skeleton"] > 1.5, row  # six lines drawn, about three in the denominator
    assert row["kept"] > 0.95 and row["precision"] > 0.95, row  # and none of it invented


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


# ---------------------------------------------------------------------------
# What the comparison could resolve
# ---------------------------------------------------------------------------


def test_an_interval_containing_zero_comes_with_what_it_could_have_resolved():
    """'No difference found' has to be distinguishable from 'too few drawings to see one'."""
    before = _report({"curves": [100, 200, 300, 400, 500]})
    after = _report({"curves": [95, 215, 290, 430, 480]})  # scattered, median small
    d = compare_real(before, after)["delta"]["curves"]
    lo, hi = d["ci95"]
    assert lo <= 0.0 <= hi  # nothing found
    assert d["halfwidth"] == pytest.approx(0.5 * (hi - lo))
    assert d["sd"] > 0 and d["n_for"] > 5  # ... and it says how many drawings it would take


def test_the_same_difference_on_every_drawing_needs_only_a_couple():
    before = _report({"curves": [100, 200, 300, 400, 500]})
    after = _report({"curves": [90, 190, 290, 390, 490]})
    d = compare_real(before, after)["delta"]["curves"]
    assert d["sd"] == 0.0 and d["n_for"] == 2


def test_no_difference_at_all_cannot_name_a_sample_size():
    r = _report({"curves": [100, 200, 300]})
    d = compare_real(r, r)["delta"]["curves"]
    assert d["median"] == 0.0 and d["n_for"] is None and d["halfwidth"] == 0.0


def test_the_needed_count_grows_with_scatter_and_shrinks_with_the_effect():
    from line2func.eval import _n_for

    assert _n_for(10.0, 10.0) > _n_for(10.0, 2.0)
    assert _n_for(1.0, 5.0) > _n_for(10.0, 5.0)
    assert _n_for(0.0, 1.0) is None and _n_for(float("nan"), 1.0) is None


def test_a_measure_missing_from_an_older_report_is_skipped_not_fatal():
    before = _report({"curves": [1.0, 2.0]})
    after = _report({"curves": [1.0, 2.0]})
    for row in before["images"]:
        del row["length_vs_skeleton"]
    delta = compare_real(before, after)["delta"]
    assert "curves" in delta and "length_vs_skeleton" not in delta


# ---------------------------------------------------------------------------
# The three paths that trace and then judge must judge alike
# ---------------------------------------------------------------------------


def test_every_path_judges_at_the_same_threshold(tmp_path, monkeypatch):
    """`demo --quality`, the web app's jobs.run_trace and eval --realset each trace and then call
    quality.assess. They pass the threshold differently - demo and jobs hand theirs straight
    through, eval resolves it first - and all three have to land on Otsu's when none is given, or
    the same drawing scores differently depending on which one measured it.

    Pinned instead of extracting a shared helper: what the three share is two calls inside
    genuinely different surroundings, and a helper taking the union of their parameters would not
    stop a fourth from diverging.
    """
    from line2func import jobs, lineart, quality

    seen = []
    real_assess = quality.assess

    def spy(curves, ink, threshold=None, **kw):
        seen.append(quality.tracer_threshold(ink, threshold))
        return real_assess(curves, ink, threshold, **kw)

    monkeypatch.setattr(quality, "assess", spy)
    folder = _folder(tmp_path, n=1)
    eval_real(folder, tolerance=2.0)

    rgb = np.repeat(np.asarray(lineart.load_rgb(folder / "d0.png"))[:, :, 0, None], 3, axis=2)
    p = {"method": "none", "scale": 1.0,
         **jobs.trace_options({"curves": 500, "quality": True}, rgb.shape[1::-1], 1.0, 4_000_000)}
    jobs.run_trace(rgb, "d0.png", p, lambda stage: None, started=0.0)

    assert len(seen) == 2, seen
    assert seen[0] == pytest.approx(seen[1]), f"eval judged at {seen[0]}, the app at {seen[1]}"
    assert seen[0] == pytest.approx(quality.auto_threshold(lineart.extract(folder / "d0.png", "none")))

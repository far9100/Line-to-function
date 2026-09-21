"""The baseline engine and the synthetic generator give exactly the recorded output (see golden.py)."""

import json

import pytest

import golden


def _recorded(path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    env = data.pop("_env", None)
    if env != golden.environment():
        pytest.skip(f"digests were recorded with {env}; this is {golden.environment()} (re-record with golden.py)")
    return data


def _check(recorded: dict, current: dict) -> None:
    changed = sorted(k for k in current if recorded.get(k) != current[k])
    missing = sorted(k for k in recorded if k not in current)
    assert not changed, f"output changed for: {', '.join(changed)}"
    assert not missing or all(k.startswith("real") for k in missing), f"not recomputed: {missing}"


def test_baseline_output_is_unchanged():
    _check(_recorded(golden.BASELINE_FILE), golden.baseline_digests())


def test_real_drawing_output_is_unchanged():
    recorded = _recorded(golden.BASELINE_FILE)
    current = golden.real_digests()
    if not current:
        pytest.skip("the real test drawing is not in this checkout")
    _check({k: v for k, v in recorded.items() if k.startswith("real")}, current)


def test_synthetic_data_is_unchanged():
    _check(_recorded(golden.SYNTH_FILE), golden.synth_digests())


def test_recording_without_the_real_drawing_keeps_its_digests(tmp_path, monkeypatch):
    """Re-recording used to drop them silently, and no test said so: this one does."""
    recorded = {"_env": golden.environment(), "line": "abc", "real1_1x": "deadbeef", "real1_2x": "cafe"}
    path = tmp_path / "baseline_digests.json"
    path.write_text(json.dumps(recorded), encoding="utf-8")
    monkeypatch.setattr(golden, "BASELINE_FILE", path)
    monkeypatch.setattr(golden, "SYNTH_FILE", tmp_path / "synth_digests.json")
    monkeypatch.setattr(golden, "DATA", tmp_path)
    monkeypatch.setattr(golden, "baseline_digests", lambda: {"line": "xyz"})
    monkeypatch.setattr(golden, "synth_digests", dict)
    monkeypatch.setattr(golden, "real_digests", dict)  # as if the drawing were not in this checkout

    monkeypatch.setattr(golden.sys, "argv", ["golden.py", "--write"])
    assert golden.main() == 0
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["line"] == "xyz"  # what could be recomputed is rewritten
    assert written["real1_1x"] == "deadbeef" and written["real1_2x"] == "cafe"  # the rest is kept

    # and when the drawing *is* there, its fresh digests win
    monkeypatch.setattr(golden, "real_digests", lambda: {"real1_1x": "new"})
    assert golden.main() == 0
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["real1_1x"] == "new" and "real1_2x" not in written

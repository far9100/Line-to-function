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

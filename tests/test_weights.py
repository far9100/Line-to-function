import hashlib
import io

import numpy as np
import pytest

from line2func import weights


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    payload = b"pretend weights" * 1000
    monkeypatch.setenv("LINE2FUNC_HOME", str(tmp_path / "cache"))
    monkeypatch.setitem(weights.WEIGHTS, "fake", {
        "file": "fake.pth", "url": "https://example.invalid/fake.pth",
        "sha256": hashlib.sha256(payload).hexdigest(), "about": "test", "license": "MIT",
    })

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    served = {"data": payload}
    monkeypatch.setattr(weights.urllib.request, "urlopen", lambda req, timeout=0: Response(served["data"]))
    return served


def test_fetch_verifies_and_caches(fake_registry):
    path = weights.fetch("fake", quiet=True)
    assert path.is_file() and weights.require("fake") == path
    # tampering is detected
    path.write_bytes(b"evil")
    with pytest.raises(weights.WeightsError):
        weights.require("fake")


def test_fetch_rejects_wrong_hash(fake_registry):
    fake_registry["data"] = b"something else"
    with pytest.raises(weights.WeightsError, match="mismatch"):
        weights.fetch("fake", quiet=True)
    assert not weights.path_for("fake").exists()
    assert not weights.path_for("fake").with_name("fake.pth.part").exists()


def test_require_explains_how_to_download(tmp_path, monkeypatch):
    monkeypatch.setenv("LINE2FUNC_HOME", str(tmp_path))
    with pytest.raises(weights.WeightsError, match="line2func.weights fetch informative"):
        weights.require("informative")
    with pytest.raises(weights.WeightsError):
        weights.path_for("nope")


def test_informative_model_if_downloaded():
    pytest.importorskip("torch")
    try:
        weights.require("informative")
    except weights.WeightsError:
        pytest.skip("informative weights not downloaded")
    from line2func.lineart_model import extract

    rgb = np.full((70, 90, 3), 230, np.uint8)
    rgb[20:50, 30:60] = 40  # a dark square: its outline should come out as ink
    ink = extract(rgb, "informative", device="cpu")
    assert ink.shape == (70, 90) and ink.dtype == np.float32
    edge = np.zeros_like(ink, bool)
    edge[18:22, 28:62] = edge[48:52, 28:62] = True
    assert ink[edge].mean() > 3 * ink[:10, :10].mean()

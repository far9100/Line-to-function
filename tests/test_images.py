"""Decoding uploaded images (lineart.open_image) and guessing line art vs photo (lineart.suggest_mode)."""

import io

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

from line2func import lineart
from line2func.curves import Curve, CurveSet
from line2func.render import render_lineart


def _encode(arr: np.ndarray, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()


def test_bytes_files_and_paths_decode_alike(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    data = _encode(rgb)
    path = tmp_path / "a.png"
    path.write_bytes(data)
    for src in (data, io.BytesIO(data), path, str(path)):
        np.testing.assert_array_equal(lineart.open_image(src), rgb)
    np.testing.assert_array_equal(lineart.load_rgb(data), rgb)


def test_exif_orientation_is_applied():
    rgb = np.zeros((20, 40, 3), np.uint8)
    rgb[:, :20] = 255  # left half white
    exif = Image.Exif()
    exif[0x0112] = 6  # shown rotated 90 degrees clockwise, as phone cameras often store photos
    out = lineart.open_image(_encode(rgb, "JPEG", exif=exif, quality=95))
    assert out.shape == (40, 20, 3)
    assert out[:20].mean() > 200 and out[20:].mean() < 50  # the left half is now on top


def test_transparency_is_composited_onto_white():
    rgba = np.zeros((10, 10, 4), np.uint8)  # transparent black ...
    rgba[5:, :, 3] = 255  # ... except the opaque bottom half
    out = lineart.open_image(_encode(rgba))
    assert out[:5].min() == 255 and out[5:].max() == 0


def test_16_bit_grayscale_is_scaled_not_clipped():
    gray = np.full((8, 8), 30000, np.uint16)
    gray[:, 4:] = 65535
    out = lineart.open_image(_encode(gray))
    assert abs(int(out[0, 0, 0]) - round(30000 / 257)) <= 1 and out[0, 7, 0] == 255


def test_size_limits():
    rgb = np.full((100, 200, 3), 200, np.uint8)
    with pytest.raises(lineart.ImageTooLarge):
        lineart.open_image(_encode(rgb), max_pixels=10_000)
    assert lineart.open_image(_encode(rgb), max_side=50).shape == (25, 50, 3)
    # a big JPEG is decoded at a reduced scale, so it fits the pixel limit
    assert lineart.open_image(_encode(rgb, "JPEG"), max_side=50, max_pixels=10_000).shape == (25, 50, 3)


def test_garbage_is_not_an_image():
    with pytest.raises(UnidentifiedImageError):
        lineart.open_image(b"this is not an image")


def _drawing(size: int = 512, width: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    curves = [Curve(rng.uniform(20, size - 20, (4, 2)), stroke=i) for i in range(12)]
    return render_lineart(CurveSet(size, size, curves), size, size, line_width=width)


def test_suggest_mode():
    lines = _drawing()
    assert lineart.suggest_mode(lines) == "lineart"
    assert lineart.suggest_mode(255 - lines) == "lineart"  # light lines on dark paper
    assert lineart.suggest_mode(np.stack([np.full_like(lines, 255), lines, lines], axis=2)) == "lineart"  # red ink
    assert lineart.suggest_mode(np.full((50, 50, 3), 255, np.uint8)) == "lineart"  # blank page

    yy, xx = np.mgrid[0:300, 0:400]
    noise = np.random.default_rng(1).normal(0, 12, (300, 400, 3))
    photo = np.stack([xx / 400 * 255, yy / 300 * 255, np.full(xx.shape, 120.0)], axis=2) + noise
    assert lineart.suggest_mode(np.clip(photo, 0, 255).astype(np.uint8)) == "photo"
    painting = np.full((300, 400, 3), 255, np.uint8)  # flat color fills on white
    painting[50:250, 50:200] = (200, 60, 60)
    painting[100:280, 220:380] = (40, 90, 200)
    assert lineart.suggest_mode(painting) == "photo"

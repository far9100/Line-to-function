"""Turn an input image into an ink map for the tracer.

Every method returns an **ink map**: float32 in ``[0, 1]``, shape ``(H, W)``,
where 1 means "line" and 0 means "paper".

* ``none``  - the input already is line art (dark lines on light paper, or the
  reverse, which is detected and inverted).
* ``canny`` - Canny edge detection. Thick lines produce an edge on each side.
* ``xdog``  - eXtended Difference of Gaussians (Winnemöller et al., 2012), edge
  term only, which gives sketch-like lines from photos.
* ``informative`` / ``informative-coarse`` - pretrained line-art network
  (Informative Drawings, Chan et al., CVPR 2022, MIT); needs PyTorch and
  ``python -m line2func.weights fetch informative``. See :mod:`line2func.lineart_model`.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage

METHODS = ("none", "canny", "xdog", "informative", "informative-coarse")


class ImageTooLarge(ValueError):
    """The image has more pixels than the caller allows."""


def _pil_to_rgb(im) -> np.ndarray:
    """A loaded PIL image as RGB uint8: alpha composited onto white, 16-bit and float scaled to 8 bits."""
    mode = im.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        # convert("RGB") would clip these to white
        arr = np.nan_to_num(np.asarray(im, dtype=np.float64))
        if mode == "F":
            arr = arr * (255.0 if arr.size and arr.max() <= 1.0 else 1.0)
        elif mode.startswith("I;16") or (arr.size and arr.max() > 255):
            arr = arr / 257.0
        gray = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if mode in ("RGBA", "LA", "PA", "RGBa", "La") or (mode == "P" and "transparency" in im.info):
        rgba = np.asarray(im.convert("RGBA")).astype(np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        return np.rint(rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return np.asarray(im.convert("RGB"))


def open_image(src, *, max_side: int | None = None, max_pixels: int | None = None) -> np.ndarray:
    """Decode an image file (path, encoded bytes or binary file object) to an RGB uint8 array.

    EXIF orientation is applied, so photos come out the way image viewers show
    them. Transparent pixels are composited onto white; 16-bit images are scaled
    to 8 bits. ``max_side`` shrinks larger images (Lanczos; big JPEGs are decoded
    at a reduced scale directly). An image that still has more than
    ``max_pixels`` pixels raises :class:`ImageTooLarge` before it is decoded.
    """
    from PIL import Image, ImageOps

    if isinstance(src, (bytes, bytearray, memoryview)):
        src = io.BytesIO(bytes(src))
    with warnings.catch_warnings():
        if max_pixels is not None:  # the caller's limit replaces Pillow's decompression-bomb warning
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        try:
            im = Image.open(src)
        except Image.DecompressionBombError as exc:
            raise ImageTooLarge(str(exc)) from None
    with im:
        w, h = im.size
        if max_side and im.format == "JPEG" and max(w, h) > max_side:
            f = max_side / max(w, h)
            im.draft(None, (max(1, int(w * f)), max(1, int(h * f))))  # 1/2, 1/4 or 1/8 scale, never below
            w, h = im.size
        if max_pixels is not None and w * h > max_pixels:
            raise ImageTooLarge(f"{w}x{h} is {w * h / 1e6:.0f} MP, more than {max_pixels / 1e6:.0f} MP")
        im.load()
        rgb = _pil_to_rgb(ImageOps.exif_transpose(im))
    if max_side and max(rgb.shape[:2]) > max_side:
        h, w = rgb.shape[:2]
        f = max_side / max(h, w)
        size = (max(1, round(w * f)), max(1, round(h * f)))
        rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))
    return np.ascontiguousarray(rgb)


def load_rgb(image) -> np.ndarray:
    """Path, encoded bytes, binary file or array -> RGB uint8 array.

    Files go through :func:`open_image` (EXIF orientation applied, transparent
    pixels composited onto white).
    """
    if isinstance(image, (str, Path, bytes, bytearray, memoryview)) or hasattr(image, "read"):
        return open_image(image)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


def to_gray(image) -> np.ndarray:
    """Luminance in ``[0, 1]`` (float32)."""
    rgb = load_rgb(image).astype(np.float32) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold for values in ``[0, 1]``; 0.5 if the values are constant."""
    hist, edges = np.histogram(np.ravel(values), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * centers)
    mu_total = m0[-1]
    valid = (w0 > 0) & (w1 > 0)
    if not np.any(valid):
        return 0.5
    between = np.zeros_like(w0)
    between[valid] = (mu_total * w0[valid] - total * m0[valid]) ** 2 / (w0[valid] * w1[valid])
    return float(edges[int(np.argmax(between)) + 1])


def _plain(gray: np.ndarray) -> np.ndarray:
    ink = 1.0 - gray
    # Mostly-dark images are light lines on dark paper: invert.
    if np.median(gray) < 0.5:
        ink = gray.copy()
    return ink.astype(np.float32)


def canny(gray: np.ndarray, sigma: float = 1.4, low: float | None = None, high: float | None = None) -> np.ndarray:
    """Binary Canny edges as an ink map. Thresholds default to data-driven values."""
    g = ndimage.gaussian_filter(gray.astype(np.float64), sigma)
    gx = ndimage.sobel(g, axis=1)
    gy = ndimage.sobel(g, axis=0)
    mag = np.hypot(gx, gy)

    # non-maximum suppression along the gradient, quantized to 4 directions
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    # gradient direction bins: 0 = along x, 1 = (+x, +y), 2 = along y, 3 = (-x, +y)
    q = np.digitize(angle, [22.5, 67.5, 112.5, 157.5]) % 4
    pad = np.pad(mag, 1)
    h, w = mag.shape
    c = (slice(1, h + 1), slice(1, w + 1))

    def shifted(dy: int, dx: int) -> np.ndarray:
        return pad[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]

    offsets = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros_like(mag, dtype=bool)
    for k, (dy, dx) in offsets.items():
        sel = q == k
        keep |= sel & (pad[c] >= shifted(dy, dx)) & (pad[c] >= shifted(-dy, -dx))
    nms = np.where(keep, mag, 0.0)

    peak = float(nms.max())
    if peak <= 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    if high is None:
        # Otsu on the normalized magnitudes separates noise ridges from real edges
        high = otsu_threshold(nms[nms > 0] / peak) * peak
    if low is None:
        low = 0.5 * high
    weak = nms >= low
    labels, _ = ndimage.label(weak, structure=np.ones((3, 3)))
    strong_labels = np.unique(labels[nms >= high])
    edges = np.isin(labels, strong_labels[strong_labels > 0])
    return edges.astype(np.float32)


def xdog(
    gray: np.ndarray,
    sigma: float = 0.8,
    k: float = 1.6,
    epsilon: float = 0.02,
    phi: float = 50.0,
) -> np.ndarray:
    """XDoG lines as an ink map.

    Classic XDoG thresholds ``G_sigma + p (G_sigma - G_{k sigma})``, whose first
    (tone) term turns dark flat areas into ink. For line extraction only the
    difference of Gaussians is kept, so flat areas of any brightness stay
    blank; its dark side is soft-thresholded: ``ink = tanh(phi (-DoG - epsilon))``.
    """
    g = gray.astype(np.float64)
    dog = ndimage.gaussian_filter(g, sigma) - ndimage.gaussian_filter(g, k * sigma)
    return np.clip(np.tanh(phi * (-dog - epsilon)), 0.0, 1.0).astype(np.float32)


def extract(image, method: str = "none") -> np.ndarray:
    """Ink map of ``image`` (path or array) using ``method`` (one of :data:`METHODS`)."""
    if method in ("informative", "informative-coarse"):
        from line2func import lineart_model

        return lineart_model.extract(load_rgb(image), method)
    gray = to_gray(image)
    if method == "none":
        return _plain(gray)
    if method == "canny":
        return canny(gray)
    if method == "xdog":
        return xdog(gray)
    raise ValueError(f"unknown line-art method {method!r}; choose from {', '.join(METHODS)}")


def suggest_mode(image) -> str:
    """Guess whether ``image`` is line art (``"lineart"``) or a photo / painting (``"photo"``).

    Judged on a box-filtered thumbnail of at most 384 px. Line art is mostly
    blank paper, and what is drawn is thin, so almost every drawn pixel lies
    next to paper. Photos have little blank area; paintings and filled
    illustrations have large drawn areas far from any paper. It is only a
    suggestion for the user interface, which lets the user override it.
    """
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(load_rgb(image)))
    factor = int(np.ceil(max(img.size) / 384))
    if factor > 1:
        img = img.reduce(factor)  # box average: thin lines stay as light strokes
    small = np.asarray(img, dtype=np.float32) / 255.0
    gray = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    paper = np.percentile(gray, 10 if np.median(gray) < 0.5 else 90)  # light-on-dark drawings too
    drawn = np.abs(gray - paper) >= 0.12
    if not drawn.any():
        return "lineart"
    blank = 1.0 - float(drawn.mean())
    solid = float(np.mean(ndimage.distance_transform_edt(drawn)[drawn] > 2.0))
    return "lineart" if blank >= 0.5 and solid <= 0.35 else "photo"

"""Turn an input image into an ink map for the tracer.

Every method returns an **ink map**: float32 in ``[0, 1]``, shape ``(H, W)``,
where 1 means "line" and 0 means "paper".

* ``none``  - the input already is line art (dark lines on light paper, or the
  reverse, which is detected and inverted).
* ``canny`` - Canny edge detection. Thick lines produce an edge on each side.
* ``xdog``  - eXtended Difference of Gaussians (Winnemöller et al., 2012), edge
  term only, which gives sketch-like lines from photos.
* ``flow``  - coherent line drawing (Kang, Lee & Chui, NPAR 2007): a difference
  of Gaussians taken across the edge tangent flow and smoothed along it, which
  gives long connected lines instead of speckle. **The default for photos**,
  and the only good one that needs nothing downloaded.
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

METHODS = ("none", "canny", "xdog", "flow", "informative", "informative-coarse")
MODEL_METHODS = ("informative", "informative-coarse")  # need PyTorch and downloaded weights
PURE_METHODS = tuple(m for m in METHODS if m not in MODEL_METHODS)  # what a browser can run


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


# The flow extractor's fixed settings; the three the detail slider moves are in DETAIL_TUNED below.
FLOW_SIGMA_G = 1.0  # px: gradient smoothing for the structure tensor the flow is read from
FLOW_SIGMA_C = 5.2  # px: how far the flow is smoothed, = 3.0 * sqrt(3) (see edge_tangent_flow)
FLOW_K = 1.6  # the DoG's second sigma, as a multiple of the first (as in xdog)
FLOW_PHI = 40.0  # steepness of the soft threshold, on gray in [0, 1] (xdog uses 50)
FLOW_ITERS = 3  # FDoG rounds: each one draws the lines found so far onto the original and looks again

DETAIL = 50.0  # default line-art detail, 0..100 (see detail_params)
# The flow settings the detail slider moves: as tuned (50), at its coarsest (0) and at its finest (100).
DETAIL_TUNED = {"sigma_e": 1.00, "sigma_m": 2.60, "tau": 0.985}
DETAIL_COARSE = {"sigma_e": 1.80, "sigma_m": 3.80, "tau": 0.950}
DETAIL_FINE = {"sigma_e": 0.80, "sigma_m": 2.20, "tau": 0.997}


def detail_params(detail: float | None = None) -> dict:
    """The :func:`flow` settings for a line-art detail of 0..100 (``None`` means :data:`DETAIL`).

    :data:`DETAIL` (50) gives the tuned values (:data:`DETAIL_TUNED`); from there the settings move to
    :data:`DETAIL_FINE` at 100 and to :data:`DETAIL_COARSE` at 0. All three move together because they
    are three views of one decision - how weak a ridge still counts as a line. Finer means a narrower
    kernel across the flow (so lines close together stay apart), a shorter run along it (less evidence
    demanded that a ridge carries on), and a ``tau`` nearer 1 (the DoG's centre and surround nearly
    cancel, so a weaker ridge survives).

    **``tau`` is moved in proportion, not linearly**, because what it does is set by how far it is from
    1, and that margin spans 0.050 to 0.003 - an order of magnitude - across the slider. Interpolated
    straight, the fine half of the slider barely moves ``tau`` while the two sigmas drop quickly, and
    the result is *less* line at 60 than at 50. Measured on three pictures, over 0, 10, ... 100, the
    proportional form gives more line at every step and the straight one does not.

    Two consequences worth knowing. The narrower kernels of a high detail are also **cheaper**, so the
    slowest setting is 0, not 100. And the fine end is deliberately tame - 100 gives about one and a
    half to two times the line of the tuned setting, not ten times - because past that what comes in is
    mostly texture, and the curve budget goes on it.
    """
    if detail is None:
        detail = DETAIL
    if not 0.0 <= detail <= 100.0:
        raise ValueError("line-art detail must be between 0 and 100")
    other = DETAIL_FINE if detail >= DETAIL else DETAIL_COARSE
    a = abs(detail - DETAIL) / DETAIL
    out = {k: (1.0 - a) * v + a * other[k] for k, v in DETAIL_TUNED.items() if k != "tau"}
    margin = (1.0 - DETAIL_TUNED["tau"]) ** (1.0 - a) * (1.0 - other["tau"]) ** a
    out["tau"] = 1.0 - margin
    return out


def edge_tangent_flow(gray: np.ndarray, sigma_g: float = FLOW_SIGMA_G,
                      sigma_c: float = FLOW_SIGMA_C) -> tuple[np.ndarray, np.ndarray]:
    """The edge tangent flow of ``gray``: unit vectors ``(tx, ty)`` along the local feature direction.

    The flow is the minor eigenvector of the smoothed structure tensor, which points along an edge
    rather than across it. Kang et al. build the same field by smoothing the tangents over several
    rounds; **rounds of un-normalized linear smoothing collapse** - three passes of sigma 3.0 are one
    pass of sigma 3 * sqrt(3) - so this is one separable Gaussian (:data:`FLOW_SIGMA_C`).

    It is a direction without a sign: ``(tx, ty)`` and ``(-tx, -ty)`` mean the same thing, and which
    one comes out flips along a curve. Everything that reads the flow has to be even in it, which is
    why :func:`_directional` samples symmetrically.
    """
    g = gray.astype(np.float64)
    gx = ndimage.gaussian_filter(g, sigma_g, order=(0, 1))
    gy = ndimage.gaussian_filter(g, sigma_g, order=(1, 0))
    e = ndimage.gaussian_filter(gx * gx, sigma_c)
    f = ndimage.gaussian_filter(gx * gy, sigma_c)
    h = ndimage.gaussian_filter(gy * gy, sigma_c)
    # the smaller eigenvalue of [[e, f], [f, h]], and (f, lam - e), the eigenvector that goes with it
    lam = 0.5 * (e + h - np.hypot(e - h, 2.0 * f))
    tx, ty = f, lam - e
    size = np.hypot(tx, ty)
    flat = size < 1e-12  # no edge here at all: any direction will do, and none of them gets any weight
    size = np.where(flat, 1.0, size)
    return np.where(flat, 1.0, tx / size), np.where(flat, 0.0, ty / size)


def _offsets(reach: float) -> np.ndarray:
    """Whole-pixel offsets from ``-reach`` to ``reach``, rounded outwards."""
    r = int(np.ceil(reach))
    return np.arange(-r, r + 1, dtype=np.float64)


def _gaussian_weights(offsets: np.ndarray, sigma: float) -> np.ndarray:
    w = np.exp(-0.5 * (offsets / sigma) ** 2)
    return w / w.sum()


def _directional(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, offsets: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """``sum_i weights[i] * img(p + offsets[i] * (dx, dy))`` at every pixel ``p``, sampled bilinearly.

    One straight line of samples per pixel, aimed along that pixel's own ``(dx, dy)``. The weights are
    symmetric, so the answer does not depend on which way round the direction points - which is what
    makes this safe to use on the unsigned flow of :func:`edge_tangent_flow`. Following a curved
    streamline instead would have to pick a way round at every step and would double back wherever the
    sign flips; the straight run is one gather per offset, and :func:`flow_dog` recovers the curve by
    re-aiming on each of its rounds.

    Samples that fall outside the picture come from an **odd reflection**, which carries the slope
    across the border instead of flattening it. Repeating the edge pixel would leave a kink there that
    a difference of Gaussians reads as a line, and a photo would come out framed: measured on a plain
    gradient, clamping drew a 0.019 line down the first column, odd reflection draws exactly nothing.
    """
    h, w = img.shape
    pad = int(np.ceil(np.abs(offsets).max())) + 1
    wide = np.pad(img, pad, mode="reflect", reflect_type="odd")
    rows, cols = np.ogrid[pad:h + pad, pad:w + pad]
    coords = np.empty((2, h, w), dtype=np.float64)
    got = np.empty((h, w), dtype=np.float64)
    acc = np.zeros((h, w), dtype=np.float64)
    for offset, weight in zip(offsets, weights):
        if offset == 0.0:
            acc += weight * img
            continue
        np.multiply(dy, offset, out=coords[0])
        coords[0] += rows
        np.multiply(dx, offset, out=coords[1])
        coords[1] += cols
        ndimage.map_coordinates(wide, coords, order=1, mode="nearest", output=got)
        np.multiply(got, weight, out=got)
        acc += got
    return acc


def flow_dog(gray: np.ndarray, tx: np.ndarray, ty: np.ndarray, sigma_e: float = 1.0,
             sigma_m: float = 2.6, tau: float = 0.985, k: float = FLOW_K, phi: float = FLOW_PHI,
             iterations: int = FLOW_ITERS) -> np.ndarray:
    """Flow-based difference of Gaussians as an ink map, given the flow from :func:`edge_tangent_flow`.

    Each round does two passes: a difference of Gaussians **across** the flow, which asks "is there a
    dark ridge here", and a Gaussian **along** it, which asks the neighbours on the same line whether
    they agree. A speck of noise has no line to agree with it and is averaged away; a real line is
    reinforced from both ends. The lines found are then drawn onto the original and the round runs
    again, which closes the places a single pass leaves dotted.

    The two Gaussians across the flow are folded into one kernel (``w_e - tau * w_k``) before any
    sampling, which halves the work. ``tau`` below 1 leaves a margin that flat and smoothly shaded
    areas cannot cross, so a sky or a cheek contributes exactly nothing: a difference of Gaussians is
    zero on anything linear, and what is left, ``(1 - tau) * gray``, has the wrong sign to be ink.
    """
    original = gray.astype(np.float64)
    img = original
    across = _offsets(2.0 * k * sigma_e)
    w_across = _gaussian_weights(across, sigma_e) - tau * _gaussian_weights(across, k * sigma_e)
    along = _offsets(2.0 * sigma_m)
    w_along = _gaussian_weights(along, sigma_m)
    nx, ny = -ty, tx  # across the flow
    ink = np.zeros_like(original)
    for _ in range(iterations):
        ridge = _directional(img, nx, ny, across, w_across)
        agreed = _directional(ridge, tx, ty, along, w_along)
        ink = np.tanh(phi * np.maximum(-agreed, 0.0))
        img = np.minimum(original, 1.0 - ink)
    return np.clip(ink, 0.0, 1.0).astype(np.float32)


def flow(gray: np.ndarray, detail: float | None = None) -> np.ndarray:
    """Coherent line drawing as an ink map: the flow, then :func:`flow_dog` at ``detail`` (0..100)."""
    tx, ty = edge_tangent_flow(gray)
    return flow_dog(gray, tx, ty, **detail_params(detail))


def extract(image, method: str = "none", detail: float | None = None) -> np.ndarray:
    """Ink map of ``image`` (path or array) using ``method`` (one of :data:`METHODS`).

    ``detail`` is the line-art detail, 0..100 (:data:`DETAIL` when it is ``None``). Only ``flow``
    reads it; the other methods have nothing it would mean.
    """
    if method in MODEL_METHODS:
        from line2func import lineart_model

        return lineart_model.extract(load_rgb(image), method)
    gray = to_gray(image)
    if method == "none":
        return _plain(gray)
    if method == "canny":
        return canny(gray)
    if method == "xdog":
        return xdog(gray)
    if method == "flow":
        return flow(gray, detail)
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


def suggest_method(image) -> str:
    """The line-art method to offer for ``image``: ``"flow"`` for a photo, ``"none"`` for line art.

    The rule the two web pages open an image with (:func:`line2func.jobs.read_image` applies it to the
    mode it has already read), so their automatic choice cannot drift apart. It follows
    :func:`suggest_mode`, and is only a suggestion: the pages let the user pick another method, and the
    command line does not guess at all - there ``--lineart`` says which.
    """
    return "flow" if suggest_mode(image) == "photo" else "none"

"""Baseline vectorizer: ink map -> skeleton -> stroke graph -> cubic curves.

Pipeline
--------
1. **Binarize** the ink map (Otsu by default; :func:`line2func.pipeline.trace`
   lowers it for line art, keeping ``reference_threshold`` for the line width
   and the faint-stroke test), add thin faint strokes below the threshold (a
   local-contrast test, :func:`faint_line_mask`) and, with
   ``very_faint_lines``, the faintest ones that are clearly lines
   (:func:`very_faint_line_mask`), drop specks smaller than a few line widths
   (with ``join_widths``, on clean paper, pieces close to other ink are judged
   with it, so a line broken into dots stays), and fill pinholes inside lines.
2. **Filled areas**: parts much thicker than the typical line are cut out and
   only their outline is traced (tagged ``fill_outline``). With
   ``solid_ratio`` set, so is ink that is merely a few line widths thick over
   some length, such as a heavy eyelash (:func:`solid_areas`).
3. **Thin** the remaining lines to a 1-px skeleton (Guo-Hall / Lam-Lee-Suen
   thinning, as in ``skimage.morphology.thin``).
4. **Graph**: skeleton pixels become a graph whose nodes are end points and
   junctions and whose edges are the pixel chains between them.
5. **Clean up**: prune short spurs that thinning leaves on thick lines, and
   merge pairs of nearby junctions (two lines crossing at a shallow angle thin
   into two T-junctions joined by a short edge).
6. **Gap closing**: pairs of end points that are close and point at each other
   are linked, so a stroke with a small break stays one stroke.
7. **Strokes**: at every junction, edges that continue each other smoothly
   are paired, so a line crossing another line stays one stroke.
8. **Fitting**: each stroke is resampled, split at sharp corners, lightly
   smoothed and fitted with Schneider's algorithm (:mod:`line2func.fit`).

Steps 5-8 each make a choice (shallow crossing, gap, junction pairing,
corner). A scorer makes them (:mod:`line2func.decisions`): the angle rules by
default here, or the learned scorer (:mod:`line2func.decision_model`), which
``pipeline.trace`` uses.

Each curve's ``confidence`` is the fraction of its length that lies on ink,
so curves that bridge a gap score lower.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from line2func.curves import Curve, CurveSet
from line2func.decisions import (
    RULES,
    CornerCand,
    CrossingCand,
    DecisionContext,
    GapCand,
    JunctionCand,
    _angle,
    _direction_from,
    exact_matching,
    greedy_pairs,
    pick_peaks,
)
from line2func.fit import fit_polyline
from line2func.geometry import evaluate
from line2func.lineart import otsu_threshold

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class BaselineParams:
    """Tuning knobs. ``None`` means "derive from the estimated line width"."""

    threshold: float | None = None  # ink threshold in [0, 1]; None = Otsu
    fit_tolerance: float = 1.0  # max distance of the fitted curve from the stroke, px
    min_component_area: float | None = None  # specks smaller than this are dropped, px²
    max_gap: float | None = None  # longest visible gap that is closed, px
    max_gap_widths: float = 4.0  # default max_gap, in estimated line widths
    gap_angle: float = 35.0  # max angle between an end's direction and the gap, degrees
    continue_angle: float = 45.0  # max bend when continuing a stroke through a junction
    corner_angle: float = 60.0  # turning angle that splits a stroke at a corner
    fill_ratio: float = 3.0  # blobs this many times thicker than a line are "filled"
    # ink at least solid_ratio line widths thick, along at least solid_length line widths, is a
    # solid area too (heavy eyelashes, brush wedges; see solid_areas). None = only fill_ratio blobs
    solid_ratio: float | None = None
    solid_length: float = 2.0
    smooth_sigma: float = 1.0  # smoothing of the pixel chain before fitting, px
    # measure the distorted zone around each junction on the junction itself (the disk inscribed in the
    # ink there) instead of using half the drawing's typical line width everywhere, which matters when
    # one drawing holds both thin and thick lines. It gives about 2-5% fewer curves at the same
    # accuracy, for a hair less raster fidelity (docs/progress.md); off is the behaviour up to 1.0
    local_trim: bool = False
    # faint strokes below the threshold (local-contrast ridge test, see faint_line_mask)
    faint_lines: bool = True
    faint_contrast: float = 0.3  # min contrast over the local background, as a fraction of the threshold
    faint_noise_sigmas: float = 4.0  # ... and at least this many times the paper's pixel noise
    faint_min_length: float = 15.0  # faint strokes shorter than this (px) are noise
    faint_band: float = 2.0  # px (plus half a line width) around strong lines where no faint strokes are looked for
    # specks and faint pieces are judged together with the pieces within this many line widths of them, so a
    # line that broke into dots and dashes stays (it is long as a whole) while lone specks go; 0 = each alone.
    # Only on clean paper, whose pixel noise (paper_noise) is below join_max_noise: on noisy paper, chains of
    # noise specks would pass as lines
    join_widths: float = 0.0
    join_max_noise: float = 0.005
    # strength of the noise filters, as a multiple of their tuned limits (the derived speck area and
    # faint_min_length; line2func.residual scales its minimum length too): 1 = as tuned, 0 = keep every speck
    # and faint piece, 2 = twice as strict
    denoise: float = 1.0
    # strokes fainter still, down to the paper's own noise, when they are clearly lines (see very_faint_lines):
    # at least this long, at most this wide, and at most one junction per this much line - in line widths
    very_faint_lines: bool = False
    very_faint_length: float = 15.0
    very_faint_width: float = 2.5
    very_faint_junction_gap: float = 17.0
    # when the ink threshold was lowered below Otsu's (pipeline.trace_threshold): Otsu's threshold, at which
    # the typical line width and the faint strokes' contrast are still judged, so faint lines now in the mask
    # do not make the lines look thinner nor faint-stroke detection more sensitive. None = the ink threshold
    reference_threshold: float | None = None
    # who decides crossings, gap links, junction pairing and corners (line2func.decisions):
    # None = the angle rules; "learned" = the bundled learned scorer; or a path to scorer weights
    decisions: str | None = None


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------


def _thin_luts() -> tuple[np.ndarray, np.ndarray]:
    """Deletion lookup tables for the two thinning sub-iterations.

    Neighbour bit order (weights of the correlation kernel below):
    bit 0 = E, 1 = NE, 2 = N, 3 = NW, 4 = W, 5 = SW, 6 = S, 7 = SE.
    """

    def bits(n: int) -> list[bool]:
        return [bool(n >> i & 1) for i in range(8)]

    g1 = np.zeros(256, bool)
    g2 = np.zeros(256, bool)
    g3 = np.zeros(256, bool)
    g3p = np.zeros(256, bool)
    for n in range(256):
        b = bits(n)
        s = sum(1 for i in (0, 2, 4, 6) if not b[i] and (b[i + 1] or b[(i + 2) % 8]))
        g1[n] = s == 1
        n1 = sum(1 for k in (1, 3, 5, 7) if b[k] or b[k - 1])
        n2 = sum(1 for k in (1, 3, 5, 7) if b[k] or b[(k + 1) % 8])
        g2[n] = min(n1, n2) in (2, 3)
        g3[n] = not ((b[1] or b[2] or not b[7]) and b[0])
        g3p[n] = not ((b[5] or b[6] or not b[3]) and b[4])
    return g1 & g2 & g3, g1 & g2 & g3p


_LUT_A, _LUT_B = _thin_luts()
_NEIGHBOUR_WEIGHTS = np.array([[8, 4, 2], [16, 0, 1], [32, 64, 128]], dtype=np.uint8)


def thin(mask: np.ndarray, max_iterations: int = 10_000) -> np.ndarray:
    """1-pixel-wide, 8-connected skeleton of a boolean mask."""
    skel = np.asarray(mask, dtype=bool).astype(np.uint8)
    count = int(skel.sum())
    for _ in range(max_iterations):
        for lut in (_LUT_A, _LUT_B):
            code = ndimage.correlate(skel, _NEIGHBOUR_WEIGHTS, mode="constant")
            skel[np.take(lut, code) & (skel == 1)] = 0
        new_count = int(skel.sum())
        if new_count == count:
            break
        count = new_count
    return skel.astype(bool)


def _groups(mask: np.ndarray, gap: float = 0.0) -> np.ndarray:
    """Labels of ``mask``'s pixels (0 elsewhere): its 8-connected pieces, with pieces at most about ``gap`` px
    apart sharing one label."""
    reach = int(round(0.5 * gap))
    grown = ndimage.binary_dilation(mask, structure=_EIGHT, iterations=reach) if reach > 0 else mask
    labels, _ = ndimage.label(grown, structure=_EIGHT)
    return np.where(mask, labels, 0) if reach > 0 else labels


def _remove_small(mask: np.ndarray, min_area: float, gap: float = 0.0) -> np.ndarray:
    """``mask`` without its pieces smaller than ``min_area`` px, counting pieces within ``gap`` px as one."""
    labels = _groups(mask, gap)
    if not labels.any():
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def faint_line_mask(
    ink: np.ndarray, strong: np.ndarray, threshold: float, line_width: float, params: "BaselineParams"
) -> np.ndarray:
    """Thin faint strokes below the threshold, to add to the ``strong`` line mask.

    A pixel is a faint-line candidate if it stands out from its *local*
    background: the white top-hat (ink minus its morphological opening at ~2
    line widths) of the lightly smoothed ink must reach ``faint_contrast`` x
    the threshold (``reference_threshold`` if set) and ``faint_noise_sigmas`` x the paper's pixel noise
    (:func:`paper_noise`). Broad shading has no ridge, and paper shading cannot
    lift noise over the limit. A band of about half a line width + ``faint_band`` px around
    strong lines is left out (blur halos and JPEG ringing there would add ghost
    curves, and would merge close parallel lines). A faint component is kept
    only if it is at least ``faint_min_length`` px long.

    Tuned on 40 hard synthetic scenes (noise, blur, JPEG, paper shading, specks)
    to add zero false pixels there, which is the price for being conservative:
    a lower ``--threshold`` recovers more faint ink on clean scans.
    """
    # contrast over the *local* background (white top-hat of the lightly smoothed ink):
    # paper shading raises the background, so absolute ink would let noise through there
    smooth = ndimage.gaussian_filter(ink, 0.5)
    ridge = local_contrast(ink, line_width, smooth)
    reference = params.reference_threshold if params.reference_threshold is not None else threshold
    noise = paper_noise(smooth, strong)
    level = max(0.05, params.faint_contrast * reference, params.faint_noise_sigmas * noise)
    # leave out a band around strong lines: blur halos and JPEG ringing live there,
    # and they would add ghost curves alongside real lines
    reach = int(round(0.5 * line_width)) + int(round(params.faint_band))
    band = ndimage.binary_dilation(strong, iterations=reach) if reach > 0 else strong
    weak = (ridge >= level) & ~band
    if weak.any():
        # real faint strokes are long; fringe bits, specks and noise chains are short (with join_widths, on clean
        # paper, the pieces of a stroke that broke into dashes count together)
        labels = _groups(weak, params.join_widths * line_width if noise < params.join_max_noise else 0.0)
        length = np.bincount(labels[thin(weak)], minlength=int(labels.max()) + 1)
        keep = length >= params.faint_min_length * params.denoise
        keep[0] = False
        weak = weak & keep[labels]
    if params.very_faint_lines:
        weak = weak | very_faint_line_mask(ink, ridge, band, line_width, params)
    return weak


def paper_contrast_ceiling(ink: np.ndarray, ridge: np.ndarray, line_width: float) -> float | None:
    """The most local contrast (``ridge``) the blank paper itself reaches: grain, JPEG, scan noise.

    Measured on the paper at least 3 line widths from any ink >= 0.04 (the 99.99th percentile, at least
    0.03). ``None`` when under 1% of the image is blank paper: then it cannot be measured.
    """
    blank = ~ndimage.binary_dilation(ink >= 0.04, iterations=max(1, int(round(3 * line_width))))
    if blank.sum() < 0.01 * blank.size:
        return None
    return max(0.03, float(np.percentile(ridge[blank], 99.99)))


def _neighbours(mask: np.ndarray) -> np.ndarray:
    return ndimage.convolve(mask.astype(np.int32), np.ones((3, 3), np.int32), mode="constant") - mask


def _prune_skeleton(skel: np.ndarray, n: int) -> np.ndarray:
    """``skel`` without its branches shorter than about ``n`` px (the rest keeps its full length)."""
    core = skel.copy()
    for _ in range(n):
        core &= _neighbours(core) >= 2
    for _ in range(n):
        core |= skel & ndimage.binary_dilation(core, structure=_EIGHT)
    return core


VERY_FAINT_CONNECT = 0.75  # pieces of a very faint line joined where the contrast stays above this x the ceiling
VERY_FAINT_PIECE = 5.0  # line widths: a shorter piece counts if it continues a kept very faint line ...
VERY_FAINT_GAP = 3.0  # ... across a gap of at most this many line widths, end to end


def _skeleton_degree(skel: np.ndarray) -> np.ndarray:
    """Every skeleton pixel's number of skeleton neighbours (a staircase's diagonal steps not counted twice)."""
    degree = np.zeros(skel.shape, dtype=np.int32)
    if skel.any():
        ys, xs = np.nonzero(skel)
        degree[ys, xs] = _pixel_graph(skel)[2]
    return degree


def very_faint_line_mask(ink: np.ndarray, ridge: np.ndarray, band: np.ndarray, line_width: float,
                         params: "BaselineParams") -> np.ndarray:
    """Strokes fainter than the faint level, down to the paper's own noise, that are clearly lines.

    Near the paper's noise, contrast cannot tell a faint line from pencil texture: in a real drawing, very
    light background strokes (local contrast 0.04-0.08) and the pencil texture on a hat (top tenth 0.06, top
    hundredth 0.13) were equally light. Shape can: a line is long, thin and hardly branches, texture is
    short, dense and criss-crossed.

    Candidates are the ``ridge`` (local contrast) above the paper's noise ceiling
    (:func:`paper_contrast_ceiling`); the pieces of one line, which dips below the ceiling here and there,
    are judged together where the contrast between them stays above :data:`VERY_FAINT_CONNECT` x the
    ceiling. A candidate that reaches 0.015 over the ceiling somewhere is kept when it is at least
    ``very_faint_length`` line widths long, at most ``very_faint_width`` wide on average, and has at most one
    junction per ``very_faint_junction_gap`` line widths once the spurs that thinning leaves (up to 2 line
    widths) are pruned. A shorter piece (from :data:`VERY_FAINT_PIECE` line widths) that continues a kept
    one, end to end across at most :data:`VERY_FAINT_GAP` line widths, is kept too. ``band`` (around strong
    lines) is left out.
    """
    ceiling = paper_contrast_ceiling(ink, ridge, line_width)
    if ceiling is None:
        return np.zeros(ink.shape, dtype=bool)
    # one faint line dips below the ceiling here and there: its pieces are joined, and judged together,
    # where the contrast stays above CONNECT x the ceiling in between (pinholes filled, so that thinning
    # does not turn them into loops)
    low = _fill_holes((ridge >= VERY_FAINT_CONNECT * ceiling) & ~band, line_width * line_width) & ~band
    if not low.any():
        return low
    labels, n = ndimage.label(low, structure=_EIGHT)
    seeded = np.bincount(labels[low & (ridge >= ceiling + 0.015)], minlength=n + 1) > 0
    skel = thin(low)
    length = np.bincount(labels[skel], minlength=n + 1).astype(np.float64)
    area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    core = _prune_skeleton(skel, int(round(2 * line_width)))
    degree = _skeleton_degree(core)
    # junctions: clusters of branching pixels, one per junction
    clusters, _ = ndimage.label(degree >= 3, structure=_EIGHT)
    at = clusters > 0
    pairs = np.unique(np.stack([labels[at], clusters[at]], axis=1), axis=0) if at.any() else np.zeros((0, 2), int)
    junctions = np.bincount(pairs[:, 0], minlength=n + 1).astype(np.float64)
    slim = area <= params.very_faint_width * line_width * np.maximum(length, 1.0)
    simple = junctions * params.very_faint_junction_gap * line_width <= length
    keep = seeded & (length >= params.very_faint_length * line_width) & slim & simple
    keep[0] = False
    # shorter pieces that continue a kept line, end to end across a small gap, belong to it
    ey, ex = np.nonzero(core & (degree == 1))
    owner = labels[ey, ex]
    piece = ~keep & slim & simple & (length >= VERY_FAINT_PIECE * line_width)
    piece[0] = False
    for _ in range(10):
        kept_ends = keep[owner]
        open_ends = piece[owner] & ~kept_ends
        if not kept_ends.any() or not open_ends.any():
            break
        d, _ = cKDTree(np.stack([ex[kept_ends], ey[kept_ends]], axis=1)).query(
            np.stack([ex[open_ends], ey[open_ends]], axis=1), distance_upper_bound=VERY_FAINT_GAP * line_width)
        joined = np.unique(owner[open_ends][np.isfinite(d)])
        if not joined.size:
            break
        keep[joined] = True
    return keep[labels]


def local_contrast(ink: np.ndarray, line_width: float, smooth: np.ndarray | None = None) -> np.ndarray:
    """How much each pixel stands out from its local background: the white top-hat (the ink minus its
    morphological opening over about 2 line widths) of the lightly smoothed ink (``smooth``, if already
    computed). Lines stand out; broad shading does not."""
    if smooth is None:
        smooth = ndimage.gaussian_filter(ink, 0.5)
    k = int(2 * round(line_width) + 3) | 1
    return smooth - ndimage.grey_opening(smooth, size=(k, k))


def auto_threshold(ink: np.ndarray) -> float:
    """The ink threshold when none is given: Otsu's, kept within [0.2, 0.8]."""
    return float(np.clip(otsu_threshold(ink), 0.2, 0.8))


def paper_noise(ink: np.ndarray, strong: np.ndarray) -> float:
    """Robust std of the pixel noise on the paper (high-pass residual, away from lines).

    Smooth shading is removed by the high-pass filter, so only grain, scan noise
    and compression artifacts count.
    """
    resid = ink - ndimage.gaussian_filter(ink, 1.0)
    paper = ~ndimage.binary_dilation(strong, iterations=3)
    vals = resid[paper]
    if vals.size < 100:
        return 0.0
    return float(1.4826 * np.median(np.abs(vals - np.median(vals))))


def _centerline(ink: np.ndarray, threshold: float | None) -> tuple[np.ndarray, float]:
    """The ink mask (specks ignored) and the length of its skeleton, px."""
    thr = threshold if threshold is not None else auto_threshold(ink)
    mask = _remove_small(np.asarray(ink) > thr, 8)
    if not mask.any():
        return mask, 0.0
    _, _, _, length = _pixel_graph(thin(mask))
    return mask, float(length)


def estimate_line_width(ink: np.ndarray, threshold: float | None = None) -> float:
    """Typical line width (px): ink area divided by centerline length, specks ignored."""
    mask, length = _centerline(ink, threshold)
    if not mask.any():
        return 0.0
    return float(mask.sum() / max(length, 1.0))


def ink_length(ink: np.ndarray, threshold: float | None = None) -> float:
    """Total length of the drawing's lines, px: the skeleton of the ink mask, specks ignored."""
    return _centerline(ink, threshold)[1]


def solid_areas(ink: np.ndarray, mask: np.ndarray, dist: np.ndarray, skel: np.ndarray, line_w: float,
                ratio: float, length: float, dark: float = 0.8, flat: float = 0.8) -> np.ndarray:
    """Solid ink at least ``ratio`` line widths thick along at least ``length`` line widths.

    The thick part is the union of the disks inscribed in the ink (``dist``, its
    distance transform) that are at least ``ratio`` x ``line_w`` across: a
    morphological opening. A piece of it counts when its centerline (``skel``)
    runs through such disks for at least ``length`` line widths, and when its
    middle is solid: the ink there is typically at least ``dark`` x the
    drawing's dark ink (the 90th percentile) and at least ``flat`` x the darkest
    ink around it. Two lines drawn close together make thick ink too, but a
    lighter one in between; a blot at a line end or a junction is too short.
    Those are left to the line tracer.
    """
    r = 0.5 * (ratio * line_w + 1.0)  # distance to the edge at the middle of ink that thick
    seeds = dist >= r
    if not seeds.any():
        return np.zeros_like(mask)
    region = mask & (ndimage.distance_transform_edt(~seeds) <= r + 1.0)
    labels, n = ndimage.label(region, structure=_EIGHT)
    run = np.bincount(labels[skel & seeds], minlength=n + 1)
    dark_ink = float(np.percentile(ink[mask], 90))
    size = 2 * int(np.ceil(r)) + 1
    keep = np.zeros(n + 1, dtype=bool)
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or run[k] < length * line_w:
            continue
        box = tuple(slice(max(0, s.start - size), s.stop + size) for s in box)
        middle = (labels[box] == k) & seeds[box]
        near = ndimage.maximum_filter(ink[box], size=size)[middle]
        v = ink[box][middle]
        keep[k] = np.median(v) >= dark * dark_ink and np.median(v / np.maximum(near, 1e-6)) >= flat
    return keep[labels]


def _fill_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    labels, n = ndimage.label(~mask)  # 4-connected background
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    small = sizes <= max_area
    small[0] = False
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small[border] = False
    return mask | small[labels]


# ---------------------------------------------------------------------------
# Skeleton graph
# ---------------------------------------------------------------------------


@dataclass
class _Edge:
    pts: np.ndarray  # (n, 2) xy pixel centers, from node a to node b
    a: int
    b: int
    closed: bool = False  # a loop with no node on it (a == b == -1)
    alive: bool = True
    _length: float | None = field(default=None, repr=False)

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = float(np.sum(np.linalg.norm(np.diff(self.pts, axis=0), axis=1)))
        return self._length

    def set_pts(self, pts: np.ndarray) -> None:
        self.pts = pts
        self._length = None

    def node(self, end: int) -> int:
        return self.a if end == 0 else self.b

    def from_end(self, end: int) -> np.ndarray:
        """Points ordered starting at ``end``."""
        return self.pts if end == 0 else self.pts[::-1]


class _Graph:
    def __init__(self, pos: list[np.ndarray], edges: list[_Edge]):
        self.pos = pos
        self.edges = edges
        # extra radius around a node where the skeleton is distorted (merged junctions)
        self.spread = [0.0] * len(pos)

    def incidence(self) -> list[list[tuple[int, int]]]:
        inc: list[list[tuple[int, int]]] = [[] for _ in self.pos]
        for i, e in enumerate(self.edges):
            if e.alive and not e.closed:
                inc[e.a].append((i, 0))
                inc[e.b].append((i, 1))
        return inc


def _pixel_graph(skel: np.ndarray):
    """Skeleton pixel adjacency with redundant diagonal links removed.

    A diagonal link is dropped when the two pixels also share an orthogonal
    neighbour in the skeleton, so staircase steps don't look like junctions.
    Returns ``(xy, neighbours, degree, length)``.
    """
    ys, xs = np.nonzero(skel)
    n = len(ys)
    idx = np.full((skel.shape[0] + 2, skel.shape[1] + 2), -1, dtype=np.int64)
    idx[ys + 1, xs + 1] = np.arange(n)
    src, dst, steps = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        nb = idx[ys + 1 + dy, xs + 1 + dx]
        m = nb >= 0
        src.append(np.nonzero(m)[0])
        dst.append(nb[m])
        steps.append(np.full(int(m.sum()), 1.0))
    for dy, dx in ((1, 1), (1, -1)):
        nb = idx[ys + 1 + dy, xs + 1 + dx]
        shared = (idx[ys + 1 + dy, xs + 1] >= 0) | (idx[ys + 1, xs + 1 + dx] >= 0)
        m = (nb >= 0) & ~shared
        src.append(np.nonzero(m)[0])
        dst.append(nb[m])
        steps.append(np.full(int(m.sum()), np.sqrt(2.0)))
    s = np.concatenate(src) if src else np.zeros(0, np.int64)
    d = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    length = float(np.concatenate(steps).sum()) if steps else 0.0
    a = np.concatenate([s, d])
    b = np.concatenate([d, s])
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    degree = np.bincount(a, minlength=n)
    bounds = np.concatenate([[0], np.cumsum(degree)])
    b_list = b.tolist()
    neighbours = [b_list[bounds[i] : bounds[i + 1]] for i in range(n)]
    xy = np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)
    return xy, neighbours, degree, length


def _trace_graph(skel: np.ndarray) -> _Graph:
    """Nodes (end points, junction clusters) and pixel-chain edges of a skeleton."""
    xy, nbrs, degree, _ = _pixel_graph(skel)
    n = len(xy)
    node_of = np.full(n, -1, dtype=np.int64)

    # junction pixels (degree >= 3) that touch form one node
    junction = degree >= 3
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.nonzero(junction)[0].tolist():
        for j in nbrs[i]:
            if junction[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    pos: list[np.ndarray] = []
    members: dict[int, list[int]] = {}
    for i in np.nonzero(junction)[0].tolist():
        members.setdefault(find(i), []).append(i)
    for group in members.values():
        node_of[group] = len(pos)
        pos.append(xy[group].mean(axis=0))
    for i in np.nonzero(degree == 1)[0].tolist():
        node_of[i] = len(pos)
        pos.append(xy[i].copy())

    edges: list[_Edge] = []
    visited = np.zeros(n, dtype=bool)
    seen_direct: set[tuple[int, int]] = set()
    for p in np.nonzero(node_of >= 0)[0].tolist():
        for q in nbrs[p]:
            if node_of[q] >= 0:
                if node_of[q] == node_of[p]:
                    continue
                key = (min(p, q), max(p, q))
                if key in seen_direct:
                    continue
                seen_direct.add(key)
                edges.append(_Edge(xy[[p, q]], int(node_of[p]), int(node_of[q])))
                continue
            if visited[q]:
                continue
            path = [p, q]
            visited[q] = True
            prev, cur = p, q
            while True:
                nxt = [r for r in nbrs[cur] if r != prev]
                if not nxt:
                    break
                r = nxt[0]
                path.append(r)
                if node_of[r] >= 0 or visited[r]:
                    break
                visited[r] = True
                prev, cur = cur, r
            end = path[-1]
            if node_of[end] < 0:  # ran into a visited chain (should not happen)
                continue
            if node_of[end] == node_of[p] and len(path) <= 4:
                continue  # tiny loop inside a junction cluster
            edges.append(_Edge(xy[path], int(node_of[p]), int(node_of[end])))

    # remaining unvisited degree-2 pixels form node-free loops
    for s in np.nonzero((degree == 2) & ~visited)[0].tolist():
        if visited[s]:
            continue
        path = [s]
        visited[s] = True
        prev, cur = -1, s
        while True:
            nxt = [r for r in nbrs[cur] if r != prev and not visited[r]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)
        if len(path) >= 3:
            edges.append(_Edge(xy[path], -1, -1, closed=True))
    return _Graph(pos, edges)


def _merge_degree_two(g: _Graph) -> None:
    """Join the two edges meeting at every node of degree 2."""
    inc = g.incidence()

    def replace(v: int, old: tuple[int, int], new: tuple[int, int]) -> None:
        lst = inc[v]
        lst[lst.index(old)] = new

    for v in range(len(g.pos)):
        if len(inc[v]) != 2:
            continue
        (e1, end1), (e2, end2) = inc[v]
        if e1 == e2:  # both ends of one edge: it becomes a closed loop
            e = g.edges[e1]
            e.closed, e.a, e.b = True, -1, -1
            inc[v] = []
            continue
        a_e, b_e = g.edges[e1], g.edges[e2]
        far1, far2 = a_e.node(1 - end1), b_e.node(1 - end2)
        p1 = a_e.from_end(1 - end1)  # far1 -> v
        p2 = b_e.from_end(end2)  # v -> far2
        a_e.set_pts(np.vstack([p1, p2[1:]]))
        old1 = (e1, 1 - end1)
        old2 = (e2, 1 - end2)
        a_e.a, a_e.b = far1, far2
        b_e.alive = False
        inc[v] = []
        replace(far1, old1, (e1, 0))
        replace(far2, old2, (e1, 1))


def _prune_spurs(g: _Graph, spur_len: float) -> None:
    for _ in range(4):
        inc = g.incidence()
        deg = [len(x) for x in inc]
        changed = False
        for e in g.edges:
            if not e.alive or e.closed or e.a == e.b or e.length >= spur_len:
                continue
            for tip, base in ((e.a, e.b), (e.b, e.a)):
                if deg[tip] == 1 and deg[base] >= 3:
                    e.alive = False
                    deg[tip] -= 1
                    deg[base] -= 1
                    changed = True
                    break
        _merge_degree_two(g)
        if not changed:
            break


def _looks_like_crossing(g: _Graph, inc, eid: int, reach: float, max_bend: float) -> bool:
    """Do two degree-3 junctions joined by edge ``eid`` form one shallow X crossing?

    True when each outer arm at one junction continues straight into an outer
    arm at the other, while the two arms at each junction do not continue
    each other (that pattern is two T-junctions, like the bar of an "H").
    """
    e = g.edges[eid]
    arms = []
    for v in (e.a, e.b):
        if len(inc[v]) != 3:
            return False
        center = g.pos[v]
        arms.append(
            [_direction_from(g.edges[i].from_end(end), center, reach) for i, end in inc[v] if i != eid]
        )
    (a1, a2), (b1, b2) = arms
    if min(_angle(a1, -a2), _angle(b1, -b2)) <= max_bend:
        return False
    straight = max(_angle(a1, -b1), _angle(a2, -b2))
    swapped = max(_angle(a1, -b2), _angle(a2, -b1))
    return min(straight, swapped) <= max_bend


def _crossing_cand(g: _Graph, inc, eid: int, reach: float, max_bend: float) -> CrossingCand:
    e = g.edges[eid]
    arms = [[((i, end), g.edges[i].from_end(end)) for i, end in inc[v] if i != eid] for v in (e.a, e.b)]
    return CrossingCand(eid, e.a, e.b, e.length, _looks_like_crossing(g, inc, eid, reach, max_bend), e.pts,
                        arms[0], arms[1])


def _merge_close_junctions(
    g: _Graph, merge_len: float, crossing_len: float, reach: float, max_bend: float,
    scorer=None, ctx: DecisionContext | None = None, recorder=None,
) -> None:
    """Contract short junction-to-junction edges.

    Edges shorter than ``merge_len`` are thinning noise and always contracted;
    edges up to ``crossing_len`` are contracted only if the scorer takes the two
    junctions for one shallow crossing (by default the angle rule
    :func:`_looks_like_crossing`).
    """
    scorer = scorer or RULES
    inc = g.incidence()
    deg = [len(x) for x in inc]
    parent = list(range(len(g.pos)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def eligible(e: _Edge) -> bool:
        return e.alive and not e.closed and e.a != e.b and deg[e.a] >= 3 and deg[e.b] >= 3

    # The crossing test reads only this snapshot of the graph (the loop below
    # changes nothing it looks at), so every candidate is decided up front.
    cands = [_crossing_cand(g, inc, eid, reach, max_bend) for eid, e in enumerate(g.edges)
             if eligible(e) and merge_len <= e.length < crossing_len]
    verdicts = scorer.crossing(cands)
    if recorder is not None:
        recorder.crossings(cands, verdicts, ctx)
    decided = {c.eid: bool(v) for c, v in zip(cands, verdicts)}

    merged = False
    for eid, e in enumerate(g.edges):
        if not eligible(e):
            continue
        if e.length < merge_len or (e.length < crossing_len and decided.get(eid, False)):
            ra, rb = find(e.a), find(e.b)
            if ra != rb:
                parent[ra] = rb
            e.alive = False
            merged = True
    if not merged:
        return
    groups: dict[int, list[int]] = {}
    for v in range(len(g.pos)):
        groups.setdefault(find(v), []).append(v)
    for root, vs in groups.items():
        if len(vs) > 1:
            center = np.mean([g.pos[v] for v in vs], axis=0)
            g.spread[root] = max(float(np.linalg.norm(g.pos[v] - center)) + g.spread[v] for v in vs)
            g.pos[root] = center
    for e in g.edges:
        if e.alive and not e.closed:
            e.a, e.b = find(e.a), find(e.b)
            if e.a == e.b and e.length < 2.0 * merge_len:
                e.alive = False  # the "eye" between two shallow crossing lines


# ---------------------------------------------------------------------------
# Strokes
# ---------------------------------------------------------------------------


def _link_gaps(
    g: _Graph, inc, links: dict, max_gap: float, max_angle: float, reach: float,
    scorer=None, ctx: DecisionContext | None = None, recorder=None,
) -> None:
    scorer = scorer or RULES
    tips, outs, ends = [], [], []
    for v, lst in enumerate(inc):
        if len(lst) != 1:
            continue
        eid, end = lst[0]
        pts = g.edges[eid].from_end(end)  # starts at the tip
        tip = pts[0]
        back = _direction_from(pts, tip, reach)
        tips.append(tip)
        outs.append(-back)
        ends.append((eid, end))
    if len(tips) < 2:
        return
    tips_arr = np.array(tips)
    limits = scorer.limits
    radius = max_gap if limits.gap_radius_scale == 1.0 else max_gap * limits.gap_radius_scale
    gate = max_angle if limits.gap_angle_gate is None else limits.gap_angle_gate
    cands = []
    for i, j in sorted(cKDTree(tips_arr).query_pairs(radius)):
        (ei, _), (ej, _) = ends[i], ends[j]
        if ei == ej and g.edges[ei].length < 3.0 * max_gap:
            continue
        gap = tips_arr[j] - tips_arr[i]
        dist = float(np.hypot(gap[0], gap[1]))
        if dist < 1.5:
            ai = aj = 0.0
        else:
            unit = gap / dist
            ai, aj = _angle(outs[i], unit), _angle(outs[j], -unit)
            if max(ai, aj) > gate:
                continue
        # with wider limits, candidates the rule would not even consider are kept, but not rule_ok
        rule_ok = max(ai, aj) <= max_angle and (radius == max_gap or dist <= max_gap)
        cands.append(GapCand(i, j, ends[i], ends[j], tips_arr[i], tips_arr[j], outs[i], outs[j], dist, ai, aj,
                             rule_ok, dist * (1.0 + (ai + aj) / 90.0),
                             g.edges[ends[i][0]].from_end(ends[i][1]), g.edges[ends[j][0]].from_end(ends[j][1])))
    scores = scorer.gap_scores(cands)
    chosen = greedy_pairs(scores, [(c.i, c.j) for c in cands])
    if recorder is not None:
        recorder.gaps(cands, chosen, ctx)
    for i, j in chosen:
        links[ends[i]] = ends[j]
        links[ends[j]] = ends[i]


def _link_junctions(
    g: _Graph, inc, links: dict, max_bend: float, reach: float,
    scorer=None, ctx: DecisionContext | None = None, recorder=None,
) -> None:
    scorer = scorer or RULES
    cands = []
    for v, lst in enumerate(inc):
        if len(lst) < 2:
            continue
        if len(lst) == 2:
            links[lst[0]], links[lst[1]] = lst[1], lst[0]
            continue
        center = g.pos[v]
        # look past the distorted zone of merged junctions
        r = reach + 2.0 * g.spread[v]
        dirs = [_direction_from(g.edges[eid].from_end(end), center, r) for eid, end in lst]
        pairs = [(i, j) for i in range(len(lst)) for j in range(i + 1, len(lst))]
        bends = np.array([_angle(dirs[i], -dirs[j]) for i, j in pairs], dtype=np.float64)
        arms = [g.edges[eid].from_end(end) for eid, end in lst]
        far = [len(inc[g.edges[eid].node(1 - end)]) for eid, end in lst]
        cands.append(JunctionCand(v, list(lst), center, g.spread[v], r, dirs, pairs, bends, arms, far))
    scores = scorer.junction_scores(cands, max_bend)
    decisions = []
    for c, (z, ends_here) in zip(cands, scores):
        if scorer.junction_policy == "exact" and ends_here is not None:
            chosen = exact_matching(z, c.pairs, ends_here, len(c.keys))
        else:
            chosen = greedy_pairs(z, c.pairs)
        decisions.append(chosen)
        for i, j in chosen:
            links[c.keys[i]], links[c.keys[j]] = c.keys[j], c.keys[i]
    if recorder is not None:
        recorder.junctions(cands, decisions, ctx)


def _assemble(g: _Graph, inc, links: dict, trim: float, info: list | None = None,
              line_dist: np.ndarray | None = None) -> list[tuple[np.ndarray, bool]]:
    """Walk linked edges into strokes; returns ``(points, closed)`` pairs.

    With ``info`` (a list), each stroke's seams are appended to it: where two
    edges were joined, as ``(arc start, arc end, "junction" | "gap")`` along the
    stroke's points (the stretch between the two edges' last and first points).

    ``trim`` is how far from a junction the skeleton is dropped. With
    ``line_dist`` (the distance transform of the lines) each junction is
    measured on its own instead, which matters when one drawing holds both thin
    and thick lines.
    """
    is_junction = [len(x) >= 3 for x in inc]
    used = [not e.alive for e in g.edges]
    strokes: list[tuple[np.ndarray, bool]] = []
    reach_at: dict[int, float] = {}

    def distorted(v: int) -> float:
        """How far from node ``v`` thinning distorts the skeleton.

        The disk inscribed in the ink at the node (``line_dist``) measures that
        zone where one width for the whole drawing cannot: a junction of thick
        lines is distorted over a longer stretch than one of thin lines, and a
        crossing is distorted further than a line's own half width.
        """
        if v not in reach_at:
            local = trim
            if line_dist is not None:
                h, w = line_dist.shape
                x = int(np.clip(round(g.pos[v][0] - 0.5), 0, w - 1))
                y = int(np.clip(round(g.pos[v][1] - 0.5), 0, h - 1))
                local = float(line_dist[y, x]) + 1.0
            reach_at[v] = local + 2.0 * g.spread[v]
        return reach_at[v]

    def walk(eid: int, entry: int):
        seq = []
        while True:
            used[eid] = True
            seq.append((eid, entry))
            nxt = links.get((eid, 1 - entry))
            if nxt is None:
                return seq, False
            if used[nxt[0]]:
                return seq, nxt == seq[0]
            eid, entry = nxt

    def foot(arm: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Extend ``arm`` (which starts next to junction ``c``) straight toward ``c``.

        Returns the point on the arm's own line closest to ``c``, so a stroke
        ending at a junction reaches the other line without bending toward
        the junction center.
        """
        m = min(len(arm), 6)
        d = arm[0] - arm[m - 1]
        n = float(np.hypot(d[0], d[1]))
        if n < 1e-9:
            return c
        d = d / n
        return arm[0] + d * max(0.0, float(np.dot(c - arm[0], d)))

    def points(seq, closed: bool) -> np.ndarray:
        chunks = []
        last = len(seq) - 1
        for k, (eid, entry) in enumerate(seq):
            e = g.edges[eid]
            p = e.from_end(entry)
            n_in, n_out = e.node(entry), e.node(1 - entry)
            # Trim the distorted skeleton around junctions. Strokes passing
            # through simply connect their trimmed arms; strokes ending at a
            # junction are extended straight to it.
            if is_junction[n_in]:
                c = g.pos[n_in]
                far = np.nonzero(np.linalg.norm(p - c, axis=1) > distorted(n_in))[0]
                p = p[far[0] :] if len(far) else c[None]
                if k == 0 and not closed and len(far):
                    p = np.vstack([foot(p, c), p])
            if is_junction[n_out]:
                c = g.pos[n_out]
                far = np.nonzero(np.linalg.norm(p - c, axis=1) > distorted(n_out))[0]
                p = p[: far[-1] + 1] if len(far) else p[:1]
                if k == last and not closed and len(far):
                    p = np.vstack([p, foot(p[::-1], c)])
            chunks.append(p)
        if info is not None:
            out = np.vstack(chunks)
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))])
            first = np.cumsum([0] + [len(c) for c in chunks])
            seams = []
            for k in range(1, len(seq) + (1 if closed else 0)):
                eid, entry = seq[k - 1]
                kind = "junction" if is_junction[g.edges[eid].node(1 - entry)] else "gap"
                if k < len(seq):
                    seams.append((float(cum[first[k] - 1]), float(cum[first[k]]), kind))
                else:  # a closed stroke's last edge joins its first one
                    seams.append((float(cum[-1]), float(cum[-1]), kind))
            info.append(seams)
            return out
        return np.vstack(chunks)

    for e in g.edges:
        if e.alive and e.closed:
            strokes.append((e.pts, True))
            if info is not None:
                info.append([])
    for eid, e in enumerate(g.edges):
        if not e.alive or e.closed:
            continue
        for end in (0, 1):
            if not used[eid] and (eid, end) not in links:
                seq, closed = walk(eid, end)
                strokes.append((points(seq, closed), closed))
    for eid, e in enumerate(g.edges):
        if not used[eid] and e.alive and not e.closed:
            seq, closed = walk(eid, 0)
            strokes.append((points(seq, closed), closed))
    return strokes


def _strokes_from_skeleton(
    skel: np.ndarray, radius: float, params: BaselineParams, close_gaps: bool,
    scorer=None, ctx: DecisionContext | None = None, recorder=None, info: list | None = None,
    line_dist: np.ndarray | None = None,
):
    scorer = scorer or RULES
    g = _trace_graph(skel)
    reach = 2.0 * radius + 3.0
    _prune_spurs(g, spur_len=2.0 * radius + 2.0)
    scale = scorer.limits.crossing_len_scale
    _merge_close_junctions(
        g,
        merge_len=2.0 * radius + 3.0,
        crossing_len=12.0 * radius + 6.0 if scale == 1.0 else 12.0 * radius * scale + 6.0,
        reach=reach,
        max_bend=params.continue_angle,
        scorer=scorer,
        ctx=ctx,
        recorder=recorder,
    )
    _merge_degree_two(g)
    inc = g.incidence()
    links: dict = {}
    if close_gaps:
        visible = params.max_gap if params.max_gap is not None else max(4.0, params.max_gap_widths * 2.0 * radius)
        # thinning pulls each skeleton end back by about half a line width,
        # so skeleton tips are one line width farther apart than the ink ends
        _link_gaps(g, inc, links, visible + 2.0 * radius, params.gap_angle, reach, scorer=scorer, ctx=ctx,
                   recorder=recorder)
    _link_junctions(g, inc, links, params.continue_angle, reach, scorer=scorer, ctx=ctx, recorder=recorder)
    return _assemble(g, inc, links, trim=radius + 1.0, info=info, line_dist=line_dist)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _resample(pts: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0.0:
        return pts[:1]
    n = max(1, int(round(total / spacing)))
    s = np.linspace(0.0, total, n + 1)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def _turn_profile(pts: np.ndarray, closed: bool, k: int, skip: int) -> tuple[np.ndarray, np.ndarray]:
    """``(sample indices, turning angle in degrees)`` along a stroke resampled at 1 px.

    The two arms are measured over ``k`` samples each, starting ``skip``
    samples away from the candidate point: thinning cuts corners into short
    diagonal chamfers, which would otherwise hide part of the turn.
    """
    n = len(pts)
    reach = k + skip
    if n < 2 * reach + 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    if closed:
        i = np.arange(n)
        v1 = pts[(i - skip) % n] - pts[(i - reach) % n]
        v2 = pts[(i + reach) % n] - pts[(i + skip) % n]
    else:
        i = np.arange(reach, n - reach)
        v1 = pts[i - skip] - pts[i - reach]
        v2 = pts[i + reach] - pts[i + skip]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-12)
    turn = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return i, turn


def _corners(pts: np.ndarray, closed: bool, k: int, skip: int, min_angle: float) -> list[int]:
    """Indices where the stroke turns by more than ``min_angle`` degrees (the angle rule)."""
    i, turn = _turn_profile(pts, closed, k, skip)
    return pick_peaks(i, turn, min_angle, len(pts), closed, k + skip)


def _smooth(pts: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    if sigma <= 0.0 or len(pts) < 3:
        return pts
    if closed:
        return ndimage.gaussian_filter1d(pts, sigma, axis=0, mode="wrap")
    pad = min(len(pts) - 1, int(3 * sigma) + 1)
    # point reflection keeps the end points fixed and straight ends straight
    left = 2 * pts[0] - pts[1 : pad + 1][::-1]
    right = 2 * pts[-1] - pts[-pad - 1 : -1][::-1]
    ext = np.vstack([left, pts, right])
    out = ndimage.gaussian_filter1d(ext, sigma, axis=0, mode="nearest")[pad : pad + len(pts)]
    out[0], out[-1] = pts[0], pts[-1]
    return out


def _fit_stroke(
    pts: np.ndarray, closed: bool, radius: float, params: BaselineParams,
    scorer=None, ctx: DecisionContext | None = None, recorder=None, seams: list | None = None, stroke: int = -1,
) -> list[np.ndarray]:
    if closed:
        pts = np.vstack([pts, pts[:1]])
    pts = _resample(pts, 1.0)
    if len(pts) < 2:
        return []
    if closed:
        pts = pts[:-1]
        if len(pts) < 3:
            return []
    skip = max(1, int(round(radius + 1)))
    if scorer is None:
        corners = _corners(pts, closed, k=4, skip=skip, min_angle=params.corner_angle)
    else:
        index, turn = _turn_profile(pts, closed, 4, skip)
        cand = CornerCand(pts, closed, index, turn, skip, 4, seams or [])
        key, threshold = scorer.corner_keys(cand, params.corner_angle)
        corners = pick_peaks(index, key, threshold, len(pts), closed, 4 + skip)
        if recorder is not None:
            recorder.corners(cand, corners, ctx, stroke)
    tol = params.fit_tolerance
    if closed and not corners:
        return fit_polyline(_smooth(pts, params.smooth_sigma, True), tol, closed=True)
    if closed:
        pts = np.vstack([np.roll(pts, -corners[0], axis=0), pts[corners[0] : corners[0] + 1]])
        corners = [c - corners[0] for c in corners[1:]]
    cuts = [0] + corners + [len(pts) - 1]
    pieces = [pts[s : e + 1] for s, e in zip(cuts[:-1], cuts[1:]) if e > s]
    joints = [(i, i + 1) for i in range(len(pieces) - 1)]
    if closed and len(pieces) > 1:
        joints.append((len(pieces) - 1, 0))
    for a, b in joints:
        pieces[a], pieces[b] = _sharpen(pieces[a], pieces[b], skip, skip + 4)
    curves: list[np.ndarray] = []
    for piece in pieces:
        if len(piece) >= 2:
            curves.extend(fit_polyline(_smooth(piece, params.smooth_sigma, False), tol))
    return curves


def _line_through(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through ``pts``: ``(point, unit direction)``."""
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    return center, vt[0]


def _sharpen(left: np.ndarray, right: np.ndarray, skip: int, reach: int) -> tuple[np.ndarray, np.ndarray]:
    """Move the corner shared by ``left[-1]``/``right[0]`` to where the two arms meet.

    Thinning cuts corners into a short chamfer; the arms (outside the chamfer)
    are fitted with lines and the corner is placed at their intersection.
    """
    if len(left) <= reach or len(right) <= reach:
        return left, right
    c1, d1 = _line_through(left[-reach - 1 : -skip])
    c2, d2 = _line_through(right[skip : reach + 1])
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 0.2:  # nearly parallel arms: leave as is
        return left, right
    diff = c2 - c1
    s = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
    corner = c1 + s * d1
    if np.linalg.norm(corner - left[-1]) > 2.0 * skip + 2.0:
        return left, right
    return np.vstack([left[:-skip], corner]), np.vstack([corner, right[skip:]])


def _support(ctrl: np.ndarray, ink: np.ndarray) -> float:
    """Fraction of points along the curve (every ~1 px) that fall on ink."""
    length = float(np.sum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1)))
    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(length)) + 1))
    p = evaluate(ctrl, t)
    h, w = ink.shape
    x = np.clip(p[:, 0].astype(np.int64), 0, w - 1)
    y = np.clip(p[:, 1].astype(np.int64), 0, h - 1)
    return float(ink[y, x].mean())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def vectorize(ink: np.ndarray, params: BaselineParams | None = None, *, scorer=None, recorder=None) -> CurveSet:
    """Trace an ink map (``(H, W)``, 1 = line) into a :class:`CurveSet`.

    ``scorer`` makes the tracer's decisions: crossings, gap links, junction
    pairing and corners (:mod:`line2func.decisions`). By default these are the
    angle rules, or the scorer named by ``params.decisions``. ``recorder``
    collects every decision candidate (training data for a learned scorer).
    """
    params = params or BaselineParams()
    if scorer is None:
        if params.decisions in (None, "rules"):
            scorer = RULES
        else:
            from line2func.decisions import load_scorer

            scorer = load_scorer(params.decisions)
    ink = np.asarray(ink, dtype=np.float32)
    if ink.ndim != 2:
        raise ValueError("ink map must be a 2-D array")
    height, width = ink.shape
    result = CurveSet(width, height, meta={"engine": "baseline"})

    thr = params.threshold if params.threshold is not None else auto_threshold(ink)
    mask = ink > thr
    if not mask.any():
        result.meta["line_width"] = 0.0
        return result

    # rough line width from the distance transform along the skeleton; the
    # median ignores filled blobs, which would inflate an area/length estimate.
    # Judged on the strong lines: at the reference threshold if the ink threshold is lower
    # (the faint lines that the lower threshold lets in are thinner, and every line gains its soft edge)
    lowered = params.reference_threshold is not None and params.reference_threshold > thr
    strong = ink > params.reference_threshold if lowered else mask
    skel0 = thin(strong)
    depth = ndimage.distance_transform_edt(strong)[skel0]
    rough_w = float(np.clip(2.0 * np.median(depth) - 1.0, 1.0, 50.0)) if depth.size else 1.0
    # specks this close to other ink are judged with it (join_widths), on clean paper only, as faint pieces are
    join = 0.0
    if params.join_widths > 0 and paper_noise(ndimage.gaussian_filter(ink, 0.5), mask) < params.join_max_noise:
        join = params.join_widths * rough_w
    if params.faint_lines:
        mask = mask | faint_line_mask(ink, mask, thr, rough_w, params)
    min_area = params.min_component_area
    if min_area is None:
        min_area = max(8.0, 2.0 * rough_w * rough_w) * params.denoise
    mask = _remove_small(mask, min_area, join)
    mask = _fill_holes(mask, max(3.0, 0.5 * rough_w * rough_w))
    if not mask.any():
        result.meta["line_width"] = 0.0
        return result

    # filled areas: pixels deeper than fill_ratio line widths, grown back to the blob
    dist = ndimage.distance_transform_edt(mask)
    fill_radius = max(4.0, params.fill_ratio * rough_w)
    seeds = dist >= fill_radius
    fill_region = np.zeros_like(mask)
    if seeds.any():
        fill_region = mask & (ndimage.distance_transform_edt(~seeds) <= fill_radius + 1.0)
        fill_region = _remove_small(fill_region, 4.0 * fill_radius * fill_radius)
    if params.solid_ratio is not None:
        if not lowered:
            solid = solid_areas(ink, mask, dist, skel0, rough_w, params.solid_ratio, params.solid_length)
        else:
            # judged at the reference threshold, like the line width, then given back the soft rim
            # that the lower ink threshold adds around them
            core = _fill_holes(strong & mask, max(3.0, 0.5 * rough_w * rough_w))
            solid = solid_areas(ink, core, ndimage.distance_transform_edt(core), skel0, rough_w,
                                params.solid_ratio, params.solid_length)
            solid = mask & ndimage.binary_dilation(solid, structure=_EIGHT, iterations=2)
        fill_region = fill_region | solid
    line_mask = _remove_small(mask & ~fill_region, min_area, join) if fill_region.any() else mask

    skel = thin(line_mask)
    _, _, _, skel_len = _pixel_graph(skel)
    line_w = float(line_mask.sum() / max(skel_len, 1.0)) if skel_len > 0 else rough_w
    radius = 0.5 * line_w
    result.meta["line_width"] = round(line_w, 2)

    support = ndimage.binary_dilation(mask, structure=_EIGHT).astype(np.float32)
    # how thick the lines are at every pixel: the trim around each junction and the scorer's
    # features both read it (the same array as dist unless filled areas were cut out)
    line_dist = ndimage.distance_transform_edt(line_mask) if fill_region.any() else dist
    ctx = None
    if scorer.needs_features or recorder is not None:
        ctx = DecisionContext(ink, mask, line_dist, thr, line_w, radius)
    scorer.begin(ctx)
    stroke_id = 0

    def emit(strokes, tags: tuple[str, ...], seams=None, decide=None) -> None:
        nonlocal stroke_id
        for k, (pts, closed) in enumerate(strokes):
            ctrls = _fit_stroke(pts, closed, radius, params, scorer=decide, ctx=ctx,
                                recorder=recorder if decide is not None else None,
                                seams=seams[k] if seams is not None else None, stroke=stroke_id)
            if not ctrls:
                continue
            for c in ctrls:
                conf = round(_support(c, support), 3)
                result.curves.append(Curve(c, stroke=stroke_id, confidence=conf, tags=tags))
            stroke_id += 1

    min_len = max(2.0, line_w)
    infos: list = []
    raw = _strokes_from_skeleton(skel, radius, params, close_gaps=True, scorer=scorer, ctx=ctx, recorder=recorder,
                                 info=infos, line_dist=line_dist if params.local_trim else None)
    keep = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) >= min_len for p, _ in raw]
    strokes = [s for s, ok in zip(raw, keep) if ok]
    seams = [s for s, ok in zip(infos, keep) if ok]
    emit(strokes, (), seams, decide=scorer)
    report = scorer.report()
    if report:
        result.meta["decisions"] = report

    if fill_region.any():
        outline = fill_region & ~ndimage.binary_erosion(fill_region, structure=_EIGHT)
        outline_strokes = _strokes_from_skeleton(thin(outline), 0.5, params, close_gaps=False)
        emit(outline_strokes, ("fill_outline",))
    return result

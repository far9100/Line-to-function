"""Closed outline loops of an ink mask (numpy only).

A filled area - a heavy eyelash, a shadow - is exported as its boundary plus
the curves drawn inside it (:mod:`line2func.fill`), and that boundary has to
come back **closed**. Thinning the edge and walking it as a skeleton does not:
the walk splits a ring wherever two rings touch and hands back open chains,
which are then closed by a straight chord, inventing the area between chord and
arc. On ``lineArt (5).jpg`` that invented 129,399 px against a true 87,337.

:func:`contours` walks the outline itself instead, so every loop closes.
"""

from __future__ import annotations

import numpy as np

# the eight neighbours, clockwise from east, as (dy, dx)
_RING = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def contours(mask: np.ndarray) -> list[np.ndarray]:
    """Closed outline loops of an ink mask, each in order, the background always on the same side.

    Moore-neighbour following: from an outline pixel, the eight neighbours are
    tried clockwise from wherever the walk came in, and the first one on the ink
    is stepped to. It closes on itself, and unlike a skeleton walk it does not
    mind an outline that is only one pixel wide - both sides of a thin line are
    then stretches of a single loop.
    """
    mask = np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = mask.shape
    flat = mask.ravel()
    steps = np.array([dy * w + dx for dy, dx in _RING], dtype=np.int64)
    back = np.array([(k + 4) % 8 for k in range(8)])  # the way back, once a step has been taken
    # a loop starts at ink whose western neighbour is paper, so the walk knows where it came
    # from. In raster order that is each piece's leftmost ink, and the ink just right of a hole
    west_paper = np.zeros_like(mask)
    west_paper[:, 1:] = mask[:, 1:] & ~mask[:, :-1]
    starts = np.nonzero(west_paper.ravel())[0]
    seen = np.zeros(h * w, dtype=bool)
    loops = []
    for start in starts.tolist():
        if seen[start]:
            continue
        # the walk enters from the west, which is background for the first outline pixel of a row
        p, entry, path = start, 4, [start]
        # the walk is decided by where it stands and where it came from, so it has closed the
        # loop as soon as that pair repeats. A thin line, whose two sides are one loop, is
        # walked up one side and down the other, and only then repeats.
        states = {p * 8 + entry}
        seen[start] = True
        while True:
            nxt = -1
            for k in range(1, 9):
                d = (entry + k) % 8
                q = p + steps[d]
                if flat[q]:
                    nxt, entry = q, back[d]
                    break
            if nxt < 0:
                break
            state = nxt * 8 + entry
            if state in states:
                break
            states.add(state)
            p = nxt
            seen[p] = True
            path.append(p)
        if len(path) >= 3:
            ys, xs = np.divmod(np.array(path, dtype=np.int64), w)
            loops.append(np.stack([xs - 1 + 0.5, ys - 1 + 0.5], axis=1).astype(np.float64))
    return loops

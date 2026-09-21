"""line2func: trace the lines in an image into cubic parametric curves."""

from line2func.curves import Curve, CurveSet
from line2func.geometry import (
    arc_length,
    bounding_box,
    evaluate,
    flatten,
    from_power,
    split,
    to_power,
)
from line2func.render import rasterize, render_lineart, render_overlay

__version__ = "1.3.0"

__all__ = [
    "Curve",
    "CurveSet",
    "arc_length",
    "bounding_box",
    "evaluate",
    "flatten",
    "from_power",
    "rasterize",
    "render_lineart",
    "render_overlay",
    "split",
    "to_power",
]

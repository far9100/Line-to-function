"""Curve data model and the ``curves.json`` file format.

``curves.json`` stores control points in image pixel coordinates (y down, see
:mod:`line2func.geometry`), together with the image size so exporters can flip
to y-up coordinates. Layout::

    {
      "format": "line2func.curves",
      "version": 1,
      "image": {"width": 640, "height": 480},
      "coordinates": "image-pixels-y-down",
      "meta": {"engine": "baseline", "line_width": 2.1},
      "curves": [
        {"id": 0, "stroke": 0, "ctrl": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
         "confidence": 1.0, "tags": [],
         "width": 2.3, "color": "#1a1a1a",                      (optional)
         "shape": {"type": "line", ...} | {"type": "arc", ...}, (optional)
         "functions": ["y=...", "x=..."]}                       (optional)
      ]
    }

Curves that share a ``stroke`` id are consecutive pieces of one drawn stroke,
listed in drawing order. The optional fields are written only when known:
``width`` (measured line width, px), ``color`` (measured ink color), ``shape``
(the curve recognized as a straight line or circular arc, see
:mod:`line2func.shapes`) and ``functions`` (the curve as Desmos equations
``y = f(x)`` / ``x = g(y)``, see :mod:`line2func.functions`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from line2func.geometry import as_ctrl

FORMAT_NAME = "line2func.curves"
FORMAT_VERSION = 1
COORDINATES = "image-pixels-y-down"


@dataclass
class Curve:
    """One cubic Bézier piece of a stroke."""

    ctrl: np.ndarray
    stroke: int = 0
    confidence: float = 1.0
    tags: tuple[str, ...] = ()
    width: float | None = None  # measured line width, px
    color: str | None = None  # measured ink color, "#rrggbb"
    shape: dict | None = None  # recognized line / arc (line2func.shapes)
    functions: list[str] | None = None  # the curve as y = f(x) / x = g(y) equations (line2func.functions)

    def __post_init__(self) -> None:
        self.ctrl = as_ctrl(self.ctrl)
        self.stroke = int(self.stroke)
        self.confidence = float(self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        self.tags = tuple(str(t) for t in self.tags)
        if self.width is not None:
            self.width = float(self.width)
            if not self.width > 0:
                raise ValueError("width must be positive")
        if self.color is not None and not (
            isinstance(self.color, str) and len(self.color) == 7 and self.color.startswith("#")
        ):
            raise ValueError(f"color must look like '#rrggbb', got {self.color!r}")
        if self.functions is not None:
            self.functions = [str(f) for f in self.functions]


@dataclass
class CurveSet:
    """All curves traced from one image."""

    width: int
    height: int
    curves: list[Curve] = field(default_factory=list)
    # free-form, JSON-serializable run info (engine, estimated line width, ...)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.width, self.height = int(self.width), int(self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image width and height must be positive")

    def __len__(self) -> int:
        return len(self.curves)

    def __iter__(self):
        return iter(self.curves)

    def strokes(self) -> dict[int, list[Curve]]:
        """Curves grouped by stroke id, in first-appearance order."""
        groups: dict[int, list[Curve]] = {}
        for c in self.curves:
            groups.setdefault(c.stroke, []).append(c)
        return groups

    @property
    def num_strokes(self) -> int:
        return len({c.stroke for c in self.curves})

    def scaled(self, factor: float, width: int, height: int) -> "CurveSet":
        """Copy with every coordinate and width multiplied by ``factor``, for an image of ``width`` x ``height``.

        Shapes are dropped (re-run :func:`line2func.shapes.recognize` at the new scale).
        """
        curves = [
            Curve(c.ctrl * factor, c.stroke, c.confidence, c.tags,
                  width=None if c.width is None else c.width * factor, color=c.color)
            for c in self.curves
        ]
        meta = dict(self.meta)
        if meta.get("line_width"):
            meta["line_width"] = round(float(meta["line_width"]) * factor, 2)
        return CurveSet(width, height, curves, meta)

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "image": {"width": self.width, "height": self.height},
            "coordinates": COORDINATES,
            "meta": dict(self.meta),
            "curves": [self._curve_dict(i, c) for i, c in enumerate(self.curves)],
        }

    @staticmethod
    def _curve_dict(i: int, c: Curve) -> dict:
        d = {
            "id": i,
            "stroke": c.stroke,
            "ctrl": [[float(x), float(y)] for x, y in c.ctrl],
            "confidence": c.confidence,
            "tags": list(c.tags),
        }
        if c.width is not None:
            d["width"] = round(c.width, 3)
        if c.color is not None:
            d["color"] = c.color
        if c.shape is not None:
            d["shape"] = c.shape
        if c.functions is not None:
            d["functions"] = list(c.functions)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CurveSet":
        if data.get("format") != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} document")
        version = data.get("version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported {FORMAT_NAME} version: {version!r}")
        if data.get("coordinates", COORDINATES) != COORDINATES:
            raise ValueError(f"unsupported coordinate system: {data['coordinates']!r}")
        image = data["image"]
        curves = [
            Curve(
                ctrl=item["ctrl"],
                stroke=item.get("stroke", 0),
                confidence=item.get("confidence", 1.0),
                tags=tuple(item.get("tags", ())),
                width=item.get("width"),
                color=item.get("color"),
                shape=item.get("shape"),
                functions=item.get("functions"),
            )
            for item in data.get("curves", [])
        ]
        return cls(
            width=image["width"], height=image["height"], curves=curves, meta=dict(data.get("meta", {}))
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def load_json(cls, path: str | Path) -> "CurveSet":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

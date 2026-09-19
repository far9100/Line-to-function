"""Golden digests: the exact output of the baseline engine and of the synthetic generator.

Refactors that must not change behaviour (the decision-scorer refactor of
``baseline.py``, new optional ``synth`` features) are checked against digests
recorded before the change:

    python tests/golden.py --write      # record (only when a change is *meant* to alter output)

``tests/test_golden.py`` recomputes and compares them. A digest is the SHA-256
of the result's JSON (full float precision) or of the image bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from line2func import geometry as g, lineart, synth  # noqa: E402
from line2func.baseline import BaselineParams, vectorize  # noqa: E402
from line2func.curves import Curve, CurveSet  # noqa: E402
from line2func.render import render_lineart  # noqa: E402

DATA = Path(__file__).parent / "data"
BASELINE_FILE = DATA / "baseline_digests.json"
SYNTH_FILE = DATA / "synth_digests.json"
REAL_DRAWING = ROOT / "81cd1da0606ffb17d34b4d0a123d4a5f_2521654532793940545.webp"
SCENES = 20  # val_v1 scenes per subset
K = 0.5522847498


def environment() -> dict[str, str]:
    """Library versions: float results (and so the digests) are only comparable between equal versions."""
    import scipy

    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(curves: CurveSet) -> str:
    return _sha(json.dumps(curves.to_dict(), sort_keys=True).encode())


def _drawn(ctrls, width=2.0, size=128, edit=None) -> np.ndarray:
    gt = CurveSet(size, size, [Curve(c, stroke=i) for i, c in enumerate(ctrls)])
    img = render_lineart(gt, size, size, line_width=width)
    if edit is not None:
        img = edit(img.copy())
    return 1.0 - img / 255.0


def _circle(cx, cy, r):
    out = []
    for a in range(4):
        t0, t1 = a * np.pi / 2, (a + 1) * np.pi / 2
        p0 = np.array([cx + r * np.cos(t0), cy + r * np.sin(t0)])
        p3 = np.array([cx + r * np.cos(t1), cy + r * np.sin(t1)])
        p1 = p0 + K * r * np.array([-np.sin(t0), np.cos(t0)])
        p2 = p3 - K * r * np.array([-np.sin(t1), np.cos(t1)])
        out.append(np.array([p0, p1, p2, p3]))
    return out


def _cut(img):
    img[:, 60:64] = 255
    return img


def _blob(img):
    yy, xx = np.mgrid[:128, :128]
    img[(yy - 90) ** 2 + (xx - 90) ** 2 < 20**2] = 0
    return img


def hand_cases() -> dict[str, np.ndarray]:
    """Small drawings for every decision the engine makes: crossings, T, H, corner, gap, fill."""
    cases = {
        "line": _drawn([g.line([10, 20], [110, 90])]),
        "circle": _drawn(_circle(64, 64, 40)),
        "t_junction": _drawn([g.line([10, 30], [118, 30]), g.line([64, 30], [64, 118])]),
        "h_bar": _drawn([g.line([50, 20], [50, 108]), g.line([60, 20], [60, 108]), g.line([50, 64], [60, 64])]),
        "corner": _drawn([g.line([20, 20], [20, 100]), g.line([20, 100], [100, 100])]),
        "gap": _drawn([g.line([10, 64], [118, 64])], edit=_cut),
        "fill": _drawn([g.line([10, 20], [118, 20])], edit=_blob),
        "wave": _drawn([np.array([[10, 64], [40, 0], [80, 128], [118, 64]], float)], width=3.0),
    }
    for angle in (90, 30, 20, 15, 10):
        t = np.tan(np.radians(angle / 2)) * 54
        cases[f"cross_{angle}"] = _drawn([g.line([10, 64 - t], [118, 64 + t]), g.line([10, 64 + t], [118, 64 - t])])
    for i in range(3):
        _, _, img = synth.random_scene(np.random.default_rng(100 + i))
        cases[f"random_{i}"] = 1.0 - img / 255.0
    return cases


def val_scene(kind: str, i: int) -> synth.Scene:
    seed = {"clean": 10_001, "hard": 10_002}[kind]
    return synth.make_scene(np.random.default_rng([seed, i]), 512, 512, kind)


def baseline_digests() -> dict[str, str]:
    out = {name: digest(vectorize(ink)) for name, ink in hand_cases().items()}
    for kind in ("clean", "hard"):
        for i in range(SCENES):
            ink = lineart.extract(val_scene(kind, i).image, "none")
            out[f"val_{kind}_{i:02d}"] = digest(vectorize(ink))
            if i < 3:  # the finest tolerance --curves traces at, and the tracing without faint strokes
                out[f"val_{kind}_{i:02d}_tol025"] = digest(vectorize(ink, BaselineParams(fit_tolerance=0.25)))
                out[f"val_{kind}_{i:02d}_nofaint"] = digest(vectorize(ink, BaselineParams(faint_lines=False)))
    return out


def real_digests() -> dict[str, str]:
    """The first real drawing at 1x and 2x (optional: skipped when the image is not present)."""
    if not REAL_DRAWING.is_file():
        return {}
    from line2func import pipeline

    rgb = lineart.load_rgb(REAL_DRAWING)
    ink = lineart.extract(rgb, "none")
    h, w = ink.shape
    big = pipeline._resize_ink(ink, (2 * w, 2 * h))
    return {"real1_1x": digest(vectorize(ink)), "real1_2x": digest(vectorize(big, BaselineParams(fit_tolerance=2.0)))}


def synth_digests() -> dict[str, str]:
    out = {}
    for kind in ("clean", "hard"):
        for i in range(10):
            scene = val_scene(kind, i)
            gt = json.dumps(scene.gt.to_dict(), sort_keys=True).encode()
            out[f"scene_{kind}_{i:02d}"] = _sha(scene.image.tobytes() + gt)
        for i in range(5):
            img, ctrl, widths = synth.single_curve_sample(np.random.default_rng([7, i]), 64, kind)
            out[f"single_{kind}_{i}"] = _sha(img.tobytes() + ctrl.tobytes() + widths.tobytes())
            img, ctrls, widths = synth.multi_curve_sample(np.random.default_rng([8, i]), 64, kind)
            out[f"multi_{kind}_{i}"] = _sha(img.tobytes() + ctrls.tobytes() + widths.tobytes())
    return out


def main() -> int:
    if "--write" not in sys.argv[1:]:
        print(__doc__)
        return 1
    DATA.mkdir(exist_ok=True)
    base = {"_env": environment(), **baseline_digests(), **real_digests()}
    BASELINE_FILE.write_text(json.dumps(base, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    syn = {"_env": environment(), **synth_digests()}
    SYNTH_FILE.write_text(json.dumps(syn, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(base)} baseline and {len(syn)} synth digests to {DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

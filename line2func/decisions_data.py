"""Training data for learned decision scorers: labelled candidates from synthetic scenes.

    python -m line2func.decisions_data --presets hard:4000,hard2:3000,thin:2000,clean:1000 \\
        --upstream rules,oracle --workers 8 --out data/decisions_v1

Each scene is generated, extracted and (like ``pipeline.trace``) enlarged 2x
when its lines are thin. It is then traced twice with a :class:`Recorder`:
* **rules upstream** - the rules decide, so later candidates look as they
  will at first;
* **oracle upstream** - the ground truth decides, so later candidates also
  look as they will once earlier decisions improve.

Candidates are generated with the wide limits a learned scorer uses, and every
one is labelled against the ground truth (:mod:`line2func.labels`).

Output: ``shard_XXXX.npz`` files (no pickles). Per decision kind ``k`` they hold:

* ``k__x`` - features (n, F) float32;
* ``k__y`` - label, int8: 1, 0, or -1 (ambiguous);
* ``k__group`` - int64 ``scene << 20 | group``: the node for junctions, the edge for crossings, the stroke for corners;
* ``k__ids`` - int32 (n, 3): the candidate's ids;
* ``k__rule`` - float32 (n, 3): the rule's score, gate and choice.

A ``meta`` JSON string holds the presets, seeds, scene numbers, upstream and
upscale factors, and the feature version. Seeds start at 3,000,000, apart from
val_v1/val_v2/tune_v1 and the neural engine's streams, so any row can be regenerated.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SEED_BASE = 3_000_000
# Fixed: a preset's seed is SEED_BASE + its position, so new presets are only ever appended - inserting
# one would re-seed every preset after it and silently change data that has already been generated.
PRESET_ORDER = ("hard", "hard2", "thin", "clean",
                "abl_jpeg", "abl_blur", "abl_noise", "abl_ink", "abl_shading", "abl_width")


class _RulesWide:
    """The rules' decisions while candidates are generated with the wide limits (for recording)."""

    def __new__(cls):
        from line2func.decisions import WIDE_LIMITS, Scorer

        class RulesWide(Scorer):
            limits = WIDE_LIMITS
            needs_features = True

            def begin(self, ctx):
                self.ctx = ctx

            def crossing(self, cands):
                # the rules only consider edges shorter than their own limit
                limit = 12.0 * self.ctx.radius + 6.0
                return [bool(c.rule and c.length < limit) for c in cands]

        return RulesWide()


def _scene(preset: str, i: int):
    from line2func import lineart, pipeline, synth

    rng = np.random.default_rng([SEED_BASE + PRESET_ORDER.index(preset), i])
    scene = synth.make_scene(rng, 512, 512, preset)
    ink = lineart.extract(scene.image, "none")
    gt = scene.gt
    factor = pipeline.choose_upscale(ink, "auto")
    if factor > 1:
        rgb = np.repeat(scene.image[:, :, None], 3, axis=2)
        big = pipeline.resize(rgb, (gt.width * factor, gt.height * factor))
        ink = lineart.extract(big, "none")
        gt = gt.scaled(float(factor), gt.width * factor, gt.height * factor)
        gt.meta["widths"] = (np.array(scene.gt.meta["widths"]) * factor).tolist()
    return ink, gt, factor


def record_scene(preset: str, i: int, upstream: str) -> dict:
    """Labelled candidate rows of one scene traced with one upstream: ``{kind: {x, y, group, ids, rule}}``."""
    from line2func import baseline
    from line2func.decision_features import Recorder
    from line2func.decisions import WIDE_LIMITS
    from line2func.labels import KINDS, GTIndex, OracleScorer, label_recording

    ink, gt, factor = _scene(preset, i)
    params = baseline.BaselineParams()
    scorer = _RulesWide() if upstream == "rules" else OracleScorer(gt, KINDS, WIDE_LIMITS)
    rec = Recorder(params)
    baseline.vectorize(ink, params, scorer=scorer, recorder=rec)
    labels = label_recording(rec, GTIndex(gt))
    out = {}
    for kind in rec.x:
        x, rule, _ = rec.table(kind)
        if kind in ("junction", "junction_end"):
            groups = [ids[0] for ids in rec.ids[kind]]
            ids = [(ids[0], ids[1], ids[2]) for ids in rec.ids[kind]]
        elif kind == "gap":
            groups = [0] * len(x)
            ids = [(a, b, -1) for a, b in rec.ids[kind]]
        elif kind == "crossing":
            groups = [ids[0] for ids in rec.ids[kind]]
            ids = [tuple(ids) for ids in rec.ids[kind]]
        else:  # corner: (stroke, sample)
            groups = [ids[0] for ids in rec.ids[kind]]
            ids = [(a, b, -1) for a, b in rec.ids[kind]]
        out[kind] = {
            "x": x,
            "y": labels[kind],
            "group": np.asarray(groups, dtype=np.int64).reshape(-1),
            "ids": np.asarray(ids, dtype=np.int32).reshape(-1, 3),
            "rule": rule,
        }
    return {"rows": out, "factor": factor}


def _work(job: tuple) -> tuple:
    shard, items = job
    rows: dict[str, dict[str, list]] = {}
    scenes = []
    for uid, (preset, i, upstream) in items:
        result = record_scene(preset, i, upstream)
        scenes.append({"uid": uid, "preset": preset, "index": i, "upstream": upstream, "upscale": result["factor"]})
        for kind, cols in result["rows"].items():
            dest = rows.setdefault(kind, {k: [] for k in ("x", "y", "group", "ids", "rule")})
            for key, value in cols.items():
                if key == "group":
                    value = (np.int64(uid) << 20) | value
                dest[key].append(value)
    return shard, rows, scenes


def write_shard(path: Path, rows: dict, scenes: list, names: dict, features_version: int) -> None:
    arrays = {}
    for kind, cols in rows.items():
        for key, parts in cols.items():
            arrays[f"{kind}__{key}"] = np.concatenate(parts) if parts else np.zeros(0)
        arrays[f"{kind}__names"] = np.array(names[kind])
    meta = {"format": "line2func.decisions_data", "version": 1, "features_version": features_version,
            "seed_base": SEED_BASE, "preset_order": list(PRESET_ORDER), "scenes": scenes}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load(folder: str | Path) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    """All shards of a data folder: ``({kind: {x, y, group, ids, rule}}, scene list)``.

    Every shard must hold the feature version this line2func computes. Shards of
    two generations in one folder would otherwise be concatenated into a table
    whose columns mean different things in different rows, and only a shape
    mismatch would give it away - which is luck, not a check.
    """
    from line2func.decision_features import FEATURES_VERSION

    rows: dict[str, dict[str, list]] = {}
    scenes = []
    for path in sorted(Path(folder).glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            version = meta.get("features_version")
            if version != FEATURES_VERSION:
                raise ValueError(f"{path} holds features version {version}, this line2func computes "
                                 f"{FEATURES_VERSION}; regenerate the folder (python -m line2func.decisions_data)")
            scenes += meta["scenes"]
            for name in data.files:
                if "__" not in name or name.endswith("__names"):
                    continue
                kind, key = name.split("__")
                rows.setdefault(kind, {}).setdefault(key, []).append(data[name])
    return {k: {c: np.concatenate(v) for c, v in cols.items()} for k, cols in rows.items()}, scenes


def main(argv: list[str] | None = None) -> int:
    from line2func.decision_features import FEATURES_VERSION, NAMES

    p = argparse.ArgumentParser(prog="python -m line2func.decisions_data", description=__doc__.splitlines()[0])
    p.add_argument("--presets", default="hard:4000,hard2:3000,thin:2000,clean:1000",
                   help="preset:count list (presets: " + ", ".join(PRESET_ORDER) + ")")
    p.add_argument("--upstream", default="rules,oracle", help="rules and/or oracle")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-shard", type=int, default=50, help="scene tracings per shard")
    p.add_argument("--out", type=Path, default=Path("data/decisions_v1"))
    args = p.parse_args(argv)

    items = []
    for part in args.presets.split(","):
        preset, count = part.split(":")
        if preset not in PRESET_ORDER:
            p.error(f"unknown preset {preset!r}")
        for i in range(int(count)):
            for upstream in args.upstream.split(","):
                items.append((preset, i, upstream.strip()))
    items = list(enumerate(items))
    jobs = [(k, items[s: s + args.per_shard]) for k, s in enumerate(range(0, len(items), args.per_shard))]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard, rows, scenes in pool.map(_work, jobs):
            write_shard(args.out / f"shard_{shard:04d}.npz", rows, scenes, NAMES, FEATURES_VERSION)
            done += len(scenes)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"shard {shard:4d}: {done}/{len(items)} tracings, {rate:.1f}/s, "
                  f"~{(len(items) - done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
    print(f"wrote {len(jobs)} shards to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

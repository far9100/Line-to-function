"""Evaluate vectorizers on synthetic ground truth.

    python -m line2func.synth valset --out data/val_v1           # once: write the standard set
    python -m line2func.eval --valset data/val_v1                    # the baseline engine
    python -m line2func.eval --ckpt runs/single/best.pt           # single-curve model vs baseline
    python -m line2func.eval --ckpt runs/m3/best.pt               # multi-curve model vs baseline (patches)
    python -m line2func.eval --ckpt runs/m3/best.pt --valset data/val_v1 --fullset data/full_v1
                                   # whole drawings: both engines, the enabling conditions,
                                   # and a blind-test kit for real drawings in data/full_v1
    python -m line2func.eval --valset data/val_v1 --decisions learned   # a decision scorer vs the rules

Scene evaluation reports the metrics of the README's Evaluation section per
subset (clean, hard, hard2, thin): F_GT@2, crossing continuity, gap closure,
fragments per stroke, curve count ratio and CPU seconds per megapixel, plus the
decision metrics. ``--decisions`` also checks a scorer's gates (G1-G7). Single-curve
evaluation compares the model with the baseline engine on the same degraded
patches.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from line2func import baseline, lineart
from line2func.curves import CurveSet
from line2func.metrics import chamfer, decision_counts, decision_scores, f_score, structure_scores
from line2func.synth import load_scene_dir

EVAL_SEED = 2_000_003  # single-curve evaluation patches (disjoint from training streams)


def _subsets(valset: Path) -> dict[str, Path]:
    if any(valset.glob("*.json")):
        return {valset.name: valset}
    subs = {p.name: p for p in sorted(valset.iterdir()) if p.is_dir() and any(p.glob("*.json"))}
    if not subs:
        raise FileNotFoundError(f"no scenes in {valset}; create them with: python -m line2func.synth valset --out {valset}")
    return subs


def eval_scenes(valset: str | Path, limit: int | None = None, vectorize=None, pass_gt: bool = False) -> dict[str, dict]:
    """Evaluation metrics per subset of a scene set; ``vectorize(ink) -> CurveSet`` (default: baseline).

    Besides the structure metrics, every subset gets the decision metrics of
    :func:`line2func.metrics.decision_scores` (pooled over its scenes), which
    also penalize wrong joins. With ``pass_gt`` the engine is called as
    ``vectorize(ink, gt)`` (oracle experiments).
    """
    vectorize = vectorize or baseline.vectorize
    results = {}
    for name, folder in _subsets(Path(valset)).items():
        scenes = load_scene_dir(folder)[:limit]
        f, frag = [], []
        cont = [0.0, 0]
        clos = [0.0, 0]
        n_pred = n_gt = 0
        seconds = megapixels = 0.0
        pooled: dict[str, float] = {}
        for png, gt in scenes:
            ink = lineart.extract(png, "none")
            t = time.perf_counter()
            pred = vectorize(ink, gt) if pass_gt else vectorize(ink)
            seconds += time.perf_counter() - t
            for key, value in decision_counts(pred, gt).items():
                pooled[key] = pooled.get(key, 0.0) + value
            megapixels += gt.width * gt.height / 1e6
            f.append(f_score(pred, gt)["f"])
            s = structure_scores(pred, gt)
            if s["crossing_checks"]:
                cont[0] += s["crossing_continuity"] * s["crossing_checks"]
                cont[1] += s["crossing_checks"]
            if s["gap_checks"]:
                clos[0] += s["gap_closure"] * s["gap_checks"]
                clos[1] += s["gap_checks"]
            if not np.isnan(s["fragments_per_stroke"]):
                frag.append(s["fragments_per_stroke"])
            n_pred += len(pred)
            n_gt += len(gt)
        results[name] = {
            "scenes": len(scenes),
            "f_gt2": float(np.mean(f)) if f else float("nan"),
            "crossing_continuity": cont[0] / cont[1] if cont[1] else float("nan"),
            "gap_closure": clos[0] / clos[1] if clos[1] else float("nan"),
            "fragments_per_stroke": float(np.mean(frag)) if frag else float("nan"),
            "curve_ratio": n_pred / max(1, n_gt),
            "seconds_per_mp": seconds / max(megapixels, 1e-9),
            "crossing_checks": cont[1],
            "gap_checks": clos[1],
            **decision_scores(pooled),
            "decision_counts": pooled,
        }
    return results


def eval_single_curve(ckpt: str | Path, samples: int = 1000, kind: str = "hard", device: str | None = None) -> dict:
    """Chamfer distance (px) of the model and of the baseline on the same patches."""
    import torch

    from line2func.model.data import SingleCurveDataset, stack
    from line2func.model.nets import load_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(ckpt, dev)
    ps = int(meta["hparams"]["patch_size"])
    ink, ctrl, _ = stack(SingleCurveDataset(EVAL_SEED, samples, patch_size=ps, kind=kind, clean_fraction=0.0), samples)
    with torch.no_grad():
        preds = [model(ink[i : i + 512].to(dev))["ctrl"].float().cpu() for i in range(0, samples, 512)]
    pred_ctrl = torch.cat(preds).numpy().astype(np.float64) * ps
    gt_ctrl = ctrl.numpy().astype(np.float64) * ps

    model_d, base_d, base_counts = [], [], []
    for i in range(samples):
        gt = [gt_ctrl[i]]
        model_d.append(chamfer([pred_ctrl[i]], gt))
        traced = baseline.vectorize(ink[i, 0].numpy())
        base_counts.append(len(traced))
        # an empty result counts as missing the whole curve
        base_d.append(chamfer(traced, gt) if len(traced) else float(ps))

    def summary(d: list[float]) -> dict:
        a = np.minimum(np.array(d), ps)
        return {
            "mean_px": float(a.mean()),
            "median_px": float(np.median(a)),
            "within_0.5px": float(np.mean(a <= 0.5)),
            "within_1px": float(np.mean(a <= 1.0)),
        }

    counts = np.array(base_counts)
    return {
        "samples": samples,
        "kind": kind,
        "model": summary(model_d),
        "baseline": summary(base_d) | {"one_curve": float(np.mean(counts == 1)), "none": float(np.mean(counts == 0))},
    }


def eval_multi_curve(ckpt: str | Path, samples: int = 1000, kind: str = "hard", device: str | None = None,
                     threshold: float = 0.5) -> dict:
    """Per-patch F@1px / F@2px of the multi-curve model and of the baseline on the same patches."""
    import torch

    from line2func.model.data import MultiCurveDataset, stack
    from line2func.model.nets import load_checkpoint

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_checkpoint(ckpt, dev)
    hp = meta["hparams"]
    ps = int(hp["patch_size"])
    ds = MultiCurveDataset(EVAL_SEED, samples, patch_size=ps, kind=kind, clean_fraction=0.0,
                           max_curves=int(hp["queries"]))
    ink, ctrl, _, mask = stack(ds, samples)
    keep, pred = [], []
    with torch.no_grad():
        for i in range(0, samples, 256):
            out = model(ink[i : i + 256].to(dev))
            keep.append((torch.sigmoid(out["logit"]) > threshold).cpu())
            pred.append(out["ctrl"].float().cpu())
    keep_np = torch.cat(keep).numpy()
    pred_np = torch.cat(pred).numpy().astype(np.float64) * ps
    gt_np = ctrl.numpy().astype(np.float64) * ps
    mask_np = mask.numpy() > 0.5

    rows = {"model": [], "baseline": []}
    for i in range(samples):
        gt = list(gt_np[i][mask_np[i]])
        candidates = {"model": list(pred_np[i][keep_np[i]]), "baseline": baseline.vectorize(ink[i, 0].numpy()).curves}
        for who, curves in candidates.items():
            f1 = f_score(curves, gt, threshold=1.0)
            f2 = f_score(curves, gt, threshold=2.0)
            rows[who].append((f1["f"], f2["f"], f1["precision"], f1["recall"], len(curves), len(gt)))

    def summary(r: list) -> dict:
        a = np.array(r, dtype=np.float64)
        return {"f1px": float(a[:, 0].mean()), "f2px": float(a[:, 1].mean()), "precision_1px": float(a[:, 2].mean()),
                "recall_1px": float(a[:, 3].mean()), "curve_ratio": float(a[:, 4].sum() / max(a[:, 5].sum(), 1)),
                "count_exact": float(np.mean(a[:, 4] == a[:, 5]))}

    return {"samples": samples, "kind": kind, "task": "multi_curve",
            "model": summary(rows["model"]), "baseline": summary(rows["baseline"])}


def _print_multi(r: dict) -> None:
    print(f"{r['samples']} {r['kind']} multi-curve patches (per-patch averages):")
    print(f"{'':<10}{'F@1px':>8}{'F@2px':>8}{'P@1px':>8}{'R@1px':>8}{'curves/GT':>11}{'count ok':>10}")
    for who in ("model", "baseline"):
        s = r[who]
        print(f"{who:<10}{s['f1px']:>8.3f}{s['f2px']:>8.3f}{s['precision_1px']:>8.3f}{s['recall_1px']:>8.3f}"
              f"{s['curve_ratio']:>11.2f}{s['count_exact']:>10.3f}")


def gate(base: dict, model: dict) -> list[dict]:
    """The conditions for enabling the neural engine (README, Evaluation), checked on synthetic scenes.

    The third condition (a blind test on real drawings, >= 60% rated better or
    tied) needs human raters; see :mod:`line2func.blindtest`.
    """
    checks = []
    if "hard" in base and "hard" in model:
        b, m = base["hard"], model["hard"]
        for key, label in (("crossing_continuity", "hard: crossing continuity >= baseline + 0.05"),
                           ("gap_closure", "hard: gap closure >= baseline + 0.05")):
            checks.append({"check": label, "model": m[key], "need": b[key] + 0.05, "ok": m[key] >= b[key] + 0.05})
        checks.append({"check": "hard: F_GT@2 >= baseline", "model": m["f_gt2"], "need": b["f_gt2"],
                       "ok": m["f_gt2"] >= b["f_gt2"]})
    if "clean" in base and "clean" in model:
        b, m = base["clean"], model["clean"]
        checks.append({"check": "clean: F_GT@2 >= baseline - 0.01", "model": m["f_gt2"], "need": b["f_gt2"] - 0.01,
                       "ok": m["f_gt2"] >= b["f_gt2"] - 0.01})
    return checks


def traced(ink: np.ndarray, scorer=None, upscale: str = "1") -> CurveSet:
    """The baseline engine with a decision scorer; ``upscale="auto"`` traces thin lines at 2x like ``pipeline.trace``."""
    from line2func import pipeline

    factor = pipeline.choose_upscale(ink, "auto") if upscale == "auto" else 1
    if factor == 1:
        return baseline.vectorize(ink, scorer=scorer)
    h, w = ink.shape
    big = pipeline._resize_ink(ink, (w * factor, h * factor))
    params = baseline.BaselineParams(fit_tolerance=float(factor))
    return baseline.vectorize(big, params, scorer=scorer).scaled(1.0 / factor, w, h)


def gate_decisions(ref: dict, cand: dict, gaps: bool = True, corners: bool = True) -> list[dict]:
    """Conditions for switching a decision scorer on, against the rules it replaces (``ref``).

    G1-G7 of the decision-scorer plan; G8 (real drawings) is checked by hand.
    ``gaps`` / ``corners``: whether the candidate changes gap linking / corner
    detection (their conditions are skipped otherwise).
    """
    checks = []

    def check(label: str, value: float, need: float, at_least: bool = True) -> None:
        ok = bool(np.isfinite(value) and (value >= need if at_least else value <= need))
        checks.append({"check": label, "model": value, "need": need, "ok": ok, "at_least": at_least})

    common = [k for k in cand if k in ref]
    for sub in ("hard", "hard2"):
        if sub in common:
            check(f"G1 {sub}: BCubed F >= rules + 0.01", cand[sub]["bcubed_f"], ref[sub]["bcubed_f"] + 0.01)
    if "hard" in common:
        r, c = ref["hard"], cand["hard"]
        # half of what perfect decisions gain on val_v1 hard (+0.032, python -m line2func.labels oracle)
        check("G2 hard: crossing continuity >= rules + 0.016", c["crossing_continuity"], r["crossing_continuity"] + 0.016)
        if gaps:
            check("G3 hard: gap closure >= rules + 0.05", c["gap_closure"], r["gap_closure"] + 0.05)
    for sub in common:
        r, c = ref[sub], cand[sub]
        check(f"G4 {sub}: BCubed P >= rules - 0.002", c["bcubed_p"], r["bcubed_p"] - 0.002)
        for key, label in (("joins_other_per100", "wrong joins / 100 strokes"),
                           ("t_false_cont", "T stems continued"), ("crossing_false_turn", "turns at crossings")):
            if np.isfinite(r[key]):
                check(f"G4 {sub}: {label} <= rules", c[key], r[key], at_least=False)
        check(f"G5 {sub}: F_GT@2 >= rules - 0.002", c["f_gt2"], r["f_gt2"] - 0.002)
        check(f"G5 {sub}: curve ratio <= rules + 0.02", c["curve_ratio"], r["curve_ratio"] + 0.02, at_least=False)
        check(f"G7 {sub}: s/MP <= 1.15 x rules", c["seconds_per_mp"], 1.15 * r["seconds_per_mp"], at_least=False)
    if "clean" in common:
        r, c = ref["clean"], cand["clean"]
        for key in ("crossing_continuity", "bcubed_f", "crossing_both_ok", "t_bar_ok"):
            if np.isfinite(r[key]):
                check(f"G5 clean: {key} >= rules - 0.005", c[key], r[key] - 0.005)
    if corners:
        if "hard2" in common:
            check("G6 hard2: corner F >= rules + 0.03", cand["hard2"]["corner_f"], ref["hard2"]["corner_f"] + 0.03)
        for sub in ("clean", "hard"):
            if sub in common:
                check(f"G6 {sub}: corner P >= rules - 0.01", cand[sub]["corner_p"], ref[sub]["corner_p"] - 0.01)
    return checks


def _print_decisions(results: dict) -> None:
    cols = [("bcubed_p", "BCubed P"), ("bcubed_r", "BCubed R"), ("bcubed_f", "BCubed F"),
            ("crossing_both_ok", "X both ok"), ("crossing_false_turn", "X turn"), ("t_bar_ok", "T bar ok"),
            ("t_false_cont", "T stem cont"), ("joins_other_per100", "joins/100"), ("corner_p", "corner P"),
            ("corner_r", "corner R")]
    print(f"{'subset':<8}" + "".join(f"{h:>12}" for _, h in cols))
    for name, r in results.items():
        print(f"{name:<8}" + "".join(f"{r[k]:>12.3f}" for k, _ in cols))


def _print_gate_decisions(checks: list[dict]) -> None:
    print("decision scorer gates:")
    for c in checks:
        sign = ">=" if c["at_least"] else "<="
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['model']:.3f} (need {sign} {c['need']:.3f})")


def _print_scenes(results: dict) -> None:
    cols = [("f_gt2", "F_GT@2"), ("crossing_continuity", "cross cont."), ("gap_closure", "gap closure"),
            ("fragments_per_stroke", "frag/stroke"), ("curve_ratio", "curve ratio"), ("seconds_per_mp", "s/MP")]
    print(f"{'subset':<8}{'scenes':>7}" + "".join(f"{h:>13}" for _, h in cols))
    for name, r in results.items():
        print(f"{name:<8}{r['scenes']:>7}" + "".join(f"{r[k]:>13.3f}" for k, _ in cols))


def _print_gate(checks: list[dict]) -> None:
    print("enabling conditions:")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['model']:.3f} (need >= {c['need']:.3f})")
    print("  [ -- ] real drawings: blind test, >= 60% better or tied (run with --fullset DIR)")


def _print_single(r: dict) -> None:
    print(f"{r['samples']} {r['kind']} patches, chamfer distance to the true curve (px):")
    print(f"{'':<10}{'mean':>8}{'median':>8}{'<=0.5px':>9}{'<=1px':>8}")
    for who in ("model", "baseline"):
        s = r[who]
        print(f"{who:<10}{s['mean_px']:>8.3f}{s['median_px']:>8.3f}{s['within_0.5px']:>9.3f}{s['within_1px']:>8.3f}")
    b = r["baseline"]
    print(f"baseline returned exactly one curve on {b['one_curve']:.1%} of patches, nothing on {b['none']:.1%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.eval", description="Evaluate vectorizers.")
    p.add_argument("--ckpt", type=Path, help="model checkpoint")
    p.add_argument("--valset", type=Path, help="scene set (e.g. data/val_v1)")
    p.add_argument("--fullset", type=Path, help="folder of real drawings: build a blind-test kit (needs --ckpt)")
    p.add_argument("--samples", type=int, default=1000, help="single-curve patches to evaluate")
    p.add_argument("--kind", default="hard", help="degradation preset for single-curve patches")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most this many scenes per subset")
    p.add_argument("--device", default=None)
    p.add_argument("--decisions", default=None, metavar="SCORER",
                   help="with --valset: compare this decision scorer (learned or a weights path) with the rules")
    p.add_argument("--upscale", choices=("1", "auto"), default="1",
                   help="with --decisions: trace thin lines at 2x like pipeline.trace (auto) or not (1)")
    p.add_argument("--json", type=Path, help="also write the results to this JSON file")
    args = p.parse_args(argv)

    if args.ckpt is not None and (args.valset is not None or args.fullset is not None):
        from line2func.model.infer import load_vectorizer

        run = load_vectorizer(str(args.ckpt), args.device)
        results = {}
        if args.valset is not None:
            print("baseline engine:")
            base = eval_scenes(args.valset, args.limit)
            _print_scenes(base)
            print("model engine:")
            model = eval_scenes(args.valset, args.limit, vectorize=run)
            _print_scenes(model)
            checks = gate(base, model)
            _print_gate(checks)
            results = {"baseline": base, "model": model, "gate": checks}
        if args.fullset is not None:
            from line2func.blindtest import make_kit

            out = args.ckpt.parent / "blindtest"
            n = make_kit(args.fullset, out, run)
            print(f"blind test kit: {n} drawings -> {out / 'index.html'}")
            print(f"  raters open it, vote, and export votes; then: python -m line2func.blindtest score {out} VOTES.json ...")
            results["blindtest_kit"] = str(out)
    elif args.ckpt is not None:
        import torch

        task = torch.load(args.ckpt, map_location="cpu", weights_only=False)["task"]
        if task == "multi_curve":
            results = eval_multi_curve(args.ckpt, args.samples, args.kind, args.device)
            _print_multi(results)
        else:
            results = eval_single_curve(args.ckpt, args.samples, args.kind, args.device)
            _print_single(results)
    elif args.valset is not None and args.decisions:
        from line2func.decisions import load_scorer

        scorer = load_scorer(args.decisions)
        print("rules:")
        ref = eval_scenes(args.valset, args.limit, vectorize=lambda ink: traced(ink, None, args.upscale))
        _print_scenes(ref)
        _print_decisions(ref)
        print(f"decisions by {args.decisions}:")
        cand = eval_scenes(args.valset, args.limit, vectorize=lambda ink: traced(ink, scorer, args.upscale))
        _print_scenes(cand)
        _print_decisions(cand)
        checks = gate_decisions(ref, cand)
        _print_gate_decisions(checks)
        results = {"rules": ref, "decisions": cand, "gate": checks}
    elif args.valset is not None:
        results = eval_scenes(args.valset, args.limit)
        _print_scenes(results)
        _print_decisions(results)
    else:
        p.error("give --valset (baseline scenes) or --ckpt (single-curve model)")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

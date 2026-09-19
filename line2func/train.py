"""Train a line2func model.

    python -m line2func.train --config configs/overfit.yaml              # quick self-check
    python -m line2func.train --config configs/single_curve.yaml --device cuda
    python -m line2func.train --config configs/multi_curve.yaml --device cuda
    python -m line2func.train --config configs/multi_curve.yaml --resume   # continue last.pt

Every run stops cleanly after ``train.max_wall_hours`` (default 36, or
``--max-hours``): it saves ``last.pt`` and exits, and ``--resume`` continues
from there. ``last.pt`` is also written every ``train.checkpoint_minutes`` and
on Ctrl+C. It holds the model, optimizer, scheduler, AMP scaler, step and RNG
states; the data stream is a pure function of (seed, sample index), so a
resumed run sees exactly the samples an uninterrupted run would have seen.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

TASK_NAMES = ("single_curve", "multi_curve")

DEFAULTS: dict = {
    "name": "run",
    "task": "single_curve",
    "seed": 0,
    "out_dir": None,  # default: runs/<name>
    "data": {"patch_size": 64, "kind": "hard", "clean_fraction": 0.25, "overfit_samples": 0},
    "model": {},  # network hyperparameters; each network has its own defaults
    "train": {
        "steps": 30000,
        "batch_size": 256,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "warmup_steps": 500,
        "min_lr_ratio": 0.05,
        "amp": "bf16",  # bf16 | fp16 | none
        "grad_clip": 1.0,
        "num_workers": 6,
        "log_every": 100,
        "eval_every": 2000,
        "val_samples": 2000,
        "checkpoint_minutes": 30,
        "max_wall_hours": 36.0,
    },
    # final-metric limits, e.g. {"max_mean_px": 1.0} or {"min_f1": 0.9}; a miss fails the run
    "check": {},
}


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None, overrides: dict | None = None) -> dict:
    """Defaults <- YAML file <- ``overrides`` (nested dicts)."""
    cfg = DEFAULTS
    if path is not None:
        import yaml

        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    cfg = _merge(cfg, overrides or {})
    if cfg["task"] not in TASK_NAMES:
        raise ValueError(f"unknown task {cfg['task']!r}; available: {', '.join(TASK_NAMES)}")
    if cfg["out_dir"] is None:
        cfg["out_dir"] = str(Path("runs") / cfg["name"])
    if cfg["train"]["amp"] not in ("bf16", "fp16", "none"):
        raise ValueError("train.amp must be bf16, fp16 or none")
    if not cfg["train"]["max_wall_hours"] > 0:
        raise ValueError("train.max_wall_hours must be positive")
    return cfg


def check_metrics(check: dict, metrics: dict) -> list[str]:
    """Failed ``max_<metric>`` / ``min_<metric>`` limits, as messages (empty = all passed)."""
    failures = []
    for key, limit in (check or {}).items():
        kind, _, name = key.partition("_")
        value = metrics.get(name, float("nan"))
        ok = value <= limit if kind == "max" else value >= limit if kind == "min" else False
        if not ok:
            failures.append(f"{name} = {value:.4f} (limit {key}: {limit})")
    return failures


def _worker_init(_worker_id: int) -> None:
    import torch

    torch.set_num_threads(1)  # one sample generator per worker; avoid oversubscribing the CPU


def _lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return f


def _save(path: Path, state: dict) -> None:
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def train(cfg: dict, device: str | None = None, resume: str | None = None) -> dict:
    """Run training; returns ``{"status", "step", "best", "metrics"}`` (``best`` = lowest val score)."""
    import torch
    from torch.utils.data import DataLoader

    from line2func.model.data import stack
    from line2func.model.nets import CKPT_FORMAT, CKPT_VERSION, build_model
    from line2func.model.tasks import make_task

    task = make_task(cfg)
    tc, dc = cfg["train"], cfg["data"]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[tc["amp"]]
    if dev.type != "cuda":
        amp_dtype = None  # CPU autocast is slow and pointless here
    model = build_model(cfg["task"], task.hparams()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(int(tc["warmup_steps"]), int(tc["steps"]), float(tc["min_lr_ratio"]))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    step, best, metrics = 0, float("inf"), {}
    if resume:
        path = out_dir / "last.pt" if resume == "auto" else Path(resume)
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        if ckpt["task"] != cfg["task"]:
            raise ValueError(f"{path} holds a {ckpt['task']} model, but the config trains {cfg['task']}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best, metrics = int(ckpt["step"]), float(ckpt["best"]), ckpt.get("metrics", {})
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        if dev.type == "cuda" and rng.get("cuda"):
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        print(f"resumed from {path} at step {step}")

    def state() -> dict:
        return {
            "format": CKPT_FORMAT,
            "version": CKPT_VERSION,
            "task": cfg["task"],
            "hparams": model.hparams,
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "best": best,
            "metrics": metrics,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    total, bs = int(tc["steps"]), int(tc["batch_size"])
    if step >= total:
        print(f"already finished ({step}/{total} steps)")
        return {"status": "done", "step": step, "best": best, "metrics": metrics}

    overfit = int(dc.get("overfit_samples") or 0)
    if overfit:
        val = stack(task.dataset(seed, overfit, overfit=overfit), overfit)
    else:
        n_val = int(tc["val_samples"])
        val = stack(task.dataset(seed + 1_000_003, n_val), n_val)

    def predict(x):
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            return model(x.to(dev))

    def evaluate() -> dict:
        model.eval()
        try:
            return task.evaluate(predict, val)
        finally:
            model.train()

    train_set = task.dataset(seed, (total - step) * bs, offset=step * bs, overfit=overfit)
    workers = int(tc["num_workers"])
    loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        pin_memory=dev.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=_worker_init if workers > 0 else None,
    )

    log_path = out_dir / "log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "loss", *task.parts, "lr", "samples_per_s", *(f"val_{k}" for k in task.val_keys)])

    cap_s = float(tc["max_wall_hours"]) * 3600.0
    ckpt_s = float(tc["checkpoint_minutes"]) * 60.0
    t_start = t_ckpt = t_log = time.monotonic()
    window = dict.fromkeys(("loss", *task.parts), 0.0)
    n_window = 0
    status = "done"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"training {cfg['task']} ({n_params:.1f}M params) on {dev} ({tc['amp'] if amp_dtype else 'fp32'}), "
          f"steps {step}->{total}, batch {bs}, workers {workers}, cap {tc['max_wall_hours']} h, out {out_dir}",
          flush=True)
    model.train()
    try:
        for batch in loader:
            batch = tuple(t.to(dev, non_blocking=True) for t in batch)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch[0])
            loss, parts = task.loss(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            window["loss"] += loss.item()
            for k in task.parts:
                window[k] += parts[k]
            n_window += 1

            evaluated = False
            if step % int(tc["eval_every"]) == 0 or step == total:
                metrics = evaluate()
                evaluated = True
                if metrics["score"] < best:
                    best = metrics["score"]
                    _save(out_dir / "best.pt", state())
            if step % int(tc["log_every"]) == 0 or evaluated:
                now = time.monotonic()
                n = max(n_window, 1)
                rate = n_window * bs / max(now - t_log, 1e-9)
                row = [step] + [round(window[k] / n, 6) for k in ("loss", *task.parts)]
                row += [f"{sched.get_last_lr()[0]:.3g}", round(rate, 1)]
                row += [round(metrics[k], 4) for k in task.val_keys] if evaluated else [""] * len(task.val_keys)
                log.writerow(row)
                log_file.flush()
                msg = f"step {step:>7}/{total}  loss {window['loss'] / n:.4f}  {rate:,.0f} samples/s"
                if evaluated:
                    msg += "  | " + task.describe(metrics)
                print(msg, flush=True)
                window = dict.fromkeys(window, 0.0)
                n_window = 0
                t_log = now

            now = time.monotonic()
            if step >= total:
                break
            if now - t_start >= cap_s:
                status = "wall_cap"
                break
            if now - t_ckpt >= ckpt_s:
                _save(out_dir / "last.pt", state())
                t_ckpt = now
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        _save(out_dir / "last.pt", state())
        log_file.close()

    hours = (time.monotonic() - t_start) / 3600.0
    if status == "wall_cap":
        print(f"stopped at the {tc['max_wall_hours']} h cap (step {step}/{total}); "
              f"continue with --resume {out_dir / 'last.pt'}")
    elif status == "interrupted":
        print(f"interrupted at step {step}; continue with --resume {out_dir / 'last.pt'}")
    else:
        print(f"finished {step} steps in {hours:.2f} h; best val score {best:.4f} -> {out_dir / 'best.pt'}")
    return {"status": status, "step": step, "best": best, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m line2func.train", description="Train a line2func model.")
    p.add_argument("--config", type=Path, required=True, help="YAML config (see configs/)")
    p.add_argument("--device", default=None, help="cuda, cuda:1, cpu (default: cuda if available)")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="continue from a checkpoint (default: <out_dir>/last.pt)")
    p.add_argument("--max-hours", type=float, default=None, help="wall-clock cap for this run (default 36)")
    p.add_argument("--out", default=None, help="output folder (default: runs/<name>)")
    args = p.parse_args(argv)
    overrides: dict = {}
    if args.max_hours is not None:
        overrides["train"] = {"max_wall_hours": args.max_hours}
    if args.out is not None:
        overrides["out_dir"] = args.out
    cfg = load_config(args.config, overrides)
    result = train(cfg, device=args.device, resume=args.resume)
    if result["status"] == "done" and cfg.get("check"):
        failures = check_metrics(cfg["check"], result["metrics"])
        print("self-check " + ("PASSED" if not failures else "FAILED: " + "; ".join(failures)))
        return 0 if not failures else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

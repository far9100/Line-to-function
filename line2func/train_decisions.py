"""Train the learned decision scorer (needs PyTorch; the result runs on numpy only).

    python -m line2func.train_decisions --data data/decisions_v1 --out runs/decisions/v1 --device cuda

One small MLP per decision kind (junction pair, arm end, gap, crossing,
corner), trained on the labelled candidates of
:mod:`line2func.decisions_data`:

* inputs clipped to their training range and standardized;
* two hidden layers of 64 (ReLU, dropout 0.1);
* binary cross-entropy with label smoothing and positive weighting;
* AdamW, batches of up to 4096 (at least ~50 steps per epoch), early
  stopping on a validation split by scene (every 10th scene);
* a scale and a shift per kind (Platt scaling: ``sigmoid(z / T + B)``),
  fitted on the validation split, calibrate the probabilities. The shift undoes
  the bias the positive weighting puts into the logits.

The weights are exported as ``decisions.npz`` (no pickles), which
:class:`line2func.decision_model.LearnedScorer` evaluates with numpy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

KINDS = ("junction", "junction_end", "gap", "crossing", "corner")
HIDDEN = 64
FORMAT = "line2func.decisions"
VERSION = 1


def _split(group: np.ndarray) -> np.ndarray:
    """True for validation rows: every 10th scene (the scene is the group's high bits)."""
    return ((group >> 20) % 10) == 0


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train_kind(x, y, group, rule, device: str, epochs: int = 30, seed: int = 0, verbose: bool = True) -> dict:
    """Fit one kind; returns its exported arrays and validation report."""
    import torch

    torch.manual_seed(seed)
    labelled = y >= 0
    x, y, group, rule = x[labelled], y[labelled].astype(np.float32), group[labelled], rule[labelled]
    val = _split(group)
    tr = ~val
    lo = np.quantile(x[tr], 0.001, axis=0).astype(np.float32)
    hi = np.quantile(x[tr], 0.999, axis=0).astype(np.float32)
    xc = np.clip(x, lo, hi)
    mu = xc[tr].mean(axis=0).astype(np.float32)
    sd = np.maximum(xc[tr].std(axis=0), 1e-6).astype(np.float32)
    xs = ((xc - mu) / sd).astype(np.float32)

    n_in = x.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(n_in, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN, 1),
    ).to(device)
    pos = float(y[tr].sum())
    neg = float(len(y[tr]) - pos)
    pos_weight = torch.tensor([min(20.0, neg / max(pos, 1.0))], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(xs[tr]).to(device)
    yt = torch.from_numpy(0.02 + 0.96 * y[tr]).to(device)  # label smoothing 0.02
    xv = torch.from_numpy(xs[val]).to(device)
    yv = torch.from_numpy(y[val]).to(device)
    best, best_state, patience = float("inf"), None, 0
    batch = int(np.clip(len(xt) // 50, 256, 4096))  # at least ~50 steps per epoch, also for the rarer kinds
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for s in range(0, len(xt), batch):
            idx = perm[s: s + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vloss = float(torch.nn.functional.binary_cross_entropy_with_logits(net(xv).squeeze(1), yv))
        if vloss < best - 1e-5:
            best, patience = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        zv = net(xv).squeeze(1).cpu().numpy().astype(np.float64)
    # Platt scaling: sigmoid(a z + b) with the validation NLL minimized
    from scipy.optimize import minimize

    yvn = y[val].astype(np.float64)

    def nll(ab):
        q = ab[0] * zv + ab[1]
        return float(np.mean(np.logaddexp(0.0, q) - yvn * q))

    fit = minimize(nll, np.array([1.0, 0.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    a, b = float(max(fit.x[0], 1e-3)), float(fit.x[1])
    best_t, best_b, best_nll = 1.0 / a, b, nll([a, b])
    pv = 1.0 / (1.0 + np.exp(-(zv / best_t + best_b)))
    acc = float(np.mean((pv >= 0.5) == (yvn > 0.5))) if len(yvn) else float("nan")
    rule_acc = float(np.mean((rule[val][:, 2] > 0.5) == (yvn > 0.5))) if len(yvn) else float("nan")
    layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    out = {"lo": lo, "hi": hi, "mu": mu, "sd": sd, "T": np.float32(best_t), "B": np.float32(best_b),
           "tau": np.float32(0.5), "delta": np.float32(0.0)}
    for k, layer in enumerate(layers):
        out[f"W{k}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        out[f"b{k}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    report = {"train": int(tr.sum()), "val": int(val.sum()), "pos_share": round(pos / max(pos + neg, 1), 4),
              "epochs": epoch + 1, "val_nll": round(best_nll, 4), "T": round(best_t, 3), "B": round(best_b, 3),
              "val_acc": round(acc, 4),
              "rule_acc": round(rule_acc, 4), "val_auc": round(_auc(yvn, pv), 4)}
    if verbose:
        print(f"  {report}", flush=True)
    return {"arrays": out, "report": report}


def export(path: Path, kinds: dict, names: dict, features_version: int, meta: dict) -> None:
    arrays = {"format": np.array(FORMAT), "version": np.array(VERSION), "features_version": np.array(features_version),
              "kinds": np.array(sorted(kinds)), "train_meta": np.array(json.dumps(meta))}
    for kind, res in kinds.items():
        arrays[f"{kind}__names"] = np.array(names[kind])
        for key, value in res["arrays"].items():
            arrays[f"{kind}__{key}"] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def main(argv: list[str] | None = None) -> int:
    from line2func.decision_features import FEATURES_VERSION, NAMES
    from line2func.decisions_data import load

    p = argparse.ArgumentParser(prog="python -m line2func.train_decisions", description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/decisions_v1"))
    p.add_argument("--out", type=Path, default=Path("runs/decisions/v1"))
    p.add_argument("--device", default=None)
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    rows, scenes = load(args.data)
    print(f"loaded {len(scenes)} tracings from {args.data} in {time.perf_counter() - t0:.0f} s")
    results = {}
    for kind in args.kinds.split(","):
        r = rows[kind]
        print(f"{kind}: {len(r['y'])} candidates, {int((r['y'] >= 0).sum())} labelled", flush=True)
        results[kind] = train_kind(r["x"], r["y"], r["group"], r["rule"], device, args.epochs)
    meta = {"data": str(args.data), "tracings": len(scenes), "reports": {k: v["report"] for k, v in results.items()},
            "seconds": round(time.perf_counter() - t0, 1)}
    export(args.out / "decisions.npz", results, NAMES, FEATURES_VERSION, meta)
    (args.out / "report.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {args.out / 'decisions.npz'} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

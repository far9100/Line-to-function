"""What differs between models: data, loss and validation metric.

A task bundles everything :mod:`line2func.train` needs to train one kind of
model, so all models share the same loop (AMP, wall-clock cap, checkpoints,
exact resume).
"""

from __future__ import annotations

import numpy as np
import torch

from line2func.model.data import MultiCurveDataset, SingleCurveDataset
from line2func.model.losses import multi_curve_loss, point_error_px, single_curve_loss


class SingleCurveTask:
    name = "single_curve"
    parts = ("ctrl", "pts", "width")
    val_keys = ("mean_px", "median_px", "acc_1px")

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ps = int(cfg["data"]["patch_size"])

    def hparams(self) -> dict:
        return {"patch_size": self.ps, **self.cfg["model"]}

    def dataset(self, seed: int, length: int, offset: int = 0, overfit: int = 0):
        d = self.cfg["data"]
        return SingleCurveDataset(seed, length, self.ps, d["kind"], d["clean_fraction"], offset, overfit)

    def loss(self, out: dict, batch: tuple):
        return single_curve_loss(out, batch[1], batch[2], self.ps)

    @torch.no_grad()
    def evaluate(self, predict, val: tuple) -> dict[str, float]:
        errors = []
        for i in range(0, len(val[0]), 512):
            out = predict(val[0][i : i + 512])
            errors.append(point_error_px(out["ctrl"], val[1][i : i + 512].to(out["ctrl"].device), self.ps).cpu())
        e = torch.cat(errors)
        m = {"mean_px": e.mean().item(), "median_px": e.median().item(), "acc_1px": (e < 1.0).float().mean().item()}
        m["score"] = m["mean_px"]  # lower is better
        return m

    def describe(self, m: dict) -> str:
        return f"val {m['mean_px']:.3f} px, acc@1px {m['acc_1px']:.3f}"


class MultiCurveTask:
    name = "multi_curve"
    parts = ("cls", "ctrl", "pts", "width")
    val_keys = ("f1", "precision", "recall", "count_acc")

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ps = int(cfg["data"]["patch_size"])
        self.queries = int(cfg["model"].get("queries", 16))

    def hparams(self) -> dict:
        return {"patch_size": self.ps, **self.cfg["model"]}

    def dataset(self, seed: int, length: int, offset: int = 0, overfit: int = 0):
        d = self.cfg["data"]
        return MultiCurveDataset(seed, length, self.ps, d["kind"], d["clean_fraction"], offset, overfit, self.queries)

    def loss(self, out: dict, batch: tuple):
        return multi_curve_loss(out, batch[1], batch[2], batch[3])

    @torch.no_grad()
    def evaluate(self, predict, val: tuple, threshold: float = 0.5) -> dict[str, float]:
        """Per-patch F@1px of curve length (predicted vs true curves), averaged."""
        from line2func.metrics import f_score

        precision, recall, f1, count_ok = [], [], [], []
        for i in range(0, len(val[0]), 512):
            out = predict(val[0][i : i + 512])
            keep = (torch.sigmoid(out["logit"]) > threshold).cpu().numpy()
            pred = out["ctrl"].cpu().numpy().astype(np.float64) * self.ps
            gt = val[1][i : i + 512].numpy().astype(np.float64) * self.ps
            mask = val[3][i : i + 512].numpy() > 0.5
            for b in range(len(pred)):
                s = f_score(list(pred[b][keep[b]]), list(gt[b][mask[b]]), threshold=1.0)
                precision.append(s["precision"])
                recall.append(s["recall"])
                f1.append(s["f"])
                count_ok.append(keep[b].sum() == mask[b].sum())
        m = {"f1": float(np.mean(f1)), "precision": float(np.mean(precision)), "recall": float(np.mean(recall)),
             "count_acc": float(np.mean(count_ok))}
        m["score"] = 1.0 - m["f1"]
        return m

    def describe(self, m: dict) -> str:
        return f"val F@1px {m['f1']:.3f} (P {m['precision']:.3f} R {m['recall']:.3f}), count acc {m['count_acc']:.3f}"


TASKS = {t.name: t for t in (SingleCurveTask, MultiCurveTask)}


def make_task(cfg: dict):
    try:
        return TASKS[cfg["task"]](cfg)
    except KeyError:
        raise ValueError(f"unknown task {cfg['task']!r}; available: {', '.join(TASKS)}") from None

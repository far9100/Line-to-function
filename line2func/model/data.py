"""On-the-fly training data.

Sample ``i`` is generated from ``numpy.random.default_rng([seed, offset + i])``,
so the data stream is reproducible, needs no disk space, and resuming at step
``k`` simply sets ``offset = k * batch_size``. The dataset is a plain
top-level class, so it pickles into Windows ``spawn`` DataLoader workers.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from line2func.synth import multi_curve_sample, single_curve_sample


class SingleCurveDataset(Dataset):
    def __init__(
        self,
        seed: int,
        length: int,
        patch_size: int = 64,
        kind: str = "hard",
        clean_fraction: float = 0.25,
        offset: int = 0,
        overfit: int = 0,
    ):
        self.seed = int(seed)
        self.length = int(length)
        self.patch_size = int(patch_size)
        self.kind = kind
        self.clean_fraction = float(clean_fraction)
        self.offset = int(offset)
        self.overfit = int(overfit)  # > 0: cycle through this many fixed samples

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        idx = self.offset + int(i)
        if self.overfit:
            idx %= self.overfit
        rng = np.random.default_rng([self.seed, idx])
        kind = "clean" if rng.random() < self.clean_fraction else self.kind
        image, ctrl, width = single_curve_sample(rng, self.patch_size, kind)
        ink = torch.from_numpy(1.0 - image.astype(np.float32) / 255.0)[None]
        return (
            ink,
            torch.from_numpy((ctrl / self.patch_size).astype(np.float32)),
            torch.from_numpy(width.astype(np.float32)),
        )


class MultiCurveDataset(Dataset):
    """Patches with up to ``max_curves`` curves, padded to a fixed size.

    Items: ``(ink (1, S, S), ctrl (N, 4, 2) patch units, width (N, 2) px, mask (N,))``.
    """

    def __init__(
        self,
        seed: int,
        length: int,
        patch_size: int = 64,
        kind: str = "hard",
        clean_fraction: float = 0.25,
        offset: int = 0,
        overfit: int = 0,
        max_curves: int = 16,
    ):
        self.seed, self.length, self.patch_size = int(seed), int(length), int(patch_size)
        self.kind, self.clean_fraction = kind, float(clean_fraction)
        self.offset, self.overfit, self.max_curves = int(offset), int(overfit), int(max_curves)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        idx = self.offset + int(i)
        if self.overfit:
            idx %= self.overfit
        rng = np.random.default_rng([self.seed, idx])
        kind = "clean" if rng.random() < self.clean_fraction else self.kind
        image, ctrls, widths = multi_curve_sample(rng, self.patch_size, kind, self.max_curves)
        n, k = self.max_curves, len(ctrls)
        ctrl = np.zeros((n, 4, 2), np.float32)
        width = np.ones((n, 2), np.float32)
        mask = np.zeros(n, np.float32)
        ctrl[:k] = ctrls / self.patch_size
        width[:k] = widths[:, None]
        mask[:k] = 1.0
        ink = torch.from_numpy(1.0 - image.astype(np.float32) / 255.0)[None]
        return ink, torch.from_numpy(ctrl), torch.from_numpy(width), torch.from_numpy(mask)


def stack(dataset: Dataset, n: int) -> tuple[torch.Tensor, ...]:
    """Materialize the first ``n`` samples as batched tensors (validation sets)."""
    items = [dataset[i] for i in range(n)]
    return tuple(torch.stack(parts) for parts in zip(*items))

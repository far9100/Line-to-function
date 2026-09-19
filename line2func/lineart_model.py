"""Pretrained line-art extraction for photos (needs PyTorch).

The generator below is ported from *Informative Drawings: Learning to Generate
Line Drawings that Convey Geometry and Semantics* (Chan, Durand, Isola,
CVPR 2022; https://github.com/carolineec/informative-drawings, MIT license,
Copyright (c) 2022 Caroline Chan), configured as in ControlNet's lineart
preprocessor: 3 residual blocks, RGB in [0, 1] -> line drawing in [0, 1]
(1 = white paper). Weights are fetched and verified by
``python -m line2func.weights fetch informative``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from line2func.weights import require

_norm = nn.InstanceNorm2d


class _ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), _norm(features), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(features, features, 3), _norm(features),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """Informative Drawings generator (attribute names match the published weights)."""

    def __init__(self, input_nc: int = 3, output_nc: int = 1, n_residual_blocks: int = 3):
        super().__init__()
        self.model0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), _norm(64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for _ in range(2):
            layers += [nn.Conv2d(cin, cin * 2, 3, stride=2, padding=1), _norm(cin * 2), nn.ReLU(inplace=True)]
            cin *= 2
        self.model1 = nn.Sequential(*layers)
        self.model2 = nn.Sequential(*[_ResidualBlock(cin) for _ in range(n_residual_blocks)])
        layers = []
        for _ in range(2):
            layers += [nn.ConvTranspose2d(cin, cin // 2, 3, stride=2, padding=1, output_padding=1),
                       _norm(cin // 2), nn.ReLU(inplace=True)]
            cin //= 2
        self.model3 = nn.Sequential(*layers)
        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Sigmoid())

    def forward(self, x):
        return self.model4(self.model3(self.model2(self.model1(self.model0(x)))))


_CACHE: dict[tuple[str, str], Generator] = {}


def load_generator(name: str = "informative", device: str | torch.device | None = None) -> Generator:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (name, str(dev))
    if key not in _CACHE:
        net = Generator()
        state = torch.load(require(name), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        _CACHE[key] = net.to(dev).eval()
    return _CACHE[key]


@torch.no_grad()
def extract(rgb: np.ndarray, name: str = "informative", device=None, max_side: int = 2048) -> np.ndarray:
    """Ink map (float32, 1 = line) of an RGB uint8 image, same size as the input.

    Images larger than ``max_side`` are processed downscaled and the result is
    scaled back, which bounds GPU memory. Sides are padded to a multiple of 4.
    """
    net = load_generator(name, device)
    dev = next(net.parameters()).device
    h, w = rgb.shape[:2]
    x = torch.from_numpy(np.array(rgb, dtype=np.uint8)).to(dev).float().permute(2, 0, 1)[None] / 255.0
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        x = F.interpolate(x, scale_factor=scale, mode="area")
    ph, pw = (-x.shape[2]) % 4, (-x.shape[3]) % 4
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    y = net(x)[:, :, : x.shape[2] - ph, : x.shape[3] - pw]
    if scale < 1.0:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return (1.0 - y[0, 0].clamp(0, 1)).float().cpu().numpy()

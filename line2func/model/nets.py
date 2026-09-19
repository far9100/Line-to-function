"""Network definitions.

The encoder is a small ResNet with CoordConv inputs (two extra channels holding
the x and y position), which makes absolute positions easy to regress. The
single-curve model reads one ``patch_size``² ink patch and predicts the four
control points of the cubic in it, plus its start and end width.
:class:`MultiCurveNet`, the one the neural engine uses, predicts a set of up
to ``queries`` cubics per patch.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_FORMAT = "line2func.ckpt"
CKPT_VERSION = 1

# control points may lie outside the patch even when the curve is inside it
CTRL_LOW, CTRL_HIGH = -0.25, 1.25


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (
            nn.Identity()
            if cin == cout and stride == 1
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class Encoder(nn.Module):
    """``(B, 1, H, W)`` ink -> ``(B, channels[-1], H / 2^(n-1), W / 2^(n-1))`` features."""

    def __init__(self, channels=(32, 64, 128, 256), blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True)
        )
        layers = []
        prev = channels[0]
        for i, c in enumerate(channels):
            layers.append(ResBlock(prev, c, stride=1 if i == 0 else 2))
            layers += [ResBlock(c, c) for _ in range(blocks - 1)]
            prev = c
        self.stages = nn.Sequential(*layers)

    @staticmethod
    def coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(self.coords(x)))


class SingleCurveNet(nn.Module):
    """Predicts one cubic: ``ctrl`` ``(B, 4, 2)`` in patch units (0-1) and ``width`` ``(B, 2)`` in px."""

    def __init__(self, patch_size: int = 64, channels=(32, 64, 128, 256), blocks: int = 2, hidden: int = 512):
        super().__init__()
        self.hparams = {"patch_size": patch_size, "channels": list(channels), "blocks": blocks, "hidden": hidden}
        self.encoder = Encoder(channels, blocks)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.pool(self.encoder(x))
        # fp32 head: bf16 keeps 8 mantissa bits, i.e. ~0.25 px steps at 64 px
        with torch.autocast(x.device.type, enabled=False):
            out = self.head(f.float())
        ctrl = torch.sigmoid(out[:, :8]).view(-1, 4, 2) * (CTRL_HIGH - CTRL_LOW) + CTRL_LOW
        width = 0.3 + F.softplus(out[:, 8:10])
        return {"ctrl": ctrl, "width": width}


class _MLP(nn.Sequential):
    def __init__(self, d: int, out: int):
        super().__init__(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, out))


class MultiCurveNet(nn.Module):
    """DETR-style set prediction of up to ``queries`` cubics in one patch.

    CNN features (stride 8) become tokens for a small Transformer encoder;
    learned queries attend to them in a Transformer decoder. Every decoder
    layer's output goes through the shared heads (for auxiliary losses); the
    last one is the prediction. Per query: ``ctrl`` ``(4, 2)`` in patch units,
    ``width`` ``(2,)`` px and ``logit`` (does this query hold a curve?).
    """

    def __init__(
        self,
        patch_size: int = 64,
        channels=(32, 64, 128, 256),
        blocks: int = 2,
        d_model: int = 256,
        heads: int = 8,
        enc_layers: int = 2,
        dec_layers: int = 4,
        queries: int = 16,
        ff: int = 1024,
    ):
        super().__init__()
        self.hparams = {
            "patch_size": patch_size, "channels": list(channels), "blocks": blocks, "d_model": d_model,
            "heads": heads, "enc_layers": enc_layers, "dec_layers": dec_layers, "queries": queries, "ff": ff,
        }
        self.encoder = Encoder(channels, blocks)
        grid = patch_size // 2 ** (len(channels) - 1)
        self.proj = nn.Conv2d(channels[-1], d_model, 1)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.tokens = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, queries, d_model) * 0.02)
        self.decoder = nn.ModuleList(
            nn.TransformerDecoderLayer(d_model, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(dec_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.ctrl_head = _MLP(d_model, 8)
        self.width_head = _MLP(d_model, 2)
        self.logit_head = nn.Linear(d_model, 1)

    def _heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        # fp32 heads: bf16 keeps 8 mantissa bits, i.e. ~0.25 px steps at 64 px
        with torch.autocast(h.device.type, enabled=False):
            h = self.norm(h.float())
            ctrl = torch.sigmoid(self.ctrl_head(h)).view(h.shape[0], h.shape[1], 4, 2)
            return {
                "ctrl": ctrl * (CTRL_HIGH - CTRL_LOW) + CTRL_LOW,
                "width": 0.3 + F.softplus(self.width_head(h)),
                "logit": self.logit_head(h).squeeze(-1),
            }

    def forward(self, x: torch.Tensor) -> dict:
        f = self.proj(self.encoder(x))  # (B, d, g, g)
        mem = self.tokens(f.flatten(2).transpose(1, 2) + self.pos)
        h = self.query.expand(x.shape[0], -1, -1)
        layers = []
        for layer in self.decoder:
            h = layer(h, mem)
            layers.append(self._heads(h))
        out = dict(layers[-1])
        out["aux"] = layers[:-1]
        return out


def build_model(task: str, hparams: dict) -> nn.Module:
    if task == "single_curve":
        return SingleCurveNet(**hparams)
    if task == "multi_curve":
        return MultiCurveNet(**hparams)
    raise ValueError(f"unknown task {task!r}")


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model stored in a checkpoint; returns ``(model in eval mode, checkpoint dict)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path} is not a line2func checkpoint")
    model = build_model(ckpt["task"], ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt

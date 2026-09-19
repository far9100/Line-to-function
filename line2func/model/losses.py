"""Losses and metrics for curve regression.

A cubic and its reverse (``P3..P0``) are the same curve, so the target is
first oriented to whichever direction is closer to the prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bernstein(t: torch.Tensor) -> torch.Tensor:
    """``(T, 4)`` cubic Bernstein basis at parameters ``t``."""
    mt = 1.0 - t
    return torch.stack([mt**3, 3 * mt * mt * t, 3 * mt * t * t, t**3], dim=1)


def curve_points(ctrl: torch.Tensor, n: int = 16) -> torch.Tensor:
    """``(B, n, 2)`` points at uniform ``t`` on each ``(B, 4, 2)`` cubic."""
    basis = bernstein(torch.linspace(0.0, 1.0, n, device=ctrl.device, dtype=ctrl.dtype))
    return torch.einsum("tk,bkd->btd", basis, ctrl)


def orient(pred: torch.Tensor, ctrl: torch.Tensor, width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse targets whose reversed control points are closer to ``pred``."""
    fwd = (pred - ctrl).abs().mean(dim=(1, 2))
    rev = (pred - ctrl.flip(1)).abs().mean(dim=(1, 2))
    flip = (rev < fwd)[:, None]
    ctrl_o = torch.where(flip[:, :, None], ctrl.flip(1), ctrl)
    width_o = torch.where(flip, width.flip(1), width)
    return ctrl_o, width_o


def single_curve_loss(
    out: dict[str, torch.Tensor], ctrl: torch.Tensor, width: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, dict[str, float]]:
    """Control-point L1 + curve-point L1 (patch units) + width L1 (px, down-weighted)."""
    ctrl_o, width_o = orient(out["ctrl"].detach(), ctrl, width)
    l_ctrl = F.l1_loss(out["ctrl"], ctrl_o)
    l_pts = F.l1_loss(curve_points(out["ctrl"]), curve_points(ctrl_o))
    l_width = F.smooth_l1_loss(out["width"], width_o) / patch_size
    loss = l_ctrl + 2.0 * l_pts + l_width
    return loss, {"ctrl": l_ctrl.item(), "pts": l_pts.item(), "width": l_width.item()}


def _pair_ctrl_l1(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise direction-invariant L1 between ``(B, Q, 4, 2)`` and ``(B, N, 4, 2)``.

    Returns ``(cost (B, Q, N), reversed (B, Q, N) bool)``.
    """
    p = pred[:, :, None]
    fwd = (p - gt[:, None]).abs().mean(dim=(-1, -2))
    rev = (p - gt.flip(2)[:, None]).abs().mean(dim=(-1, -2))
    return torch.minimum(fwd, rev), rev < fwd


@torch.no_grad()
def hungarian_match(out: dict, ctrl: torch.Tensor, mask: torch.Tensor, w_ctrl: float = 5.0, w_cls: float = 1.0):
    """Optimal query <-> target assignment per sample (DETR).

    Returns ``(batch_idx, query_idx, target_idx, reversed)`` index tensors.
    """
    from scipy.optimize import linear_sum_assignment

    l1, rev = _pair_ctrl_l1(out["ctrl"], ctrl)
    cost = w_ctrl * l1 - w_cls * torch.sigmoid(out["logit"])[:, :, None]
    cost, rev, counts = cost.cpu().numpy(), rev.cpu(), mask.sum(dim=1).long().tolist()
    b_idx, q_idx, t_idx = [], [], []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        rows, cols = linear_sum_assignment(cost[b, :, :n])
        b_idx += [b] * len(rows)
        q_idx += rows.tolist()
        t_idx += cols.tolist()
    dev = ctrl.device
    b_t = torch.tensor(b_idx, dtype=torch.long)
    q_t = torch.tensor(q_idx, dtype=torch.long)
    t_t = torch.tensor(t_idx, dtype=torch.long)
    flipped = rev[b_t, q_t, t_t] if len(b_idx) else torch.zeros(0, dtype=torch.bool)
    return b_t.to(dev), q_t.to(dev), t_t.to(dev), flipped.to(dev)


def _set_loss(out: dict, ctrl, width, mask, neg_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    b, q, t, flipped = hungarian_match(out, ctrl, mask)
    target_cls = torch.zeros_like(out["logit"])
    target_cls[b, q] = 1.0
    weights = torch.where(target_cls > 0, 1.0, neg_weight)
    l_cls = F.binary_cross_entropy_with_logits(out["logit"], target_cls, weight=weights, reduction="sum")
    l_cls = l_cls / weights.sum()
    if len(b) == 0:
        zero = out["ctrl"].sum() * 0.0
        return l_cls + zero, {"cls": l_cls.item(), "ctrl": 0.0, "pts": 0.0, "width": 0.0}
    p_ctrl = out["ctrl"][b, q]
    g_ctrl = ctrl[b, t]
    g_ctrl = torch.where(flipped[:, None, None], g_ctrl.flip(1), g_ctrl)
    g_width = width[b, t]
    g_width = torch.where(flipped[:, None], g_width.flip(1), g_width)
    l_ctrl = F.l1_loss(p_ctrl, g_ctrl)
    l_pts = F.l1_loss(curve_points(p_ctrl), curve_points(g_ctrl))
    l_width = F.smooth_l1_loss(out["width"][b, q], g_width) * 0.05
    loss = 2.0 * l_cls + 5.0 * l_ctrl + 5.0 * l_pts + l_width
    return loss, {"cls": l_cls.item(), "ctrl": l_ctrl.item(), "pts": l_pts.item(), "width": l_width.item()}


def multi_curve_loss(
    out: dict, ctrl: torch.Tensor, width: torch.Tensor, mask: torch.Tensor, neg_weight: float = 0.2
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hungarian-matched set loss on the final and every auxiliary decoder layer."""
    loss, parts = _set_loss(out, ctrl, width, mask, neg_weight)
    for aux in out.get("aux", []):
        loss = loss + _set_loss(aux, ctrl, width, mask, neg_weight)[0]
    return loss, parts


@torch.no_grad()
def point_error_px(pred_ctrl: torch.Tensor, ctrl: torch.Tensor, patch_size: int, n: int = 32) -> torch.Tensor:
    """``(B,)`` mean distance in px between matching points of prediction and target."""
    ctrl_o, _ = orient(pred_ctrl, ctrl, torch.zeros(len(ctrl), 2, device=ctrl.device))
    diff = curve_points(pred_ctrl, n) - curve_points(ctrl_o, n)
    return diff.norm(dim=-1).mean(dim=1) * patch_size

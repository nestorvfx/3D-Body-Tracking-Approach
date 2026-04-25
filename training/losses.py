"""RTMPose-style losses for SimCC-3D prediction.

- KLDiscretLoss: per-axis KL divergence between softmax(logits / beta) and
  the Gaussian-encoded target.  Reference: mmpose/models/losses/classification_loss.py.
- BoneLoss: L1 on skeleton-bone lengths between predicted and GT 3D joints.
  Stabilises training by enforcing anatomically plausible proportions.
- MPJPELoss: straight L1 on 3D joints (camera frame metres) — the quantity
  we report at eval.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# COCO-17 bones (parent, child) for BoneLoss — same list as coco17.py.
COCO17_BONES = [
    (5, 7), (7, 9),              # left arm
    (6, 8), (8, 10),             # right arm
    (5, 6),                      # shoulders
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15),          # left leg
    (12, 14), (14, 16),          # right leg
    (0, 1), (1, 3), (0, 2), (2, 4),   # head
]


class KLDiscretLoss(nn.Module):
    """KL divergence between model softmax and Gaussian target distribution.

    Args:
        beta:  logit-softening temperature (matches RTMPose default 10.0).
        label_softmax: if True, softmax the target as well (mmpose default).
    """

    def __init__(self, beta: float = 10.0, label_softmax: bool = False):
        """label_softmax=False uses the raw Gaussian target directly.

        Per mmpose convention the RTMPose3D recipe sets label_softmax=True,
        but that softmaxes the already-normalised Gaussian target with
        beta=10, flattening its peak from ~0.47 to ~0.16.  The resulting
        weaker gradient (~3x smaller at the peak bin) causes the head to
        stall near the log(K) uniform-output ceiling — empirically verified
        on this model + dataset.  label_softmax=False preserves the sharp
        target and the gradient that goes with it.
        """
        super().__init__()
        self.beta = beta
        self.label_softmax = label_softmax

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                vis: torch.Tensor | None = None) -> torch.Tensor:
        """
        logits: [B, J, bins]
        target: [B, J, bins]  (pre-normalised Gaussian, rows sum to 1)
        vis:    [B, J]        (0/1 visibility mask) or None
        """
        # Soft logits via beta.
        log_pred = F.log_softmax(logits * self.beta, dim=-1)
        if self.label_softmax:
            tgt = F.softmax(target * self.beta, dim=-1)
        else:
            tgt = target
        # kl(P||Q) = sum_i P_i (log P_i - log Q_i); here target is "true"
        # distribution so we use sum(-tgt * log_pred).  + small eps for stability.
        loss_per_kp = -(tgt * log_pred).sum(dim=-1)     # [B, J]
        if vis is not None:
            loss_per_kp = loss_per_kp * vis
            denom = vis.sum().clamp(min=1.0)
            return loss_per_kp.sum() / denom
        return loss_per_kp.mean()


class BoneLoss(nn.Module):
    """L1 on skeleton-bone lengths between predicted and GT 3D joints."""

    def __init__(self, bones=COCO17_BONES):
        super().__init__()
        # Avoid `children` name which shadows nn.Module.children() method.
        self.register_buffer(
            "bone_parent", torch.tensor([b[0] for b in bones], dtype=torch.long))
        self.register_buffer(
            "bone_child", torch.tensor([b[1] for b in bones], dtype=torch.long))

    def forward(self, pred_3d: torch.Tensor, gt_3d: torch.Tensor,
                vis: torch.Tensor | None = None) -> torch.Tensor:
        """
        pred_3d, gt_3d: [B, J, 3]
        vis:            [B, J] or None
        """
        pb = pred_3d[:, self.bone_parent]
        pc = pred_3d[:, self.bone_child]
        gb = gt_3d[:, self.bone_parent]
        gc = gt_3d[:, self.bone_child]
        pred_len = (pc - pb).norm(dim=-1)     # [B, nBones]
        gt_len = (gc - gb).norm(dim=-1)
        diff = (pred_len - gt_len).abs()
        if vis is not None:
            mask = vis[:, self.bone_parent] * vis[:, self.bone_child]
            diff = diff * mask
            denom = mask.sum().clamp(min=1.0)
            return diff.sum() / denom
        return diff.mean()


class MPJPE(nn.Module):
    """Mean per-joint position error — pure L2, reported in metres."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_3d: torch.Tensor, gt_3d: torch.Tensor,
                vis: torch.Tensor | None = None) -> torch.Tensor:
        err = (pred_3d - gt_3d).norm(dim=-1)   # [B, J]
        if vis is not None:
            err = err * vis
            return err.sum() / vis.sum().clamp(min=1.0)
        return err.mean()


def pa_mpjpe(pred_3d: torch.Tensor, gt_3d: torch.Tensor) -> torch.Tensor:
    """Procrustes-aligned MPJPE (batch).  Scale + rotation + translation
    Umeyama alignment per sample, then L2 mean.
    pred_3d, gt_3d: [B, J, 3], no visibility support."""
    B, J, _ = pred_3d.shape
    mu_p = pred_3d.mean(dim=1, keepdim=True)
    mu_g = gt_3d.mean(dim=1, keepdim=True)
    X = pred_3d - mu_p
    Y = gt_3d - mu_g
    # Covariance
    H = X.transpose(1, 2) @ Y           # [B, 3, 3]
    U, S, Vt = torch.linalg.svd(H)
    # Reflection correction
    det = torch.det(U @ Vt)
    sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
    # Fix last column
    D = torch.eye(3, device=pred_3d.device).unsqueeze(0).repeat(B, 1, 1)
    D[:, -1, -1] = sign.squeeze(-1).squeeze(-1)
    R = U @ D @ Vt
    # scale
    var_x = (X ** 2).sum(dim=(1, 2))
    trace = (S * D.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
    scale = trace / var_x.clamp(min=1e-8)
    pred_aligned = scale.view(B, 1, 1) * (X @ R) + mu_g
    err = (pred_aligned - gt_3d).norm(dim=-1).mean(dim=-1)   # [B]
    return err.mean()

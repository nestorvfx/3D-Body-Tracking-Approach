"""MobileNetV4-Conv-S backbone + RTMPose3D-style SimCC-3D head.

Apache-2.0 clean.  Paper references:
  MobileNetV4:      https://arxiv.org/abs/2404.10518
  RTMPose:          https://arxiv.org/abs/2303.07399
  RTMPose3D head:   https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose3d

The RTMPose3D SimCC head predicts THREE 1D probability distributions per
joint — over X, Y, Z bins — and decodes each via soft-argmax.  We use a
GAP -> Linear -> split to three heads layout (same as RTMPose), with a
small per-axis MLP.

Built for 256x192 inputs.  Bin counts from training/lib/simcc3d.py:
  Wb = 384, Hb = 512, Db = 512  (input 192x256, split_ratio = 2).
Bins are a multiple of the head hidden-dim-friendly size.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCC3DHead(nn.Module):
    """RTMPose3D head lite: three parallel 1D classifiers.

    Input:  [B, feat_dim]   (pooled feature)
    Output: {x: [B, J, Wb], y: [B, J, Hb], z: [B, J, Db]}
    """

    def __init__(self, feat_dim: int, num_joints: int = 17,
                 bins: tuple[int, int, int] = (384, 512, 512),
                 hidden: int = 256):
        super().__init__()
        self.num_joints = num_joints
        self.bins = bins
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.head_x = nn.Linear(hidden, num_joints * bins[0])
        self.head_y = nn.Linear(hidden, num_joints * bins[1])
        self.head_z = nn.Linear(hidden, num_joints * bins[2])

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(feat)
        B = h.shape[0]
        J = self.num_joints
        Wb, Hb, Db = self.bins
        return {
            "x_logits": self.head_x(h).view(B, J, Wb),
            "y_logits": self.head_y(h).view(B, J, Hb),
            "z_logits": self.head_z(h).view(B, J, Db),
        }


class RTMPose3DModel(nn.Module):
    """Backbone + RTMPose3D SimCC-3D head.

    Backbone via timm.  Default: MobileNetV4-Conv-S (Apache-2.0).

    Design note: we deliberately do NOT use global-average-pooling.  Empirical
    verification (see training/diagnostics) with GAP showed the model failing
    to escape the log(K) uniform-output loss ceiling for all three axes — the
    pooled 1280-dim vector has no spatial *where*-signal, so the head cannot
    predict per-joint bin distributions that depend on the person's
    configuration in the image.  Canonical mmpose RTMPose3D uses a spatial
    feature map (C, H, W) and a token-based head for the same reason.

    Our minimal-invasive version: preserve the feature map via global_pool='',
    reduce channels with a 1x1 conv, flatten spatially, and feed that larger
    vector into the existing SimCC3DHead.  Params go from 9M -> ~12M, which
    is still mobile-friendly.
    """

    def __init__(self,
                 backbone: str = "mobilenetv4_conv_small.e2400_r224_in1k",
                 num_joints: int = 17,
                 bins: tuple[int, int, int] = (384, 512, 512),
                 head_hidden: int = 256,
                 proj_channels: int = 256,
                 cond_dim: int = 6,
                 pretrained: bool = True):
        super().__init__()
        self.cond_dim = cond_dim
        # global_pool='' returns the raw [B, C, H, W] feature map.
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="")
        # Probe spatial + channel dims via a dummy forward.
        self.backbone.eval()
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, 256, 192))
        if feat.dim() != 4:
            raise RuntimeError(
                f"Expected 4D feature map from backbone, got shape {feat.shape}. "
                f"Ensure global_pool='' is supported by this timm model.")
        C, H, W = feat.shape[1:]
        self.proj = nn.Sequential(
            nn.Conv2d(C, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.GELU(),
        )
        self.feat_dim = proj_channels * H * W
        print(f"[model] spatial feat map {C}x{H}x{W} -> "
              f"proj {proj_channels}x{H}x{W} -> flat {self.feat_dim}  "
              f"(+ cond_dim={cond_dim} + 1 Tz-feedback)")
        # SimCC head sees spatial features + bbox/focal conditioning (CLIFF)
        # + predicted log(T_z) (BLADE-lite, CVPR 2025).  The T_z feedback
        # closes the loop: depth is predicted first, then the joint head
        # conditions its 2D/Z coordinates on the known metric scale.
        # BLADE (Wang et al. 2025) demonstrates this recovers 4-8 mm MPJPE
        # on close-range images — the regime where perspective distortion
        # matters most and weak-perspective HMR fails.
        self.head = SimCC3DHead(self.feat_dim + cond_dim + 1,
                                num_joints, bins, head_hidden)
        # Dedicated root-depth head: factorized RootNet-style (Moon ICCV 2019 +
        # Zhang 2021 factorization).  Predicts TWO log-scalars:
        #   log_gamma_size — body-size correction (captures adult/child; a kid
        #                    at 2m looks identical to an adult at 3m; this
        #                    scalar resolves the ambiguity via learned visual
        #                    cues — head/body ratio, proportions, clothing).
        #   log_gamma_pose — pose-compactness correction (sitting/crouching
        #                    changes bbox area without changing depth).
        # Final root_z = k_prior × exp(log_gamma_size) × exp(log_gamma_pose)
        # where k_prior = sqrt(fx · fy · A_real / A_bbox) is computed
        # analytically from the camera intrinsics + bbox (pinhole geometry,
        # A_real = 2m × 2m canonical human box).
        #
        # Why this beats raw root_z regression (which our v1 did):
        #   - k_prior absorbs most of the variance (geometric prior)
        #   - gammas are bounded scalars around 1.0 → stable training
        #   - Outlier samples (CMU BVH bugs with root_z=90000m) produce
        #     gamma=30000 which the L1 on log_gamma caps gracefully
        #   - Explicit body-size head forces the network to use visual cues
        #     for the adult/child ambiguity instead of memorizing depths
        #
        # Reference: Moon et al. "Camera Distance-aware Top-down Approach"
        #   ICCV 2019; Zhang 2021 factorized correction factors.
        self.root_z_head = nn.Sequential(
            nn.Linear(proj_channels + cond_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),          # [log_gamma_size, log_gamma_pose]
        )

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor | None = None,
                k_prior: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
        """x:       [B, 3, H, W] input crop.
        cond:     [B, cond_dim] — bbox (cx,cy,w,h normalised to full image) +
                   (fx, fy) normalised by full-image diagonal.  Training ALWAYS
                   supplies cond; pass None for sanity / export checks only.
        k_prior:  [B] — geometric depth prior sqrt(fx·fy·A_real/A_bbox), metres.
                   Computed in the dataloader; combined with the model's
                   predicted log_gamma factors to form the final root_z.
                   Pass None to skip combining (e.g. feature-extraction mode).

        Returns dict with keys:
          x_logits / y_logits / z_logits — SimCC distributions (unchanged)
          log_gamma_size  [B] — predicted body-size correction (log-space)
          log_gamma_pose  [B] — predicted pose-compactness correction
          root_z          [B] — combined absolute metres = k_prior · exp(size+pose)
                                 (only present when k_prior is given)
        """
        B = x.shape[0]
        if cond is None:
            cond = x.new_zeros((B, self.cond_dim))
        feat = self.backbone(x)              # [B, C, H, W]
        feat = self.proj(feat)               # [B, P, H, W]
        feat_flat = feat.flatten(1)          # [B, P*H*W]

        # Step 1 — predict T_z first (BLADE pipeline, CVPR 2025).  GAP pool
        # + CLIFF cond → factorized RootNet scalars.  Runs BEFORE the SimCC
        # head so its output can feed back as conditioning.
        feat_gap = feat.mean(dim=(2, 3))     # [B, P]
        feat_rz = torch.cat([feat_gap, cond], dim=-1)
        rz_out = self.root_z_head(feat_rz)                    # [B, 2]
        log_gamma_size = rz_out[:, 0]
        log_gamma_pose = rz_out[:, 1]

        # Step 2 — compute predicted T_z and feed it back to the SimCC head.
        # Stop-gradient on the feedback path: the SimCC joint loss must not
        # push the root_z head in a direction that only helps 2D prediction
        # at the cost of absolute depth.  The root_z head is trained only by
        # its own log-L1 loss.
        if k_prior is not None:
            root_z = k_prior * torch.exp(log_gamma_size + log_gamma_pose)
            # Log-transform + zero-centre (typical root_z ≈ 3 m →
            # log ≈ 1.1, so subtract 1.0) for stable feeding into the head.
            tz_feedback = (torch.log(torch.clamp(root_z, min=0.1)) - 1.0
                           ).detach().unsqueeze(-1)           # [B, 1]
        else:
            # Export / debug path — pass zero feedback.
            root_z = None
            tz_feedback = feat.new_zeros((B, 1))

        cond_plus_tz = torch.cat([cond, tz_feedback], dim=-1)  # [B, cond+1]
        feat_simcc = torch.cat([feat_flat, cond_plus_tz], dim=-1)
        out = self.head(feat_simcc)
        out["log_gamma_size"] = log_gamma_size
        out["log_gamma_pose"] = log_gamma_pose
        if root_z is not None:
            out["root_z"] = root_z
        return out


def build_model(name: str = "mnv4s", num_joints: int = 17,
                bins: tuple[int, int, int] = (384, 512, 512),
                pretrained: bool = True) -> RTMPose3DModel:
    presets = {
        "mnv4s": "mobilenetv4_conv_small.e2400_r224_in1k",
        "mnv4m": "mobilenetv4_conv_medium.e500_r256_in1k",
    }
    backbone = presets.get(name, name)
    return RTMPose3DModel(backbone=backbone, num_joints=num_joints,
                          bins=bins, pretrained=pretrained)


def decode_simcc(logits_x, logits_y, logits_z, input_wh, z_range_m=2.17, root_z=None,
                 mode: str = "soft_argmax"):
    """Decode SimCC-3D logits to pixel + depth.

    Args:
        logits_{x,y,z}: [B, J, bins_*]
        input_wh:        (W, H) of the CROPPED INPUT image.
        z_range_m:       metric z range ±z_range_m relative to root.
        root_z:          [B] per-sample root Z in camera frame (metres).
        mode:            "soft_argmax" (default) — weighted mean of bin indices
                         by softmax(logits).  Sub-pixel accurate when the model's
                         distribution is sharp.
                         "argmax" — pick the peak bin directly.  Eliminates the
                         uniform-background bias that pulls soft-argmax toward
                         the center when the model is uncertain, at the cost of
                         bin-resolution quantisation (0.5 px for 384 bins / 192 W).

    Returns:
        kps_2d:  [B, J, 2] pixel coords in the crop frame.
        kps_z:   [B, J] depth in metres (absolute, if root_z given).
    """
    W, H = input_wh
    Bx = logits_x.shape[-1]; By = logits_y.shape[-1]; Bz = logits_z.shape[-1]
    if mode == "argmax":
        ex = logits_x.argmax(dim=-1).float()
        ey = logits_y.argmax(dim=-1).float()
        ez = logits_z.argmax(dim=-1).float()
    else:
        px = F.softmax(logits_x, dim=-1)
        py = F.softmax(logits_y, dim=-1)
        pz = F.softmax(logits_z, dim=-1)
        idx_x = torch.arange(Bx, device=logits_x.device, dtype=torch.float32)
        idx_y = torch.arange(By, device=logits_y.device, dtype=torch.float32)
        idx_z = torch.arange(Bz, device=logits_z.device, dtype=torch.float32)
        ex = (px * idx_x).sum(dim=-1)      # [B, J]
        ey = (py * idx_y).sum(dim=-1)
        ez = (pz * idx_z).sum(dim=-1)
    # bin centres -> pixel coords using split_ratio=2 convention from simcc3d.
    split_ratio = 2.0
    pix_u = ex / split_ratio
    pix_v = ey / split_ratio
    # z: bin -> normalised [-1, 1] -> relative metric -> absolute if root given.
    z_norm = (ez / Bz) * 2.0 - 1.0
    z_rel = z_norm * z_range_m
    if root_z is not None:
        z_abs = z_rel + root_z.unsqueeze(-1)
    else:
        z_abs = z_rel
    return torch.stack([pix_u, pix_v], dim=-1), z_abs


if __name__ == "__main__":
    m = build_model("mnv4s", pretrained=False)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
    x = torch.randn(2, 3, 256, 192)
    with torch.no_grad():
        out = m(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")

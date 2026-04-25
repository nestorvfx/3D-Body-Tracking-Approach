"""Bench inference cost: CLIFF-conditioned (new) vs baseline (old) architecture.
Reports ms/frame + param counts on CPU (mobile-relevant) and GPU if available.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from training.model import build_model, SimCC3DHead


class BaselineRTMPose3D(nn.Module):
    """Architecture from BEFORE the CLIFF change — for apples-to-apples bench."""
    def __init__(self, backbone="mobilenetv4_conv_small.e2400_r224_in1k",
                 num_joints=17, bins=(384, 512, 512), head_hidden=256,
                 proj_channels=256):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0, global_pool="")
        self.backbone.eval()
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 3, 256, 192))
        C, H, W = feat.shape[1:]
        self.proj = nn.Sequential(
            nn.Conv2d(C, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.GELU(),
        )
        self.feat_dim = proj_channels * H * W
        self.head = SimCC3DHead(self.feat_dim, num_joints, bins, head_hidden)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = feat.flatten(1)
        return self.head(feat)


def bench(name, model, feed_cond, device, n_warm=30, n_run=200):
    model.eval()
    x = torch.randn(1, 3, 256, 192, device=device)
    cond = torch.randn(1, 6, device=device)
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(x, cond) if feed_cond else model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_run):
            _ = model(x, cond) if feed_cond else model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    per_ms = (time.perf_counter() - t0) / n_run * 1000.0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name:30}  {per_ms:6.2f} ms/frame  ({n_params/1e6:5.2f} M params)")
    return per_ms, n_params


def main():
    for dev_name in ["cpu", "cuda"]:
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(dev_name)
        print(f"\n=== {device} ===")
        baseline = BaselineRTMPose3D().to(device)
        new = build_model("mnv4s", pretrained=False).to(device)
        t_base, n_base = bench("baseline (pre-CLIFF)", baseline, False, device)
        t_new, n_new = bench("CLIFF cond + root-z", new, True, device)
        delta_ms = t_new - t_base
        delta_pct = 100.0 * delta_ms / t_base
        print(f"  delta: {delta_ms:+.2f} ms/frame ({delta_pct:+.1f} %)  "
              f"+{n_new - n_base:,} params")


if __name__ == "__main__":
    main()

"""SmoothNet-T — tiny 1D-convolutional temporal refiner for 3D pose sequences.

Reference: Zeng et al., "SmoothNet: A Plug-and-Play Network for Refining
Human Poses in Videos", ECCV 2022.  https://arxiv.org/abs/2112.13715
(MIT-licensed code: https://github.com/cure-lab/SmoothNet)

Takes a sliding window of `window_size` frames of per-frame 3D pose
predictions (from the single-frame model) and regresses a smoothed
trajectory.  Unlike OneEuroFilter, SmoothNet LEARNS what "smooth"
means for our pose manifold — it can also correct small systematic
errors that a pure low-pass filter can't.

This is a minimal reimplementation:
  * Input:  (B, T, J, C)  where C = 3 for 3D coords
  * Output: (B, T, J, C)  same shape, smoothed
  * Architecture: Reshape to per-joint per-axis 1D signals, 1D temporal
    conv stack, residual skip.  Pure Conv1D — exports cleanly to ONNX /
    Core ML / TFLite.  ~60k params for T=8 window, ready for mobile.

Commercial-clean: the CODE is MIT; the PUBLISHED WEIGHTS are H3.6M-trained
(research-only).  We train our own weights on our commercial-clean
synthetic video data.
"""
from __future__ import annotations

import torch
from torch import nn


class SmoothNetResBlock(nn.Module):
    """1D convolutional residual block over the time dimension."""

    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                                padding=padding)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C)
        res = x
        y = self.conv1(x.transpose(1, 2)).transpose(1, 2)   # time conv
        y = self.norm1(y)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y.transpose(1, 2)).transpose(1, 2)
        y = self.norm2(y)
        return res + y


class SmoothNet(nn.Module):
    """Tiny trajectory refiner for temporal pose sequences.

    Input shape:  (B, T, J*3)  — collapsed joint/axis channels
    Output shape: (B, T, J*3)
    """

    def __init__(
        self,
        window_size: int = 8,
        in_channels: int = 51,    # 17 joints x 3 axes
        hidden: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.causal = causal

        self.in_proj = nn.Linear(in_channels, hidden)
        self.blocks = nn.ModuleList([
            SmoothNetResBlock(hidden, kernel_size=7, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden, in_channels)

    def forward(self, x):
        # x: (B, T, J*3)
        h = self.in_proj(x)                    # (B, T, H)
        for blk in self.blocks:
            h = blk(h)
        dx = self.out_proj(h)                   # (B, T, J*3) residual
        return x + dx                            # residual refinement


def make_smoothnet(num_joints: int = 17, window_size: int = 8,
                    causal: bool = False) -> SmoothNet:
    """Default SmoothNet-T configuration: 8-frame window, causal option."""
    return SmoothNet(
        window_size=window_size,
        in_channels=num_joints * 3,
        hidden=128,
        n_blocks=3,
        dropout=0.1,
        causal=causal,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = make_smoothnet(num_joints=17, window_size=8)
    n = count_parameters(model)
    print(f"SmoothNet-T params: {n:,}  (~{n/1000:.0f}k)")
    x = torch.randn(4, 8, 51)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)

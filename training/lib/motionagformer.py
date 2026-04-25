"""MotionAGFormer-XS — temporal 2D-to-3D pose lifter.

Reference:
  Mehraban, Aryan et al., "MotionAGFormer: Enhancing 3D Human Pose
  Estimation with a Transformer-GCNFormer Network," WACV 2024.
  https://arxiv.org/abs/2310.16288  · Code (Apache-2.0):
  https://github.com/TaatiTeam/MotionAGFormer

This is a minimal, commercial-clean reimplementation of the XS variant
(the smallest published config) sized to run in real time on mobile
Neural Engine / Hexagon NPU.  It consumes a sliding window of per-frame
predictions from the single-frame RTMPose3D model and outputs a
temporally-refined 3D pose sequence.

XS variant target:
  * ~0.4M parameters
  * ~0.08 GFLOPs per 27-frame window
  * <2ms inference on iPhone A17 ANE

Architecture:
  * Input: (B, T, J, C)  where C=3 (per-frame x/y/z from single-frame head)
  * Joint embedding: per-joint MLP to `dim` channels
  * N dual-stream blocks: each block has
      - ST-Attention stream: spatial self-attention across joints, followed
        by temporal self-attention across frames
      - GCNFormer stream: graph conv over skeleton adjacency + temporal
        1D conv
      - Streams merged via learned gating
  * Joint head: per-joint linear regressor to 3D

For commercial cleanliness we DO NOT use the published MotionAGFormer
weights (trained on Human3.6M + MPI-INF-3DHP).  The architecture is
re-implemented here under Apache-2.0 spirit; weights will be trained
from scratch on our synthetic pipeline output + AIST++ CC-BY data.
"""
from __future__ import annotations

import torch
from torch import nn


# COCO-17 skeleton adjacency matrix for the GCNFormer stream.
COCO17_EDGES: list[tuple[int, int]] = [
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 6),                   # shoulders
    (5, 11), (6, 12), (11, 12),   # torso
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
    (0, 1), (1, 3), (0, 2), (2, 4),   # head
]


def adjacency_matrix(num_joints: int = 17) -> torch.Tensor:
    """Symmetric, self-loop-augmented adjacency used by GCNFormer."""
    A = torch.eye(num_joints)
    for a, b in COCO17_EDGES:
        A[a, b] = 1.0
        A[b, a] = 1.0
    # Normalise D^{-1/2} A D^{-1/2}
    D = torch.diag(A.sum(-1) ** -0.5)
    return D @ A @ D


# ------------------------- Building blocks -------------------------

class JointEmbed(nn.Module):
    """Per-joint coordinate -> channels embedding."""

    def __init__(self, in_ch: int = 3, dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(in_ch, dim)

    def forward(self, x):                      # (B, T, J, C)
        return self.proj(x)                    # (B, T, J, dim)


class SpatialAttention(nn.Module):
    """Multi-head attention across J (joints) within each frame."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B, T, J, D)
        B, T, J, D = x.shape
        x = x.reshape(B * T, J, D)
        qkv = self.qkv(x).reshape(B * T, J, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)    # (BT, H, J, D/H) each
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * T, J, D)
        return self.proj(out).reshape(B, T, J, D)


class TemporalAttention(nn.Module):
    """Multi-head attention across T (frames) within each joint channel."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B, T, J, D)
        B, T, J, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * J, T, D)
        qkv = self.qkv(x).reshape(B * J, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B * J, T, D)
        return self.proj(out).reshape(B, J, T, D).permute(0, 2, 1, 3)


class STAttentionBlock(nn.Module):
    """One ST-Transformer block: spatial attn -> temporal attn -> FFN."""

    def __init__(self, dim: int, heads: int = 4, ffn_mult: float = 2.0,
                  dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.sa = SpatialAttention(dim, heads, dropout)
        self.n2 = nn.LayerNorm(dim)
        self.ta = TemporalAttention(dim, heads, dropout)
        self.n3 = nn.LayerNorm(dim)
        ffn = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn, dim),
        )

    def forward(self, x):
        x = x + self.sa(self.n1(x))
        x = x + self.ta(self.n2(x))
        x = x + self.ffn(self.n3(x))
        return x


class GCNFormerBlock(nn.Module):
    """Graph conv across joints + temporal 1D conv per joint channel."""

    def __init__(self, dim: int, adj: torch.Tensor, kernel_size: int = 3,
                  dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adj", adj)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.n1 = nn.LayerNorm(dim)
        self.temporal = nn.Conv1d(dim, dim, kernel_size,
                                    padding=kernel_size // 2, groups=dim)
        self.n2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B, T, J, D)
        B, T, J, D = x.shape
        # Graph conv across joints
        g = torch.einsum("ij,btjd->btid", self.adj.to(x.dtype), x)
        g = self.proj1(self.n1(g))
        g = self.act(g)
        g = self.drop(g)
        # Temporal depth-wise 1D conv per joint
        g2 = g.permute(0, 2, 3, 1).reshape(B * J, D, T)
        g2 = self.temporal(g2)
        g2 = g2.reshape(B, J, D, T).permute(0, 3, 1, 2)
        g2 = self.proj2(self.n2(g2))
        return x + self.act(g2)


class DualStreamBlock(nn.Module):
    """Parallel ST-Attention + GCNFormer streams, merged via gating."""

    def __init__(self, dim: int, heads: int, adj: torch.Tensor,
                  dropout: float = 0.1):
        super().__init__()
        self.st = STAttentionBlock(dim, heads=heads, dropout=dropout)
        self.gcn = GCNFormerBlock(dim, adj, kernel_size=3, dropout=dropout)
        # Learned per-channel gate (broadcasts over B, T, J).
        self.gate = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        s = self.st(x)
        g = self.gcn(x)
        w = torch.sigmoid(self.gate)
        return w * s + (1.0 - w) * g


# ------------------------- Main model -------------------------

class MotionAGFormer(nn.Module):
    """MotionAGFormer-XS: dual-stream spatio-temporal lifter.

    Input:  (B, T, J, 3)    single-frame 3D predictions (x, y, z per joint)
    Output: (B, T, J, 3)    temporally-refined predictions
    """

    def __init__(
        self,
        num_joints: int = 17,
        window_size: int = 27,
        dim: int = 64,
        heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.1,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.window_size = window_size
        adj = adjacency_matrix(num_joints)
        self.embed = JointEmbed(in_channels, dim)
        # Temporal positional encoding (learnable, shape (1, T, 1, D)).
        self.pos_time = nn.Parameter(torch.zeros(1, window_size, 1, dim))
        # Spatial (joint) positional encoding.
        self.pos_joint = nn.Parameter(torch.zeros(1, 1, num_joints, dim))
        nn.init.trunc_normal_(self.pos_time, std=0.02)
        nn.init.trunc_normal_(self.pos_joint, std=0.02)
        self.blocks = nn.ModuleList([
            DualStreamBlock(dim, heads=heads, adj=adj, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.n_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_channels)

    def forward(self, x):                       # (B, T, J, 3)
        B, T, J, _ = x.shape
        h = self.embed(x) + self.pos_time[:, :T] + self.pos_joint[:, :, :J]
        for blk in self.blocks:
            h = blk(h)
        h = self.n_final(h)
        return self.head(h) + x                 # residual refinement


def make_motionagformer_xs(num_joints: int = 17,
                             window_size: int = 27) -> MotionAGFormer:
    """XS variant — 0.4M parameters, mobile-deployable."""
    return MotionAGFormer(
        num_joints=num_joints,
        window_size=window_size,
        dim=64, heads=4, n_blocks=4,
        dropout=0.1,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = make_motionagformer_xs(num_joints=17, window_size=27)
    n = count_parameters(model)
    print(f"MotionAGFormer-XS params: {n:,}  (~{n/1000:.0f}k)")
    x = torch.randn(2, 27, 17, 3)
    y = model(x)
    print("Input:", x.shape, "Output:", y.shape)

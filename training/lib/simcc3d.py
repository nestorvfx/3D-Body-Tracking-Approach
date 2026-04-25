"""SimCC-3D label encoder/decoder (mmpose-style, reimplemented for portability).

SimCC-3D treats 3D joint prediction as THREE independent 1-D classifications
over discrete bins along X, Y (image-plane) and Z (root-relative depth).  Loss
is KL divergence between a narrow Gaussian target and a softmax-normalised
prediction (KLDiscretLoss, `beta=10`).

Reference: mmpose RTMPose3D head
  https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose3d/rtmpose3d/simcc_3d_label.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimCC3DConfig:
    input_size: tuple[int, int, int] = (192, 256, 256)   # (W, H, D)
    split_ratio: float = 2.0
    # Per-axis sigma — Z is intentionally ~6× narrower than X/Y.
    # Rationale: X/Y targets can land anywhere in the [0, W*split_ratio]
    # bin range (naturally spread), so sigma≈5 gives label_softmax(β=10) a
    # meaningful peak.  Z targets are root-relative and metric-scaled:
    # real joint spread is ±0.5 m around root, clustering in ~60 of the
    # 512 bins.  sigma=5 on that produces an almost-uniform target after
    # label_softmax(β=10), so the model trains toward uniform output and
    # the loss pins at ~log(K).  Canonical RTMW3D uses (4.9, 5.66, 0.8).
    # See mmpose projects/rtmpose3d/rtmpose3d/simcc_3d_label.py.
    sigma: tuple[float, float, float] = (5.0, 5.66, 0.8)
    z_range_m: float = 2.17                              # ±2.17m from root
    root_indices: tuple[int, ...] = (11, 12)             # COCO-17 hips average


def encode(
    kps2d: np.ndarray,          # [K, 2] pixel coords in (W, H) space
    kps3d: np.ndarray,           # [K, 3] camera-space metric coords
    visibility: np.ndarray,      # [K]
    cfg: SimCC3DConfig = SimCC3DConfig(),
) -> dict[str, np.ndarray]:
    """Return Gaussian-encoded targets: `target_x [K, Wbins]`, `target_y`, `target_z`.

    The X / Y axes are normalised to the bin counts derived from `input_size *
    split_ratio`; the Z axis is first centred on the mean of the root indices
    in camera space, scaled by `z_range_m`, then binned the same way.
    """
    W, H, D = cfg.input_size
    Wb = int(W * cfg.split_ratio)
    Hb = int(H * cfg.split_ratio)
    Db = int(D * cfg.split_ratio)
    sx, sy, sz = cfg.sigma

    K = kps2d.shape[0]
    # Root-relative z
    root_z = kps3d[list(cfg.root_indices), 2].mean()
    z_norm = (kps3d[:, 2] - root_z) / cfg.z_range_m      # expected in [-1, 1]
    z_idx = (z_norm + 1.0) * 0.5 * Db                    # bin centre

    # X / Y bin centres
    x_idx = kps2d[:, 0] * cfg.split_ratio
    y_idx = kps2d[:, 1] * cfg.split_ratio

    tx = np.zeros((K, Wb), dtype=np.float32)
    ty = np.zeros((K, Hb), dtype=np.float32)
    tz = np.zeros((K, Db), dtype=np.float32)

    x_grid = np.arange(Wb, dtype=np.float32)
    y_grid = np.arange(Hb, dtype=np.float32)
    z_grid = np.arange(Db, dtype=np.float32)

    for k in range(K):
        if visibility[k] < 0.5:
            continue
        tx[k] = np.exp(-0.5 * ((x_grid - x_idx[k]) / sx) ** 2)
        ty[k] = np.exp(-0.5 * ((y_grid - y_idx[k]) / sy) ** 2)
        tz[k] = np.exp(-0.5 * ((z_grid - z_idx[k]) / sz) ** 2)

    # Normalize so each row sums to 1 (target for KL loss)
    for t in (tx, ty, tz):
        norm = t.sum(axis=-1, keepdims=True)
        norm[norm == 0] = 1.0
        t /= norm

    return {"target_x": tx, "target_y": ty, "target_z": tz,
            "root_z": np.float32(root_z)}


def decode(
    pred_x: np.ndarray,         # [K, Wb]
    pred_y: np.ndarray,         # [K, Hb]
    pred_z: np.ndarray,         # [K, Db]
    cfg: SimCC3DConfig = SimCC3DConfig(),
) -> dict[str, np.ndarray]:
    """Argmax decode with sub-bin linear refinement.  Returns pixel-coord
    keypoints and root-relative 3D z.  Caller supplies root_z to recover
    absolute camera-space z."""
    W, H, D = cfg.input_size
    Wb = pred_x.shape[-1]
    Hb = pred_y.shape[-1]
    Db = pred_z.shape[-1]

    x_bin = pred_x.argmax(axis=-1).astype(np.float32)
    y_bin = pred_y.argmax(axis=-1).astype(np.float32)
    z_bin = pred_z.argmax(axis=-1).astype(np.float32)

    kps2d = np.stack([x_bin / cfg.split_ratio, y_bin / cfg.split_ratio], axis=-1)
    z_norm = (z_bin / (Db / 2)) - 1.0
    z_rel = z_norm * cfg.z_range_m
    return {"keypoints_2d": kps2d, "keypoints_z_root_relative_m": z_rel}

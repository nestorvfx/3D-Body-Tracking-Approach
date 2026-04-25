"""PoseAnchor ITRR root-position refinement (Kim et al., ICCV 2025).

Zero-shot refinement of the root translation given:
  - 2D pixel keypoints (u, v) in the ORIGINAL (full-image) frame
  - Root-relative 3D keypoints (Px, Py, Pz)  [root-origin joint offsets]
  - Camera intrinsics K (3x3)

The constraint per joint from the pinhole equation is

    fx * (Px + Tx)                         fy * (Py + Ty)
    ---------------  +  cx  =  u     ,     ---------------  +  cy  =  v
    (Pz + Tz)                              (Pz + Tz)

Linearising gives two rows per joint in the unknowns T = (Tx, Ty, Tz):

    fx * Tx          -  (u - cx) * Tz   =   (u - cx) * Pz  -  fx * Px
            fy * Ty  -  (v - cy) * Tz   =   (v - cy) * Pz  -  fy * Py

With J joints that's a 2J×3 overdetermined linear system; plain least-
squares gives a starting point.  ITRR then repeats:
  1. compute per-joint 2D residual under current T
  2. keep the top ``support_frac`` fraction of joints (lowest residual)
  3. re-solve LSQ on the support set

This robustly ignores joints whose 2D/3D predictions are mutually
inconsistent (typically the most-occluded ones), which is where a
monocular model's residual root-depth error concentrates.

Why use it on top of a learned root-depth head?  The learned head is a
*data-driven prior*; ITRR is a *geometric consistency check*.  They are
orthogonal, and PoseAnchor (ICCV 2025) reports +5-15 mm MPJPE on top of
several pre-existing root-aware models without retraining.

Cost: one 2J×3 pseudoinverse per iteration.  At J=17, n_iters=5 this is
~0.5 ms on CPU.  Not ideal for 100 FPS mobile but trivial for server-
side eval; on-device you can skip or drop to n_iters=1.
"""
from __future__ import annotations

import numpy as np


def itrr_refine_root(
    kps2d: np.ndarray,         # [J, 2] full-image pixel coords
    kps3d_rel: np.ndarray,     # [J, 3] root-relative 3D in camera frame (metres)
    K: np.ndarray,             # [3, 3] camera intrinsics
    vis: np.ndarray | None = None,     # [J] 0/1 visibility mask, optional
    n_iters: int = 5,
    support_frac: float = 0.7,
    min_support: int = 4,
) -> np.ndarray:
    """Return the refined root translation ``T`` (3-vec, metres) that
    best explains the 2D observations given the root-relative 3D pose.

    This is a stateless utility — no model needed at call time."""
    kps2d = np.asarray(kps2d, dtype=np.float64)
    kps3d_rel = np.asarray(kps3d_rel, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    J = kps2d.shape[0]

    if vis is None:
        valid_mask = np.ones(J, dtype=bool)
    else:
        valid_mask = np.asarray(vis).astype(bool)
    if valid_mask.sum() < min_support:
        # Not enough valid joints — fall back to centroid trick.
        return np.array([0.0, 0.0, float(np.median(kps3d_rel[:, 2]))],
                        dtype=np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = kps2d[:, 0]
    v = kps2d[:, 1]
    Px = kps3d_rel[:, 0]
    Py = kps3d_rel[:, 1]
    Pz = kps3d_rel[:, 2]

    zero_J = np.zeros(J)
    # Rows of A for the u-constraint: [fx, 0, -(u-cx)]
    A_u = np.stack([fx * np.ones(J), zero_J, -(u - cx)], axis=1)
    # Rows for the v-constraint:      [0, fy, -(v-cy)]
    A_v = np.stack([zero_J, fy * np.ones(J), -(v - cy)], axis=1)
    b_u = (u - cx) * Pz - fx * Px
    b_v = (v - cy) * Pz - fy * Py

    # Stack: first J rows = u-constraints, next J rows = v-constraints.
    A = np.concatenate([A_u, A_v], axis=0)            # [2J, 3]
    b = np.concatenate([b_u, b_v], axis=0)            # [2J]
    joint_idx = np.concatenate([np.arange(J), np.arange(J)])

    # Mask invalid joints out of the initial solve.
    init_rows = np.concatenate([valid_mask, valid_mask])
    T, *_ = np.linalg.lstsq(A[init_rows], b[init_rows], rcond=None)

    # ITRR iterations — keep only the support_frac-best joints each step.
    k_support = max(min_support, int(round(valid_mask.sum() * support_frac)))
    for _ in range(n_iters):
        # 2D residual per joint (ℓ2 over the two rows).
        r = b - A @ T                                 # [2J]
        # Aggregate u,v residuals per joint.
        per_joint = np.zeros(J)
        per_joint += r[:J] ** 2
        per_joint += r[J:] ** 2
        per_joint = np.sqrt(per_joint)
        # Only consider initially-valid joints for support selection.
        per_joint[~valid_mask] = np.inf
        support = np.argsort(per_joint)[:k_support]
        rows = np.concatenate([support, support + J])
        T_new, *_ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
        if np.allclose(T_new, T, atol=1e-5):
            T = T_new
            break
        T = T_new
    return T.astype(np.float64)


__all__ = ["itrr_refine_root"]

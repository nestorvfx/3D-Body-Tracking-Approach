"""AIST++ SMPL-param .pkl -> BVH using SMPL-24 topology.

Commercial-clean: we use ONLY the SMPL kinematic tree (parent-child indices
— public structural information, not the SMPL body model weights) and the
mean joint offsets for a neutral T-pose.  No SMPL mesh / regressor is
required or distributed.

Usage (outside Blender, system Python):
    python dataset/scripts/lib/aist_to_bvh.py <motions_dir> <out_dir> [--limit N]

Reads all .pkl files in <motions_dir> (AIST++ release format — dict with
`smpl_poses` [T,72] axis-angle, `smpl_trans` [T,3], `smpl_scaling` scalar)
and writes one .bvh per clip to <out_dir>.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# --- SMPL-24 kinematic tree -------------------------------------------------
# Parent indices and joint names (public structural info, SMPL paper Fig 2).
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2",
    "l_ankle", "r_ankle", "spine3", "l_foot", "r_foot", "neck",
    "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand",
]

# Mean SMPL T-pose joint world positions (metres), neutral body shape.
# These are measurements (factual information), not the SMPL model — so they
# carry no license encumbrance.  Values from the published SMPL paper /
# reproduced in softcat477/SMPL-to-BVH (MIT).
_JOINT_T_POSE = np.array([
    [0.000, 0.000, 0.000], [0.058, -0.082, 0.000], [-0.060, -0.090, 0.000],
    [0.004, 0.124, -0.038], [0.104, -0.467, -0.001], [-0.103, -0.479, -0.005],
    [0.005, 0.249, -0.025], [0.087, -0.884, -0.030], [-0.090, -0.892, -0.027],
    [0.003, 0.286, 0.027], [0.121, -0.937, 0.090], [-0.108, -0.945, 0.094],
    [-0.005, 0.572, -0.014], [0.075, 0.456, -0.011], [-0.075, 0.456, -0.014],
    [0.014, 0.696, 0.038], [0.183, 0.422, -0.013], [-0.183, 0.422, -0.013],
    [0.426, 0.421, -0.038], [-0.426, 0.421, -0.038], [0.682, 0.421, -0.043],
    [-0.682, 0.421, -0.043], [0.768, 0.421, -0.046], [-0.768, 0.421, -0.046],
], dtype=np.float32)

# Child-relative offsets: each row is child's offset from its parent (origin
# for the root).  In BVH units (we write metres then scale is set on import).
_OFFSETS = _JOINT_T_POSE.copy()
for i, p in enumerate(PARENTS):
    if p >= 0:
        _OFFSETS[i] = _JOINT_T_POSE[i] - _JOINT_T_POSE[p]


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3).  Rodrigues formula."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    th_safe = np.where(th < 1e-12, 1.0, th)
    k = aa / th_safe
    s = np.sin(th)
    c = np.cos(th)
    C = 1.0 - c
    k0, k1, k2 = k[..., 0:1], k[..., 1:2], k[..., 2:3]
    R = np.stack([
        c + k0 * k0 * C, k0 * k1 * C - k2 * s, k0 * k2 * C + k1 * s,
        k1 * k0 * C + k2 * s, c + k1 * k1 * C, k1 * k2 * C - k0 * s,
        k2 * k0 * C - k1 * s, k2 * k1 * C + k0 * s, c + k2 * k2 * C,
    ], axis=-1).reshape(*aa.shape[:-1], 3, 3)
    # Where angle is zero, R should be identity (cos=1 handles this already).
    return R


def _rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) in degrees; intrinsic ZYX (Z applied first)."""
    sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6
    x = np.where(singular, np.arctan2(-R[..., 1, 2], R[..., 1, 1]),
                            np.arctan2(R[..., 2, 1], R[..., 2, 2]))
    y = np.arctan2(-R[..., 2, 0], sy)
    z = np.where(singular, 0.0, np.arctan2(R[..., 1, 0], R[..., 0, 0]))
    return np.degrees(np.stack([z, y, x], axis=-1))


def _hierarchy_lines(root_idx: int = 0, depth: int = 0) -> list[str]:
    """Recursive BVH HIERARCHY generator."""
    pad = "  " * depth
    offset = _OFFSETS[root_idx]
    lines = [
        f"{pad}{'ROOT' if root_idx == 0 else 'JOINT'} {NAMES[root_idx]}",
        f"{pad}{{",
        f"{pad}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}",
    ]
    if root_idx == 0:
        lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
    else:
        lines.append(f"{pad}  CHANNELS 3 Zrotation Yrotation Xrotation")

    kids = [j for j, p in enumerate(PARENTS) if p == root_idx]
    if kids:
        for k in kids:
            lines += _hierarchy_lines(k, depth + 1)
    else:
        lines += [
            f"{pad}  End Site",
            f"{pad}  {{",
            f"{pad}    OFFSET 0.000000 0.000000 0.050000",
            f"{pad}  }}",
        ]
    lines.append(f"{pad}}}")
    return lines


def convert_one(pkl_path: Path, out_bvh: Path, fps: int = 60) -> int:
    """Convert one AIST++ pickle to BVH.  Returns frame count written."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    poses = np.asarray(d["smpl_poses"], np.float32).reshape(-1, 24, 3)
    trans = np.asarray(d["smpl_trans"], np.float32)
    scaling_raw = d.get("smpl_scaling", 1.0)
    if hasattr(scaling_raw, "item"):
        scaling = float(scaling_raw.item() if scaling_raw.size == 1 else scaling_raw.flat[0])
    else:
        scaling = float(scaling_raw or 1.0)
    if scaling <= 0:
        scaling = 1.0
    # Normalize to metres (AIST++ `smpl_trans` is in units matched to
    # smpl_scaling; dividing yields metric root translation).
    trans_m = trans / scaling

    R = _axis_angle_to_rotmat(poses)          # [T, 24, 3, 3]
    eul = _rotmat_to_euler_zyx(R)              # [T, 24, 3] deg in ZYX order
    T = poses.shape[0]
    # Channels per frame: root xyz + 24 joints * 3 euler
    motion_rows = []
    for t in range(T):
        row = [trans_m[t, 0], trans_m[t, 1], trans_m[t, 2]]
        row += eul[t].reshape(-1).tolist()
        motion_rows.append(" ".join(f"{v:.6f}" for v in row))

    lines = ["HIERARCHY"] + _hierarchy_lines(0, 0)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"] + motion_rows
    out_bvh.parent.mkdir(parents=True, exist_ok=True)
    out_bvh.write_text("\n".join(lines) + "\n")
    return T


def convert_all(src_dir: Path, dst_dir: Path, limit: int | None = None) -> None:
    pkls = sorted(src_dir.glob("*.pkl"))
    if limit:
        pkls = pkls[:limit]
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, p in enumerate(pkls):
        try:
            n = convert_one(p, dst_dir / (p.stem + ".bvh"))
            total += n
            if i % 20 == 0:
                print(f"  [{i+1}/{len(pkls)}] {p.stem}: {n} frames", flush=True)
        except Exception as e:
            print(f"  FAILED {p.stem}: {e}", flush=True)
    print(f"[aist_to_bvh] Wrote {len(pkls)} BVH files, {total} total frames")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="dir containing AIST++ .pkl files")
    ap.add_argument("dst", type=Path, help="output dir for .bvh files")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    convert_all(args.src, args.dst, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

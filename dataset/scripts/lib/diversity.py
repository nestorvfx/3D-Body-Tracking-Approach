"""Diversity metrics for the rendered dataset.

Computed from 3D keypoint annotations (no Blender, no VPoser) so the metrics
can run on system Python with only numpy.  The goal isn't to compete with
research-grade pose priors; it's to give the iteration loop a fast,
quantitative signal that says "these 20 samples are all standing-walking" vs
"these 20 samples cover a diverse pose manifold."

Metrics:
  * APD (average pairwise distance) of root-relative joint configurations
  * covariance-log-determinant of PCA-32 features (hull-volume proxy)
  * per-joint angle entropy (bone direction distribution)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_poses_from_pilot(pilot_dir: Path) -> np.ndarray:
    """Gather all kept-frame 3D keypoints from a pilot output directory.

    Returns array [N, 17, 3] of world-space keypoints in metres.
    """
    poses = []
    for seq_dir in sorted(pilot_dir.glob("seq_*")):
        labels_path = seq_dir / "labels.json"
        if not labels_path.exists():
            continue
        labels = json.loads(labels_path.read_text())
        for fa in labels["frames"]:
            kps = fa["keypoints_3d_world_m"]
            if any(k["x"] is None for k in kps):
                continue
            arr = np.array([[k["x"], k["y"], k["z"]] for k in kps], dtype=np.float32)
            poses.append(arr)
    if not poses:
        return np.empty((0, 17, 3), dtype=np.float32)
    return np.stack(poses, axis=0)


def _root_relative(poses: np.ndarray) -> np.ndarray:
    """Subtract mid-hip (average of kps 11 and 12) from every joint."""
    if poses.shape[0] == 0:
        return poses
    root = (poses[:, 11] + poses[:, 12]) * 0.5  # mid-hip
    return poses - root[:, None, :]


def _bone_directions(poses: np.ndarray) -> np.ndarray:
    """Compute unit vectors along each COCO-17 skeleton edge for each pose."""
    # COCO-17 edges (same as lib.coco17.COCO17_SKELETON)
    edges = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
             (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (0, 1), (1, 3), (0, 2), (2, 4)]
    bones = []
    for a, b in edges:
        v = poses[:, b] - poses[:, a]
        norms = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
        bones.append(v / norms)
    return np.stack(bones, axis=1)          # [N, E, 3]


def apd(poses: np.ndarray, sample: int | None = 256) -> float:
    """Average pairwise Euclidean distance between root-relative configurations."""
    if poses.shape[0] < 2:
        return 0.0
    rel = _root_relative(poses).reshape(poses.shape[0], -1)   # [N, 51]
    n = rel.shape[0]
    if sample and n > sample:
        idx = np.random.default_rng(0).choice(n, sample, replace=False)
        rel = rel[idx]
        n = sample
    # Mean of upper triangle
    diffs = rel[:, None, :] - rel[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    mask = np.triu(np.ones_like(dists, dtype=bool), k=1)
    return float(dists[mask].mean())


def log_covariance_volume(poses: np.ndarray, d_pca: int = 16) -> float:
    """log|Σ| of the top-d_pca PCA features — proxy for pose hull volume.

    Higher = more diverse.  Uses a pseudo-inverse-safe computation so that
    rank-deficient batches return a finite (but large-negative) number.
    """
    if poses.shape[0] < 4:
        return float("-inf")
    rel = _root_relative(poses).reshape(poses.shape[0], -1)
    rel -= rel.mean(axis=0, keepdims=True)
    # SVD-based PCA
    try:
        U, S, Vt = np.linalg.svd(rel, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("-inf")
    k = min(d_pca, S.shape[0])
    eigs = (S[:k] ** 2) / max(1, rel.shape[0] - 1)
    eps = 1e-9
    return float(np.sum(np.log(eigs + eps)))


def bone_direction_entropy(poses: np.ndarray, n_bins: int = 20) -> float:
    """Mean Shannon entropy of bone-direction angular components across edges."""
    if poses.shape[0] == 0:
        return 0.0
    bones = _bone_directions(poses)           # [N, E, 3]
    ents = []
    for axis in range(3):
        vals = bones[..., axis]               # [N, E]
        for e in range(vals.shape[1]):
            hist, _ = np.histogram(vals[:, e], bins=n_bins, range=(-1.0, 1.0))
            p = hist / max(1, hist.sum())
            p = p[p > 0]
            ents.append(-np.sum(p * np.log(p)))
    return float(np.mean(ents))


def per_joint_coverage(poses: np.ndarray) -> dict[str, float]:
    """Per-joint bounding-box diagonals of 3D positions (metres)."""
    if poses.shape[0] == 0:
        return {}
    names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
    rel = _root_relative(poses)
    diag = (rel.max(axis=0) - rel.min(axis=0))
    diag_m = np.linalg.norm(diag, axis=-1)
    return {n: float(d) for n, d in zip(names, diag_m)}


def diversity_report(pilot_dir: str | Path) -> dict:
    pilot_dir = Path(pilot_dir)
    poses = _load_poses_from_pilot(pilot_dir)
    rep = {
        "n_frames": int(poses.shape[0]),
        "apd_m": apd(poses),
        "log_pca_volume": log_covariance_volume(poses),
        "bone_dir_entropy": bone_direction_entropy(poses),
        "per_joint_range_m": per_joint_coverage(poses),
    }
    return rep

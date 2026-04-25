"""Multi-teacher keypoint agreement filter (scaffold).

Consensus rule: for each joint, keep if >= `min_agree` teachers fall within
`sigma_px` pixels of each other.  Drop entire frame if fewer than
`min_joint_consensus` joints have consensus.
"""
from __future__ import annotations

import numpy as np


def agreement_mask(
    kps_per_teacher: np.ndarray,      # [T, J, 2] pixel coords per teacher
    sigma_px: float = 8.0,
    min_agree: int = 2,
) -> np.ndarray:
    """Return boolean [J] mask of joints with teacher agreement."""
    T, J, _ = kps_per_teacher.shape
    mask = np.zeros((J,), dtype=bool)
    for j in range(J):
        pts = kps_per_teacher[:, j, :]                   # [T, 2]
        dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)   # [T, T]
        # count teachers each teacher agrees with (self included)
        agree = (dists < sigma_px).sum(axis=1)
        mask[j] = (agree.max() >= min_agree)
    return mask


def frame_consensus_score(
    kps_per_teacher: np.ndarray,
    sigma_px: float = 8.0,
    min_agree: int = 2,
    min_joint_consensus: int = 10,
) -> tuple[bool, float, np.ndarray]:
    """Return (keep, agreement_ratio, per_joint_mask)."""
    mask = agreement_mask(kps_per_teacher, sigma_px, min_agree)
    ratio = float(mask.mean())
    return bool(mask.sum() >= min_joint_consensus), ratio, mask

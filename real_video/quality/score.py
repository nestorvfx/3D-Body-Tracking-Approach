"""Per-frame quality score (scaffold)."""
from __future__ import annotations

import numpy as np


def laplacian_variance(gray: np.ndarray) -> float:
    """Motion-blur proxy: variance of Laplacian.  Higher = sharper."""
    import cv2  # type: ignore
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_quality(
    *,
    agreement_ratio: float,       # from ensemble.agreement.frame_consensus_score
    n_visible: int,               # labeled joints this frame
    bbox_area: float,             # person bbox area, pixels
    motion_blur: float,           # laplacian_variance normalised to [0,1] where 1=blurry
    view_angle_rad: float,        # 0 = fronto-parallel, π = from behind
) -> float:
    s = 0.35 * np.clip(agreement_ratio, 0, 1)
    s += 0.20 * min(n_visible / 17.0, 1.0)
    s += 0.15 * np.clip(np.sqrt(bbox_area) / 200.0, 0, 1)
    s += 0.15 * (1.0 - np.clip(motion_blur, 0, 1))
    s += 0.15 * float(np.cos(view_angle_rad))
    return float(s)

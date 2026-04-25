"""Temporal consistency filters (scaffold).

Implements:
  - OneEuroFilter for smoothing 2D keypoint trajectories (MediaPipe's choice).
  - Bone-length constancy filter: median bone lengths over a window; drop
    frames where any bone deviates > threshold.
  - Velocity plausibility: drop frames where joint speed > N sigma.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class OneEuroFilter:
    """1D OneEuroFilter (per joint per axis).
    See https://gery.casiez.net/1euro/ — Casiez 2012."""
    freq: float = 30.0
    mincutoff: float = 1.0
    beta: float = 0.007
    dcutoff: float = 1.0
    _x_prev: float = 0.0
    _dx_prev: float = 0.0
    _t_prev: float = 0.0
    _initialized: bool = False

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if not self._initialized:
            self._x_prev = x; self._dx_prev = 0.0; self._t_prev = t
            self._initialized = True
            return x
        dt = max(1e-6, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        dx_hat = self._alpha(self.dcutoff, dt) * dx + \
                 (1 - self._alpha(self.dcutoff, dt)) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1 - alpha) * self._x_prev
        self._x_prev = x_hat; self._dx_prev = dx_hat; self._t_prev = t
        return x_hat


def bone_length_filter(
    kps_sequence: np.ndarray,         # [T, J, 3]
    bone_pairs: list[tuple[int, int]],
    max_deviation: float = 0.15,
) -> np.ndarray:
    """Return [T] boolean mask of frames whose bone lengths are within
    `max_deviation` of the per-bone median over the sequence."""
    T = kps_sequence.shape[0]
    per_bone_lengths = []
    for a, b in bone_pairs:
        d = np.linalg.norm(kps_sequence[:, a] - kps_sequence[:, b], axis=-1)  # [T]
        per_bone_lengths.append(d)
    per_bone_lengths = np.stack(per_bone_lengths, axis=0)                      # [B, T]
    medians = np.median(per_bone_lengths, axis=1, keepdims=True)               # [B, 1]
    deviations = np.abs(per_bone_lengths - medians) / (medians + 1e-9)
    frame_ok = (deviations < max_deviation).all(axis=0)
    return frame_ok


def velocity_filter(
    kps_sequence: np.ndarray,         # [T, J, 3] metric
    fps: float = 30.0,
    max_speed_m_per_s: float = 10.0,
) -> np.ndarray:
    """Drop frames where any joint's instantaneous speed exceeds threshold."""
    T = kps_sequence.shape[0]
    vel = np.diff(kps_sequence, axis=0) * fps                                  # [T-1, J, 3]
    speed = np.linalg.norm(vel, axis=-1)                                        # [T-1, J]
    ok = (speed.max(axis=-1) < max_speed_m_per_s)
    # Pad to length T; first frame assumed OK
    return np.concatenate([[True], ok], axis=0)

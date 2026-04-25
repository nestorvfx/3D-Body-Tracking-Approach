"""OneEuroFilter — causal, low-latency smoothing of noisy time series.

Reference: G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems," CHI 2012.
Public-domain reference implementation: https://gery.casiez.net/1euro/

Why this filter: MediaPipe's BlazePose pipeline uses it to eliminate
frame-to-frame joint wobble in real-time inference.  It is explicitly
designed for high-jitter low-latency data (hand tracking, head tracking),
which is exactly the profile of per-frame 3D keypoints from a single-frame
model.

Key property: the cutoff frequency is modulated by signal VELOCITY —
slow motion gets heavy smoothing (removes jitter), fast motion gets light
smoothing (preserves responsiveness).  No fixed latency ceiling.

This file is portable PyTorch/numpy-free so it can run:
  * in Python during training-eval
  * on CPU during on-device postprocessing (reimplementable in 30 lines
    of Swift/Kotlin/C++ for mobile deployment).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class OneEuroFilter1D:
    """Single-channel OneEuroFilter.  Stateful: call `filter(x, t)` each frame."""
    freq: float = 30.0                  # expected update rate (Hz)
    min_cutoff: float = 1.0             # minimum cutoff frequency (Hz)
    beta: float = 0.007                 # cutoff slope (the "velocity sensitivity")
    d_cutoff: float = 1.0               # cutoff for derivative estimate
    _x_prev: float | None = None
    _dx_prev: float = 0.0
    _t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float | None = None) -> float:
        if self._x_prev is None:
            # Initialise on first sample
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = float(t) if t is not None else 0.0
            return self._x_prev

        if t is None:
            dt = 1.0 / self.freq
            t = (self._t_prev or 0.0) + dt
        else:
            dt = max(1e-6, t - (self._t_prev or t))

        # Estimate derivative
        dx = (x - self._x_prev) / dt
        dx_hat = self._alpha(self.d_cutoff, dt) * dx + \
                  (1.0 - self._alpha(self.d_cutoff, dt)) * self._dx_prev
        self._dx_prev = dx_hat

        # Speed-adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._t_prev = t
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilter3D:
    """Vectorised OneEuroFilter over a fixed number of (x, y, z) joints.

    Shape: filter an array of shape (J, 3) per call.  Maintains J * 3
    independent 1D filters internally.
    """

    def __init__(
        self,
        num_joints: int = 17,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self.num_joints = num_joints
        self._filters = [
            OneEuroFilter1D(freq=freq, min_cutoff=min_cutoff,
                             beta=beta, d_cutoff=d_cutoff)
            for _ in range(num_joints * 3)
        ]

    def filter(self, kps: "Sequence[Sequence[float]]", t: float | None = None):
        """`kps` is an iterable of J length-3 coordinate triples.  Returns
        a list of J smoothed (x, y, z) tuples."""
        out: list[tuple[float, float, float]] = []
        for j in range(self.num_joints):
            xs = kps[j]
            x = self._filters[j * 3 + 0].filter(float(xs[0]), t)
            y = self._filters[j * 3 + 1].filter(float(xs[1]), t)
            z = self._filters[j * 3 + 2].filter(float(xs[2]), t)
            out.append((x, y, z))
        return out

    def reset(self) -> None:
        for f in self._filters:
            f.reset()


# Common presets tuned for pose keypoints.
# `min_cutoff` low + `beta` moderate = aggressive smoothing on still frames,
# responsive on fast motion.  These values replicate MediaPipe's defaults.
PRESETS = {
    "body_3d":  {"min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0},
    "body_2d":  {"min_cutoff": 0.5, "beta": 0.01, "d_cutoff": 1.0},
    "hand":     {"min_cutoff": 2.0, "beta": 0.1,  "d_cutoff": 1.0},
    "head":     {"min_cutoff": 1.5, "beta": 0.05, "d_cutoff": 1.0},
}


def make_filter(preset: str = "body_3d", num_joints: int = 17,
                 freq: float = 30.0) -> OneEuroFilter3D:
    cfg = PRESETS.get(preset, PRESETS["body_3d"])
    return OneEuroFilter3D(num_joints=num_joints, freq=freq, **cfg)

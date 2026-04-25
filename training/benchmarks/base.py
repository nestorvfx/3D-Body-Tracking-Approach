"""Benchmark protocol + shared utilities.

Every external benchmark implements the Benchmark protocol defined here.
The CLI in run.py takes any Benchmark and evaluates a checkpoint against
it.  Add a new commercial-clean benchmark by creating a module that
exposes ``build_benchmark(data_root, **kwargs) -> Benchmark``.

Keypoint convention: our model outputs COCO-17 body joints in the camera
frame.  Each benchmark provides a mapping list of length 17 — one entry
per COCO-17 index — giving either:
  * an integer  (benchmark's own joint index for the matching COCO joint)
  * -1          (no correspondence; masked out of metrics)

MPJPE and PA-MPJPE are then computed only on the matched joints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


# COCO-17 body keypoint names (the layout our model outputs).
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class BenchmarkSample:
    """One evaluation sample yielded by a Benchmark."""
    sample_id: str                 # e.g. "TS1/img_000001"
    image_rgb: np.ndarray          # uint8 HxWx3
    bbox_xywh: tuple[int, int, int, int]   # person bbox in full image pixels
    camera_K: np.ndarray           # [3,3] intrinsics (pixels)
    gt_kps3d_cam: np.ndarray       # [N_bench, 3] camera-frame metres, ORIGINAL benchmark joint layout
    # Per-joint visibility in the ORIGINAL benchmark joint layout.  Some
    # benchmarks don't provide occlusion flags (all joints considered
    # visible); in that case this is all-ones.
    gt_vis: np.ndarray             # [N_bench] 0/1 float
    meta: dict                     # arbitrary per-sample info (subject, etc.)


class Benchmark(Protocol):
    """Duck-typed benchmark interface."""

    name: str
    # Length-17 list.  Element i is the benchmark's own joint index that
    # corresponds to COCO-17 joint i, or -1 if no correspondence.
    coco17_to_bench: list[int]
    # Name list for the benchmark's native layout (for logging).
    bench_joint_names: list[str]

    def __len__(self) -> int: ...
    def iter_samples(self) -> Iterator[BenchmarkSample]: ...


def pick_matched(kps_xyz: np.ndarray, mapping: list[int]) -> np.ndarray | None:
    """Extract the 17 COCO-mapped joints from a benchmark's N-joint array.

    Returns [17, 3] with rows of NaN for any COCO index with mapping == -1.
    Callers should mask those out of metric computation.
    """
    if len(mapping) != 17:
        raise ValueError(f"mapping len {len(mapping)} != 17")
    out = np.full((17, 3), np.nan, dtype=np.float32)
    for i, j in enumerate(mapping):
        if j >= 0:
            out[i] = kps_xyz[j]
    return out


def valid_joint_mask(kps17: np.ndarray,
                      vis17: np.ndarray | None = None) -> np.ndarray:
    """Return a [17] bool mask of joints usable for metric computation."""
    m = ~np.isnan(kps17).any(axis=-1)
    if vis17 is not None:
        m &= (vis17 > 0.5)
    return m

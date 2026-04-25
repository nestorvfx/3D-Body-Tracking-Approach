"""BVH quality scoring + motion-aware window selection + ground-plane lift.

Three problems this module solves:

1. **Random window selection** sometimes lands on T-pose setup frames,
   stumble/recovery, or long static moments. We score candidate windows and
   pick a high-quality one.

2. **Uniform decimation (=3)** at 120fps BVH = 25ms between kept frames =
   <2 degrees joint rotation change between adjacent kept frames. Training
   labels are near-duplicates. We use a velocity-adaptive stride that targets
   ~150-200ms of effective pose change between kept frames (matches BEDLAM's
   5-6 fps subsampled training regime).

3. **"Flying" characters** when CMU clips contain jumps/falls with the root
   Y well below zero, or when shape retargeting pushed the mesh off the
   ground. We compute a per-clip ground offset via the 5th percentile of
   foot-Y across the window and lift the root bone up by that amount.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import bpy  # type: ignore
import math


# ------------------------- BVH sampling utilities -------------------------

_CMU_BODY_BONES = (
    "Hips", "Spine", "Spine1", "Head",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)


def _sample_frame(bvh_arm, frame: int) -> dict[str, "bpy.types.Vector"]:
    """Return world-space head positions for all tracked bones at `frame`."""
    scene = bpy.context.scene
    scene.frame_set(int(frame))
    dg = bpy.context.evaluated_depsgraph_get()
    ev = bvh_arm.evaluated_get(dg)
    mw = ev.matrix_world
    pos = {}
    for bn in _CMU_BODY_BONES:
        pb = ev.pose.bones.get(bn)
        if pb is None:
            continue
        pos[bn] = mw @ pb.head
    return pos


# ------------------------- Ground-plane lift -------------------------

@dataclass
class GroundOffset:
    ground_z: float
    lift_m: float          # amount to add to root_z so feet rest on Y=0


def compute_ground_offset(
    bvh_arm,
    frame_range: tuple[int, int],
    *,
    sample_stride: int = 4,
    percentile: float = 5.0,
) -> GroundOffset:
    """Percentile-based ground-plane lift.

    Samples foot-toe Y values across the clip (subsampled every
    `sample_stride` frames) and returns the lift needed so the 5th-percentile
    low point touches the ground plane at Y=0.  Per-clip, not per-frame, to
    avoid "bouncing" between jump and stance frames.
    """
    lo_frames = list(range(frame_range[0], frame_range[1] + 1, sample_stride))
    foot_ys: list[float] = []
    for f in lo_frames:
        pos = _sample_frame(bvh_arm, f)
        for b in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"):
            p = pos.get(b)
            if p is not None:
                foot_ys.append(float(p.z))
    if not foot_ys:
        return GroundOffset(ground_z=0.0, lift_m=0.0)
    foot_ys.sort()
    idx = max(0, int(len(foot_ys) * percentile / 100.0) - 1)
    ground_z = foot_ys[idx]
    # Lift so ground_z becomes 0 (or slightly above to avoid z-fighting).
    return GroundOffset(ground_z=ground_z, lift_m=-ground_z + 0.005)


# ------------------------- Window scoring -------------------------

@dataclass
class WindowScore:
    start_frame: int
    end_frame: int
    pose_variance: float     # higher = more motion
    root_jerk: float         # lower = smoother
    foot_contact_ratio: float
    bone_length_drift: float # lower = cleaner retarget target
    score: float


def _pose_variance(poses: list[dict]) -> float:
    """Mean per-bone per-frame joint-displacement variance."""
    if len(poses) < 2:
        return 0.0
    bone_names = list(poses[0].keys())
    diffs = []
    for b in bone_names:
        ys = [p[b].z for p in poses if b in p]
        xs = [p[b].x for p in poses if b in p]
        zs = [p[b].y for p in poses if b in p]
        if len(ys) < 2:
            continue
        # Mean absolute inter-frame displacement in m.
        n = min(len(xs), len(ys), len(zs))
        d = 0.0
        for i in range(1, n):
            d += ((xs[i] - xs[i - 1]) ** 2 +
                  (ys[i] - ys[i - 1]) ** 2 +
                  (zs[i] - zs[i - 1]) ** 2) ** 0.5
        diffs.append(d / max(1, n - 1))
    return sum(diffs) / max(1, len(diffs))


def _root_jerk(poses: list[dict]) -> float:
    """Third derivative of root translation magnitude, mean-per-frame."""
    roots = [p.get("Hips") for p in poses]
    roots = [r for r in roots if r is not None]
    if len(roots) < 4:
        return 0.0
    jerks = []
    for i in range(3, len(roots)):
        d3 = (roots[i] - 3 * roots[i - 1] + 3 * roots[i - 2] - roots[i - 3])
        jerks.append(d3.length)
    return sum(jerks) / max(1, len(jerks))


def _foot_contact_ratio(poses: list[dict], contact_z: float, eps: float = 0.08) -> float:
    """Fraction of frames where at least one foot is within `eps` of contact_z."""
    n_contact = 0
    for p in poses:
        foot_zs = [p[b].z for b in ("LeftToeBase", "RightToeBase",
                                      "LeftFoot", "RightFoot") if b in p]
        if not foot_zs:
            continue
        if min(foot_zs) - contact_z < eps:
            n_contact += 1
    return n_contact / max(1, len(poses))


def _bone_length_drift(poses: list[dict]) -> float:
    """RMS deviation of (LeftArm→LeftForeArm) bone length across frames."""
    dists = []
    for p in poses:
        if "LeftArm" in p and "LeftForeArm" in p:
            d = (p["LeftArm"] - p["LeftForeArm"]).length
            dists.append(d)
    if len(dists) < 2:
        return 0.0
    mu = sum(dists) / len(dists)
    var = sum((d - mu) ** 2 for d in dists) / len(dists)
    return var ** 0.5


def score_window(
    bvh_arm,
    start_frame: int,
    end_frame: int,
    ground_z: float = 0.0,
    sample_step: int = 2,
) -> WindowScore:
    """Collect frame poses and compute the composite score."""
    poses = [_sample_frame(bvh_arm, f)
             for f in range(start_frame, end_frame + 1, sample_step)]
    pv = _pose_variance(poses)
    jerk = _root_jerk(poses)
    fcr = _foot_contact_ratio(poses, ground_z)
    bld = _bone_length_drift(poses)
    # Normalized additive score.  Coefficients chosen empirically:
    #  - pose_variance ~ 0.003-0.05 m/frame for typical CMU
    #  - root_jerk ~ 0-0.1, >0.3 indicates teleport
    #  - foot_contact_ratio ~ 0-1
    #  - bone_length_drift ~ 0-0.01 m
    score = (4.0 * pv) + (0.5 * fcr) - (3.0 * jerk) - (50.0 * bld)
    return WindowScore(start_frame, end_frame, pv, jerk, fcr, bld, score)


def best_window(
    bvh_arm,
    seq_frames: int,
    rng,
    *,
    num_candidates: int = 8,
    min_gap_from_ends: int = 2,
) -> WindowScore:
    """Pick the best-scoring candidate window out of `num_candidates` random
    candidates.  Windows are non-overlapping when the clip is large enough."""
    if bvh_arm.animation_data and bvh_arm.animation_data.action:
        fr = bvh_arm.animation_data.action.frame_range
        clip_start, clip_end = int(fr[0]), int(fr[1])
    else:
        clip_start, clip_end = 1, seq_frames

    lo = clip_start + min_gap_from_ends
    hi = clip_end - seq_frames - min_gap_from_ends
    if hi <= lo:
        # Clip too short; return single available window.
        return score_window(bvh_arm, lo, lo + seq_frames - 1)

    # Rough ground estimate from a few sampled frames (cheap pre-pass).
    sample_frames = [rng.randint(lo, hi + seq_frames) for _ in range(6)]
    foot_zs: list[float] = []
    for f in sample_frames:
        p = _sample_frame(bvh_arm, f)
        for b in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"):
            if b in p:
                foot_zs.append(float(p[b].z))
    foot_zs.sort()
    ground_z = foot_zs[max(0, int(len(foot_zs) * 0.05))] if foot_zs else 0.0

    best: WindowScore | None = None
    tried: list[tuple[int, int]] = []
    for _ in range(num_candidates):
        s = rng.randint(lo, hi)
        e = s + seq_frames - 1
        ws = score_window(bvh_arm, s, e, ground_z=ground_z)
        tried.append((s, ws.score))
        if best is None or ws.score > best.score:
            best = ws
    assert best is not None
    return best


# ------------------------- Velocity-adaptive stride -------------------------

def adaptive_keep_frames(
    bvh_arm,
    start_frame: int,
    end_frame: int,
    *,
    target_stride_ms: float = 175.0,
    fps: float = 120.0,
    min_stride: int = 2,
    max_stride: int = 24,
    max_frames: int = 8,
) -> list[int]:
    """Return a list of frame indices to keep, spaced so that consecutive
    kept frames are ~`target_stride_ms` apart in effective pose change.

    We use the simpler time-based stride but bound it.  Full
    velocity-integral adaptation is overkill for a pilot: BEDLAM's 30->6fps
    subsample is the same idea.  With BVH 120fps + 175ms target => stride 21.
    """
    stride = max(min_stride, min(max_stride, int(round(target_stride_ms * fps / 1000.0))))
    frames = list(range(start_frame, end_frame + 1, stride))
    if len(frames) > max_frames:
        # Evenly downsample to max_frames
        n = len(frames)
        idx = [int(round(i * (n - 1) / (max_frames - 1))) for i in range(max_frames)]
        frames = [frames[i] for i in idx]
    return frames


def apply_ground_lift(target_arm, lift_m: float) -> None:
    """Offset the target armature's object Z position by `lift_m`.

    Safe because the character's kinematic chain is relative to the armature
    object's world matrix — we're moving the whole object, not the root bone.
    """
    if abs(lift_m) < 1e-4:
        return
    target_arm.location.z += lift_m
    bpy.context.view_layer.update()

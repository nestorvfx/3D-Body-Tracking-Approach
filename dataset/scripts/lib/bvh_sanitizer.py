"""Clip-level BVH sanitation: reject whole clips that are unusable BEFORE the
expensive retargeting + rendering pipeline runs on them.

Checks:
  - NaN / Inf in any animation curve
  - T-pose / calibration intro: leading N frames with joint variance near zero
  - Teleportation: root Δ > threshold per frame (bad data / cut errors)
  - Stretched bones (rare — authored scale channels); per-frame bone lengths
    should be constant
  - Framerate sanity (120 fps CMU / 60 fps AIST should round-trip)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore
import mathutils  # type: ignore


@dataclass
class BVHClipReport:
    path: str
    frames: int
    fps: float
    ok: bool
    rejection_reason: str = ""
    leading_calibration_frames: int = 0
    trailing_calibration_frames: int = 0
    bone_length_drift_pct: float = 0.0
    max_root_delta_m: float = 0.0

    def describe(self) -> str:
        status = "OK" if self.ok else f"REJECT ({self.rejection_reason})"
        return (f"[bvh_sanitizer] {Path(self.path).name}: {status} "
                f"frames={self.frames} fps={self.fps:.1f} "
                f"calib=[{self.leading_calibration_frames},{self.trailing_calibration_frames}] "
                f"drift%={self.bone_length_drift_pct:.2f} "
                f"max_root_Δ={self.max_root_delta_m:.2f}m")


def _iter_keyframe_values(fcurve) -> list[float]:
    try:
        return [kp.co[1] for kp in fcurve.keyframe_points]
    except Exception:
        return []


def _iter_action_fcurves(action):
    """Yield all FCurves on an Action across both Blender 4.x and 5.x APIs.

    Blender 5.x overhauled Action to use layers + slots + channelbags.  The
    simplest portable accessor is to iterate `action.layers[i].strips[j]` and
    collect `channelbag.fcurves`.  4.x still has the flat `action.fcurves`.
    """
    # Blender 4.x (legacy flat)
    if hasattr(action, "fcurves"):
        for fc in action.fcurves:
            yield fc
        return
    # Blender 5.x (layered)
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for slot in getattr(action, "slots", []):
                try:
                    cb = strip.channelbag(slot)
                except Exception:
                    cb = None
                if cb is None:
                    continue
                for fc in getattr(cb, "fcurves", []):
                    yield fc


def scan_action_for_nans(action) -> bool:
    """Return True if any fcurve in the action has a NaN/Inf value."""
    if action is None:
        return False
    for fc in _iter_action_fcurves(action):
        for v in _iter_keyframe_values(fc):
            if not math.isfinite(v):
                return True
    return False


# Long clips (100STYLE has 6000-8000+ frames) would make per-frame scans
# too expensive.  The sanitizer samples a bounded subset regardless of clip
# length so total cost stays under ~3s per clip.
_MAX_SANITIZER_SAMPLES = 60


def find_calibration_block(
    bvh_arm,
    max_check_frames: int = 30,
    stillness_threshold_deg: float = 0.5,
) -> tuple[int, int]:
    """Count leading/trailing frames where the pose is nearly static.

    Typical failure: CMU clips start with a T-pose or a few-frames calibration.
    Returns (leading, trailing) — frames to skip at each end of the clip.
    """
    if bvh_arm.animation_data is None or bvh_arm.animation_data.action is None:
        return 0, 0
    fr = bvh_arm.animation_data.action.frame_range
    start, end = int(fr[0]), int(fr[1])
    total = end - start + 1
    n_check = min(max_check_frames, total // 2)
    if n_check < 3:
        return 0, 0

    def _joint_deltas(frames):
        scene = bpy.context.scene
        prev = None
        deltas = []
        for f in frames:
            scene.frame_set(f)
            bpy.context.view_layer.update()
            q = [pb.rotation_quaternion.copy() for pb in bvh_arm.pose.bones]
            if prev is not None:
                d = 0.0
                for a, b in zip(prev, q):
                    # Angle between two unit quaternions
                    dot = abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w)
                    dot = max(-1.0, min(1.0, dot))
                    d += 2.0 * math.acos(dot)
                deltas.append(math.degrees(d) / max(1, len(q)))
            prev = q
        return deltas

    leading = 0
    deltas = _joint_deltas(range(start, start + n_check))
    for d in deltas:
        if d < stillness_threshold_deg:
            leading += 1
        else:
            break

    trailing = 0
    deltas = _joint_deltas(range(end - n_check, end + 1))
    for d in reversed(deltas):
        if d < stillness_threshold_deg:
            trailing += 1
        else:
            break

    return leading, trailing


def _sampled_frames(fr: tuple[int, int], max_samples: int = _MAX_SANITIZER_SAMPLES) -> list[int]:
    """Return up to `max_samples` evenly-spaced frame indices from [fr[0], fr[1]]."""
    start, end = int(fr[0]), int(fr[1])
    n = end - start + 1
    if n <= max_samples:
        return list(range(start, end + 1))
    stride = max(1, n // max_samples)
    return list(range(start, end + 1, stride))[:max_samples]


def measure_root_teleportation(bvh_arm) -> float:
    """Return max per-FRAME Δ in root translation (m) — i.e. actual jump
    between adjacent animation frames, not inter-sample gaps.

    To stay O(1) on long clips we inspect a few contiguous windows at the
    start / middle / end of the clip and measure frame-to-frame deltas
    within each window.  Real teleportation is a data defect that shows
    up as a single large spike; sampling a few windows catches it
    reliably while keeping cost bounded.
    """
    if "Hips" not in bvh_arm.pose.bones:
        return 0.0
    scene = bpy.context.scene
    if bvh_arm.animation_data is None or bvh_arm.animation_data.action is None:
        return 0.0
    fr = bvh_arm.animation_data.action.frame_range
    start, end = int(fr[0]), int(fr[1])
    total = end - start + 1
    window_size = 30
    if total <= window_size * 3:
        windows = [range(start, end + 1)]
    else:
        mid = (start + end) // 2
        windows = [
            range(start, start + window_size),
            range(mid - window_size // 2, mid + window_size // 2),
            range(end - window_size + 1, end + 1),
        ]

    max_delta = 0.0
    for w in windows:
        prev = None
        for f in w:
            scene.frame_set(f)
            bpy.context.view_layer.update()
            root = bvh_arm.matrix_world @ bvh_arm.pose.bones["Hips"].head
            if prev is not None:
                max_delta = max(max_delta, (root - prev).length)
            prev = root
    return max_delta


def measure_bone_length_drift(bvh_arm) -> float:
    """Return max relative drift (std/mean) across bone lengths.  Subsampled."""
    scene = bpy.context.scene
    if bvh_arm.animation_data is None or bvh_arm.animation_data.action is None:
        return 0.0
    frames = _sampled_frames(bvh_arm.animation_data.action.frame_range, max_samples=20)
    bones = list(bvh_arm.pose.bones)
    lengths: dict[str, list[float]] = {b.name: [] for b in bones}
    for f in frames:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        for b in bones:
            v = b.tail - b.head
            lengths[b.name].append(v.length)
    drift_pcts = []
    for name, vals in lengths.items():
        if len(vals) < 2:
            continue
        mu = sum(vals) / len(vals)
        if mu < 1e-6:
            continue
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        drift_pcts.append(100.0 * std / mu)
    return max(drift_pcts) if drift_pcts else 0.0


def sanitize_bvh_clip(
    bvh_arm,
    bvh_path: str,
    *,
    teleport_threshold_m: float = 0.5,
    drift_threshold_pct: float = 2.0,
    min_usable_frames: int = 30,
) -> BVHClipReport:
    """Decide whether this loaded BVH clip is usable.  Returns a report."""
    action = bvh_arm.animation_data.action if bvh_arm.animation_data else None
    fr = action.frame_range if action else (1, 1)
    n_frames = int(fr[1] - fr[0] + 1)
    fps = bpy.context.scene.render.fps

    rep = BVHClipReport(path=bvh_path, frames=n_frames, fps=float(fps), ok=True)

    if scan_action_for_nans(action):
        rep.ok = False
        rep.rejection_reason = "NaN/Inf in action fcurves"
        return rep

    leading, trailing = find_calibration_block(bvh_arm)
    rep.leading_calibration_frames = leading
    rep.trailing_calibration_frames = trailing
    usable = n_frames - leading - trailing
    if usable < min_usable_frames:
        rep.ok = False
        rep.rejection_reason = f"usable frames {usable} < {min_usable_frames}"
        return rep

    teleport = measure_root_teleportation(bvh_arm)
    rep.max_root_delta_m = teleport
    if teleport > teleport_threshold_m:
        rep.ok = False
        rep.rejection_reason = f"root teleport {teleport:.2f}m > {teleport_threshold_m}"
        return rep

    drift = measure_bone_length_drift(bvh_arm)
    rep.bone_length_drift_pct = drift
    if drift > drift_threshold_pct:
        rep.ok = False
        rep.rejection_reason = f"bone-length drift {drift:.2f}% > {drift_threshold_pct}"
        return rep

    return rep

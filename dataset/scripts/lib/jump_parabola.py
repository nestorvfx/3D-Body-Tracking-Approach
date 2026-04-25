"""Distinguish a real ballistic jump from a floating-artifact retarget error.

Idea (GroundLink SIGGRAPH Asia 2023, https://csr.bu.edu/groundlink/):
when both feet are off the ground for several frames, fit a parabola to the
pelvis vertical position.  Real jumps follow `z(t) = z0 + v0*t - 0.5 g t^2`,
i.e., ballistic motion; fit residual (R²) is near 1.0.  Retarget errors
produce flat or noisy height profiles with R² far from 1.0.

Usage:
    events = scan_airborne_windows(armature, f_start, f_end)
    for ev in events:
        if ev.is_real_jump:
            pass                         # leave alone
        else:
            apply_foot_lock_or_reject(ev)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import bpy  # type: ignore
import mathutils  # type: ignore


@dataclass
class AirborneEvent:
    start_frame: int
    end_frame: int
    max_height_m: float
    parabola_r2: float
    is_real_jump: bool


def _pelvis_z(armature) -> float:
    pb = armature.pose.bones.get("pelvis") or armature.pose.bones.get("root")
    if pb is None:
        return 0.0
    return (armature.matrix_world @ pb.head).z


def _any_foot_airborne(armature, z_threshold: float) -> bool:
    for name in ("foot.L", "foot.R"):
        pb = armature.pose.bones.get(name)
        if pb is None:
            continue
        if (armature.matrix_world @ pb.head).z > z_threshold:
            continue
        return False   # at least one foot grounded
    return True


def _fit_parabola(ys: list[float]) -> float:
    """Return R² of quadratic least-squares fit to `ys` vs index.
    1.0 = perfect parabola, 0.0 = no correlation."""
    n = len(ys)
    if n < 4:
        return 0.0
    xs = list(range(n))
    sx = sum(xs); sx2 = sum(x * x for x in xs); sx3 = sum(x ** 3 for x in xs); sx4 = sum(x ** 4 for x in xs)
    sy = sum(ys); sxy = sum(x * y for x, y in zip(xs, ys)); sx2y = sum((x * x) * y for x, y in zip(xs, ys))
    # Solve 3x3 normal equations for a*x^2 + b*x + c
    M = [
        [sx4, sx3, sx2],
        [sx3, sx2, sx],
        [sx2, sx,  n],
    ]
    rhs = [sx2y, sxy, sy]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                 - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                 + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    D = det3(M)
    if abs(D) < 1e-12:
        return 0.0

    def replace_col(col, new):
        return [row[:col] + [new[i]] + row[col+1:] for i, row in enumerate(M)]

    a = det3(replace_col(0, rhs)) / D
    b = det3(replace_col(1, rhs)) / D
    c = det3(replace_col(2, rhs)) / D

    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x * x + b * x + c)) ** 2 for x, y in zip(xs, ys))
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def scan_airborne_windows(
    armature,
    frame_start: int,
    frame_end: int,
    *,
    z_threshold_m: float = 0.08,
    min_airborne_frames: int = 4,
    r2_real_jump_threshold: float = 0.85,
) -> list[AirborneEvent]:
    """Find contiguous spans where both feet are above `z_threshold_m` and
    classify each as a real ballistic jump (R² above threshold) or an
    artifact (R² below)."""
    scene = bpy.context.scene
    airborne_mask: list[bool] = []
    pelvis_zs: list[float] = []

    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        airborne_mask.append(_any_foot_airborne(armature, z_threshold_m))
        pelvis_zs.append(_pelvis_z(armature))

    events: list[AirborneEvent] = []
    i = 0
    while i < len(airborne_mask):
        if not airborne_mask[i]:
            i += 1
            continue
        j = i
        while j < len(airborne_mask) and airborne_mask[j]:
            j += 1
        run_len = j - i
        if run_len >= min_airborne_frames:
            window = pelvis_zs[i:j]
            r2 = _fit_parabola(window)
            events.append(AirborneEvent(
                start_frame=frame_start + i,
                end_frame=frame_start + j - 1,
                max_height_m=max(window),
                parabola_r2=r2,
                is_real_jump=r2 >= r2_real_jump_threshold,
            ))
        i = j

    return events

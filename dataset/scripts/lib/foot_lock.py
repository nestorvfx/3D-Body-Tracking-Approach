"""Velocity-threshold foot lock for post-retargeting drift.

Problem: CMU BVH retargeted to MPFB without IK produces foot slide — the
character appears to glide across the floor when the pelvis translation
doesn't perfectly match foot-ground contact.

Solution: detect "foot plant" frames heuristically (ankle low AND slow) and
lock the foot position to its first-contact world coord until the detected
release frame.  Implemented as per-frame Copy Location constraints that
target an Empty placed at the planted position.

References:
  - MediaPipe / OneEuroFilter contact logic
  - DeepMimic foot plant detection (Peng et al. https://arxiv.org/abs/1804.02717)
  - GroundLink foot contact inference, SIGGRAPH Asia 2023
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import bpy  # type: ignore
import mathutils  # type: ignore


@dataclass
class FootPlantEvent:
    bone_name: str            # "foot.L" or "foot.R"
    start_frame: int
    end_frame: int
    pinned_world_pos: tuple[float, float, float]


def _foot_world_pos(armature, bone_name: str) -> "mathutils.Vector":
    pb = armature.pose.bones.get(bone_name)
    if pb is None:
        return mathutils.Vector((0.0, 0.0, 0.0))
    return armature.matrix_world @ pb.head


def detect_foot_plants(
    armature,
    frame_start: int,
    frame_end: int,
    *,
    z_threshold_m: float = 0.05,
    velocity_threshold_m: float = 0.06,      # per-frame displacement
    min_plant_frames: int = 4,
) -> list[FootPlantEvent]:
    """Scan the retargeted animation for foot-plant windows.

    A plant is defined as a contiguous run of frames where BOTH:
      - foot Z < z_threshold_m (close to ground)
      - frame-to-frame foot Δ < velocity_threshold_m (not moving)
    lasting at least `min_plant_frames`.
    """
    scene = bpy.context.scene
    events: list[FootPlantEvent] = []

    for bone_name in ("foot.L", "foot.R"):
        if bone_name not in armature.pose.bones:
            continue
        positions: list[mathutils.Vector] = []
        for f in range(frame_start, frame_end + 1):
            scene.frame_set(f)
            bpy.context.view_layer.update()
            positions.append(_foot_world_pos(armature, bone_name).copy())

        # Mark frames as "planted" when both criteria hold.
        planted_mask: list[bool] = [False] * len(positions)
        for i in range(len(positions)):
            low = positions[i].z < z_threshold_m
            if i == 0:
                fast = False
            else:
                fast = (positions[i] - positions[i - 1]).length >= velocity_threshold_m
            planted_mask[i] = low and not fast

        # Run-length encode planted spans >= min_plant_frames.
        i = 0
        while i < len(planted_mask):
            if not planted_mask[i]:
                i += 1
                continue
            j = i
            while j < len(planted_mask) and planted_mask[j]:
                j += 1
            run_len = j - i
            if run_len >= min_plant_frames:
                anchor = positions[i + run_len // 2]     # mid-plant position
                events.append(FootPlantEvent(
                    bone_name=bone_name,
                    start_frame=frame_start + i,
                    end_frame=frame_start + j - 1,
                    pinned_world_pos=(float(anchor.x), float(anchor.y), float(anchor.z)),
                ))
            i = j

    return events


def apply_foot_plants(armature, events: Sequence[FootPlantEvent]) -> list[object]:
    """Create pin Empties + Copy Location constraints on the foot bones.

    Returns the list of created empties (caller can remove after rendering).
    The constraints have their `influence` keyframed per frame inside the
    plant window so only active frames are pinned.
    """
    created: list[object] = []
    scene = bpy.context.scene

    for ev in events:
        # Create an empty at the plant position.
        empty = bpy.data.objects.new(
            f"_footpin_{ev.bone_name}_{ev.start_frame}",
            None,
        )
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.05
        empty.location = ev.pinned_world_pos
        scene.collection.objects.link(empty)
        created.append(empty)

        pb = armature.pose.bones[ev.bone_name]
        c = pb.constraints.new("COPY_LOCATION")
        c.name = f"_footpin_{ev.start_frame}_{ev.end_frame}"
        c.target = empty
        c.target_space = "WORLD"
        c.owner_space = "WORLD"
        c.influence = 0.0
        c.keyframe_insert("influence", frame=ev.start_frame - 1)
        c.influence = 1.0
        c.keyframe_insert("influence", frame=ev.start_frame)
        c.keyframe_insert("influence", frame=ev.end_frame)
        c.influence = 0.0
        c.keyframe_insert("influence", frame=ev.end_frame + 1)

    return created


def cleanup_foot_plants(armature, created_empties: Sequence[object]) -> None:
    for pb in armature.pose.bones:
        stale = [c for c in pb.constraints if c.name.startswith("_footpin_")]
        for c in stale:
            try:
                pb.constraints.remove(c)
            except Exception:
                pass
    for empty in created_empties:
        try:
            bpy.data.objects.remove(empty, do_unlink=True)
        except Exception:
            pass

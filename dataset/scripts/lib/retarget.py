"""CMU BVH → MPFB "default" rig retargeting, headless.

Process:
  1. Import the BVH as a temporary armature.
  2. Set target rig pose bones to XYZ rotation mode.
  3. Add Copy Rotation (LOCAL-LOCAL) constraints from BVH bones → MPFB bones.
  4. Add Copy Location (WORLD-WORLD) from BVH Hips → MPFB root.
  5. Bake visual keying into the target armature's action.
  6. Delete the BVH armature.
"""
from __future__ import annotations

from pathlib import Path

import bpy  # type: ignore

from .cmu_bvh import CMU_TO_MPFB_DEFAULT, CMU_BVH_GLOBAL_SCALE


def retarget_bvh_to_mpfb(
    bvh_path: str | Path,
    target_armature: object,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    frame_step: int = 1,
) -> tuple[int, int]:
    """Apply a BVH motion to `target_armature` (MPFB default rig).

    Returns (bake_start, bake_end) frame range actually baked."""
    bvh_path = str(bvh_path)

    # Deselect everything before the import so we know what's new.
    bpy.ops.object.select_all(action="DESELECT")
    existing = set(bpy.data.objects)

    bpy.ops.import_anim.bvh(
        filepath=bvh_path,
        global_scale=CMU_BVH_GLOBAL_SCALE,
        rotate_mode="NATIVE",
        axis_forward="-Z",
        axis_up="Y",
        use_fps_scale=False,
        update_scene_fps=True,
        update_scene_duration=True,
    )
    new_objs = [o for o in bpy.data.objects if o not in existing]
    bvh_arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
    if bvh_arm is None:
        raise RuntimeError(f"BVH import produced no armature from {bvh_path}")

    action = bvh_arm.animation_data.action if bvh_arm.animation_data else None
    if action is None:
        raise RuntimeError("BVH armature has no animation data")
    full_start = int(action.frame_range[0])
    full_end = int(action.frame_range[1])
    bake_start = start_frame if start_frame is not None else full_start
    bake_end = end_frame if end_frame is not None else full_end
    bake_start = max(bake_start, full_start)
    bake_end = min(bake_end, full_end)

    # Ensure target pose bones are in XYZ euler for clean baked keys.
    for pb in target_armature.pose.bones:
        pb.rotation_mode = "XYZ"

    # Wire constraints BVH → target.
    added_constraints = []
    for src_name, dst_name in CMU_TO_MPFB_DEFAULT.items():
        if src_name not in bvh_arm.pose.bones:
            continue
        if dst_name not in target_armature.pose.bones:
            continue
        pb = target_armature.pose.bones[dst_name]
        c = pb.constraints.new("COPY_ROTATION")
        c.name = f"_retarget_rot_{src_name}"
        c.target = bvh_arm
        c.subtarget = src_name
        c.target_space = "LOCAL"
        c.owner_space = "LOCAL"
        added_constraints.append((pb, c.name))

    # Root translation: Hips world → root world.
    if "Hips" in bvh_arm.pose.bones and "root" in target_armature.pose.bones:
        root = target_armature.pose.bones["root"]
        loc = root.constraints.new("COPY_LOCATION")
        loc.name = "_retarget_loc_Hips"
        loc.target = bvh_arm
        loc.subtarget = "Hips"
        loc.target_space = "WORLD"
        loc.owner_space = "WORLD"
        added_constraints.append((root, loc.name))

    # Bake into target_armature.
    bpy.context.view_layer.objects.active = target_armature
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    target_armature.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")

    bpy.ops.nla.bake(
        frame_start=bake_start,
        frame_end=bake_end,
        step=frame_step,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=False,   # creates a fresh action for the baked keys
        bake_types={"POSE"},
    )

    bpy.ops.object.mode_set(mode="OBJECT")

    # Delete the source BVH armature + its action.
    bvh_data = bvh_arm.data
    bpy.data.objects.remove(bvh_arm, do_unlink=True)
    try:
        bpy.data.armatures.remove(bvh_data)
    except Exception:
        pass
    try:
        bpy.data.actions.remove(action)
    except Exception:
        pass

    return bake_start, bake_end

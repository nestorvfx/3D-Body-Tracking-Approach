"""Fast BVH -> MPFB retargeting via live Copy Rotation constraints.

The old `retarget.py` was slow because it called ``bpy.ops.nla.bake`` across the
entire BVH clip (100-500 frames) just to extract one pose.  That op writes a
keyframe per frame per bone (thousands of fcurve inserts) and then clears
constraints — the fcurve churn dominates.

Insight: we never needed the baked keyframes.  The constraints themselves (Copy
Rotation LOCAL-LOCAL per bone + Copy Location WORLD on root) are what Blender
evaluates at render time.  If we leave them live and just drive
``scene.frame_set(f)``, Blender's depsgraph evaluates the BVH action, evaluates
the target armature through the constraints, and we get the correct pose for
FREE every frame — no bake required.

Cost: ~10-30 ms/frame (dominated by depsgraph eval), vs ~16 s/sample with bake.

Usage:
    bvh_arm = load_bvh(path)
    ctx = RetargetContext(bvh_arm, target_arm)   # installs constraints once
    for f in frames:
        ctx.apply_pose(f)                         # scene.frame_set + eval
    ctx.cleanup()                                  # removes constraints + BVH
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import bpy  # type: ignore

from .cmu_bvh import CMU_TO_MPFB_DEFAULT, CMU_BVH_GLOBAL_SCALE
from .source_mappings import (
    mapping_for_source, detect_source_from_bvh, root_rotation_safe,
    copy_rotation_space,
)
from . import ik_retarget


def load_bvh(path: str | Path, source: str | None = None) -> object:
    """Import a BVH as a temporary armature. Returns the armature object.

    If `source` is None, auto-detects based on bone-name heuristics
    (cmu | 100style | aistpp).  The source determines the global scale factor
    and the bone-name mapping used by RetargetContext.
    """
    if source is None:
        source = detect_source_from_bvh(str(path))
    _, global_scale = mapping_for_source(source)
    bpy.ops.object.select_all(action="DESELECT")
    existing = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(
        filepath=str(path),
        global_scale=global_scale,
        rotate_mode="NATIVE",
        axis_forward="-Z",
        axis_up="Y",
        use_fps_scale=False,
        update_scene_fps=True,
        update_scene_duration=True,
    )
    new_objs = [o for o in bpy.data.objects if o not in existing]
    arm = next((o for o in new_objs if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"BVH import produced no armature from {path}")
    arm["_mocap_source"] = source
    return arm


@dataclass
class RetargetContext:
    """Installs live Copy Rotation / Copy Location constraints from a BVH
    armature onto an MPFB target armature.  `apply_pose(frame)` just shifts the
    scene frame and forces depsgraph evaluation; Blender's constraint stack
    handles the rest-pose-aware rotation transfer for free."""

    bvh_arm: object
    target_arm: object
    bone_map: dict[str, str] = field(default_factory=dict)
    _added: list[tuple[object, str]] = field(default_factory=list)
    frame_range: tuple[int, int] = (1, 1)
    use_ik_limbs: bool = True             # SOTA: drive end-effector via IK
    _ik_rig: object = None
    # Rest-pose-offset root translation (computed at __post_init__ time):
    # SOURCE hips REST world position and TARGET root REST world position.
    # We translate the target ARMATURE OBJECT each frame by the delta between
    # source hips current world position and source hips rest world position.
    # This keeps relative hip motion correct without depending on absolute
    # coordinate alignment between the two armatures.
    _src_hips_rest_world: object = None
    _tgt_arm_rest_location: object = None

    # MPFB target bones whose Copy Rotation we SKIP when use_ik_limbs is True.
    # These bones are IN the IK chain (controlled by the IK solver), so adding
    # Copy Rotation would fight IK.
    #
    # Per-source tuning: AIST++ benefits from upperarm01/upperleg01 also being
    # IK-controlled (its SMPL kinematic-tree bone axes don't align with MPFB,
    # so Copy Rotation introduces 40-60° extra arm/leg error).  CMU/100STYLE
    # benefit from Copy Rotation on upperarm01/upperleg01 because they have
    # rest-pose axes that align better with MPFB's.
    _IK_OVERRIDDEN_DEFAULT: set[str] = field(default_factory=set)   # none — allow Copy Rotation on everything, IK fine-tunes
    _IK_OVERRIDDEN_AISTPP = {
        "upperarm01.L", "upperarm01.R",
        "lowerarm01.L", "lowerarm01.R",
        "upperleg01.L", "upperleg01.R",
        "lowerleg01.L", "lowerleg01.R",
    }

    def __post_init__(self) -> None:
        # Populate bone_map from source if empty.
        source = self.bvh_arm.get("_mocap_source", "cmu")
        if not self.bone_map:
            self.bone_map, _ = mapping_for_source(source)
        rotation_space = copy_rotation_space(source)
        # Make sure target pose bones are in XYZ euler for predictable behaviour.
        for pb in self.target_arm.pose.bones:
            pb.rotation_mode = "XYZ"

        # Clear any stale retarget constraints from prior runs.
        self._remove_prior_retarget_constraints()

        tgt_bones = self.target_arm.pose.bones
        src_bones = self.bvh_arm.pose.bones

        # Per-bone Copy Rotation — except for limb bones driven by IK.
        ik_override = (self._IK_OVERRIDDEN_AISTPP if source == "aistpp"
                        else self._IK_OVERRIDDEN_DEFAULT)
        for src, dst in self.bone_map.items():
            if src == "Hips" or src not in src_bones or dst not in tgt_bones:
                continue
            if self.use_ik_limbs and dst in ik_override:
                continue
            pb = tgt_bones[dst]
            c = pb.constraints.new("COPY_ROTATION")
            c.name = f"_fast_rot_{src}"
            c.target = self.bvh_arm
            c.subtarget = src
            c.target_space = rotation_space
            c.owner_space = rotation_space
            self._added.append((pb, c.name))

        # Root translation: Copy Location WORLD-WORLD from SOURCE root bone
        # to MPFB root.  The source root bone name varies per schema
        # (CMU "Hips", 100STYLE "Hips", AIST++ "pelvis") so we look it up
        # from the bone_map rather than hardcoding.  Rest-pose Z mismatch
        # between rigs becomes a fixed per-character offset absorbed by
        # Procrustes at validation time and invisible in rendering.
        source_root = next(
            (src for src, dst in self.bone_map.items() if dst == "root"),
            None,
        )
        if source_root and source_root in src_bones and "root" in tgt_bones:
            root = tgt_bones["root"]
            c = root.constraints.new("COPY_LOCATION")
            c.name = f"_fast_loc_{source_root}"
            c.target = self.bvh_arm
            c.subtarget = source_root
            c.target_space = "WORLD"
            c.owner_space = "WORLD"
            self._added.append((root, c.name))

        # Copy root rotation from source Hips — only for sources where this
        # is known to align with MPFB's root rest axes.  CMU works; 100STYLE,
        # AIST++, MHAD do not (rest-pose axis mismatch flips the character
        # 90° regardless of LOCAL vs WORLD constraint space).  The character
        # will miss source yaw but stays upright, which is the correct
        # trade-off for training data where camera yaw is randomised anyway.
        source = self.bvh_arm.get("_mocap_source", "cmu")
        if (root_rotation_safe(source) and source_root
                and source_root in src_bones and "root" in tgt_bones):
            root = tgt_bones["root"]
            c = root.constraints.new("COPY_ROTATION")
            c.name = f"_fast_rot_root_{source_root}"
            c.target = self.bvh_arm
            c.subtarget = source_root
            c.target_space = rotation_space
            c.owner_space = rotation_space
            self._added.append((root, c.name))

        # Install IK on arm + leg chains targeting source end-effectors.
        # This corrects for bone-length and body-width mismatch between the
        # thin source mocap skeletons and MPFB's clothed character mesh —
        # without IK, source poses with hands at the navel end up with
        # MPFB wrists CLIPPING into the abdomen.
        if self.use_ik_limbs:
            self._ik_rig = ik_retarget.add_limb_ik(
                self.target_arm, self.bvh_arm, arm=True, leg=True)

        if self.bvh_arm.animation_data and self.bvh_arm.animation_data.action:
            fr = self.bvh_arm.animation_data.action.frame_range
            self.frame_range = (int(fr[0]), int(fr[1]))

    def _remove_prior_retarget_constraints(self) -> None:
        for pb in self.target_arm.pose.bones:
            stale = [c for c in pb.constraints if c.name.startswith("_fast_")
                     or c.name.startswith("_retarget_")
                     or c.name.startswith("_ik_")]
            for c in stale:
                pb.constraints.remove(c)
        # Remove orphan IK target empties from prior runs
        for o in list(bpy.data.objects):
            if o.name.startswith("_iktgt_"):
                try:
                    bpy.data.objects.remove(o, do_unlink=True)
                except Exception:
                    pass

    def apply_pose(self, frame: int) -> None:
        """Drive the BVH action to `frame` and force a depsgraph update."""
        bpy.context.scene.frame_set(int(frame))
        bpy.context.view_layer.update()

    def cleanup(self) -> None:
        """Remove all installed constraints and the BVH armature datablock."""
        if self._ik_rig is not None:
            try:
                self._ik_rig.cleanup()
            except Exception:
                pass
            self._ik_rig = None
        for pb, cname in self._added:
            c = pb.constraints.get(cname)
            if c:
                try:
                    pb.constraints.remove(c)
                except Exception:
                    pass
        self._added.clear()

        # Remove BVH armature + its action.
        try:
            action = None
            if self.bvh_arm.animation_data:
                action = self.bvh_arm.animation_data.action
            data = self.bvh_arm.data
            bpy.data.objects.remove(self.bvh_arm, do_unlink=True)
            if data is not None:
                try:
                    bpy.data.armatures.remove(data)
                except Exception:
                    pass
            if action is not None:
                try:
                    bpy.data.actions.remove(action)
                except Exception:
                    pass
        except Exception as e:
            print(f"[fast_retarget] cleanup ignored: {e}")

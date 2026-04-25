"""IK-based limb retargeting that prevents hand-into-torso and foot-into-leg
clipping caused by bone-length and body-width mismatches.

Problem: rotation-only Copy Rotation propagates source bone rotations to the
target rig, but MPFB's bones have DIFFERENT lengths and the MPFB mesh has a
different body width than the thin source-mocap skeleton.  When the source
puts a hand "at the navel", MPFB's longer arm + thicker torso ends up with
the wrist INSIDE the abdomen.

Solution (Auto-Rig Pro / Rokoko approach): drive the end-effector via IK
targeting the source bone's world position.  The IK solver finds joint
rotations that put the MPFB wrist where the source wrist is, regardless of
the proportion mismatch.

Topology:
  - MPFB chain: clavicle.L -> upperarm01.L -> upperarm02.L -> upperarm03.L
                -> lowerarm01.L -> lowerarm02.L -> wrist.L
  - We add IK on `lowerarm02.L` with chain_length=4 (covers lowerarm02,
    lowerarm01, upperarm03, upperarm02; clavicle and upperarm01 keep their
    Copy Rotation drivers for shoulder orientation).
  - Target = an Empty driven by Copy Location from the source `LeftHand` bone.

For legs analogously: IK on `lowerleg02.L` with target = source `LeftFoot`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import bpy  # type: ignore


# (target_chain_end_bone, ik_chain_length, MPFB target bone name we look up
#  the source equivalent for via the source's bone_map)
#
# The source bone name varies by source schema:
#   CMU       — LeftHand / RightHand / LeftFoot / RightFoot
#   100STYLE  — LeftWrist / RightWrist / LeftAnkle / RightAnkle
#   AIST++    — l_wrist   / r_wrist   / l_ankle   / r_ankle
#
# We look up the correct source bone name by inverting the source's bone_map
# (find the source key that maps to the MPFB target bone we care about).
ARM_IK_SETUP = {
    # chain_length=3 covers (lowerarm02, lowerarm01, upperarm03).
    # upperarm02/01 are OUT of the IK chain; upperarm01 gets Copy Rotation
    # (per source) so the shoulder direction follows source.  This reduces
    # IK's freedom to place the elbow in arbitrary positions.
    "L": ("lowerarm02.L", 3, "wrist.L"),
    "R": ("lowerarm02.R", 3, "wrist.R"),
}
LEG_IK_SETUP = {
    "L": ("lowerleg02.L", 3, "foot.L"),
    "R": ("lowerleg02.R", 3, "foot.R"),
}


def _find_source_bone_for_target(source_arm, target_mpfb_bone: str) -> str | None:
    """Look up the source-side bone name whose mapping points to the named
    MPFB bone.  Source's bone map is read from `source_mappings`."""
    from .source_mappings import mapping_for_source
    src = source_arm.get("_mocap_source", "cmu")
    bone_map, _ = mapping_for_source(src)
    for src_name, dst_name in bone_map.items():
        if dst_name == target_mpfb_bone:
            return src_name
    return None


def _create_target_empty(name: str, source_arm, source_bone_name: str,
                          mode: str = "head") -> object:
    """Create an Empty whose world location follows source_arm.pose.bones[source_bone_name].
    """
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(empty)
    # Use Copy Location WORLD with the source pose bone's HEAD via Empty parenting:
    # Blender's Copy Location supports a bone subtarget; the location follows the
    # bone's HEAD position by default, in WORLD space.
    c = empty.constraints.new("COPY_LOCATION")
    c.name = "_iktgt_follow"
    c.target = source_arm
    c.subtarget = source_bone_name
    c.target_space = "WORLD"
    c.owner_space = "WORLD"
    # `head_tail` controls whether to follow head (0) or tail (1).
    try:
        c.head_tail = 0.0 if mode == "head" else 1.0
    except Exception:
        pass
    return empty


def _add_ik(target_arm, target_bone_name: str, target_empty, chain_length: int,
             pole: object | None = None, pole_angle: float = 0.0) -> str:
    """Add an IK constraint on the named pose bone targeting `target_empty`.
    Returns constraint name."""
    pb = target_arm.pose.bones.get(target_bone_name)
    if pb is None:
        return ""
    c = pb.constraints.new("IK")
    cname = f"_ik_{target_bone_name}"
    c.name = cname
    c.target = target_empty
    c.chain_count = chain_length
    c.use_tail = True
    c.use_stretch = False
    c.use_rotation = True
    if pole is not None:
        c.pole_target = pole
        c.pole_angle = pole_angle
    return cname


@dataclass
class IKRig:
    target_arm: object
    source_arm: object
    empties: list[object]
    constraints: list[tuple[object, str]]

    def cleanup(self) -> None:
        for pb, cname in self.constraints:
            c = pb.constraints.get(cname)
            if c:
                try:
                    pb.constraints.remove(c)
                except Exception:
                    pass
        for e in self.empties:
            try:
                bpy.data.objects.remove(e, do_unlink=True)
            except Exception:
                pass


def add_limb_ik(target_arm, source_arm, *,
                  arm: bool = True, leg: bool = True) -> IKRig:
    """Install IK on arm and/or leg chains.  Returns IKRig for cleanup."""
    empties: list[object] = []
    constraints: list[tuple[object, str]] = []
    tgt_bones = target_arm.pose.bones
    src_bones = source_arm.pose.bones

    if arm:
        for side, (chain_end, length, mpfb_target_bone) in ARM_IK_SETUP.items():
            if chain_end not in tgt_bones:
                continue
            src_bone = _find_source_bone_for_target(source_arm, mpfb_target_bone)
            if src_bone is None or src_bone not in src_bones:
                continue
            empty = _create_target_empty(
                f"_iktgt_arm_{side}", source_arm, src_bone, mode="head")
            empties.append(empty)
            cname = _add_ik(target_arm, chain_end, empty, length)
            if cname:
                constraints.append((tgt_bones[chain_end], cname))

    if leg:
        for side, (chain_end, length, mpfb_target_bone) in LEG_IK_SETUP.items():
            if chain_end not in tgt_bones:
                continue
            src_bone = _find_source_bone_for_target(source_arm, mpfb_target_bone)
            if src_bone is None or src_bone not in src_bones:
                continue
            empty = _create_target_empty(
                f"_iktgt_leg_{side}", source_arm, src_bone, mode="head")
            empties.append(empty)
            cname = _add_ik(target_arm, chain_end, empty, length)
            if cname:
                constraints.append((tgt_bones[chain_end], cname))

    return IKRig(target_arm=target_arm, source_arm=source_arm,
                  empties=empties, constraints=constraints)

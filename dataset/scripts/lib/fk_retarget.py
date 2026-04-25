"""BVH -> MPFB retargeter using Thomas Larsson's `retarget_bvh` algorithm.

Reference: https://bitbucket.org/Diffeomorphic/retarget_bvh (retarget.py CBoneAnim).

CORE IDEA (quoted from retarget_bvh header):
    T_b = A_b @ T'_b   =>   A_b = T_b @ T'_b^-1
    M_b = M'_b @ A_b
    L_b = B_b @ M_p^-1 @ A_b @ M'_b

Where:
    T_b        target bone world rotation, both rigs in T-pose (setup)
    T'_b       source bone world rotation, both rigs in T-pose (setup)
    A_b        per-bone axis-correction matrix, cached once
    M'_b       source bone world rotation at frame F (animated)
    M_b        target bone world rotation at frame F (derived)
    M_p        target bone's parent world rotation at frame F
    B_b        target rest relative inverted: trgBone.matrix_local^-1 @ parent.matrix_local
    L_b        target matrix_basis (what we write each frame)

Why this works where a naive "set pose = source world rotation" fails:
    * A-pose rest on target with T-pose source leaves huge rest-to-pose rotations
      at wrists/ankles, wrecking linear-blend skin weights that expect twist
      distribution across sub-bones.  The A_b matrix absorbs the per-bone rest
      mismatch algebraically WITHOUT requiring a rest-pose bake, preserving
      the A-pose skin binding.
    * Twist-distribution bones (upperarm02, lowerarm02, etc.) in the MPFB
      default rig are now driven via chain mapping — source joint maps to a
      LIST of target bones, and the rotation is spread across them using
      quaternion slerp so each twist bone carries a share of the roll.

Chain distribution:
    For source joint X mapped to target chain [t1, t2, ..., tN]:
      * Each tN receives matrix_basis = slerp(identity, per-bone-formula, 1/N).
      * Compounded through the chain, the end bone lands at the full target
        world rotation (= source world rotation + axis correction).
      * For single-target mappings (N=1), slerp factor is 1.0 (no-op).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import bpy         # type: ignore
import mathutils   # type: ignore

from .source_mappings import mapping_for_source, detect_source_from_bvh


# ------------------------------------------------------------------------
# BVH loading
# ------------------------------------------------------------------------

def load_bvh(path: str | Path, source: str | None = None) -> object:
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


# ------------------------------------------------------------------------
# Per-bone cached matrices
# ------------------------------------------------------------------------

@dataclass
class BoneEntry:
    """One entry per SOURCE joint.  Holds primary target + optional twist target.

    The twist bone is applied via swing-twist decomposition: decompose the
    full rotation (quaternion) around the bone's local +Y axis (along-bone),
    apply the SWING part to the primary bone, apply the TWIST part to the
    twist bone's matrix_basis.  This spreads forearm/upper-arm roll across
    two bones as the MPFB skin weights expect, avoiding candy-wrapper
    artifacts at the wrist / elbow / knee.
    """
    src_name: str
    tgt_name: str                 # primary target bone (e.g. upperarm01.L)
    twist_tgt_name: str | None    # twist sub-bone (e.g. upperarm02.L) or None
    is_root: bool

    # Cached at setup (retarget_bvh nomenclature):
    aMatrix: "mathutils.Matrix"   # 3x3: srcWorldRot_T^-1 @ trgWorldRot_T
    bMatrix: "mathutils.Matrix"   # 4x4: trgBone.matrix_local^-1 @ parent.matrix_local

    # Sorted-order depth (for parents-first processing)
    depth: int


@dataclass
class RetargetContext:
    bvh_arm: object
    target_arm: object
    entries: list[BoneEntry] = field(default_factory=list)
    frame_range: tuple[int, int] = (1, 1)
    _src_root_ref_world: "mathutils.Vector | None" = None
    _tgt_root_rest_world: "mathutils.Vector | None" = None

    # Auto-detected source-character forward-facing vector at first animated
    # frame.  Computed from LeftShoulder->RightShoulder cross with world up.
    # Consumers (camera rigs) use this instead of assuming -Y facing.
    src_facing: "mathutils.Vector | None" = None

    # Ground-snap offset: Z amount to shift root so the lowest foot across
    # the animation sits at Z=0.  Computed once per clip at setup.
    _floor_z_offset: float = 0.0

    # Cache of source bone rest world matrices so we don't recompute every frame.
    _src_rest_world: dict[str, "mathutils.Matrix"] = field(default_factory=dict)
    # Cache of target bone rest world matrices (native A-pose).
    _tgt_rest_world: dict[str, "mathutils.Matrix"] = field(default_factory=dict)

    # Lookup: target bone name -> target's parent bone name within the
    # target armature's actual parent chain (as used by retarget_bvh's bMatrix).
    _tgt_parent_name: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        src_kind = self.bvh_arm.get("_mocap_source", "cmu")
        bone_map, _ = mapping_for_source(src_kind)

        # Force target rotation to quaternion for well-defined matrix_basis writes.
        for pb in self.target_arm.pose.bones:
            pb.rotation_mode = "QUATERNION"

        self._build_entries(bone_map)

        # Pose fingers into a relaxed half-closed curl.  MPFB rest has fingers
        # splayed in A-pose; retargeting doesn't drive finger bones (mocap
        # BVHs don't animate them), so they'd stay rigidly splayed.  Setting
        # a slight curl per finger joint produces a natural hanging hand.
        self._pose_fingers_relaxed()

        if self.bvh_arm.animation_data and self.bvh_arm.animation_data.action:
            fr = self.bvh_arm.animation_data.action.frame_range
            self.frame_range = (int(fr[0]), int(fr[1]))

        # Sample source's first-animated-frame root head as the "natural
        # standing reference" for root-translation math.  BVH rests are
        # usually at origin (character buried); first frame is the standing
        # pose height and positional origin.
        root_entry = next((e for e in self.entries if e.is_root), None)
        if root_entry is not None:
            bpy.context.scene.frame_set(self.frame_range[0])
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            src_eval = self.bvh_arm.evaluated_get(dg)
            spb = src_eval.pose.bones.get(root_entry.src_name)
            if spb is not None:
                self._src_root_ref_world = (
                    self.bvh_arm.matrix_world @ spb.head).copy()

        # Auto-detect source character's forward-facing direction at frame 1.
        # Sources differ in convention (CMU faces -Y, 100STYLE/AIST++ face +Y).
        # Camera rigs use this to place the FRONT camera on the character's
        # anterior side regardless of source.
        self.src_facing = self._detect_source_facing()

        # Compute a floor offset so the lowest foot across the animation
        # rests at Z=0.  Sampled on a sparse subset of frames for speed.
        self._floor_z_offset = self._compute_floor_offset()

    def _detect_source_facing(self) -> "mathutils.Vector | None":
        """Return a unit Vector pointing in the source character's forward
        direction at the first animated frame.  Uses cross(shoulder_right_to_left,
        world_up) = body_forward.  Falls back to -Y if shoulders unavailable.
        """
        src_arm = self.bvh_arm
        # Find left/right shoulder bones via the bone map.
        src_kind = src_arm.get("_mocap_source", "cmu")
        bone_map, _ = mapping_for_source(src_kind)
        l_name = next((k for k, v in bone_map.items()
                       if (v[0] if isinstance(v, list) else v) == "clavicle.L"), None)
        r_name = next((k for k, v in bone_map.items()
                       if (v[0] if isinstance(v, list) else v) == "clavicle.R"), None)
        # Fall back to upper-arm bones if clavicle isn't mapped.
        if l_name is None:
            l_name = next((k for k, v in bone_map.items()
                           if (v[0] if isinstance(v, list) else v) == "upperarm01.L"), None)
            r_name = next((k for k, v in bone_map.items()
                           if (v[0] if isinstance(v, list) else v) == "upperarm01.R"), None)
        if not (l_name and r_name):
            return mathutils.Vector((0.0, -1.0, 0.0))

        bpy.context.scene.frame_set(self.frame_range[0])
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        src_eval = src_arm.evaluated_get(dg)
        lpb = src_eval.pose.bones.get(l_name)
        rpb = src_eval.pose.bones.get(r_name)
        if lpb is None or rpb is None:
            return mathutils.Vector((0.0, -1.0, 0.0))

        l_head = src_arm.matrix_world @ lpb.head
        r_head = src_arm.matrix_world @ rpb.head
        body_right = (r_head - l_head)
        if body_right.length < 1e-4:
            return mathutils.Vector((0.0, -1.0, 0.0))
        body_right.normalize()
        # Right-handed coordinate system: forward = up × right.
        up = mathutils.Vector((0.0, 0.0, 1.0))
        forward = up.cross(body_right)
        if forward.length < 1e-4:
            return mathutils.Vector((0.0, -1.0, 0.0))
        return forward.normalized()

    def _compute_floor_offset(self) -> float:
        """Sample feet Z across the animation; return -min_foot_Z (so when
        added to root Z, the lowest foot sits on Z=0).  Uses sparse sampling
        (every 10th frame) to stay fast.
        """
        arm = self.target_arm
        mw = arm.matrix_world

        # Temporarily apply poses to feet bones to find lowest Z.
        # We'll apply the full retarget to a sparse subset of frames.
        foot_names = [n for n in ("foot.L", "foot.R", "toe1-1.L", "toe1-1.R")
                      if n in arm.pose.bones]
        if not foot_names:
            return 0.0

        min_z = float("inf")
        fr_start, fr_end = self.frame_range
        step = max(1, (fr_end - fr_start) // 40)  # ~40 samples
        saved_basis = {pb.name: pb.matrix_basis.copy()
                       for pb in arm.pose.bones}
        saved_loc = {pb.name: pb.location.copy() for pb in arm.pose.bones}
        saved_rot = {pb.name: pb.rotation_quaternion.copy()
                     for pb in arm.pose.bones}
        try:
            for f in range(fr_start, fr_end + 1, step):
                self._apply_pose_no_floor(f)
                for fn in foot_names:
                    pb = arm.pose.bones[fn]
                    head_z = (mw @ pb.head).z
                    tail_z = (mw @ pb.tail).z
                    min_z = min(min_z, head_z, tail_z)
        finally:
            for pb in arm.pose.bones:
                if pb.name in saved_rot:
                    pb.rotation_quaternion = saved_rot[pb.name]
                if pb.name in saved_loc:
                    pb.location = saved_loc[pb.name]
            bpy.context.view_layer.update()

        if min_z == float("inf"):
            return 0.0
        return -min_z

    def _pose_fingers_relaxed(self) -> None:
        """Bake a slight curl into finger joints so hands don't look splayed
        when the mocap (which doesn't animate fingers) leaves them at A-pose
        rest.  MPFB finger convention: bones curl around the LOCAL X axis.

        finger1 = thumb, finger2-5 = index..pinky; joints 1/2/3 = prox/mid/dist.
        Bake as matrix_basis so it persists through the animation (fingers
        are unmapped, so the per-frame retarget won't overwrite their basis).
        """
        import math
        arm = self.target_arm
        # Approximate curl angles per joint (radians).  Thumb curls opposite
        # direction (local -X), so it's handled separately.
        finger_curl = {
            1: (-0.15, -0.10, -0.10),   # thumb: -X for flexion
            2: (0.25, 0.40, 0.30),      # index
            3: (0.30, 0.45, 0.35),      # middle
            4: (0.30, 0.45, 0.35),      # ring
            5: (0.30, 0.45, 0.35),      # pinky
        }
        for side in ("L", "R"):
            for finger, angles in finger_curl.items():
                for joint, angle in enumerate(angles, start=1):
                    bname = f"finger{finger}-{joint}.{side}"
                    pb = arm.pose.bones.get(bname)
                    if pb is None:
                        continue
                    pb.rotation_mode = "QUATERNION"
                    pb.rotation_quaternion = mathutils.Quaternion(
                        (1.0, 0.0, 0.0), angle)
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------
    # Setup: pose both rigs to T-pose, cache aMatrix/bMatrix per entry.
    # ------------------------------------------------------------------------

    def _build_entries(self, bone_map: dict[str, list[str]]) -> None:
        src_arm = self.bvh_arm
        tgt_arm = self.target_arm
        src_bones = src_arm.pose.bones
        tgt_bones = tgt_arm.pose.bones
        src_mw = src_arm.matrix_world
        tgt_mw = tgt_arm.matrix_world

        # Cache source & target rest world matrices (independent of any pose).
        for pb in src_bones:
            self._src_rest_world[pb.name] = (
                src_mw @ pb.bone.matrix_local).copy()
        for pb in tgt_bones:
            self._tgt_rest_world[pb.name] = (
                tgt_mw @ pb.bone.matrix_local).copy()
            parent_bone = pb.bone.parent
            if parent_bone is not None:
                self._tgt_parent_name[pb.name] = parent_bone.name

        # Pose BOTH rigs into canonical T-pose so we can sample aMatrix.
        # Source BVHs from CMU/100STYLE/AIST++ rest IN T-pose — no posing
        # needed.  MPFB rests in A-pose; we pose it per-bone to align each
        # mapped target bone's world Y-axis with the corresponding source
        # bone's world Y-axis.
        saved_basis = {pb.name: pb.matrix_basis.copy() for pb in tgt_bones}

        # Build the ordered list of (src, tgt) pairs, parents-first over
        # the TARGET chain.
        # Build source-joint entries.  Each entry has a PRIMARY target
        # (first in chain) and optionally a TWIST sub-bone (second in chain).
        # Twist sub-bones get the Y-axis-twist part of the rotation; primary
        # gets the swing.  Skin weights on MPFB's twist bones (upperarm02 etc.)
        # then distribute the forearm/upper-arm roll smoothly — without this,
        # a 90° wrist pronation produces a "candy-wrapper" pinch at the wrist.
        ordered: list[tuple[str, str, str | None, int]] = []  # (src, primary, twist, depth)
        for src_name, tgt_chain in bone_map.items():
            if src_name not in src_bones:
                continue
            if not isinstance(tgt_chain, list):
                tgt_chain = [tgt_chain]
            primary = tgt_chain[0]
            twist = tgt_chain[1] if len(tgt_chain) > 1 else None
            if primary not in tgt_bones:
                continue
            if twist is not None and twist not in tgt_bones:
                twist = None
            d = self._bone_depth(tgt_bones[primary].bone)
            ordered.append((src_name, primary, twist, d))
        ordered.sort(key=lambda x: x[3])

        # Pose target to match source's per-bone world Y-axis (T-pose alignment).
        # Process parents-first, updating depsgraph per depth level.
        #
        # SKIP the ROOT bone — some source rigs (SMPL/AIST++) have a pelvis
        # bone with an unusual rest orientation (e.g. pointing forward-down
        # at 51° off vertical).  Forcing MPFB's root to match that rotates
        # the whole character into an anterior-pelvic-tilt posture that
        # propagates down the spine chain.  Keep target root at its A-pose
        # rest; the aMatrix will absorb the source's rest orientation.
        current_depth = -1
        for src_name, tgt_name, twist_name, depth in ordered:
            if tgt_name == "root":
                continue
            if depth != current_depth:
                if current_depth >= 0:
                    bpy.context.view_layer.update()
                current_depth = depth

            src_pb = src_bones[src_name]
            tgt_pb = tgt_bones[tgt_name]
            # We want the target bone's world rotation to match the source
            # bone's REST world rotation (source is already T-pose at rest).
            src_rest_rot = self._src_rest_world[src_name].to_3x3()
            # Preserve the current (pose-chain-computed) head translation.
            cur_world = tgt_mw @ tgt_pb.matrix
            new_world = src_rest_rot.to_4x4()
            new_world.translation = cur_world.translation
            tgt_pb.matrix = tgt_mw.inverted() @ new_world
        bpy.context.view_layer.update()

        # Now both rigs are in canonical T-pose — sample aMatrix per bone.
        for src_name, tgt_name, twist_name, depth in ordered:
            src_pb = src_bones[src_name]
            tgt_pb = tgt_bones[tgt_name]

            src_world_T = self._src_rest_world[src_name].to_3x3()
            tgt_world_T = (tgt_mw @ tgt_pb.matrix).to_3x3()
            aMatrix = src_world_T.inverted() @ tgt_world_T

            tgt_bone_mat_local = tgt_pb.bone.matrix_local
            parent = tgt_pb.bone.parent
            if parent is not None:
                bMatrix = tgt_bone_mat_local.inverted() @ parent.matrix_local
            else:
                bMatrix = tgt_bone_mat_local.inverted()

            is_root = (tgt_name == "root")
            self.entries.append(BoneEntry(
                src_name=src_name,
                tgt_name=tgt_name,
                twist_tgt_name=twist_name,
                is_root=is_root,
                aMatrix=aMatrix.to_4x4(),
                bMatrix=bMatrix,
                depth=depth,
            ))

            if is_root:
                self._tgt_root_rest_world = (
                    tgt_mw @ tgt_pb.bone.matrix_local).translation.copy()

        # Restore target back to A-pose (NO bake — we keep native rest).
        for pb in tgt_bones:
            if pb.name in saved_basis:
                pb.matrix_basis = saved_basis[pb.name]
        bpy.context.view_layer.update()

        self.entries.sort(key=lambda e: e.depth)

    @staticmethod
    def _bone_depth(bone) -> int:
        d = 0
        b = bone.parent
        while b is not None:
            d += 1
            b = b.parent
        return d

    # ------------------------------------------------------------------------
    # Per-frame retarget
    # ------------------------------------------------------------------------

    def apply_pose(self, frame: int) -> None:
        """Apply retarget for `frame` WITH floor snap applied to root Z."""
        self._apply_pose_no_floor(frame)
        # Floor snap: shift the whole character up so the lowest foot
        # (sampled across the whole clip at setup time) rests on Z=0.
        if abs(self._floor_z_offset) > 1e-6:
            root_pb = self.target_arm.pose.bones.get("root")
            if root_pb is not None:
                root_pb.location.z += self._floor_z_offset
                bpy.context.view_layer.update()

    def _apply_pose_no_floor(self, frame: int) -> None:
        scene = bpy.context.scene
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()

        dg = bpy.context.evaluated_depsgraph_get()
        src_eval = self.bvh_arm.evaluated_get(dg)
        src_mw = self.bvh_arm.matrix_world
        tgt_mw = self.target_arm.matrix_world
        tgt_mw_inv = tgt_mw.inverted()

        current_depth = -1

        for e in self.entries:
            if e.depth != current_depth:
                if current_depth >= 0:
                    bpy.context.view_layer.update()
                current_depth = e.depth

            src_pb = src_eval.pose.bones.get(e.src_name)
            tgt_pb = self.target_arm.pose.bones.get(e.tgt_name)
            if src_pb is None or tgt_pb is None:
                continue

            # retarget_bvh: M_b = M'_b @ A_b (world space).
            srcMatrix_world = (src_mw @ src_pb.matrix).to_3x3().to_4x4()
            trgMatrix_world = srcMatrix_world @ e.aMatrix

            trgMatrix_armature = tgt_mw_inv @ trgMatrix_world

            parent_name = self._tgt_parent_name.get(e.tgt_name)
            if parent_name is not None:
                parent_tgt = self.target_arm.pose.bones.get(parent_name)
                parent_world_arm = parent_tgt.matrix if parent_tgt else (
                    mathutils.Matrix.Identity(4))
            else:
                parent_world_arm = mathutils.Matrix.Identity(4)

            mat1 = parent_world_arm.inverted() @ trgMatrix_armature
            mat2 = e.bMatrix @ mat1

            q_full = mat2.to_quaternion()

            # Apply FULL rotation to primary bone (swing + twist).  Sub-bones
            # (upperarm02, lowerarm02, upperleg02, lowerleg02) stay at identity
            # and follow via the kinematic chain.  Mesh near the joint is
            # weighted primarily to the primary bone and needs the full
            # rotation including twist — splitting twist to a sub-bone left
            # visible candy-wrap at the hip / shoulder where the mesh attaches.
            #
            # Measured evidence: swing-twist split produced ~150° twist delta
            # on hip/shoulder primary bones (measure_joints.py baseline),
            # which visually manifested as "legs near hips twisted".
            tgt_pb.rotation_quaternion = q_full
            if e.twist_tgt_name is not None:
                twist_pb = self.target_arm.pose.bones.get(e.twist_tgt_name)
                if twist_pb is not None:
                    twist_pb.rotation_quaternion = mathutils.Quaternion()

            # Root translation
            if e.is_root:
                src_root_world = src_mw @ src_pb.head
                if (self._src_root_ref_world is not None and
                        self._tgt_root_rest_world is not None):
                    delta = src_root_world - self._src_root_ref_world
                    tgt_target_world = self._tgt_root_rest_world + delta
                    tgt_target_arm = tgt_mw_inv @ tgt_target_world
                    rest_head_arm = tgt_pb.bone.matrix_local.translation
                    tgt_pb.location = tgt_target_arm - rest_head_arm

        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------------

    def cleanup(self) -> None:
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
        except Exception as exc:
            print(f"[fk_retarget] cleanup: {exc}")

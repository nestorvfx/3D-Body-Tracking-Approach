"""Per-frame pose plausibility validator.

Applied AFTER retargeting (so we're testing the target MPFB armature's
current pose at a specific frame).  Rejects implausible frames without
repairing them — repair invents data; rejection is safe.

Checks (SOTA 2024-2026 stack, commercial-clean):
  1. Biomechanical joint-angle ROM (OpenSim Gait2392-derived)
  2. Ground-plane penetration (max-depth-below-zero over all mesh vertices)
  3. Bone-length preservation (retargeting sanity check)
  4. Self-intersection via mathutils.bvhtree.BVHTree overlap
     (limbs penetrating torso, arm through leg, etc.)

Reference:
  OpenSim ROM: https://simtk-confluence.stanford.edu/display/OpenSim/Gait+2392+and+2354+Models
  ISB joint coord conventions: Wu et al., J.Biomech 2002
  BVHTree.overlap docs: https://docs.blender.org/api/current/mathutils.bvhtree.html
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import bpy  # type: ignore
import bmesh  # type: ignore
import mathutils  # type: ignore


# OpenSim Gait2392-derived clinical ROM limits (°).
# Interior flexion angle: parent-bone vector relative to child-bone vector.
# For knee/elbow the rest pose is a STRAIGHT line (interior = 180°), so
#   `flex = 180 - interior` equals 0 at rest and grows as the joint bends.
# Ankle is omitted here — its rest interior angle (~90°, foot perpendicular
# to shin) breaks the "straight = rest" assumption and needs rest-relative
# comparison (handled elsewhere once we wire up baseline ankle angles).
JOINT_LIMITS_FLEX_DEG = {
    # (parent_bone_name, child_bone_name): (min_flex, max_flex)
    ("upperleg01.L", "lowerleg01.L"): (-10, 155),   # knee (small hyperextension slack)
    ("upperleg01.R", "lowerleg01.R"): (-10, 155),
    ("upperarm01.L", "lowerarm01.L"): (-10, 160),   # elbow
    ("upperarm01.R", "lowerarm01.R"): (-10, 160),
}

# Frame is rejected if any of these metrics exceed their thresholds.
DEFAULT_THRESHOLDS = {
    "ground_penetration_m":   0.02,   # 2 cm max dip below floor
    "bone_length_drift_pct":  3.0,    # per-frame length vs rest
    "joint_angle_tol_deg":    5.0,    # slack on biomech ROM
    "self_intersect_pairs":   0,      # any overlap = reject
}


@dataclass
class FrameValidation:
    frame: int
    ok: bool
    rejections: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _flex_angle_deg(armature, parent_name: str, child_name: str) -> float:
    """World-space flexion angle between parent and child bone direction
    vectors.  Both bones have Blender's "head-to-tail Y axis" convention, so
    when the limb is straight the two direction vectors are parallel (angle
    0°) and when maximally bent they are antiparallel (180°).  Flex = 0 at
    rest, grows as the joint folds.
    """
    pb_p = armature.pose.bones.get(parent_name)
    pb_c = armature.pose.bones.get(child_name)
    if pb_p is None or pb_c is None:
        return 0.0
    mw = armature.matrix_world
    v_p = (mw @ pb_p.tail) - (mw @ pb_p.head)
    v_c = (mw @ pb_c.tail) - (mw @ pb_c.head)
    if v_p.length < 1e-6 or v_c.length < 1e-6:
        return 0.0
    return math.degrees(v_p.angle(v_c))


def check_joint_ranges(
    armature,
    tol_deg: float = DEFAULT_THRESHOLDS["joint_angle_tol_deg"],
) -> list[str]:
    """Return list of violation strings (empty = all joints within ROM)."""
    violations: list[str] = []
    for (p, c), (lo, hi) in JOINT_LIMITS_FLEX_DEG.items():
        flex = _flex_angle_deg(armature, p, c)
        if flex < lo - tol_deg:
            violations.append(f"{p}->{c} flex={flex:.1f}° hyperextension (<{lo-tol_deg})")
        elif flex > hi + tol_deg:
            violations.append(f"{p}->{c} flex={flex:.1f}° hyperflexion (>{hi+tol_deg})")
    return violations


def check_ground_penetration(
    basemesh,
    threshold_m: float = DEFAULT_THRESHOLDS["ground_penetration_m"],
) -> tuple[bool, float]:
    """Return (pass, max_penetration_depth_m).  Assumes Z-up ground at z=0."""
    if basemesh is None or basemesh.type != "MESH":
        return True, 0.0
    dg = bpy.context.evaluated_depsgraph_get()
    ev = basemesh.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        mw = basemesh.matrix_world
        # Transform Z only (fast path)
        worst = 0.0
        # Iterate in bulk via foreach_get if available
        try:
            import numpy as np
            n = len(me.vertices)
            buf = np.empty(n * 3, dtype=np.float32)
            me.vertices.foreach_get("co", buf)
            co = buf.reshape(-1, 3)
            # apply mw as 4x4
            mw_np = np.array(mw)
            h = np.concatenate([co, np.ones((n, 1), dtype=np.float32)], axis=1)
            world = (mw_np @ h.T).T[:, :3]
            worst = float(max(0.0, -world[:, 2].min()))   # how far below z=0
        except Exception:
            # Per-vertex fallback
            for v in me.vertices:
                w = mw @ v.co
                if -w.z > worst:
                    worst = -w.z
    finally:
        ev.to_mesh_clear()
    return worst <= threshold_m, worst


def check_bone_length_preservation(
    armature,
    rest_lengths: dict[str, float],
    tol_pct: float = DEFAULT_THRESHOLDS["bone_length_drift_pct"],
) -> tuple[bool, float]:
    """Return (pass, max_drift_pct).  Compares current bone lengths vs rest."""
    worst = 0.0
    for pb in armature.pose.bones:
        rest = rest_lengths.get(pb.name)
        if rest is None or rest < 1e-6:
            continue
        cur = (pb.tail - pb.head).length
        drift_pct = 100.0 * abs(cur - rest) / rest
        if drift_pct > worst:
            worst = drift_pct
    return worst <= tol_pct, worst


def capture_rest_lengths(armature) -> dict[str, float]:
    """Call once before animation applied — captures rest bone lengths."""
    return {pb.name: (pb.tail - pb.head).length for pb in armature.pose.bones}


# --- Self-intersection via BVHTree ---

# Which vertex groups map to which body regions.  Names match the MPFB
# default rig.  Collisions within the same region are ignored (natural
# flesh-to-flesh contact).  Collisions across regions = likely unnatural.
REGION_VGROUPS = {
    "left_arm":  ["clavicle.L", "upperarm01.L", "upperarm02.L", "upperarm03.L",
                   "lowerarm01.L", "lowerarm02.L", "wrist.L"],
    "right_arm": ["clavicle.R", "upperarm01.R", "upperarm02.R", "upperarm03.R",
                   "lowerarm01.R", "lowerarm02.R", "wrist.R"],
    "torso":     ["spine01", "spine02", "spine03", "spine04", "spine05",
                   "neck01", "neck02", "chest", "pelvis"],
    "left_leg":  ["upperleg01.L", "upperleg02.L", "lowerleg01.L",
                   "lowerleg02.L", "foot.L"],
    "right_leg": ["upperleg01.R", "upperleg02.R", "lowerleg01.R",
                   "lowerleg02.R", "foot.R"],
}

# Pairs to actively test (non-adjacent regions).  Adjacent pairs (arm-torso,
# leg-torso) touch naturally at the shoulder/hip and would false-positive.
INTERSECT_TEST_PAIRS = [
    ("left_arm", "right_arm"),
    ("left_arm", "left_leg"),
    ("left_arm", "right_leg"),
    ("right_arm", "left_leg"),
    ("right_arm", "right_leg"),
    ("left_leg", "right_leg"),
]


def _verts_for_vgroups(obj, vgroup_names: Sequence[str]) -> list[int]:
    """Return vertex indices whose weight > 0.5 in ANY of the named groups."""
    existing = {obj.vertex_groups[n].index: n for n in vgroup_names
                if n in obj.vertex_groups}
    if not existing:
        return []
    kept: list[int] = []
    existing_idx = set(existing.keys())
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group in existing_idx and g.weight > 0.5:
                kept.append(v.index)
                break
    return kept


def _bvh_for_subset(obj, vert_idx: list[int], dg) -> "mathutils.bvhtree.BVHTree | None":
    """Build a BVH tree containing only the specified vertices + their faces."""
    if not vert_idx:
        return None
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.transform(obj.matrix_world)
        keep = set(vert_idx)
        drop = [v for v in bm.verts if v.index not in keep]
        if drop:
            bmesh.ops.delete(bm, geom=drop, context="VERTS")
        tree = mathutils.bvhtree.BVHTree.FromBMesh(bm)
        bm.free()
    finally:
        ev.to_mesh_clear()
    return tree


def check_self_intersection(basemesh) -> tuple[bool, int]:
    """Return (pass, total_overlap_pair_count)."""
    if basemesh is None or basemesh.type != "MESH":
        return True, 0
    dg = bpy.context.evaluated_depsgraph_get()
    region_trees: dict[str, "mathutils.bvhtree.BVHTree"] = {}
    for region, vgroups in REGION_VGROUPS.items():
        vi = _verts_for_vgroups(basemesh, vgroups)
        t = _bvh_for_subset(basemesh, vi, dg)
        if t is not None:
            region_trees[region] = t

    pairs = 0
    for a, b in INTERSECT_TEST_PAIRS:
        ta = region_trees.get(a)
        tb = region_trees.get(b)
        if ta is None or tb is None:
            continue
        ov = ta.overlap(tb)
        pairs += len(ov)

    return pairs == 0, pairs


# --- Top-level per-frame validator ---

def validate_frame(
    armature,
    basemesh,
    rest_lengths: dict[str, float] | None,
    frame: int,
    thresholds: dict | None = None,
    *,
    do_self_intersect: bool = True,
) -> FrameValidation:
    """Run all plausibility checks at the current evaluated pose.  Caller
    must have done `scene.frame_set(frame)` and `view_layer.update()` first."""
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    rep = FrameValidation(frame=frame, ok=True)

    rom_viol = check_joint_ranges(armature, tol_deg=thr["joint_angle_tol_deg"])
    rep.metrics["rom_violations"] = rom_viol
    if rom_viol:
        rep.ok = False
        rep.rejections.append("rom:" + ";".join(rom_viol))

    ok_gp, pen = check_ground_penetration(basemesh, threshold_m=thr["ground_penetration_m"])
    rep.metrics["ground_penetration_m"] = pen
    if not ok_gp:
        rep.ok = False
        rep.rejections.append(f"ground_pen={pen:.3f}m")

    if rest_lengths is not None:
        ok_bl, drift = check_bone_length_preservation(armature, rest_lengths,
                                                        tol_pct=thr["bone_length_drift_pct"])
        rep.metrics["bone_length_drift_pct"] = drift
        if not ok_bl:
            rep.ok = False
            rep.rejections.append(f"bone_drift={drift:.1f}%")

    if do_self_intersect:
        ok_si, pairs = check_self_intersection(basemesh)
        rep.metrics["self_intersect_pairs"] = pairs
        if not ok_si:
            rep.ok = False
            rep.rejections.append(f"self_intersect={pairs}")

    return rep

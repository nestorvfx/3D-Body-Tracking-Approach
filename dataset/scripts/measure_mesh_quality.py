"""Measure mesh-quality at joint regions: vertex-to-bone distance collapse.

For each joint, find all body-mesh vertices within a radius of the joint
in REST pose.  For each such vertex, compute:
  d_rest = distance from vertex to the nearest bone segment (head-tail) at rest
  d_pose = distance at the target frame (worst-bend)
  collapse_mm = max(0, d_rest - d_pose) * 1000

Large positive collapse_mm = vertex pinched toward the bone (bad).
Negative would mean the vertex bulged away (usually fine).

Output CSV: frame, joint, vertex_idx, d_rest_mm, d_pose_mm, collapse_mm,
            vertex_world_pose_x/y/z
Plus summary per-joint: max collapse_mm, count > 2mm, count > 5mm.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import bpy
import mathutils

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ensure_mpfb():
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except Exception:
        pass
    import importlib, sys as _sys
    pkg = importlib.import_module("bl_ext.user_default.mpfb")
    _sys.modules["mpfb"] = pkg
    for s in ("humanservice", "targetservice", "assetservice", "locationservice"):
        _sys.modules[f"mpfb.services.{s}"] = importlib.import_module(
            f"bl_ext.user_default.mpfb.services.{s}")


ensure_mpfb()

from lib.mpfb_build import build_character
from lib.fk_retarget import RetargetContext, load_bvh
from lib.render_setup import clear_scene
from lib.source_mappings import detect_source_from_bvh


JOINTS = {
    "hip_L":      ("upperleg01.L",  "lowerleg01.L", 0.15),
    "hip_R":      ("upperleg01.R",  "lowerleg01.R", 0.15),
    "knee_L":     ("lowerleg01.L",  "foot.L",       0.12),
    "knee_R":     ("lowerleg01.R",  "foot.R",       0.12),
    "shoulder_L": ("upperarm01.L",  "lowerarm01.L", 0.15),
    "shoulder_R": ("upperarm01.R",  "lowerarm01.R", 0.15),
    "elbow_L":    ("lowerarm01.L",  "wrist.L",      0.10),
    "elbow_R":    ("lowerarm01.R",  "wrist.R",      0.10),
}


def seg_distance(p: mathutils.Vector, a: mathutils.Vector, b: mathutils.Vector) -> float:
    ab = b - a
    if ab.length_squared < 1e-10:
        return (p - a).length
    t = (p - a).dot(ab) / ab.length_squared
    t = max(0.0, min(1.0, t))
    closest = a + ab * t
    return (p - closest).length


def min_dist_to_bones(p: mathutils.Vector, bones: list[tuple[mathutils.Vector, mathutils.Vector]]) -> float:
    return min(seg_distance(p, a, b) for a, b in bones)


def gather_body_vertices(arm_obj, name_prefix: str):
    """Return (obj, world_verts_rest) for the body mesh.  We work on the
    mesh at REST pose to find candidate vertices near each joint."""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.name.startswith(name_prefix):
            continue
        if "body" not in obj.name.lower():
            continue
        # Rest pose vertex positions (in object space + world transform).
        mw = obj.matrix_world
        rest_verts = [mw @ v.co.copy() for v in obj.data.vertices]
        return obj, rest_verts
    return None, []


def evaluated_world_verts(obj):
    """Return the CURRENT pose world vertex positions (from depsgraph)."""
    dg = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    mesh = eval_obj.to_mesh()
    mw = obj.matrix_world
    verts = [mw @ v.co.copy() for v in mesh.vertices]
    eval_obj.to_mesh_clear()
    return verts


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bvh_path = args[0]
    out_csv = Path(args[1])
    seed = int(args[2]) if len(args) > 2 else 42

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    bm, arm = build_character(seed, with_assets=False)
    bvh = load_bvh(bvh_path, source=None)
    ctx = RetargetContext(bvh, arm)

    body_obj, rest_verts = gather_body_vertices(arm, f"subject_{seed:04d}")
    if body_obj is None:
        print("no body mesh found")
        sys.exit(1)
    # Allow forcing LBS for comparison: pass "lbs" as arg[3].
    force_mode = args[3].lower() if len(args) > 3 else None
    for m in body_obj.modifiers:
        if m.type == "ARMATURE":
            if force_mode == "lbs":
                m.use_deform_preserve_volume = False
            elif force_mode == "dqs":
                m.use_deform_preserve_volume = True
            print(f"  armature mode: preserve_volume={m.use_deform_preserve_volume}")
    print(f"[mesh-qual] body mesh: {body_obj.name}, {len(rest_verts)} verts")

    # Disable non-armature modifiers (subdiv, subsurf) for measurement so
    # evaluated mesh matches base vertex count.
    disabled_mods = []
    for m in body_obj.modifiers:
        if m.type != "ARMATURE":
            if m.show_viewport or m.show_render:
                disabled_mods.append((m, m.show_viewport, m.show_render))
                m.show_viewport = False
                m.show_render = False
    bpy.context.view_layer.update()

    # Gather joint center + bone segments at REST
    mw_arm = arm.matrix_world
    joint_rest_center = {}
    joint_bones = {}
    for joint, (parent_bone, child_bone, _r) in JOINTS.items():
        p = arm.pose.bones.get(parent_bone)
        c = arm.pose.bones.get(child_bone)
        if p is None:
            continue
        h = mw_arm @ p.bone.matrix_local @ mathutils.Vector((0.0, 0.0, 0.0))
        # simpler: use rest head position
        h_rest = (mw_arm @ p.bone.matrix_local).translation
        joint_rest_center[joint] = h_rest
        # Bone segments near joint: parent bone (head->tail at rest)
        segs = []
        pb = arm.pose.bones.get(parent_bone)
        if pb is not None:
            pa = (mw_arm @ pb.bone.matrix_local).translation
            pb_b = pb.bone
            # tail world in armature coords
            pt = (mw_arm @ pb_b.matrix_local).translation + \
                 (mw_arm.to_3x3() @ (pb_b.tail_local - pb_b.head_local))
            segs.append((pa, pt))
        if child_bone:
            cb = arm.pose.bones.get(child_bone)
            if cb is not None:
                ca = (mw_arm @ cb.bone.matrix_local).translation
                cb_b = cb.bone
                ct = (mw_arm @ cb_b.matrix_local).translation + \
                     (mw_arm.to_3x3() @ (cb_b.tail_local - cb_b.head_local))
                segs.append((ca, ct))
        joint_bones[joint] = segs

    # For each joint, find vertices within radius in REST
    joint_verts = {}
    joint_rest_d = {}
    for joint, (_p, _c, radius) in JOINTS.items():
        if joint not in joint_rest_center:
            continue
        center = joint_rest_center[joint]
        segs = joint_bones[joint]
        idx_list = []
        d_list = []
        for i, v in enumerate(rest_verts):
            if (v - center).length <= radius:
                d_rest = min_dist_to_bones(v, segs)
                idx_list.append(i)
                d_list.append(d_rest)
        joint_verts[joint] = idx_list
        joint_rest_d[joint] = d_list
        print(f"  joint {joint}: {len(idx_list)} vertices within {radius}m of rest center")

    # Find worst-bend frame per joint (reuse logic from render_joint_closeups).
    start, end = ctx.frame_range
    step = max(1, (end - start) // 60)
    worst = {j: (start, -1.0) for j in JOINTS}
    for f in range(start, end + 1, step):
        ctx.apply_pose(f)
        bpy.context.view_layer.update()
        for joint, (parent_bone, child_bone, _r) in JOINTS.items():
            if child_bone is None:
                continue
            p = arm.pose.bones.get(parent_bone)
            c = arm.pose.bones.get(child_bone)
            if p is None or c is None:
                continue
            pa = (mw_arm @ p.tail) - (mw_arm @ p.head)
            ca = (mw_arm @ c.tail) - (mw_arm @ c.head)
            if pa.length < 1e-5 or ca.length < 1e-5:
                continue
            pa.normalize(); ca.normalize()
            d = max(-1.0, min(1.0, pa.dot(ca)))
            ang = math.degrees(math.acos(d))
            if ang > worst[joint][1]:
                worst[joint] = (f, ang)

    # Measure collapse at each joint's worst-bend frame
    rows = []
    for joint, (parent_bone, child_bone, _r) in JOINTS.items():
        if joint not in joint_verts:
            continue
        frame, bend_angle = worst[joint]
        if bend_angle < 0:
            continue
        ctx.apply_pose(frame)
        bpy.context.view_layer.update()
        # Current bone segments (pose)
        segs_pose = []
        for bn in (parent_bone, child_bone):
            pb = arm.pose.bones.get(bn) if bn else None
            if pb is None:
                continue
            segs_pose.append((mw_arm @ pb.head, mw_arm @ pb.tail))
        pose_verts = evaluated_world_verts(body_obj)
        idx_list = joint_verts[joint]
        rest_d_list = joint_rest_d[joint]
        for vi, d_rest in zip(idx_list, rest_d_list):
            vpose = pose_verts[vi]
            d_pose = min_dist_to_bones(vpose, segs_pose)
            collapse_mm = (d_rest - d_pose) * 1000.0
            rows.append({
                "frame": frame,
                "joint": joint,
                "vertex_idx": vi,
                "d_rest_mm": round(d_rest * 1000.0, 2),
                "d_pose_mm": round(d_pose * 1000.0, 2),
                "collapse_mm": round(collapse_mm, 2),
                "vx": round(vpose.x, 3),
                "vy": round(vpose.y, 3),
                "vz": round(vpose.z, 3),
            })

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "joint", "vertex_idx",
                                           "d_rest_mm", "d_pose_mm",
                                           "collapse_mm", "vx", "vy", "vz"])
        w.writeheader()
        w.writerows(rows)

    # Summary
    by_joint: dict[str, list[float]] = {}
    for r in rows:
        by_joint.setdefault(r["joint"], []).append(r["collapse_mm"])
    print("\n=== Mesh collapse summary ===")
    print(f"{'joint':<12s}  {'frame':>5s}  {'mean_mm':>8s}  {'max_mm':>7s}  "
          f"{'>2mm':>6s}  {'>5mm':>6s}  {'n':>5s}")
    any_fail = False
    for joint, deltas in sorted(by_joint.items()):
        mean = sum(deltas) / len(deltas)
        mx = max(deltas)
        over2 = sum(1 for d in deltas if d > 2.0)
        over5 = sum(1 for d in deltas if d > 5.0)
        red = mx > 2.0
        any_fail = any_fail or red
        frame = next((r["frame"] for r in rows if r["joint"] == joint), 0)
        print(f"{joint:<12s}  {frame:>5d}  {mean:>8.2f}  {mx:>7.2f}  "
              f"{over2:>6d}  {over5:>6d}  {len(deltas):>5d}  "
              f"{'FAIL' if red else 'ok'}")
    print(f"\nCSV: {out_csv}")
    print(f"OVERALL: {'FAIL (max collapse > 2mm)' if any_fail else 'PASS'}")


if __name__ == "__main__":
    main()

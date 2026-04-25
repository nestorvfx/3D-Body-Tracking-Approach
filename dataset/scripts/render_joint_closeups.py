"""Render close-up crops of each joint (hip/knee/shoulder/elbow/wrist)
at the frame where that joint is at its WORST (most extreme bend) so
mesh artifacts are maximally visible.

For each source/joint combination:
  1. Scan all animation frames to find the frame where the joint has
     maximum bend.
  2. Render a 384x384 crop tightly framed on that joint from 2 angles
     (front + side).
  3. Save to out_dir/<source>_<joint>_<angle>.png

Usage:
  blender --background --python render_joint_closeups.py -- \
      <bvh_path> <hdri_path> <out_dir> [seed=42]
"""
from __future__ import annotations

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
from lib.render_setup import clear_scene, configure_render, set_world_hdri
from lib.source_mappings import detect_source_from_bvh


# For each joint: (target-bone for joint head, pair bone for bend computation,
# approx framing radius in meters)
JOINTS = {
    "hip_L":      ("upperleg01.L", "lowerleg01.L", 0.22),
    "hip_R":      ("upperleg01.R", "lowerleg01.R", 0.22),
    "knee_L":     ("lowerleg01.L", "foot.L",       0.20),
    "knee_R":     ("lowerleg01.R", "foot.R",       0.20),
    "shoulder_L": ("upperarm01.L", "lowerarm01.L", 0.22),
    "shoulder_R": ("upperarm01.R", "lowerarm01.R", 0.22),
    "elbow_L":    ("lowerarm01.L", "wrist.L",      0.18),
    "elbow_R":    ("lowerarm01.R", "wrist.R",      0.18),
    "wrist_L":    ("wrist.L",      None,           0.12),
    "wrist_R":    ("wrist.R",      None,           0.12),
}


def bend_at(arm, parent_name, child_name):
    """Bend angle in degrees between parent bone along and child bone along
    (head-to-tail world vectors).  For wrist (no child), return 0.
    """
    if child_name is None:
        return 0.0
    mw = arm.matrix_world
    pb_p = arm.pose.bones.get(parent_name)
    pb_c = arm.pose.bones.get(child_name)
    if pb_p is None or pb_c is None:
        return 0.0
    pa = (mw @ pb_p.tail) - (mw @ pb_p.head)
    ca = (mw @ pb_c.tail) - (mw @ pb_c.head)
    if pa.length < 1e-5 or ca.length < 1e-5:
        return 0.0
    pa.normalize(); ca.normalize()
    d = max(-1.0, min(1.0, pa.dot(ca)))
    return math.degrees(math.acos(d))


def find_worst_frames(ctx, arm, frame_range):
    """For each joint, find the frame where bend angle is MAXIMAL (most
    extreme deformation).  For twist-prone joints (shoulder/hip), also
    try max twist frame.
    """
    start, end = frame_range
    step = max(1, (end - start) // 80)   # ~80 samples across the clip
    best = {j: (start, 0.0) for j in JOINTS}
    for f in range(start, end + 1, step):
        ctx.apply_pose(f)
        bpy.context.view_layer.update()
        for j, (parent_name, child_name, _radius) in JOINTS.items():
            ang = bend_at(arm, parent_name, child_name)
            # Maximum bend = min dot, maximum acos result (0=folded, 180=straight).
            # We want most "folded" => smallest angle (approaching 0 degrees
            # actually means fully folded... but my bend_at returns:
            # dot(-parent.along, child.along) is 1 for folded, so acos returns 0.
            # Wait, parent.along points tail-ward, child.along tail-ward. If
            # folded, parent.along and child.along are nearly opposite
            # (child going back the way parent came), dot ~= -1, acos ~= 180°.
            # If extended, parent.along and child.along same direction,
            # dot ~= 1, acos ~= 0°.
            # So extreme bend = angle approaches 180 (if limb folds back).
            # For a bent knee, the angle from my formula is ~90-140.
            # We want MAX bend = MAX angle.
            if ang > best[j][1]:
                best[j] = (f, ang)
    return best


def target_facing(arm):
    mw = arm.matrix_world
    up = mathutils.Vector((0, 0, 1))
    for (ln, rn) in (("upperleg01.L", "upperleg01.R"),
                     ("upperarm01.L", "upperarm01.R")):
        l = arm.pose.bones.get(ln); r = arm.pose.bones.get(rn)
        if l is None or r is None:
            continue
        lh = mw @ l.head; rh = mw @ r.head
        br = mathutils.Vector((rh.x - lh.x, rh.y - lh.y, 0.0))
        if br.length < 1e-4:
            continue
        br.normalize()
        fwd = up.cross(br)
        if fwd.length > 1e-4:
            return fwd.normalized()
    return mathutils.Vector((0, -1, 0))


def render_closeup(cam_obj, joint_pos, direction, radius, out_path):
    """Render a 384x384 crop framed tight on joint_pos, viewed along `direction`."""
    cam_obj.location = (joint_pos.x + direction.x * (radius * 4.0),
                        joint_pos.y + direction.y * (radius * 4.0),
                        joint_pos.z + direction.z * (radius * 4.0))
    fwd = (joint_pos - mathutils.Vector(cam_obj.location)).normalized()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
    cam_obj.data.lens = 85        # tight crop
    bpy.context.scene.render.resolution_x = 384
    bpy.context.scene.render.resolution_y = 384
    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bvh_path = args[0]
    hdri_path = args[1]
    out_dir = Path(args[2]); out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args[3]) if len(args) > 3 else 42

    clear_scene()
    bm, arm = build_character(seed, with_assets=True)
    bvh = load_bvh(bvh_path, source=None)
    ctx = RetargetContext(bvh, arm)

    src_kind = detect_source_from_bvh(bvh_path)
    source_tag = {"cmu": "cmu", "aistpp": "aistpp", "100style": "100style"}.get(src_kind, src_kind)

    cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 85
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_world_hdri(hdri_path, strength=1.3)
    configure_render(resolution=(384, 384), engine="BLENDER_EEVEE",
                     samples=64, output_path=str(out_dir / "_.png"))

    worst = find_worst_frames(ctx, arm,
                               (ctx.frame_range[0], ctx.frame_range[1]))
    print(f"worst-bend frames: {[(j, f, round(a,1)) for j, (f, a) in worst.items()]}")

    for joint, (bone_name, _child, radius) in JOINTS.items():
        frame, _bend = worst[joint]
        ctx.apply_pose(frame)
        bpy.context.view_layer.update()

        pb = arm.pose.bones.get(bone_name)
        if pb is None:
            continue
        joint_pos = arm.matrix_world @ pb.head

        # Front view (along character's facing) and side view (perpendicular).
        facing = target_facing(arm)
        up = mathutils.Vector((0, 0, 1))
        side = up.cross(facing).normalized()

        for angle_name, direction in [("front", facing), ("side", side)]:
            out = out_dir / f"{source_tag}_{joint}_{angle_name}.png"
            render_closeup(cam_obj, joint_pos, direction, radius, out)
            print(f"  saved {out.name} (frame={frame})")


if __name__ == "__main__":
    main()

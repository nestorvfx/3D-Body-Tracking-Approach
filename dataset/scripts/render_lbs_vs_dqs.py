"""Render joint close-ups with BOTH LBS and DQS in one script so the same
character/clip/frame are compared under identical conditions.

For each joint, renders:
  <out>/<source>_<joint>_{lbs,dqs}_{front,side}.png

Output is designed for side-by-side composite via make_lbs_dqs_grid.py.
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


JOINTS = {
    "hip_L":      ("upperleg01.L", "lowerleg01.L", 0.22),
    "hip_R":      ("upperleg01.R", "lowerleg01.R", 0.22),
    "knee_L":     ("lowerleg01.L", "foot.L",       0.20),
    "knee_R":     ("lowerleg01.R", "foot.R",       0.20),
    "shoulder_L": ("upperarm01.L", "lowerarm01.L", 0.22),
    "shoulder_R": ("upperarm01.R", "lowerarm01.R", 0.22),
    "elbow_L":    ("lowerarm01.L", "wrist.L",      0.18),
    "elbow_R":    ("lowerarm01.R", "wrist.R",      0.18),
}


def bend_at(arm, parent_name, child_name):
    if child_name is None:
        return 0.0
    mw = arm.matrix_world
    p = arm.pose.bones.get(parent_name)
    c = arm.pose.bones.get(child_name)
    if p is None or c is None:
        return 0.0
    pa = (mw @ p.tail) - (mw @ p.head)
    ca = (mw @ c.tail) - (mw @ c.head)
    if pa.length < 1e-5 or ca.length < 1e-5:
        return 0.0
    pa.normalize(); ca.normalize()
    return math.degrees(math.acos(max(-1.0, min(1.0, pa.dot(ca)))))


def find_worst_frames(ctx, arm):
    start, end = ctx.frame_range
    step = max(1, (end - start) // 60)
    best = {j: (start, 0.0) for j in JOINTS}
    for f in range(start, end + 1, step):
        ctx.apply_pose(f)
        bpy.context.view_layer.update()
        for j, (pn, cn, _r) in JOINTS.items():
            ang = bend_at(arm, pn, cn)
            if ang > best[j][1]:
                best[j] = (f, ang)
    return best


def target_facing(arm):
    mw = arm.matrix_world
    up = mathutils.Vector((0, 0, 1))
    for ln, rn in (("upperleg01.L", "upperleg01.R"),
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


def set_preserve_volume(arm_obj, value: bool):
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) == arm_obj:
                mod.use_deform_preserve_volume = value


def render_cell(cam_obj, joint_pos, direction, radius, out_path):
    cam_obj.location = (joint_pos.x + direction.x * radius * 4.0,
                        joint_pos.y + direction.y * radius * 4.0,
                        joint_pos.z + direction.z * radius * 4.0)
    fwd = (joint_pos - mathutils.Vector(cam_obj.location)).normalized()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
    cam_obj.data.lens = 85
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
    tag = {"cmu": "cmu", "aistpp": "aistpp", "100style": "100style"}.get(src_kind, src_kind)

    cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 85
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_world_hdri(hdri_path, strength=1.3)
    configure_render(resolution=(384, 384), engine="BLENDER_EEVEE",
                     samples=64, output_path=str(out_dir / "_.png"))

    worst = find_worst_frames(ctx, arm)
    print(f"[worst] {[(j, f, round(a,1)) for j, (f, a) in worst.items()]}")

    for mode in ("lbs", "dqs"):
        set_preserve_volume(arm, value=(mode == "dqs"))
        for joint, (bone_name, _child, radius) in JOINTS.items():
            frame, _bend = worst[joint]
            ctx.apply_pose(frame)
            bpy.context.view_layer.update()
            pb = arm.pose.bones.get(bone_name)
            if pb is None:
                continue
            joint_pos = arm.matrix_world @ pb.head

            facing = target_facing(arm)
            up = mathutils.Vector((0, 0, 1))
            side = up.cross(facing).normalized()

            for angle_name, direction in (("front", facing), ("side", side)):
                out = out_dir / f"{tag}_{joint}_{mode}_{angle_name}.png"
                render_cell(cam_obj, joint_pos, direction, radius, out)
                print(f"  saved {out.name}")


if __name__ == "__main__":
    main()

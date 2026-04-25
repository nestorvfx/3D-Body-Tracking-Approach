"""Render a range of frames to sweep a clip."""
import sys, math
from pathlib import Path
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def ensure_mpfb():
    try: bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except: pass
    import importlib, sys as _sys
    pkg = importlib.import_module("bl_ext.user_default.mpfb")
    _sys.modules["mpfb"] = pkg
    for s in ("humanservice","targetservice","assetservice","locationservice"):
        _sys.modules[f"mpfb.services.{s}"] = importlib.import_module(f"bl_ext.user_default.mpfb.services.{s}")

ensure_mpfb()
from lib.mpfb_build import build_character
from lib.fk_retarget import RetargetContext as FKRetargetContext, load_bvh
from lib.render_setup import clear_scene, configure_render, set_world_hdri

args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
bvh_path = args[0]
hdri_path = args[1]
out_dir = Path(args[2])
frames = [int(x) for x in args[3:]] if len(args) > 3 else [1, 30, 60, 100, 150]
out_dir.mkdir(parents=True, exist_ok=True)

clear_scene()
bm, arm = build_character(42, with_assets=False)
bvh = load_bvh(bvh_path, source=None)
ctx = FKRetargetContext(bvh, arm)

cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 50
cam_obj = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

set_world_hdri(hdri_path, strength=1.5)
configure_render(resolution=(1280, 720), engine="BLENDER_EEVEE", samples=32,
                  output_path=str(out_dir / "_.png"))

import mathutils

for f in frames:
    ctx.apply_pose(f)
    mw = arm.matrix_world
    root_w = mw @ arm.pose.bones["root"].head

    # Camera: front (toward -Y) offset 3m south. Use fixed distance but use height 1.3m
    cam_obj.location = (root_w.x, root_w.y - 3.0, 1.3)
    fwd = mathutils.Vector((0, 1, -0.15)).normalized()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(out_dir / f"F_{f:04d}.png")
    bpy.ops.render.render(write_still=True)

    cam_obj.location = (root_w.x + 3.0, root_w.y, 1.3)
    fwd = mathutils.Vector((-1, 0, -0.15)).normalized()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(out_dir / f"S_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"saved {f}")

"""Render a single source at 4 specified frames with auto-facing FRONT camera.

Usage:
  blender --background --python render_grid.py -- <bvh_path> <hdri> <out_dir> <frame1> <frame2> <frame3> <frame4> [seed]
Produces: out_dir/F_<frame>.png per frame.
"""
import sys
from pathlib import Path
import bpy, mathutils

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
out_dir = Path(args[2]); out_dir.mkdir(parents=True, exist_ok=True)
frames = [int(x) for x in args[3:7]]
seed = int(args[7]) if len(args) > 7 else 42

clear_scene()
bm, arm = build_character(seed, with_assets=True)
bvh = load_bvh(bvh_path, source=None)
ctx = FKRetargetContext(bvh, arm)

cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 50
cam_obj = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

set_world_hdri(hdri_path, strength=1.5)
configure_render(resolution=(640, 960), engine="BLENDER_EEVEE", samples=64,
                 output_path=str(out_dir / "_.png"))


def target_facing():
    mw = arm.matrix_world
    up = mathutils.Vector((0, 0, 1))
    for (ln, rn) in (("upperleg01.L", "upperleg01.R"),
                     ("upperarm01.L", "upperarm01.R")):
        lpb = arm.pose.bones.get(ln); rpb = arm.pose.bones.get(rn)
        if lpb and rpb:
            l = mw @ lpb.head; r = mw @ rpb.head
            br = mathutils.Vector((r.x-l.x, r.y-l.y, 0.0))
            if br.length > 1e-4:
                br.normalize()
                fwd = up.cross(br)
                if fwd.length > 1e-4:
                    return fwd.normalized()
    return mathutils.Vector((0, -1, 0))


for f in frames:
    ctx.apply_pose(f)
    mw = arm.matrix_world
    root_w = mw @ arm.pose.bones["root"].head
    facing = target_facing()
    cam_obj.location = (root_w.x + facing.x * 3.0,
                        root_w.y + facing.y * 3.0,
                        root_w.z + 0.3)
    fwd_vec = -facing.copy(); fwd_vec.z = -0.05; fwd_vec.normalize()
    cam_obj.rotation_euler = fwd_vec.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.render.filepath = str(out_dir / f"F_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"saved {f}")

"""Render from DIRECTLY IN FRONT to verify arm-torso clearance."""
import sys, random, math
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
out_dir = Path(args[2]) if len(args) > 2 else HERE.parent / "output" / "front_view"
out_dir.mkdir(parents=True, exist_ok=True)

clear_scene()
bm, arm = build_character(42, with_assets=True)
bvh = load_bvh(bvh_path, source=None)
ctx = FKRetargetContext(bvh, arm)

# Fixed FRONT camera
cam_data = bpy.data.cameras.new("FrontCam")
cam_data.lens = 50
cam_obj = bpy.data.objects.new("FrontCam", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

set_world_hdri(hdri_path, strength=1.5)
configure_render(resolution=(1280, 720), engine="BLENDER_EEVEE", samples=64,
                  output_path=str(out_dir / "front.png"))

import mathutils

def target_facing(arm_obj) -> mathutils.Vector:
    """Target forward direction.  Uses HIP joints (upperleg01 heads) as the
    primary reference — hips are structurally fixed and rotate as a rigid
    body with the pelvis, so they give a robust yaw reading even in extreme
    dance poses where the shoulders may be twisted independently.

    Falls back to shoulder joints if hips are unavailable.  Always projects
    to the XY plane so a pelvic tilt doesn't flip the forward direction.
    """
    mw = arm_obj.matrix_world
    up = mathutils.Vector((0.0, 0.0, 1.0))

    def _compute(lname, rname):
        l = arm_obj.pose.bones.get(lname)
        r = arm_obj.pose.bones.get(rname)
        if l is None or r is None:
            return None
        lh = mw @ l.head
        rh = mw @ r.head
        bright = mathutils.Vector(((rh.x - lh.x), (rh.y - lh.y), 0.0))
        if bright.length < 1e-4:
            return None
        bright.normalize()
        fwd = up.cross(bright)
        return fwd.normalized() if fwd.length > 1e-4 else None

    # Prefer hips; fall back to shoulders.
    for pair in (("upperleg01.L", "upperleg01.R"),
                 ("upperarm01.L", "upperarm01.R")):
        result = _compute(*pair)
        if result is not None:
            return result
    return mathutils.Vector((0.0, -1.0, 0.0))


for f in [100, 142, 180, 220, 260]:
    ctx.apply_pose(f)
    mw = arm.matrix_world
    root_w = mw @ arm.pose.bones["root"].head

    # Facing is re-detected from the TARGET armature each frame, since the
    # retargeted root rotation yaws with the source animation.
    facing = target_facing(arm)
    up = mathutils.Vector((0.0, 0.0, 1.0))
    body_right = facing.cross(up).normalized()

    def place_cam(direction, name):
        cam_obj.location = (root_w.x + direction.x * 3.0,
                            root_w.y + direction.y * 3.0,
                            root_w.z + 0.3)
        fwd_vec = -direction.copy()
        fwd_vec.z = -0.05
        fwd_vec.normalize()
        cam_obj.rotation_euler = fwd_vec.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.render.filepath = str(out_dir / f"{name}_{f:04d}.png")
        bpy.ops.render.render(write_still=True)

    place_cam(facing,        "FRONT")
    place_cam(-facing,       "REAR")
    place_cam(body_right,    "SIDE")

    print(f"Saved FRONT/REAR/SIDE for frame {f} (facing={tuple(round(v,2) for v in facing)})")

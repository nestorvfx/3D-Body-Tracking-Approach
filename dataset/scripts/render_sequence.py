"""Render N consecutive frames of a BVH retarget, with DQS-enabled mesh,
auto-facing FRONT camera.  Output: PNG per frame.  Use for sequence
viewing + GIF compositing.

Usage:
  blender --background --python render_sequence.py -- \
      <bvh_path> <hdri_path> <out_dir> <start_frame> <n_frames> [seed=42]
"""
from __future__ import annotations

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


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bvh_path = args[0]
    hdri_path = args[1]
    out_dir = Path(args[2]); out_dir.mkdir(parents=True, exist_ok=True)
    start_frame = int(args[3])
    n_frames = int(args[4])
    seed = int(args[5]) if len(args) > 5 else 42

    clear_scene()
    bm, arm = build_character(seed, with_assets=True)
    bvh = load_bvh(bvh_path, source=None)
    ctx = RetargetContext(bvh, arm)

    cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 50
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_world_hdri(hdri_path, strength=1.3)
    # Moderate resolution so 100 frames renders in reasonable time.
    configure_render(resolution=(640, 960), engine="BLENDER_EEVEE",
                     samples=32, output_path=str(out_dir / "_.png"))

    # Sample the CAMERA placement once at the first frame of the sequence,
    # then keep it fixed (so motion reads naturally across the clip — the
    # character moves through frame, camera doesn't chase).
    ctx.apply_pose(start_frame)
    bpy.context.view_layer.update()
    root_w = arm.matrix_world @ arm.pose.bones["root"].head
    facing = target_facing(arm)
    # Camera at the character's initial front, slightly raised.
    cam_obj.location = (root_w.x + facing.x * 3.5,
                        root_w.y + facing.y * 3.5,
                        root_w.z + 0.4)
    fwd = -facing.copy(); fwd.z = -0.05; fwd.normalize()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
    print(f"[seq] camera at {tuple(round(v,2) for v in cam_obj.location)}, "
          f"facing {tuple(round(v,2) for v in facing)}")

    # Clamp to actual clip range.
    clip_start, clip_end = ctx.frame_range
    frames = [f for f in range(start_frame, start_frame + n_frames)
              if clip_start <= f <= clip_end]
    print(f"[seq] rendering {len(frames)} frames "
          f"({frames[0]}..{frames[-1]}) from clip range ({clip_start}..{clip_end})")

    for i, f in enumerate(frames):
        ctx.apply_pose(f)
        bpy.context.scene.render.filepath = str(out_dir / f"F_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(frames)}] saved F_{i:04d}.png (frame={f})")


if __name__ == "__main__":
    main()

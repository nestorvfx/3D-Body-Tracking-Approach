"""Render source BVH armature (stick-figure) and retargeted MPFB character
side-by-side at the same frame with the same camera.

For each frame: two renders are produced — one from the source armature
(with all meshes hidden so we see just the octahedral bones), one from the
retargeted character.  Output both into out_dir/src_<frame>.png and
tgt_<frame>.png; main() also composites them side by side.

Usage:
  blender --background --python render_side_by_side.py -- \
      <bvh_path> <hdri_path> <out_dir> <frame1> <frame2> ... [seed=42]
"""
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


def target_facing(arm_obj):
    """Forward direction from hip joint positions, projected to XY."""
    mw = arm_obj.matrix_world
    up = mathutils.Vector((0, 0, 1))
    for (ln, rn) in (("upperleg01.L", "upperleg01.R"),
                     ("upperarm01.L", "upperarm01.R")):
        l = arm_obj.pose.bones.get(ln); r = arm_obj.pose.bones.get(rn)
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


def place_camera(cam_obj, root_w, facing):
    cam_obj.location = (root_w.x + facing.x * 3.0,
                        root_w.y + facing.y * 3.0,
                        root_w.z + 0.3)
    fwd = -facing.copy(); fwd.z = -0.05; fwd.normalize()
    cam_obj.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bvh_path = args[0]
    hdri_path = args[1]
    out_dir = Path(args[2]); out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    seed = 42
    for a in args[3:]:
        try:
            frames.append(int(a))
        except ValueError:
            pass
    # Last arg is seed if it's an int and exceeds typical frame (>=1000).
    # Simpler: assume last arg is seed only if an extra int is given.
    # Take all as frames, default seed=42.

    clear_scene()
    bm, arm = build_character(seed, with_assets=True)
    bvh = load_bvh(bvh_path, source=None)
    ctx = RetargetContext(bvh, arm)

    cam_data = bpy.data.cameras.new("Cam"); cam_data.lens = 50
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    set_world_hdri(hdri_path, strength=1.5)
    configure_render(resolution=(640, 960), engine="BLENDER_EEVEE", samples=32,
                     output_path=str(out_dir / "_.png"))

    # Find the source "hips" bone name (varies by source kind).
    src_kind = bvh.get("_mocap_source", "cmu")
    SRC_HIPS = {"cmu": "Hips", "aistpp": "pelvis", "100style": "Hips"}
    src_hips_name = SRC_HIPS.get(src_kind, "Hips")

    for f in frames:
        # --- 1) Apply retarget + determine the shared camera placement.
        ctx.apply_pose(f)
        bpy.context.view_layer.update()
        mw_arm = arm.matrix_world
        root_w = mw_arm @ arm.pose.bones["root"].head
        facing = target_facing(arm)

        # --- 1a) Shift source armature so src hips colocate with target root.
        # At this frame source Hips world = bvh.matrix_world @ src_hips.head.
        # To make them equal we set bvh.location so the displacement works.
        src_hips_local = bvh.pose.bones[src_hips_name].head
        # Target is root_w.  source hip in world currently = mw_bvh @ src_hips_local.
        # We want mw_bvh @ src_hips_local = root_w.  Since mw_bvh = identity*loc,
        # that means bvh.location = root_w - src_hips_local.
        bvh.location = (root_w.x - src_hips_local.x,
                        root_w.y - src_hips_local.y,
                        root_w.z - src_hips_local.z)
        bpy.context.view_layer.update()

        place_camera(cam_obj, root_w, facing)

        # --- 2) Render TARGET with meshes visible, source skeleton hidden.
        bvh.hide_viewport = True; bvh.hide_render = True
        arm.hide_viewport = True   # we don't want to see the rig; just the mesh
        arm.hide_render = True     # but mesh objects have Armature modifier,
                                    # so hiding armature's render doesn't hide mesh.
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.startswith(f"subject_{seed:04d}"):
                obj.hide_render = False
        bpy.context.scene.render.filepath = str(out_dir / f"tgt_{f:04d}.png")
        bpy.ops.render.render(write_still=True)

        # --- 3) Render SOURCE stick-figure only (character meshes hidden).
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.startswith(f"subject_{seed:04d}"):
                obj.hide_render = True
        _stick_figure_draw(bvh, f, out_dir / f"src_{f:04d}.png", cam_obj)
        # Restore target visibility for next frame.
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.startswith(f"subject_{seed:04d}"):
                obj.hide_render = False

        # Restore target visibility.
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.startswith(f"subject_{seed:04d}"):
                obj.hide_render = False
        print(f"saved tgt_{f:04d}.png + src_{f:04d}.png")


def _stick_figure_draw(src_arm, frame: int, out_path: Path, cam_obj):
    """Render a stick-figure of the source armature using mesh cylinders
    between each bone's head and tail, plus spheres at joints."""
    import bmesh
    mw = src_arm.matrix_world
    print(f"  src_arm matrix_world translation: {mw.translation}")

    # Build bmesh with one cylinder per bone + spheres at joint positions.
    bm = bmesh.new()
    JOINT_R = 0.04
    BONE_R = 0.02
    dbg_printed = False

    unique_joints = set()
    for pb in src_arm.pose.bones:
        h = mw @ pb.head
        t = mw @ pb.tail
        if not dbg_printed:
            print(f"  first pb '{pb.name}' head={tuple(round(v,3) for v in h)}")
            dbg_printed = True
        unique_joints.add((round(h.x, 4), round(h.y, 4), round(h.z, 4)))
        unique_joints.add((round(t.x, 4), round(t.y, 4), round(t.z, 4)))

        # Cylinder from head to tail.
        v = t - h
        length = v.length
        if length < 1e-5:
            continue
        direction = v.normalized()
        # Build a cylinder along Z then rotate to direction.
        mid = (h + t) * 0.5
        # Rotation from +Z to direction.
        up = mathutils.Vector((0.0, 0.0, 1.0))
        if direction.dot(up) > 0.9999:
            rot = mathutils.Matrix.Identity(3)
        elif direction.dot(up) < -0.9999:
            rot = mathutils.Matrix.Rotation(3.14159265, 3, mathutils.Vector((1, 0, 0)))
        else:
            axis = up.cross(direction)
            axis.normalize()
            angle = math.acos(up.dot(direction))
            rot = mathutils.Matrix.Rotation(angle, 3, axis)
        tform = mathutils.Matrix.Translation(mid) @ rot.to_4x4() @ \
                mathutils.Matrix.Scale(length * 0.5, 4, (0, 0, 1))
        bmesh.ops.create_cone(bm, segments=10, radius1=BONE_R, radius2=BONE_R,
                                depth=2.0, matrix=tform)

    for (x, y, z) in unique_joints:
        bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6,
                                   radius=JOINT_R,
                                   matrix=mathutils.Matrix.Translation((x, y, z)))

    mesh_data = bpy.data.meshes.new("bvh_sticks")
    bm.to_mesh(mesh_data)
    bm.free()
    stick_obj = bpy.data.objects.new("bvh_sticks_obj", mesh_data)
    # Link to scene collection AND view layer's active collection.
    try:
        bpy.context.collection.objects.link(stick_obj)
    except Exception:
        bpy.context.scene.collection.objects.link(stick_obj)

    # Ensure unhidden in every way.
    stick_obj.hide_viewport = False
    stick_obj.hide_render = False
    stick_obj.hide_select = False
    print(f"  stick-figure: {len(mesh_data.vertices)} verts, "
          f"{len(mesh_data.polygons)} faces")

    # Bright orange Principled BSDF — simpler than emission for EEVEE.
    mat = bpy.data.materials.new("bvh_stick_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        # Clear and rebuild if default wasn't there.
        for n in list(mat.node_tree.nodes):
            mat.node_tree.nodes.remove(n)
        out = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = (1.0, 0.3, 0.05, 1.0)
    # Emission too so it pops against dark scene
    if "Emission Color" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.3, 0.05, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 3.0
    stick_obj.data.materials.append(mat)

    bpy.context.view_layer.update()

    bpy.context.scene.render.filepath = str(out_path)
    bpy.ops.render.render(write_still=True)

    # Clean up.
    bpy.data.objects.remove(stick_obj, do_unlink=True)
    bpy.data.meshes.remove(mesh_data)
    bpy.data.materials.remove(mat)


if __name__ == "__main__":
    main()

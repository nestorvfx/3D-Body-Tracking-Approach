"""Render configuration helpers: world HDRI, camera framing, engine settings.

For the pilot we use BLENDER_EEVEE_NEXT (Blender 4.2+/5.x) at 1280x720 which
matches BEDLAM-class training input resolution. A later Cycles pass can be
invoked for a photoreal tail.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import bpy  # type: ignore
import mathutils  # type: ignore


def configure_render(
    *,
    resolution: tuple[int, int] = (1280, 720),
    engine: str = "BLENDER_EEVEE",
    samples: int = 32,
    output_path: str | Path,
) -> None:
    scene = bpy.context.scene
    # Blender 4.2 used 'BLENDER_EEVEE_NEXT'; 5.x folded it back to 'BLENDER_EEVEE'.
    valid = {item.identifier for item in
             scene.render.bl_rna.properties["engine"].enum_items}
    if engine not in valid:
        for fallback in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
            if fallback in valid:
                print(f"[render_setup] engine {engine!r} unavailable; using {fallback!r}")
                engine = fallback
                break
    scene.render.engine = engine
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(output_path)
    # AGX (or Filmic fallback) gives film-like tonemapping with preserved
    # highlights — much closer to BEDLAM / Cycles output than "Standard",
    # which looks washed and clip-happy.
    try:
        scene.view_settings.view_transform = "AGX"
    except Exception:
        scene.view_settings.view_transform = "Filmic"
    try:
        scene.view_settings.look = "Medium High Contrast"
    except Exception:
        pass

    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        # Try OptiX for RTX GPU; fall back to CPU silently.
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "OPTIX"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type in ("OPTIX", "CUDA")
            scene.cycles.device = "GPU"
        except Exception as e:
            print(f"[render_setup] GPU init failed, using CPU: {e}")
    else:
        # EEVEE Next samples
        scene.eevee.taa_render_samples = samples
        # Raytracing + SSR give realistic skin reflections / depth.
        try:
            scene.eevee.use_raytracing = True
        except AttributeError:
            pass
        try:
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
        except AttributeError:
            pass
        # Enable subsurface scattering sample count (EEVEE-Next does it
        # automatically when BSDF has subsurface).
        try:
            scene.eevee.sss_samples = 16
        except AttributeError:
            pass


def set_world_hdri(hdri_path: str | Path, *, strength: float = 1.0, rotation_z: float = 0.0) -> None:
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    mapping = nt.nodes.new("ShaderNodeMapping")
    tex_coord = nt.nodes.new("ShaderNodeTexCoord")

    bg.inputs["Strength"].default_value = strength
    mapping.inputs["Rotation"].default_value[2] = rotation_z

    env.image = bpy.data.images.load(str(hdri_path))
    env.image.colorspace_settings.name = "Linear Rec.709"

    nt.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_camera(
    *,
    focal_length_mm: float = 50.0,
    sensor_width_mm: float = 36.0,
    name: str = "PilotCamera",
) -> object:
    existing = bpy.data.objects.get(name)
    if existing:
        bpy.data.objects.remove(existing, do_unlink=True)
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = focal_length_mm
    cam_data.sensor_width = sensor_width_mm
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def frame_armature(
    camera: object,
    armature: object,
    *,
    rng: random.Random,
    distance_range: tuple[float, float] = (3.0, 6.0),
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
    pitch_range: tuple[float, float] = (-0.35, 0.35),
    target_height: float = 1.0,
) -> tuple[float, float, float]:
    """Place the camera on a sphere around the armature centre, framed at head height.
    Returns (yaw, pitch, distance) used."""
    # World-space AABB centre of the armature — fall back to armature origin.
    try:
        bpy.context.view_layer.update()
        xs, ys, zs = [], [], []
        for pb in armature.pose.bones:
            p = armature.matrix_world @ pb.head
            xs.append(p.x); ys.append(p.y); zs.append(p.z)
        cx = sum(xs) / len(xs) if xs else 0.0
        cy = sum(ys) / len(ys) if ys else 0.0
        cz_min = min(zs) if zs else 0.0
        cz_max = max(zs) if zs else 1.8
        target = mathutils.Vector((cx, cy, cz_min + (cz_max - cz_min) * target_height))
    except Exception:
        target = mathutils.Vector(armature.matrix_world.to_translation())
        target.z += 1.5

    yaw = rng.uniform(*yaw_range)
    pitch = rng.uniform(*pitch_range)
    dist = rng.uniform(*distance_range)

    cam_pos = target + mathutils.Vector((
        dist * math.cos(pitch) * math.sin(yaw),
        -dist * math.cos(pitch) * math.cos(yaw),
        dist * math.sin(pitch),
    ))
    camera.location = cam_pos
    # Point the camera at the target.
    forward = (target - cam_pos).normalized()
    rot_quat = forward.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()
    return yaw, pitch, dist


def clear_scene(keep_camera: bool = False) -> None:
    for obj in list(bpy.data.objects):
        if keep_camera and obj.type == "CAMERA":
            continue
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception:
            pass
    for coll in ("meshes", "armatures", "materials", "images", "actions", "textures", "curves"):
        bag = getattr(bpy.data, coll, None)
        if bag is None:
            continue
        for item in list(bag):
            if item.users == 0:
                try:
                    bag.remove(item)
                except Exception:
                    pass

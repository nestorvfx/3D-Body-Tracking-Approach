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
    # In Blender 5.1 the scene.render.engine enum_items list is stale/filtered
    # and does NOT enumerate CYCLES even when the Cycles addon is loaded.
    # Assigning to scene.render.engine directly works, though; the enum
    # check was silently downgrading Cycles requests to EEVEE.  Try-assign
    # instead of enum-query.
    for candidate in (engine, "CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = candidate
            if candidate != engine:
                print(f"[render_setup] engine {engine!r} failed; using {candidate!r}")
            engine = candidate
            break
        except TypeError:
            continue
    else:
        raise RuntimeError(f"No render engine accepted (tried {engine}, CYCLES, EEVEE)")
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


def update_hdri_params(*, strength: float | None = None,
                         rotation_z: float | None = None) -> None:
    """Fast per-sample tweak of HDRI strength / rotation WITHOUT rebuilding
    the world node tree.  Preserves `scene.render.use_persistent_data`'s
    cache — every call to `set_world_hdri` wipes it and forces a full
    scene upload on the next render (~2 s on a 5060 Ti at 256x192), so
    you want to call `set_world_hdri` ONCE per (seed, clip) and this
    helper for per-sample variety within that window.
    """
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return
    for n in world.node_tree.nodes:
        if strength is not None and n.type == "BACKGROUND":
            try:
                n.inputs["Strength"].default_value = float(strength)
            except Exception:
                pass
        if rotation_z is not None and n.type == "MAPPING":
            try:
                n.inputs["Rotation"].default_value[2] = float(rotation_z)
            except Exception:
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


def configure_gt_passes(*,
                         depth_out_dir: str | Path | None,
                         mask_out_dir: str | Path | None = None,
                         ) -> tuple[object | None, object | None]:
    """Enable the Z-depth and (optionally) object-index mask passes, and
    wire compositor File Output nodes that write EXR/PNG sidecars per
    rendered frame.  Call ONCE per character (after configure_render).

    Blender 5.1 compositor API (verified against release notes migration
    guide and experimentally): scene.node_tree was removed and replaced
    with scene.compositing_node_group, which must be created as a
    standalone data-block and assigned.  CompositorNodeComposite was
    removed entirely — NodeGroupOutput + an explicit interface socket
    is the replacement.  CompositorNodeOutputFile's API changed too:
    `base_path`/`file_slots` → `directory`/`file_name`, and
    `format.media_type` MUST be set before `format.file_format`.

    Returns (depth_file_output_node_or_None, mask_file_output_node_or_None).
    D-PoSE (arXiv 2410.04889) reports +3.0 mm PA-MPJPE on 3DPW when
    depth is wired as auxiliary training supervision.
    """
    scene = bpy.context.scene
    scene.render.use_compositing = True                  # replaces use_nodes flag
    # use_nodes is deprecated & no-op in 5.x; set anyway for 4.x back-compat.
    try:
        scene.use_nodes = True
    except Exception:
        pass

    # Create or retrieve the compositor node group (5.x: independent data-block).
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        tree = bpy.data.node_groups.new(
            name=f"Comp_{scene.name}", type="CompositorNodeTree")
        scene.compositing_node_group = tree
    for n in list(tree.nodes):
        tree.nodes.remove(n)

    # Replacement for the removed CompositorNodeComposite: NodeGroupOutput
    # + a declared "Image" interface socket.  The compositor reads from
    # this output as the final render result.
    rl = tree.nodes.new("CompositorNodeRLayers")
    grp_out = tree.nodes.new("NodeGroupOutput")
    # Declare the interface socket if absent.
    has_image_socket = False
    try:
        for item in tree.interface.items_tree:
            if getattr(item, "in_out", None) == "OUTPUT" \
                    and getattr(item, "socket_type", "") == "NodeSocketColor":
                has_image_socket = True
                break
    except Exception:
        pass
    if not has_image_socket:
        try:
            tree.interface.new_socket(
                name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        except Exception:
            pass
    tree.links.new(rl.outputs["Image"], grp_out.inputs[0])

    vl = scene.view_layers[0]
    vl.use_pass_z = depth_out_dir is not None
    # Object-index pass was removed from the Render Layers compositor
    # output in Blender 5.x (only Cryptomatte remains).  Mask routing
    # via cryptomatte requires a separate decoder at training-time,
    # which is out of scope for this commit.  Disable mask output on
    # 5.x and fall back gracefully.  Leave the plumbing in place so
    # masks can be re-enabled once the cryptomatte path is written.
    try:
        vl.use_pass_object_index = mask_out_dir is not None
    except AttributeError:
        pass

    depth_fout = None
    if depth_out_dir is not None:
        depth_fout = tree.nodes.new("CompositorNodeOutputFile")
        depth_fout.directory = str(depth_out_dir)
        depth_fout.file_name = "depth_"         # sample_id injected per-call
        depth_fout.format.media_type = "IMAGE"  # MUST precede file_format in 5.x
        depth_fout.format.file_format = "OPEN_EXR"
        depth_fout.format.color_depth = "16"
        depth_fout.format.color_mode = "BW"
        depth_fout.format.exr_codec = "ZIP"
        # Depth socket: 5.0+ renamed "Z" → "Depth".
        tree.links.new(rl.outputs["Depth"], depth_fout.inputs[0])

    mask_fout = None
    if mask_out_dir is not None and "IndexOB" in rl.outputs:
        # Legacy 4.x path — only reachable if IndexOB socket is present.
        mask_fout = tree.nodes.new("CompositorNodeOutputFile")
        mask_fout.directory = str(mask_out_dir)
        mask_fout.file_name = "mask_"
        mask_fout.format.media_type = "IMAGE"
        mask_fout.format.file_format = "PNG"
        mask_fout.format.color_mode = "BW"
        mask_fout.format.color_depth = "8"
        mask_fout.format.compression = 15
        tree.links.new(rl.outputs["IndexOB"], mask_fout.inputs[0])

    return depth_fout, mask_fout


def set_gt_sample_path(file_output_node, sample_id: str) -> None:
    """Point a File Output node at the next sample by stamping the
    sample_id into its file_name.  Blender auto-appends the current
    scene frame number as suffix — we strip/rename after render."""
    if file_output_node is None:
        return
    file_output_node.file_name = f"{sample_id}_"


def add_shadow_catcher_plane(size_m: float = 8.0) -> object:
    """Add a large horizontal plane at Z=0 with Cycles shadow-catcher
    material.  Gives synthetic subjects a grounded floor shadow without
    fighting the HDRI — significantly improves photometric realism on
    outdoor/studio HDRIs and closes the sim-to-real gap for foot/leg
    visibility cues.  BEDLAM uses the same pattern in their Unreal
    renders.

    Returns the plane object (caller does not need to track it; the
    plane is wiped with clear_scene() between characters).
    """
    existing = bpy.data.objects.get("ShadowCatcherPlane")
    if existing:
        return existing
    bpy.ops.mesh.primitive_plane_add(size=size_m, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "ShadowCatcherPlane"
    try:
        plane.is_shadow_catcher = True
    except AttributeError:
        pass
    try:
        # Cycles visibility: keep camera visibility so shadow lands on
        # the plane, but don't contribute to reflections.
        plane.visible_glossy = False
        plane.visible_transmission = False
    except AttributeError:
        pass
    return plane


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

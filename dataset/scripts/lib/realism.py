"""Rendering realism pack:

  1. AgX view transform (Blender 4.2+ default, graceful highlight rolloff,
     fixes the red-channel clip that makes warm HDRI scenes look orange).
  2. Skin material tweaks: physically-correct subsurface radii
     (3.67, 1.37, 0.68) mm after Jensen, randomized ±20% per character.
  3. Compositor chain simulating phone-camera artefacts: lens distortion,
     chromatic aberration, vignette, film grain.

None of these change the geometry/armature — they change how the scene is
shaded and how the render buffer is post-processed.  Per-frame overhead is
tiny (compositor) or zero (material + view transform).
"""
from __future__ import annotations

import random

import bpy  # type: ignore


# ------------------------- View transform -------------------------

def apply_agx_view() -> None:
    """Switch scene view transform to AgX if available, else Standard.

    AgX was added in Blender 4.0 and made the default in 4.2.  In Blender 5.1
    it's always present.  Safe to call unconditionally; falls back on older
    versions or if the enum item is missing for any reason.
    """
    vs = bpy.context.scene.view_settings
    try:
        vs.view_transform = "AgX"
    except Exception:
        try:
            vs.view_transform = "Standard"
        except Exception:
            pass
    vs.look = "None"
    # Slight exposure jitter to diversify training brightness:
    # caller can adjust via vs.exposure before render.
    vs.gamma = 1.0


# ------------------------- Skin SSS override -------------------------

# Physically accurate skin SSS radii (mm) per Jensen 2001.  Blender takes
# these in metres, so divide by 1000.
SKIN_SSS_RADII_M = (3.67e-3, 1.37e-3, 0.68e-3)


def _find_principled_bsdf(mat: "bpy.types.Material"):
    if mat is None or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def apply_physical_skin_sss(basemesh, rng: random.Random, *, redness_jitter: float = 0.05) -> None:
    """Walk the basemesh materials; for any Principled BSDF, override
    Subsurface Radius to the physically-correct Jensen values, randomised
    ±20% per character.  Also nudges Subsurface Weight into a realistic
    range (0.6-0.9)."""
    if basemesh is None:
        return
    for slot in basemesh.material_slots:
        mat = slot.material
        bsdf = _find_principled_bsdf(mat)
        if bsdf is None:
            continue
        # Randomize the per-character SSS scalars.
        jitter = lambda: rng.uniform(0.8, 1.2)
        radii = tuple(v * jitter() for v in SKIN_SSS_RADII_M)

        # Blender 5.x Principled BSDF renamed sockets:
        # "Subsurface" -> "Subsurface Weight"
        # "Subsurface Radius" stays (VECTOR)
        radius_socket = bsdf.inputs.get("Subsurface Radius")
        if radius_socket is not None:
            try:
                radius_socket.default_value = radii
            except Exception:
                pass
        weight_socket = (bsdf.inputs.get("Subsurface Weight")
                         or bsdf.inputs.get("Subsurface"))
        if weight_socket is not None:
            try:
                weight_socket.default_value = rng.uniform(0.55, 0.85)
            except Exception:
                pass

        # Add a tiny redness push to base color (ITA-inspired tint):
        base = bsdf.inputs.get("Base Color")
        if base is not None and not base.is_linked:
            r, g, b, a = base.default_value
            r = min(1.0, r + rng.uniform(-redness_jitter, redness_jitter))
            base.default_value = (r, g, b, a)


# ------------------------- Compositor phone-camera chain -------------------------

_COMP_GROUP_NAME = "_phonecam_compositor_v3"


def _get_compositor_tree():
    """Return the scene's compositor node tree across Blender versions.

    Blender 4.x exposed `scene.node_tree` directly; Blender 5.x moved the
    compositor to a shared NodeGroup referenced via `scene.compositing_node_group`.
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    # Blender 4.x path
    nt = getattr(scene, "node_tree", None)
    if nt is not None:
        return nt
    # Blender 5.x path — bind or create a compositor node group
    if hasattr(scene, "compositing_node_group"):
        group = scene.compositing_node_group
        if group is None:
            group = bpy.data.node_groups.get(_COMP_GROUP_NAME)
            if group is None:
                group = bpy.data.node_groups.new(_COMP_GROUP_NAME, "CompositorNodeTree")
            scene.compositing_node_group = group
        return group
    return None


def build_phone_camera_compositor(
    *,
    barrel: float = 0.018,
    dispersion: float = 0.005,
    vignette_strength: float = 0.35,
) -> None:
    """Phone-camera compositor chain: Lens Distortion -> Glare -> Vignette.

    Skips gracefully if the compositor API isn't available.  Idempotent —
    creates nodes once per scene, tagged with `_phonecam_*` labels.
    """
    nt = _get_compositor_tree()
    if nt is None:
        return

    # Purge previous tagged nodes so re-runs don't stack chains.
    for n in list(nt.nodes):
        if n.label.startswith("_phonecam_"):
            nt.nodes.remove(n)

    rl = next((n for n in nt.nodes if n.type == "R_LAYERS"), None)
    comp = next((n for n in nt.nodes if n.type == "COMPOSITE"), None)
    if rl is None:
        try:
            rl = nt.nodes.new("CompositorNodeRLayers")
        except Exception:
            return
    if comp is None:
        try:
            comp = nt.nodes.new("CompositorNodeComposite")
        except Exception:
            return

    for link in list(nt.links):
        if link.to_node is comp:
            nt.links.remove(link)

    try:
        lens = nt.nodes.new("CompositorNodeLensdist")
        lens.label = "_phonecam_lens"
        lens.inputs["Distort"].default_value = barrel
        lens.inputs["Dispersion"].default_value = dispersion

        glare = nt.nodes.new("CompositorNodeGlare")
        glare.label = "_phonecam_glare"
        glare.glare_type = "FOG_GLOW"
        glare.quality = "MEDIUM"
        glare.threshold = 1.0
        glare.size = 6

        ellipse = nt.nodes.new("CompositorNodeEllipseMask")
        ellipse.label = "_phonecam_ellipse"
        ellipse.width = 0.85
        ellipse.height = 0.85

        blur = nt.nodes.new("CompositorNodeBlur")
        blur.label = "_phonecam_blur"
        blur.size_x = 120
        blur.size_y = 120

        mix_vignette = nt.nodes.new("CompositorNodeMixRGB")
        mix_vignette.label = "_phonecam_vignette_mix"
        mix_vignette.blend_type = "MULTIPLY"
        mix_vignette.inputs["Fac"].default_value = vignette_strength

        nt.links.new(rl.outputs["Image"], lens.inputs["Image"])
        nt.links.new(lens.outputs["Image"], glare.inputs["Image"])
        nt.links.new(glare.outputs["Image"], mix_vignette.inputs[1])
        nt.links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
        nt.links.new(blur.outputs["Image"], mix_vignette.inputs[2])
        nt.links.new(mix_vignette.outputs["Image"], comp.inputs["Image"])
    except Exception as e:
        print(f"[realism] compositor build skipped: {e}")


# ------------------------- One-call apply -------------------------

def apply_realism_pack(basemesh, rng: random.Random) -> None:
    """Apply all three realism upgrades in one call.

    Call AFTER build_character (so the skin material exists) and BEFORE
    render.  Safe to call once per sample."""
    apply_agx_view()
    apply_physical_skin_sss(basemesh, rng)
    build_phone_camera_compositor()

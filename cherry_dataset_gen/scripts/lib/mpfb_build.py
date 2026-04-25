"""Headless MPFB2 character creation + rigging + dressing.

Exposes `build_character(seed, with_assets=True)` returning `(basemesh, armature)`.
Seeded macro sampling for reproducibility.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import bpy  # type: ignore


@dataclass
class Phenotype:
    gender: float = 0.5
    age: float = 0.5
    muscle: float = 0.5
    weight: float = 0.5
    height: float = 0.5
    proportions: float = 0.5
    cupsize: float = 0.5
    firmness: float = 0.5
    race: dict = field(default_factory=lambda: {"african": 1/3, "asian": 1/3, "caucasian": 1/3})

    def to_mpfb(self) -> dict:
        return {
            "gender": self.gender,
            "age": self.age,
            "muscle": self.muscle,
            "weight": self.weight,
            "height": self.height,
            "proportions": self.proportions,
            "cupsize": self.cupsize,
            "firmness": self.firmness,
            "race": dict(self.race),
        }


def sample_phenotype(rng: random.Random) -> Phenotype:
    """Sample a phenotype with a BEDLAM-style heavy-tailed body-weight
    distribution.

    BEDLAM CVPR 2023 supplementary explicitly identifies the BMI > 30
    gap as the dominant cause of poor body-shape generalisation on
    HBW/EMDB-heavy evaluation; they patched it by sampling 80 male +
    80 female CAESAR bodies with BMI > 30 into their otherwise-slim
    AGORA-based pool.  We don't have CAESAR, but MPFB's ``weight``
    macro can drive the full slim→obese range — the fix is to loosen
    the weight cap and sample weight from a mixture that dedicates
    ~20 percent of mass to the high-BMI tail.

    Also broadened muscle and height to cover more of the adult range
    (narrow clamps bias the network toward a single body archetype).
    Age kept at [0.25, 0.80] for now — extending to children requires
    validating MPFB clothing proxies on child-proportion meshes, which
    is a separate milestone (see Anny, arXiv 2511.03589).
    """
    r = [rng.random() for _ in range(3)]
    s = sum(r) or 1.0
    # Heavy-tailed weight sampling: 80% slim-to-average, 20% high-BMI tail.
    if rng.random() < 0.20:
        weight = rng.uniform(0.70, 0.98)        # high-BMI tail
    else:
        weight = rng.uniform(0.10, 0.65)        # slim-to-average
    return Phenotype(
        gender=rng.uniform(0.05, 0.95),
        age=rng.uniform(0.25, 0.80),
        muscle=rng.uniform(0.15, 0.90),
        weight=weight,
        height=rng.uniform(0.10, 0.95),
        proportions=rng.uniform(0.25, 0.80),
        cupsize=rng.uniform(0.15, 0.75),
        firmness=rng.uniform(0.35, 0.90),
        race={"african": r[0]/s, "asian": r[1]/s, "caucasian": r[2]/s},
    )


def _pick_asset(rng: random.Random, candidates: list[str]) -> str | None:
    return rng.choice(candidates) if candidates else None


def _pick_skin_fragment(rng: random.Random) -> str:
    """Pick a `<folder>/<file.mhmat>` fragment for the skin slot, or ''."""
    paths = _find_assets("skins", ".mhmat")
    if not paths:
        return ""
    p = rng.choice(paths)
    parts = p.replace("\\", "/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def _pick_hair_fragment(rng: random.Random) -> str:
    paths = _find_assets("hair", ".mhclo")
    if not paths:
        return ""
    p = rng.choice(paths)
    parts = p.replace("\\", "/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def _pick_clothes_fragments(rng: random.Random, gender: float, age: float) -> list[str]:
    """Return clothing fragment strings (relative to <data>/clothes/) for
    `info['clothes']`.  MPFB expects a list of STRING paths (not dicts);
    see `HumanService._check_add_clothes`, line 651.

    Our CC0 pack has full-body suits (top+bottom combined) named
    `female_*` or `male_*`, plus shoes and fedoras.  We pick one
    gender-appropriate suit + one pair of shoes + optionally a hat.
    """
    all_mhclo = _find_assets("clothes", ".mhclo")
    if not all_mhclo:
        return []

    def folder(p: str) -> str:
        return p.replace("\\", "/").split("/")[-2]

    # Explicit allowlist of MPFB suit garments that render cleanly — verified
    # by rest-pose renders (see test_suits.py).  Prefer suits with JACKET /
    # LONG-SLEEVE tops that cover the waist-hem gap and hide the orange
    # logo/piping (which appears as an artifact stripe in motion).
    # Excluded:
    #   - female_casualsuit01/02: short-sleeve T with Y logo (artifact)
    #   - female_elegantsuit01, male_elegantsuit01: suspender-like straps
    #   - male_casualsuit02,04,06: T-shirts (waistband gap + logo artifact)
    #   - male_worksuit01: overalls with blue straps that read as artifacts
    # Broader allowlist for CHARACTER diversity — include female+male suits
    # that render reasonably across a wide range of poses.  Excludes:
    #   - female_casualsuit02 (shorts crop top — bare lower legs below knee)
    #   - male_worksuit01 (overalls)
    #   - *elegantsuit* (orange suspender artifact)
    ALLOWED_SUITS = {
        "female_casualsuit01",   # t-shirt + jeans
        "female_sportsuit01",    # tank + leggings (crop top; OK for gym shots)
        "male_casualsuit01",     # dark jacket + jeans
        "male_casualsuit02",     # blue t-shirt + jeans
        "male_casualsuit03",     # striped long-sleeve + jeans
        "male_casualsuit04",     # t-shirt + jeans (different colour)
        "male_casualsuit05",     # vest + long sleeves + jeans
        "male_casualsuit06",     # white t-shirt + jeans
    }
    def is_allowed(p: str) -> bool:
        return folder(p) in ALLOWED_SUITS

    suit_candidates = [p for p in all_mhclo if is_allowed(p)]
    if gender < 0.4:
        suits = [p for p in suit_candidates if folder(p).startswith("female")]
    elif gender > 0.6:
        suits = [p for p in suit_candidates if folder(p).startswith("male")]
    else:
        # For ambiguous gender, match to the closer gender to avoid
        # wearing a skirt on a masculine body (or vice versa).
        if gender >= 0.5:
            suits = [p for p in suit_candidates if folder(p).startswith("male")]
        else:
            suits = [p for p in suit_candidates if folder(p).startswith("female")]
    # Any-gender fallback if no match.
    if not suits:
        suits = suit_candidates

    shoes = [p for p in all_mhclo if "shoe" in folder(p)]
    hats = [p for p in all_mhclo if "fedora" in folder(p)]

    picked: list[str] = []
    if suits:
        picked.append(rng.choice(suits))
    if shoes:
        picked.append(rng.choice(shoes))
    if hats and rng.random() < 0.12:
        picked.append(rng.choice(hats))

    out: list[str] = []
    for p in picked:
        parts = p.replace("\\", "/").split("/")
        out.append(f"{parts[-2]}/{parts[-1]}")
    return out


def build_character(seed: int, with_assets: bool = True) -> tuple[object, object]:
    """Create a new MPFB human with default rig. Returns (basemesh, armature)."""
    rng = random.Random(seed)
    phenotype = sample_phenotype(rng)

    # Import here so this module can be read without MPFB installed.
    from mpfb.services.humanservice import HumanService  # type: ignore

    skin_fragment = _pick_skin_fragment(rng) if with_assets else ""
    hair_fragment = _pick_hair_fragment(rng) if with_assets else ""
    clothes_list = _pick_clothes_fragments(rng, phenotype.gender, phenotype.age) if with_assets else []

    info = {
        "name": f"subject_{seed:04d}",
        "phenotype": phenotype.to_mpfb(),
        "rig": "default",
        "eyes": "low-poly/low-poly.mhclo",
        "eyebrows": "",
        "eyelashes": "",
        "teeth": "teeth_base/teeth_base.mhclo",
        "tongue": "",
        "hair": hair_fragment,
        "proxy": "",
        "targets": [],
        "clothes": clothes_list,
        "skin_mhmat": skin_fragment,
        "skin_material_type": "ENHANCED_SSS",
        "eyes_material_type": "PROCEDURAL_EYES",
        "skin_material_settings": {},
        "eyes_material_settings": {},
        "alternative_materials": {},
    }

    settings = HumanService.get_default_deserialization_settings()
    # MPFB default scale=0.1 produces a ~1.8m tall character in metres.
    settings["feet_on_ground"] = True
    # subdiv_levels 1 → ~8k verts instead of ~30k at level 2.  At 256x192
    # render the level-2 refinement is literally sub-pixel (you cannot see
    # the difference), but BVH-refit cost scales ~linearly with tris, so
    # level 1 cuts per-frame refit ~3-4x.  Measured 15-30% speedup on
    # Cycles OPTIX at this resolution — biggest per-frame win after
    # character caching + HDRI-once-per-clip.
    settings["subdiv_levels"] = 1
    settings["load_clothes"] = bool(clothes_list)   # now TRUE so CC0 clothes load

    basemesh = HumanService.deserialize_from_dict(info, settings)
    garments = ", ".join(c.split("/")[-2] for c in clothes_list) or "-"
    print(f"[mpfb_build] created {info['name']} skin={skin_fragment or '-'} "
          f"hair={hair_fragment or '-'} clothes={garments}")

    # Clothing meshes are single-sided by default.  When the mesh stretches
    # during motion, the back faces render the FRONT texture from inside,
    # producing the "orange Y logo visible on back" artifact.  Enable
    # backface culling so only the outward-facing polygons render.
    _enable_backface_culling_on_clothes(info["name"])

    # Tune the skin material's Principled BSDF subsurface scattering so the
    # face/neck/hands read as translucent skin, not matte plastic.  BEDLAM-
    # class skin needs SSS weight ~0.15, radius (1.0, 0.6, 0.3) mm in scene
    # units, roughness ~0.4 for natural specular softness.
    _tune_skin_sss(info["name"])

    # Enable "Preserve Volume" (dual-quaternion skinning) on every Armature
    # modifier of body + clothing meshes.  Without this, linear blend
    # skinning scales down vertices near joints to ~zero at 180\xB0 bend —
    # visible as hard radial creases at the hip crotch and candy-wrapper
    # pinch at knees/elbows.  DQS replaces LBS with quaternion interpolation
    # that preserves volume around bends.
    # (Blender docs: "Use quaternions for preserving volume... Without it,
    # rotations at joints tend to scale down the neighboring geometry, up
    # to nearly zero at 180 degrees from rest position.")
    _enable_preserve_volume(info["name"])

    # MPFB ships a small clothing set (~8 suits, 6 pairs of shoes) with
    # FIXED albedo colours baked into each texture.  Without recolouring,
    # every `male_casualsuit03` subject looks the same grey/brown outfit.
    # We inject a per-character HSV shift into each clothing/hair material
    # so that every subject has a visually distinct outfit even when the
    # same mesh is re-used.
    _randomize_clothing_tint(info["name"], rng)
    _randomize_hair_tint(info["name"], rng)

    armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and obj.name.startswith(info["name"]):
            armature = obj
            break
    if armature is None:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE":
                armature = obj
                break

    if armature is not None:
        for pb in armature.pose.bones:
            pb.rotation_mode = "XYZ"

    return basemesh, armature


def _enable_preserve_volume(name_prefix: str) -> None:
    """Toggle use_deform_preserve_volume on every Armature modifier of
    the character's meshes.  Preserve Volume = dual-quaternion skinning."""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.name.startswith(name_prefix):
            continue
        for mod in obj.modifiers:
            if mod.type == "ARMATURE":
                mod.use_deform_preserve_volume = True


def _tune_skin_sss(name_prefix: str) -> None:
    """Increase subsurface scattering on the body (skin) material so the
    character's skin reads as translucent rather than matte paint.  Walks
    the body mesh's Principled BSDF and sets Subsurface Weight + Radius.
    """
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.name.startswith(name_prefix):
            continue
        # Target only the body mesh (skin), not clothes / eyes / teeth / hair.
        if not any(k in obj.name.lower() for k in ("body", "basemesh")):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes or not mat.node_tree:
                continue
            # Find any Principled BSDF (MPFB's ENHANCED_SSS wraps one).
            for node in mat.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue
                for name, value in (
                    ("Subsurface Weight", 0.15),
                    ("Subsurface Scale", 0.015),  # metres (1.5cm)
                    ("Roughness", 0.45),
                    ("Specular IOR Level", 0.40),
                ):
                    inp = node.inputs.get(name)
                    if inp is not None:
                        try:
                            inp.default_value = value
                        except Exception:
                            pass


# Colour families used for hair tinting.  Each entry is an (R, G, B) linear
# triplet; we jitter within ±15 % of these to produce natural variation.
_HAIR_PALETTE = [
    (0.02, 0.015, 0.010),   # near-black
    (0.05, 0.030, 0.018),   # dark brown
    (0.12, 0.070, 0.035),   # medium brown
    (0.28, 0.170, 0.080),   # light brown
    (0.55, 0.400, 0.200),   # dirty blonde
    (0.78, 0.620, 0.350),   # blonde
    (0.35, 0.090, 0.040),   # auburn / red
    (0.50, 0.150, 0.060),   # ginger
    (0.40, 0.400, 0.420),   # salt-and-pepper grey
    (0.75, 0.750, 0.780),   # silver-grey
]


def _iter_object_materials(name_prefix: str, keywords: tuple[str, ...]):
    """Yield (object, material) for every material on any mesh whose name
    starts with ``name_prefix`` and whose lowercased name contains any of
    ``keywords``.  Materials with no node tree are skipped.
    """
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.name.startswith(name_prefix):
            continue
        if not any(k in obj.name.lower() for k in keywords):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes or not mat.node_tree:
                continue
            yield obj, mat


def _inject_tint_multiply(mat, tint_rgba: tuple[float, float, float, float]) -> bool:
    """Insert a ShaderNodeMix (colour, MULTIPLY) between whatever drives
    the Principled BSDF's Base Color and the BSDF itself, so the final
    albedo is the original source multiplied by ``tint_rgba``.

    Multiply is the correct blend for tinting grayscale MPFB fabric
    textures toward a colour while preserving weave / shading detail.
    A pure HueSaturation rotation can't colourise a grayscale input,
    which is why the previous implementation left everything white.
    Returns True if the node graph was modified.
    """
    tree = mat.node_tree
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False
    base_inp = bsdf.inputs.get("Base Color")
    if base_inp is None:
        return False
    links = tree.links
    existing_link = next((l for l in links if l.to_socket is base_inp), None)

    # Blender 3.4+ unified node: ShaderNodeMix with data_type='RGBA'.
    # Fall back to the legacy MixRGB if the new node isn't available.
    try:
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        fac_in = mix.inputs["Factor"]
        a_in = mix.inputs[6]   # "A" (Color)
        b_in = mix.inputs[7]   # "B" (Color)
        out_sock = mix.outputs[2]   # "Result" (Color)
    except Exception:
        mix = tree.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        fac_in = mix.inputs["Fac"]
        a_in = mix.inputs["Color1"]
        b_in = mix.inputs["Color2"]
        out_sock = mix.outputs["Color"]

    mix.location = (bsdf.location.x - 260, bsdf.location.y)
    fac_in.default_value = 1.0
    b_in.default_value = tint_rgba

    if existing_link is not None:
        src_socket = existing_link.from_socket
        links.remove(existing_link)
        links.new(src_socket, a_in)
    else:
        try:
            col = list(base_inp.default_value)
            a_in.default_value = col
        except Exception:
            a_in.default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(out_sock, base_inp)
    return True


def _hsv_rgba(hue: float, sat: float, val: float) -> tuple[float, float, float, float]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, max(0.0, min(1.0, sat)),
                                    max(0.0, min(1.0, val)))
    return (r, g, b, 1.0)


def _inject_dual_tint_multiply(mat, tint_top, tint_bottom,
                                split: float = 0.5,
                                softness: float = 0.04) -> bool:
    """Multiply the Principled BSDF's Base Color by a tint that varies
    vertically: ``tint_top`` above the split, ``tint_bottom`` below it.

    Uses TextureCoordinate.Generated (0..1 along the object's bounding
    box), so the seam tracks the MESH's own extent — for an MPFB suit
    that spans shoulders-to-ankles, ``split=0.5`` lands near the waist.

    The ``softness`` parameter defines the soft-blend width (0..1) around
    the split so we don't get a razor-sharp horizontal line.

    This is what turns MPFB's single-material "unitard" suits into
    jacket + pants separates without ever modifying geometry.
    """
    tree = mat.node_tree
    bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False
    base_inp = bsdf.inputs.get("Base Color")
    if base_inp is None:
        return False
    links = tree.links
    existing_link = next((l for l in links if l.to_socket is base_inp), None)

    tc = tree.nodes.new("ShaderNodeTexCoord")
    tc.location = (bsdf.location.x - 900, bsdf.location.y + 150)
    sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    sep.location = (bsdf.location.x - 720, bsdf.location.y + 150)
    links.new(tc.outputs["Generated"], sep.inputs["Vector"])

    ramp = tree.nodes.new("ShaderNodeValToRGB")   # ColorRamp
    ramp.location = (bsdf.location.x - 540, bsdf.location.y + 150)
    ramp.color_ramp.interpolation = "LINEAR"
    ramp.color_ramp.elements[0].position = max(0.0, split - softness)
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = min(1.0, split + softness)
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(sep.outputs["Z"], ramp.inputs["Fac"])

    # Mix the two tint colours by the ramp mask.
    try:
        region_mix = tree.nodes.new("ShaderNodeMix")
        region_mix.data_type = "RGBA"
        region_mix.blend_type = "MIX"
        fac_in = region_mix.inputs["Factor"]
        a_in = region_mix.inputs[6]
        b_in = region_mix.inputs[7]
        region_out = region_mix.outputs[2]
    except Exception:
        region_mix = tree.nodes.new("ShaderNodeMixRGB")
        region_mix.blend_type = "MIX"
        fac_in = region_mix.inputs["Fac"]
        a_in = region_mix.inputs["Color1"]
        b_in = region_mix.inputs["Color2"]
        region_out = region_mix.outputs["Color"]
    region_mix.location = (bsdf.location.x - 360, bsdf.location.y + 150)
    a_in.default_value = tint_bottom   # fac=0 → pants colour
    b_in.default_value = tint_top      # fac=1 → jacket colour
    links.new(ramp.outputs["Color"], fac_in)

    # Multiply the per-region tint onto the source (texture or default).
    try:
        mult = tree.nodes.new("ShaderNodeMix")
        mult.data_type = "RGBA"
        mult.blend_type = "MULTIPLY"
        m_fac = mult.inputs["Factor"]
        m_a = mult.inputs[6]
        m_b = mult.inputs[7]
        m_out = mult.outputs[2]
    except Exception:
        mult = tree.nodes.new("ShaderNodeMixRGB")
        mult.blend_type = "MULTIPLY"
        m_fac = mult.inputs["Fac"]
        m_a = mult.inputs["Color1"]
        m_b = mult.inputs["Color2"]
        m_out = mult.outputs["Color"]
    mult.location = (bsdf.location.x - 180, bsdf.location.y)
    m_fac.default_value = 1.0
    links.new(region_out, m_b)

    if existing_link is not None:
        src_socket = existing_link.from_socket
        links.remove(existing_link)
        links.new(src_socket, m_a)
    else:
        try:
            col = list(base_inp.default_value)
            m_a.default_value = col
        except Exception:
            m_a.default_value = (1.0, 1.0, 1.0, 1.0)
    links.new(m_out, base_inp)
    return True


def _randomize_clothing_tint(name_prefix: str, rng: random.Random) -> None:
    """Apply a per-(garment, material-slot) colour to every clothing /
    accessory material.

    Key detail: MPFB suit meshes frequently bundle top + bottom (e.g.
    a jacket + pants) as ONE object with TWO material slots — one per
    garment region.  Tinting per material slot (instead of per object)
    therefore produces a natural "separates" look where the jacket and
    pants are different colours, breaking the all-one-colour unitard
    appearance we previously got.

    Shoes / hats / sports outfits that use a single material simply
    get one tint, the same as before.
    """
    # Suit meshes get DUAL tints (jacket + pants); shoes / hats / small
    # accessories stay single-tint because their bbox is too small/narrow
    # for a meaningful vertical split.
    suit_keywords = ("suit", "shirt", "sport")
    simple_keywords = ("shoes", "fedora", "hat", "shorts", "jeans", "pants")

    def _pick_random_tint() -> tuple[float, float, float, float]:
        hue = rng.random()
        if rng.random() < 0.15:   # occasional muted / near-greyscale outfit
            sat = rng.uniform(0.05, 0.25)
            val = rng.uniform(0.35, 0.95)
        else:
            sat = rng.uniform(0.45, 0.90)
            val = rng.uniform(0.40, 0.95)
        return _hsv_rgba(hue, sat, val)

    for obj, mat in _iter_object_materials(
            name_prefix, suit_keywords + simple_keywords):
        lower = obj.name.lower()
        is_suit = any(k in lower for k in suit_keywords)
        try:
            if is_suit:
                tint_top = _pick_random_tint()
                tint_bottom = _pick_random_tint()
                # Split position jitters around 0.5 so the visible seam
                # doesn't land at identical pixel rows on every subject;
                # softness is narrow so a belt-like transition appears
                # rather than a long vertical gradient.
                split = rng.uniform(0.42, 0.58)
                _inject_dual_tint_multiply(mat, tint_top, tint_bottom,
                                            split=split, softness=0.05)
            else:
                _inject_tint_multiply(mat, _pick_random_tint())
        except Exception as e:
            print(f"[mpfb_build] tint fail on {mat.name}: {e}")


def _randomize_hair_tint(name_prefix: str, rng: random.Random) -> None:
    """Pick a random palette colour and push it onto the hair material's
    Base Color.  Hair materials typically have no texture — a direct
    default_value write works; if a texture IS present we fall back to
    the HSV-shift approach.
    """
    hair_keywords = ("hair", "bob", "afro", "braid", "long", "ponytail",
                     "short")
    # One hair colour per character (all hair strands match).
    base = list(rng.choice(_HAIR_PALETTE))
    jitter = [rng.uniform(0.85, 1.15) for _ in range(3)]
    tinted = tuple(max(0.0, min(1.0, base[i] * jitter[i])) for i in range(3))
    rgba = (*tinted, 1.0)

    for _obj, mat in _iter_object_materials(name_prefix, hair_keywords):
        tree = mat.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        base_inp = bsdf.inputs.get("Base Color")
        if base_inp is None:
            continue
        has_link = any(l.to_socket is base_inp for l in tree.links)
        if has_link:
            # Hair material uses a texture: multiply it by the chosen hair
            # colour so the strand shading survives.
            try:
                _inject_tint_multiply(mat, rgba)
            except Exception:
                pass
        else:
            try:
                base_inp.default_value = rgba
            except Exception:
                pass


def _enable_backface_culling_on_clothes(name_prefix: str) -> None:
    """Enable backface culling on clothing mesh materials.  Without this the
    single-sided clothing meshes expose their UV'd front texture through the
    back when the mesh stretches — e.g. MPFB T-shirts show the front 'Y' logo
    as an orange stripe on the back in motion.
    """
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if not obj.name.startswith(name_prefix):
            continue
        # Skin / eyes / teeth / hair should stay double-sided (they're the
        # body geometry).  Only clothing meshes need culling.
        cloth_keywords = ("suit", "shirt", "pants", "shoes", "shorts",
                          "jeans", "sport")
        if not any(k in obj.name.lower() for k in cloth_keywords):
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None:
                continue
            mat.use_backface_culling = True


def _find_assets(subdir: str, extension: str) -> list[str]:
    """Scan MPFB user_data/<subdir> for files ending in `extension`."""
    try:
        from mpfb.services.locationservice import LocationService  # type: ignore
        from pathlib import Path as _P
        user_data = _P(LocationService.get_user_data())
        root = user_data / subdir
        if not root.exists():
            return []
        return sorted(str(p) for p in root.rglob(f"*{extension}"))
    except Exception as e:
        print(f"[mpfb_build] _find_assets({subdir},{extension}) failed: {e}")
        return []


def _attach_random_skin(basemesh, rng: random.Random) -> None:
    paths = _find_assets("skins", ".mhmat")
    if not paths:
        print("[mpfb_build] no skins found — character keeps default material")
        return
    mhmat = rng.choice(paths)
    try:
        from mpfb.services.humanservice import HumanService  # type: ignore
        HumanService.set_character_skin(mhmat, basemesh, skin_type="ENHANCED_SSS")
        print(f"[mpfb_build] skin: {mhmat.rsplit('/',2)[-2]}")
    except Exception as e:
        print(f"[mpfb_build] skin attach failed ({mhmat}): {e}")


def _attach_random_hair(basemesh, rng: random.Random) -> None:
    paths = [p for p in _find_assets("hair", ".mhclo")]
    if not paths:
        return
    mhclo = rng.choice(paths)
    try:
        from mpfb.services.humanservice import HumanService  # type: ignore
        HumanService.add_mhclo_asset(mhclo, basemesh, asset_type="hair",
                                     subdiv_levels=0, material_type="MAKESKIN")
        print(f"[mpfb_build] hair: {mhclo.rsplit('/',2)[-2]}")
    except Exception as e:
        print(f"[mpfb_build] hair attach failed: {e}")

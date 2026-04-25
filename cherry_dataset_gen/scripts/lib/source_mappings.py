"""Per-source BVH bone name -> MPFB2 default-rig bone mappings.

The MPFB default rig has twist-distribution sub-bones (upperarm01/02,
lowerarm01/02, upperleg01/02, lowerleg01/02).  Skin weights assume these
carry part of each joint's rotation — leaving them at rest causes mesh
tearing at the wrist / ankle (the "candy-wrapper" LBS artifact).

Each semantic joint maps to a LIST of target bones.  The retargeter
distributes the source rotation across the list via quaternion slerp
(1/N per bone) so that world orientation at the end of the chain equals
the source's joint orientation, with twist split evenly for clean skin.

MPFB arm chain:   shoulder01 -> upperarm01 -> upperarm02 -> lowerarm01 -> lowerarm02 -> wrist
MPFB leg chain:   pelvis.L   -> upperleg01 -> upperleg02 -> lowerleg01 -> lowerleg02 -> foot

clavicle / shoulder01 split is finicky; source "collar/shoulder" bones map
to [clavicle, shoulder01] so both bones absorb the shoulder girdle rotation.
"""
from __future__ import annotations


# ==============================================================================
# CMU Mocap (cgspeed/daz BVH)
# Hierarchy: Hips -> {LHipJoint, RHipJoint, LowerBack}
#            LowerBack -> Spine -> Spine1 -> {Neck, LeftShoulder, RightShoulder}
#            Neck -> Neck1 -> Head
#            LeftShoulder -> LeftArm -> LeftForeArm -> LeftHand
#            LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToeBase
# ==============================================================================
CMU_TO_MPFB_DEFAULT: dict[str, list[str]] = {
    "Hips":          ["root"],
    "LowerBack":     ["spine05"],
    "Spine":         ["spine03"],
    "Spine1":        ["spine01"],
    "Neck":          ["neck02"],
    "Neck1":         ["neck01"],
    "Head":          ["head"],
    "LeftShoulder":  ["clavicle.L", "shoulder01.L"],
    "LeftArm":       ["upperarm01.L", "upperarm02.L"],
    "LeftForeArm":   ["lowerarm01.L", "lowerarm02.L"],
    # LeftHand deliberately unmapped — CMU BVH doesn't animate the hand
    # meaningfully, so its rest orientation would persist through the clip
    # and look like "hand sticking out sideways".
    "RightShoulder": ["clavicle.R", "shoulder01.R"],
    "RightArm":      ["upperarm01.R", "upperarm02.R"],
    "RightForeArm":  ["lowerarm01.R", "lowerarm02.R"],
    "LeftUpLeg":     ["upperleg01.L", "upperleg02.L"],
    "LeftLeg":       ["lowerleg01.L", "lowerleg02.L"],
    "LeftFoot":      ["foot.L"],
    "LeftToeBase":   ["toe1-1.L"],
    "RightUpLeg":    ["upperleg01.R", "upperleg02.R"],
    "RightLeg":      ["lowerleg01.R", "lowerleg02.R"],
    "RightFoot":     ["foot.R"],
    "RightToeBase":  ["toe1-1.R"],
}
CMU_BVH_GLOBAL_SCALE = 0.0712


# ==============================================================================
# 100STYLE (Mason et al., SCA 2022)
# Hierarchy: Hips -> {Chest, RightHip, LeftHip}
#            Chest -> Chest2 -> Chest3 -> Chest4 -> {Neck, LeftCollar, RightCollar}
#            LeftCollar -> LeftShoulder -> LeftElbow -> LeftWrist
# ==============================================================================
MPFB_100STYLE_MAPPING: dict[str, list[str]] = {
    "Hips":          ["root"],
    "Chest":         ["spine05"],
    "Chest2":        ["spine04"],
    "Chest3":        ["spine03"],
    "Chest4":        ["spine01"],
    "Neck":          ["neck01"],
    "Head":          ["head"],
    "LeftCollar":    ["clavicle.L", "shoulder01.L"],
    "LeftShoulder":  ["upperarm01.L", "upperarm02.L"],
    "LeftElbow":     ["lowerarm01.L", "lowerarm02.L"],
    "LeftWrist":     ["wrist.L"],
    "RightCollar":   ["clavicle.R", "shoulder01.R"],
    "RightShoulder": ["upperarm01.R", "upperarm02.R"],
    "RightElbow":    ["lowerarm01.R", "lowerarm02.R"],
    "RightWrist":    ["wrist.R"],
    "LeftHip":       ["upperleg01.L", "upperleg02.L"],
    "LeftKnee":      ["lowerleg01.L", "lowerleg02.L"],
    "LeftAnkle":     ["foot.L"],
    "LeftToe":       ["toe1-1.L"],
    "RightHip":      ["upperleg01.R", "upperleg02.R"],
    "RightKnee":     ["lowerleg01.R", "lowerleg02.R"],
    "RightAnkle":    ["foot.R"],
    "RightToe":      ["toe1-1.R"],
}
MPFB_100STYLE_SCALE = 0.0112


# ==============================================================================
# AIST++ (SMPL→BVH via smpl2bvh; uses SMPL-24 skeleton)
# Joints: pelvis, spine{1,2,3}, l/r_{collar, shoulder, elbow, wrist, hip, knee,
#         ankle, foot}, neck, head
# ==============================================================================
AISTPP_TO_MPFB_DEFAULT: dict[str, list[str]] = {
    "pelvis":       ["root"],
    "spine1":       ["spine05"],
    "spine2":       ["spine03"],
    "spine3":       ["spine01"],
    "neck":         ["neck01"],
    "head":         ["head"],
    "l_collar":     ["clavicle.L", "shoulder01.L"],
    "l_shoulder":   ["upperarm01.L", "upperarm02.L"],
    "l_elbow":      ["lowerarm01.L", "lowerarm02.L"],
    "l_wrist":      ["wrist.L"],
    "r_collar":     ["clavicle.R", "shoulder01.R"],
    "r_shoulder":   ["upperarm01.R", "upperarm02.R"],
    "r_elbow":      ["lowerarm01.R", "lowerarm02.R"],
    "r_wrist":      ["wrist.R"],
    "l_hip":        ["upperleg01.L", "upperleg02.L"],
    "l_knee":       ["lowerleg01.L", "lowerleg02.L"],
    "l_ankle":      ["foot.L"],
    "l_foot":       ["toe1-1.L"],
    "r_hip":        ["upperleg01.R", "upperleg02.R"],
    "r_knee":       ["lowerleg01.R", "lowerleg02.R"],
    "r_ankle":      ["foot.R"],
    "r_foot":       ["toe1-1.R"],
}
AISTPP_SCALE = 1.122


# Placeholder for Berkeley MHAD
MHAD_TO_MPFB_DEFAULT: dict[str, list[str]] = {}
MHAD_SCALE = 0.001


def mapping_for_source(source: str) -> tuple[dict[str, list[str]], float]:
    """Return (bone_map, global_scale) for the given motion source.

    `bone_map` keys are source BVH bone names; values are LISTS of target
    MPFB bone names, ordered root-to-tip within the semantic joint.  The
    retargeter distributes rotation across the list to keep twist bones
    animated (needed for clean LBS at wrists/ankles).
    """
    if source == "cmu":
        return CMU_TO_MPFB_DEFAULT, CMU_BVH_GLOBAL_SCALE
    if source == "100style":
        return MPFB_100STYLE_MAPPING, MPFB_100STYLE_SCALE
    if source == "aistpp":
        return AISTPP_TO_MPFB_DEFAULT, AISTPP_SCALE
    if source == "mhad":
        return MHAD_TO_MPFB_DEFAULT, MHAD_SCALE
    raise KeyError(f"Unknown motion source: {source}")


def detect_source_from_bvh(bvh_path: str) -> str:
    """Heuristic: inspect the BVH header for diagnostic bone names."""
    try:
        with open(bvh_path, "r", errors="ignore") as f:
            head = f.read(4096)
    except Exception:
        return "cmu"
    if "Chest2" in head and "LeftCollar" in head:
        return "100style"
    if "LHipJoint" in head or "LowerBack" in head:
        return "cmu"
    if "l_collar" in head or "pelvis" in head.lower():
        return "aistpp"
    return "cmu"

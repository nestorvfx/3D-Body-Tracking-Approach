"""COCO-17 skeleton definition and MPFB2 default-rig → COCO-17 mapping.

COCO-17 keypoints (order matters — matches the MS COCO person keypoint format):
  0  nose
  1  left_eye
  2  right_eye
  3  left_ear
  4  right_ear
  5  left_shoulder
  6  right_shoulder
  7  left_elbow
  8  right_elbow
  9  left_wrist
  10 right_wrist
  11 left_hip
  12 right_hip
  13 left_knee
  14 right_knee
  15 left_ankle
  16 right_ankle
"""

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO17_SKELETON = [
    (5, 7), (7, 9),                      # left arm
    (6, 8), (8, 10),                     # right arm
    (5, 6),                              # shoulders
    (5, 11), (6, 12), (11, 12),          # torso
    (11, 13), (13, 15),                  # left leg
    (12, 14), (14, 16),                  # right leg
    (0, 1), (1, 3), (0, 2), (2, 4),      # head
]

# Flat list of the 17 MPFB bone names in COCO-17 order — used by camera_rig
# for fast keypoint-world-position queries during framing checks.
COCO17_BONES = [
    "head",         # 0  nose
    "eye.L",        # 1  left_eye
    "eye.R",        # 2  right_eye
    "eye.L",        # 3  left_ear (head-end of eye bone)
    "eye.R",        # 4  right_ear
    "clavicle.L",   # 5  left_shoulder
    "clavicle.R",   # 6  right_shoulder
    "lowerarm01.L", # 7  left_elbow
    "lowerarm01.R", # 8  right_elbow
    "wrist.L",      # 9  left_wrist
    "wrist.R",      # 10 right_wrist
    "upperleg01.L", # 11 left_hip
    "upperleg01.R", # 12 right_hip
    "lowerleg01.L", # 13 left_knee
    "lowerleg01.R", # 14 right_knee
    "foot.L",       # 15 left_ankle
    "foot.R",       # 16 right_ankle
]

# MPFB2 "default" rig bone names → COCO-17 slot.
# Strategy: each COCO joint reads either the HEAD or TAIL of the named bone.
# Verified against mpfb2/src/mpfb/data/rigs/default/rig.default.json (bone list)
# and MakeHuman default rig conventions.
#
# Format: coco_idx -> (bone_name, "head" | "tail")
MPFB_DEFAULT_TO_COCO17 = {
    0:  ("head",         "tail"),   # nose — head bone tail is approx face center
    1:  ("eye.L",        "tail"),   # left_eye
    2:  ("eye.R",        "tail"),   # right_eye
    3:  ("eye.L",        "head"),   # left_ear — approximation from eye bone head
    4:  ("eye.R",        "head"),   # right_ear
    5:  ("clavicle.L",   "tail"),   # left_shoulder (= upperarm01.L head)
    6:  ("clavicle.R",   "tail"),   # right_shoulder
    7:  ("lowerarm01.L", "head"),   # left_elbow
    8:  ("lowerarm01.R", "head"),   # right_elbow
    9:  ("wrist.L",      "head"),   # left_wrist
    10: ("wrist.R",      "head"),   # right_wrist
    11: ("upperleg01.L", "head"),   # left_hip
    12: ("upperleg01.R", "head"),   # right_hip
    13: ("lowerleg01.L", "head"),   # left_knee
    14: ("lowerleg01.R", "head"),   # right_knee
    15: ("foot.L",       "head"),   # left_ankle
    16: ("foot.R",       "head"),   # right_ankle
}

# Fallback candidates if the primary bone is missing. MPFB2 bone naming has
# evolved; tolerate alternatives rather than fail hard.
MPFB_BONE_ALIASES = {
    "eye.L":        ["eye.L", "eye_left"],
    "eye.R":        ["eye.R", "eye_right"],
    "clavicle.L":   ["clavicle.L", "shoulder01.L"],
    "clavicle.R":   ["clavicle.R", "shoulder01.R"],
    "lowerarm01.L": ["lowerarm01.L", "forearm.L"],
    "lowerarm01.R": ["lowerarm01.R", "forearm.R"],
    "wrist.L":      ["wrist.L", "hand.L"],
    "wrist.R":      ["wrist.R", "hand.R"],
    "upperleg01.L": ["upperleg01.L", "thigh.L"],
    "upperleg01.R": ["upperleg01.R", "thigh.R"],
    "lowerleg01.L": ["lowerleg01.L", "shin.L", "calf.L"],
    "lowerleg01.R": ["lowerleg01.R", "shin.R", "calf.R"],
    "foot.L":       ["foot.L"],
    "foot.R":       ["foot.R"],
    "head":         ["head"],
}


def resolve_bone(armature, wanted: str) -> str | None:
    """Return the first alias of `wanted` that exists on `armature`, else None."""
    pose_bones = armature.pose.bones
    for name in MPFB_BONE_ALIASES.get(wanted, [wanted]):
        if name in pose_bones:
            return name
    return None


def get_coco17_world(armature) -> list[tuple[float, float, float] | None]:
    """Return 17 world-space 3D positions (or None for missing bones) for the
    currently posed `armature` object. Assumes scene is at the desired frame."""
    import mathutils  # type: ignore  # Blender bundled
    mat = armature.matrix_world
    out: list[tuple[float, float, float] | None] = []
    for i in range(17):
        wanted, end = MPFB_DEFAULT_TO_COCO17[i]
        resolved = resolve_bone(armature, wanted)
        if resolved is None:
            out.append(None)
            continue
        pb = armature.pose.bones[resolved]
        local = pb.head if end == "head" else pb.tail
        world = mat @ local
        out.append((float(world.x), float(world.y), float(world.z)))
    return out


def project_to_pixels(scene, camera, world_points):
    """Project a list of world-space points to normalized image coords [0,1]
    via bpy_extras.world_to_camera_view. Returns list of (u, v, depth, inside)
    tuples, one per input (None-preserving)."""
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore
    import mathutils  # type: ignore
    results = []
    for p in world_points:
        if p is None:
            results.append(None)
            continue
        co = world_to_camera_view(scene, camera, mathutils.Vector(p))
        u, v, depth = float(co.x), float(co.y), float(co.z)
        inside = (0.0 <= u <= 1.0) and (0.0 <= v <= 1.0) and depth > 0.0
        # Blender's world_to_camera_view returns v with bottom-left origin;
        # flip to image-space (top-left origin) for downstream consumers.
        results.append((u, 1.0 - v, depth, inside))
    return results

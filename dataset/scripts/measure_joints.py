"""Measure source-vs-target joint angles using WORLD-SPACE BONE ALONG-VECTORS
only.  This is invariant to per-rig differences in bone head positions and
parent chain structure (MPFB has upperleg01/02/lowerleg01; CMU has
LeftUpLeg/LeftLeg).

Measurements:
  BEND at a joint = angle between parent_bone_along and child_bone_along
                    in world space.  Same for source and target.  If the
                    retarget is correct, delta should be near 0°.
  TWIST at a bone  = rotation around the bone's own along-axis relative
                    to its parent bone's tangent axis.

CSV columns: frame, joint, kind (bend|twist), src_deg, tgt_deg, delta_deg
"""
from __future__ import annotations

import csv
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
from lib.render_setup import clear_scene
from lib.source_mappings import detect_source_from_bvh


# For each joint, define (parent_bone_along, child_bone_along) PER RIG.
# BEND is measured as angle between these two vectors — so if retarget
# preserves bone directions, src and tgt give identical angles.
TGT_BEND = {
    "knee_L":     ("upperleg01.L", "lowerleg01.L"),
    "knee_R":     ("upperleg01.R", "lowerleg01.R"),
    "elbow_L":    ("upperarm01.L", "lowerarm01.L"),
    "elbow_R":    ("upperarm01.R", "lowerarm01.R"),
    "hip_L":      ("spine05",      "upperleg01.L"),
    "hip_R":      ("spine05",      "upperleg01.R"),
    "shoulder_L": ("spine01",      "upperarm01.L"),
    "shoulder_R": ("spine01",      "upperarm01.R"),
    "ankle_L":    ("lowerleg01.L", "foot.L"),
    "ankle_R":    ("lowerleg01.R", "foot.R"),
    "spine_mid":  ("spine05",      "spine03"),
    # `neck` is the bend at the base of the neck.  MPFB has neck01 + neck02;
    # different sources map to different MPFB bones (CMU: Neck→neck02,
    # AIST++: neck→neck01, 100STYLE: Neck→neck01).  Per-source neck target
    # bone is specified in SRC_NECK_TGT below.
}
# Per-source target bone for the "neck" joint.
SRC_NECK_TGT = {"cmu": "neck02", "aistpp": "neck01", "100style": "neck01"}

SRC_BEND = {
    "cmu": {
        "knee_L":     ("LeftUpLeg",  "LeftLeg"),
        "knee_R":     ("RightUpLeg", "RightLeg"),
        "elbow_L":    ("LeftArm",    "LeftForeArm"),
        "elbow_R":    ("RightArm",   "RightForeArm"),
        "hip_L":      ("LowerBack",  "LeftUpLeg"),
        "hip_R":      ("LowerBack",  "RightUpLeg"),
        "shoulder_L": ("Spine1",     "LeftArm"),
        "shoulder_R": ("Spine1",     "RightArm"),
        "ankle_L":    ("LeftLeg",    "LeftFoot"),
        "ankle_R":    ("RightLeg",   "RightFoot"),
        "spine_mid":  ("LowerBack",  "Spine"),
        "neck":       ("Spine1",     "Neck"),
    },
    "aistpp": {
        "knee_L":     ("l_hip",     "l_knee"),
        "knee_R":     ("r_hip",     "r_knee"),
        "elbow_L":    ("l_shoulder","l_elbow"),
        "elbow_R":    ("r_shoulder","r_elbow"),
        "hip_L":      ("spine1",    "l_hip"),
        "hip_R":      ("spine1",    "r_hip"),
        "shoulder_L": ("spine3",    "l_shoulder"),
        "shoulder_R": ("spine3",    "r_shoulder"),
        "ankle_L":    ("l_knee",    "l_ankle"),
        "ankle_R":    ("r_knee",    "r_ankle"),
        "spine_mid":  ("spine1",    "spine2"),
        "neck":       ("spine3",    "neck"),
    },
    "100style": {
        "knee_L":     ("LeftHip",    "LeftKnee"),
        "knee_R":     ("RightHip",   "RightKnee"),
        "elbow_L":    ("LeftShoulder","LeftElbow"),
        "elbow_R":    ("RightShoulder","RightElbow"),
        "hip_L":      ("Chest",      "LeftHip"),
        "hip_R":      ("Chest",      "RightHip"),
        "shoulder_L": ("Chest4",     "LeftShoulder"),
        "shoulder_R": ("Chest4",     "RightShoulder"),
        "ankle_L":    ("LeftKnee",   "LeftAnkle"),
        "ankle_R":    ("RightKnee",  "RightAnkle"),
        "spine_mid":  ("Chest",      "Chest2"),
        "neck":       ("Chest4",     "Neck"),
    },
}

TWIST_JOINTS = ["hip_L", "hip_R", "shoulder_L", "shoulder_R"]
TGT_TWIST_BONE = {
    "hip_L":       "upperleg01.L",
    "hip_R":       "upperleg01.R",
    "shoulder_L":  "upperarm01.L",
    "shoulder_R":  "upperarm01.R",
}
# End-of-chain twist bones — where swing-twist decomposition puts the twist.
# If the overall chain rotation was preserved, measuring twist on the END
# of the chain (after primary + sub-bones) should show a near-zero delta.
TGT_CHAIN_END = {
    "hip_L":       "lowerleg01.L",
    "hip_R":       "lowerleg01.R",
    "shoulder_L":  "lowerarm01.L",
    "shoulder_R":  "lowerarm01.R",
}
SRC_TWIST_BONE = {
    "cmu": {
        "hip_L": "LeftUpLeg", "hip_R": "RightUpLeg",
        "shoulder_L": "LeftArm", "shoulder_R": "RightArm",
    },
    "aistpp": {
        "hip_L": "l_hip", "hip_R": "r_hip",
        "shoulder_L": "l_shoulder", "shoulder_R": "r_shoulder",
    },
    "100style": {
        "hip_L": "LeftHip", "hip_R": "RightHip",
        "shoulder_L": "LeftShoulder", "shoulder_R": "RightShoulder",
    },
}


def along(arm, name: str) -> mathutils.Vector | None:
    pb = arm.pose.bones.get(name)
    if pb is None:
        return None
    mw = arm.matrix_world
    v = (mw @ pb.tail) - (mw @ pb.head)
    if v.length < 1e-5:
        return None
    return v.normalized()


def angle_between(a: mathutils.Vector, b: mathutils.Vector) -> float:
    d = max(-1.0, min(1.0, a.dot(b)))
    return math.degrees(math.acos(d))


def twist_deg(arm, bone_name: str) -> float | None:
    """Signed twist around bone's along-axis, referenced to world +Z
    projected perpendicular to the bone's along-axis.

    Rig-invariant: both source and target rigs can have different rest-pose
    X-axis orientations, but projecting world +Z gives the same reference
    for both.  If the retarget preserves the bone's full rotation, src and
    tgt twist measurements should match within a few degrees.
    """
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return None
    mw = arm.matrix_world
    mat = (mw @ pb.matrix).to_3x3()
    along_v = mat.col[1].normalized()
    tangent = mat.col[0].normalized()

    # World +Z projected onto plane perpendicular to along_v.
    world_up = mathutils.Vector((0.0, 0.0, 1.0))
    ref_proj = world_up - world_up.dot(along_v) * along_v
    if ref_proj.length < 1e-3:
        # Bone is nearly vertical — use world +Y as fallback.
        world_fwd = mathutils.Vector((0.0, 1.0, 0.0))
        ref_proj = world_fwd - world_fwd.dot(along_v) * along_v
        if ref_proj.length < 1e-3:
            return None
    ref_proj.normalize()
    d = max(-1.0, min(1.0, tangent.dot(ref_proj)))
    ang = math.degrees(math.acos(d))
    if ref_proj.cross(tangent).dot(along_v) < 0:
        ang = -ang
    return ang


def signed_delta(a: float, b: float) -> float:
    d = a - b
    while d > 180: d -= 360
    while d < -180: d += 360
    return d


def sample_frames(bvh, n: int) -> list[int]:
    if bvh.animation_data and bvh.animation_data.action:
        fr = bvh.animation_data.action.frame_range
        s, e = int(fr[0]), int(fr[1])
    else:
        s, e = 1, 100
    if e <= s or n <= 1:
        return [s]
    step = max(1, (e - s) // (n - 1))
    frames = list(range(s, e + 1, step))[:n]
    if frames and frames[-1] != e:
        frames[-1] = e
    return frames


def main():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    bvh_path = args[0]
    out_csv = Path(args[1])
    n_frames = int(args[2]) if len(args) > 2 else 20
    seed = int(args[3]) if len(args) > 3 else 42

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clear_scene()
    src_kind = detect_source_from_bvh(bvh_path)
    src_map = SRC_BEND.get(src_kind)
    src_twist_map = SRC_TWIST_BONE.get(src_kind, {})
    if src_map is None:
        print(f"unknown source: {src_kind}")
        sys.exit(1)

    bvh = load_bvh(bvh_path, source=src_kind)
    bm, arm = build_character(seed, with_assets=False)
    ctx = RetargetContext(bvh, arm)
    frames = sample_frames(bvh, n_frames)
    print(f"[measure] {bvh_path} ({src_kind}), seed={seed}, n={len(frames)}")

    rows = []
    for f in frames:
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        src_eval = bvh.evaluated_get(dg)

        src_bend = {}
        for joint, (pn, cn) in src_map.items():
            pa = along(src_eval, pn)
            ca = along(src_eval, cn)
            if pa is None or ca is None:
                continue
            src_bend[joint] = angle_between(pa, ca)

        src_tw = {}
        for joint in TWIST_JOINTS:
            bn = src_twist_map.get(joint)
            if bn is None:
                continue
            t = twist_deg(src_eval, bn)
            if t is not None:
                src_tw[joint] = t

        ctx.apply_pose(f)
        bpy.context.view_layer.update()

        for joint, (pn, cn) in TGT_BEND.items():
            if joint not in src_bend:
                continue
            pa = along(arm, pn); ca = along(arm, cn)
            if pa is None or ca is None:
                continue
            tgt_a = angle_between(pa, ca)
            rows.append({
                "frame": f, "joint": joint, "kind": "bend",
                "src_deg": round(src_bend[joint], 2),
                "tgt_deg": round(tgt_a, 2),
                "delta_deg": round(abs(tgt_a - src_bend[joint]), 2),
            })

        # neck (source-specific target bone)
        if "neck" in src_bend:
            neck_tgt = SRC_NECK_TGT.get(src_kind, "neck01")
            pa = along(arm, "spine01"); ca = along(arm, neck_tgt)
            if pa is not None and ca is not None:
                tgt_a = angle_between(pa, ca)
                rows.append({
                    "frame": f, "joint": "neck", "kind": "bend",
                    "src_deg": round(src_bend["neck"], 2),
                    "tgt_deg": round(tgt_a, 2),
                    "delta_deg": round(abs(tgt_a - src_bend["neck"]), 2),
                })

        for joint in TWIST_JOINTS:
            if joint not in src_tw:
                continue
            tgt_name = TGT_TWIST_BONE[joint]
            tt = twist_deg(arm, tgt_name)
            if tt is None:
                continue
            rows.append({
                "frame": f, "joint": joint, "kind": "twist",
                "src_deg": round(src_tw[joint], 2),
                "tgt_deg": round(tt, 2),
                "delta_deg": round(abs(signed_delta(tt, src_tw[joint])), 2),
            })

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "joint", "kind",
                                           "src_deg", "tgt_deg", "delta_deg"])
        w.writeheader()
        w.writerows(rows)

    by_jk: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by_jk.setdefault((r["joint"], r["kind"]), []).append(r["delta_deg"])

    print(f"\n=== Joint-angle delta summary ===")
    print(f"{'joint':<22s}  {'kind':<6s}  {'mean':>7s}  {'max':>7s}  {'n':>4s}  status")
    any_fail = False
    for (joint, kind), deltas in sorted(by_jk.items()):
        mean = sum(deltas) / len(deltas)
        mx = max(deltas)
        red = (mean > 5.0) or (mx > 15.0)
        any_fail = any_fail or red
        print(f"{joint:<22s}  {kind:<6s}  {mean:>7.2f}  {mx:>7.2f}  "
              f"{len(deltas):>4d}  {'FAIL' if red else 'ok'}")
    print(f"\nCSV: {out_csv}")
    print(f"OVERALL: {'FAIL' if any_fail else 'PASS'}")


if __name__ == "__main__":
    main()

"""Blender-side retargeting validator — no rendering, just numerical QC.

For each motion clip:
  1. Import BVH source armature.
  2. Build a minimal MPFB character + run our RetargetContext pipeline.
  3. Sample N frames; read world-space joint positions from BOTH armatures
     at each frame (source and retargeted target).
  4. Extract a common joint subset (15 canonical joints).
  5. Procrustes-align target points onto source (Umeyama, with scale).
  6. Compute per-joint MPJPE (mm), bone-length drift, per-frame worst joint.
  7. Dump `retarget_report_<tag>.json` with per-clip + per-joint metrics.

Cost: ~1-2 s per clip (20 frame samples × 2 armatures × ~50 ms per
`frame_set`).  Thousands of clips per hour on one machine.

Usage (from repo root):
    "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
        --background --python dataset/scripts/retarget_validator.py -- \
        <output_tag> [max_clips_per_source]
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy  # type: ignore
import numpy as np  # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ensure_mpfb_enabled() -> None:
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except Exception:
        pass
    import importlib, sys as _sys
    try:
        pkg = importlib.import_module("bl_ext.user_default.mpfb")
    except Exception:
        return
    _sys.modules["mpfb"] = pkg
    for svc in ("humanservice", "targetservice", "assetservice", "locationservice"):
        try:
            _sys.modules[f"mpfb.services.{svc}"] = importlib.import_module(
                f"bl_ext.user_default.mpfb.services.{svc}")
        except Exception:
            pass


ensure_mpfb_enabled()

from lib.mpfb_build import build_character                 # noqa: E402
from lib.fk_retarget import FKRetargetContext, load_bvh     # noqa: E402
from lib.render_setup import clear_scene                    # noqa: E402
from lib.motion_loader import load_all_clips                # noqa: E402
from lib.source_mappings import detect_source_from_bvh       # noqa: E402


REPO = HERE.parent
ASSETS = REPO / "assets"
OUT_DIR = REPO / "output" / "retarget_validation"


# ------------------------- Joint-mapping tables -------------------------
#
# For each source schema, map a canonical 15-joint skeleton to:
#   (source_bone_name, target_mpfb_bone_name, "head"|"tail")
#
# We intentionally exclude toes/fingers (no CMU equivalents across sources)
# and head/neck sub-bones (vary per rig).

CANONICAL_JOINTS = [
    "pelvis", "spine_mid",
    "neck", "head",
    "l_shoulder", "r_shoulder",
    "l_elbow", "r_elbow",
    "l_wrist", "r_wrist",
    "l_hip", "r_hip",
    "l_knee", "r_knee",
    "l_ankle", "r_ankle",
]


# Bone-direction pairs: each entry defines a "bone" as
#   (source_parent_joint, source_child_joint) -> (target_parent, target_child)
# Each pair gives an angular error metric that is invariant to bone length.
# These are the most-watched bones for retargeting fidelity.
ANGULAR_BONES = [
    "spine_lower",   # pelvis -> spine_mid
    "spine_upper",   # spine_mid -> neck
    "head",          # neck -> head
    "l_upperarm", "r_upperarm",   # shoulder -> elbow
    "l_forearm",  "r_forearm",    # elbow -> wrist
    "l_thigh",    "r_thigh",      # hip -> knee
    "l_shin",     "r_shin",       # knee -> ankle
]
ANGULAR_BONE_PAIRS = {
    "spine_lower": ("pelvis",     "spine_mid"),
    "spine_upper": ("spine_mid",  "neck"),
    "head":        ("neck",       "head"),
    "l_upperarm":  ("l_shoulder", "l_elbow"),
    "r_upperarm":  ("r_shoulder", "r_elbow"),
    "l_forearm":   ("l_elbow",    "l_wrist"),
    "r_forearm":   ("r_elbow",    "r_wrist"),
    "l_thigh":     ("l_hip",      "l_knee"),
    "r_thigh":     ("r_hip",      "r_knee"),
    "l_shin":      ("l_knee",     "l_ankle"),
    "r_shin":      ("r_knee",     "r_ankle"),
}

# Each entry: (source_bone, source_end, target_bone, target_end)
CMU_MAP = {
    "pelvis":      ("Hips",          "head", "root",          "head"),
    "spine_mid":   ("Spine",         "head", "spine02",       "head"),
    "neck":        ("Neck",          "head", "neck01",        "head"),
    "head":        ("Head",          "head", "head",          "head"),
    "l_shoulder":  ("LeftShoulder",  "tail", "clavicle.L",    "tail"),
    "r_shoulder":  ("RightShoulder", "tail", "clavicle.R",    "tail"),
    "l_elbow":     ("LeftForeArm",   "head", "lowerarm01.L",  "head"),
    "r_elbow":     ("RightForeArm",  "head", "lowerarm01.R",  "head"),
    "l_wrist":     ("LeftHand",      "head", "wrist.L",       "head"),
    "r_wrist":     ("RightHand",     "head", "wrist.R",       "head"),
    "l_hip":       ("LeftUpLeg",     "head", "upperleg01.L",  "head"),
    "r_hip":       ("RightUpLeg",    "head", "upperleg01.R",  "head"),
    "l_knee":      ("LeftLeg",       "head", "lowerleg01.L",  "head"),
    "r_knee":      ("RightLeg",      "head", "lowerleg01.R",  "head"),
    "l_ankle":     ("LeftFoot",      "head", "foot.L",        "head"),
    "r_ankle":     ("RightFoot",     "head", "foot.R",        "head"),
}

STYLE100_MAP = {
    "pelvis":      ("Hips",          "head", "root",          "head"),
    "spine_mid":   ("Chest2",        "head", "spine02",       "head"),
    "neck":        ("Neck",          "head", "neck01",        "head"),
    "head":        ("Head",          "head", "head",          "head"),
    "l_shoulder":  ("LeftCollar",    "tail", "clavicle.L",    "tail"),
    "r_shoulder":  ("RightCollar",   "tail", "clavicle.R",    "tail"),
    "l_elbow":     ("LeftElbow",     "head", "lowerarm01.L",  "head"),
    "r_elbow":     ("RightElbow",    "head", "lowerarm01.R",  "head"),
    "l_wrist":     ("LeftWrist",     "head", "wrist.L",       "head"),
    "r_wrist":     ("RightWrist",    "head", "wrist.R",       "head"),
    "l_hip":       ("LeftHip",       "head", "upperleg01.L",  "head"),
    "r_hip":       ("RightHip",      "head", "upperleg01.R",  "head"),
    "l_knee":      ("LeftKnee",      "head", "lowerleg01.L",  "head"),
    "r_knee":      ("RightKnee",     "head", "lowerleg01.R",  "head"),
    "l_ankle":     ("LeftAnkle",     "head", "foot.L",        "head"),
    "r_ankle":     ("RightAnkle",    "head", "foot.R",        "head"),
}

AISTPP_MAP = {
    "pelvis":      ("pelvis",     "head", "root",          "head"),
    "spine_mid":   ("spine2",     "head", "spine02",       "head"),
    "neck":        ("neck",       "head", "neck01",        "head"),
    "head":        ("head",       "head", "head",          "head"),
    "l_shoulder":  ("l_collar",   "tail", "clavicle.L",    "tail"),
    "r_shoulder":  ("r_collar",   "tail", "clavicle.R",    "tail"),
    "l_elbow":     ("l_elbow",    "head", "lowerarm01.L",  "head"),
    "r_elbow":     ("r_elbow",    "head", "lowerarm01.R",  "head"),
    "l_wrist":     ("l_wrist",    "head", "wrist.L",       "head"),
    "r_wrist":     ("r_wrist",    "head", "wrist.R",       "head"),
    "l_hip":       ("l_hip",      "head", "upperleg01.L",  "head"),
    "r_hip":       ("r_hip",      "head", "upperleg01.R",  "head"),
    "l_knee":      ("l_knee",     "head", "lowerleg01.L",  "head"),
    "r_knee":      ("r_knee",     "head", "lowerleg01.R",  "head"),
    "l_ankle":     ("l_ankle",    "head", "foot.L",        "head"),
    "r_ankle":     ("r_ankle",    "head", "foot.R",        "head"),
}

SOURCE_MAPS = {"cmu": CMU_MAP, "100style": STYLE100_MAP, "aistpp": AISTPP_MAP}


# ------------------------- Helpers -------------------------

def _bone_world_point(arm, bone_name, end):
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return None
    mw = arm.matrix_world
    p = pb.head if end == "head" else pb.tail
    return tuple(mw @ p)


def _sample_joints(arm, bone_spec, frames):
    """Return array (T, J, 3) sampled at given frames.  `bone_spec` is a list
    of (bone_name, end).  Missing bones → NaN row that is excluded later."""
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    out = np.full((len(frames), len(bone_spec), 3), np.nan, dtype=np.float64)
    for fi, f in enumerate(frames):
        scene.frame_set(int(f))
        bpy.context.view_layer.update()
        arm_eval = arm.evaluated_get(dg)
        mw = arm_eval.matrix_world
        for bi, (bname, end) in enumerate(bone_spec):
            pb = arm_eval.pose.bones.get(bname)
            if pb is None:
                continue
            p = pb.head if end == "head" else pb.tail
            w = mw @ p
            out[fi, bi] = (float(w.x), float(w.y), float(w.z))
    return out


def umeyama(P, Q, with_scale=True):
    """P, Q: (N, 3) matched points.  Return (s, R, t) such that
    s * R @ P.T + t[:, None] best matches Q.T (least squares)."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    mu_P = P.mean(0)
    mu_Q = Q.mean(0)
    Pc = P - mu_P
    Qc = Q - mu_Q
    H = Pc.T @ Qc / P.shape[0]
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = (Vt.T @ S @ U.T).astype(np.float64)
    var_P = (Pc ** 2).sum() / P.shape[0]
    s = float((D * np.diag(S)).sum() / max(1e-12, var_P)) if with_scale else 1.0
    t = mu_Q - s * R @ mu_P
    return s, R, t


def apply_rt(points, s, R, t):
    """Apply (s, R, t) to (..., 3) points."""
    shape = points.shape
    flat = points.reshape(-1, 3)
    out = (s * (R @ flat.T)).T + t
    return out.reshape(shape)


def bone_length_drift(arm, frames, bones):
    """Max relative drift of rest-length across sampled frames for given bones."""
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    lengths: dict[str, list[float]] = {b: [] for b in bones}
    for f in frames:
        scene.frame_set(int(f))
        bpy.context.view_layer.update()
        arm_eval = arm.evaluated_get(dg)
        for b in bones:
            pb = arm_eval.pose.bones.get(b)
            if pb is None:
                continue
            lengths[b].append((pb.tail - pb.head).length)
    drift_pcts = []
    for vs in lengths.values():
        if len(vs) < 2:
            continue
        mu = float(np.mean(vs))
        if mu < 1e-6:
            continue
        std = float(np.std(vs))
        drift_pcts.append(100.0 * std / mu)
    return max(drift_pcts) if drift_pcts else 0.0


# ------------------------- Per-clip validation -------------------------

def validate_clip(clip, source: str, n_samples: int = 20) -> dict:
    """Load a BVH, retarget to a fresh MPFB character, compute metrics."""
    jmap = SOURCE_MAPS.get(source)
    if jmap is None:
        return {"error": f"no map for source={source}"}

    clear_scene()
    basemesh, armature = build_character(seed=777, with_assets=False)
    if armature is None:
        return {"error": "no MPFB armature"}

    bvh_arm = load_bvh(str(clip.path), source=source)
    ctx = FKRetargetContext(bvh_arm, armature)

    fr = ctx.frame_range
    if fr[1] <= fr[0] + 5:
        return {"error": f"clip too short: {fr}"}
    frames = np.linspace(fr[0] + 2, fr[1] - 2, n_samples, dtype=int)

    src_spec = [(jmap[j][0], jmap[j][1]) for j in CANONICAL_JOINTS]
    tgt_spec = [(jmap[j][2], jmap[j][3]) for j in CANONICAL_JOINTS]

    # Sample source + target TOGETHER per frame, because FK writes to
    # target are transient (reset by the next frame_set).  We must call
    # apply_pose(f) between frame_set and bone-read.
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    src = np.full((len(frames), len(src_spec), 3), np.nan, dtype=np.float64)
    tgt = np.full((len(frames), len(tgt_spec), 3), np.nan, dtype=np.float64)

    for fi, f in enumerate(frames):
        ctx.apply_pose(int(f))                     # sets frame AND writes FK

        # Read source joints
        src_eval = bvh_arm.evaluated_get(dg)
        src_mw = src_eval.matrix_world
        for bi, (bname, end) in enumerate(src_spec):
            pb = src_eval.pose.bones.get(bname)
            if pb is None:
                continue
            p = pb.head if end == "head" else pb.tail
            w = src_mw @ p
            src[fi, bi] = (float(w.x), float(w.y), float(w.z))

        # Read target joints (after FK has been applied)
        tgt_eval = armature.evaluated_get(dg)
        tgt_mw_r = tgt_eval.matrix_world
        for bi, (bname, end) in enumerate(tgt_spec):
            pb = tgt_eval.pose.bones.get(bname)
            if pb is None:
                continue
            p = pb.head if end == "head" else pb.tail
            w = tgt_mw_r @ p
            tgt[fi, bi] = (float(w.x), float(w.y), float(w.z))

    # Mask out NaN rows (missing joints on either side)
    valid_mask = ~(np.isnan(src).any(axis=-1) | np.isnan(tgt).any(axis=-1))
    n_valid = int(valid_mask.sum())
    if n_valid < 10:
        ctx.cleanup()
        return {"error": f"too few valid joint samples: {n_valid}"}

    # Flatten valid joint samples for Procrustes alignment
    s_flat = src[valid_mask]
    t_flat = tgt[valid_mask]
    s_u, R_u, t_u = umeyama(t_flat, s_flat, with_scale=True)
    tgt_aligned = np.full_like(tgt, np.nan)
    tgt_aligned[valid_mask] = apply_rt(t_flat, s_u, R_u, t_u)

    err_mm = np.linalg.norm(src - tgt_aligned, axis=-1) * 1000.0  # (T, J)
    err_valid = err_mm[valid_mask]

    # Per-joint errors (averaged over valid frames only)
    per_joint_mean: dict[str, float] = {}
    for j, name in enumerate(CANONICAL_JOINTS):
        col_valid = valid_mask[:, j]
        if col_valid.sum() == 0:
            per_joint_mean[name] = float("nan")
        else:
            per_joint_mean[name] = float(err_mm[:, j][col_valid].mean())

    worst_joint = max(
        (v, k) for k, v in per_joint_mean.items() if not math.isnan(v)
    )[1] if per_joint_mean else "?"

    # Bone-direction angular error (independent of bone length, scale, rest pose):
    # for each "bone" defined as parent_joint -> child_joint, compute the
    # unit vector on source vs target, take arccos(dot).  This directly
    # measures rotational fidelity.
    joint_idx = {n: i for i, n in enumerate(CANONICAL_JOINTS)}

    def _angular_err(bone_name: str) -> float:
        a, b = ANGULAR_BONE_PAIRS[bone_name]
        ai, bi = joint_idx[a], joint_idx[b]
        sv = src[:, bi] - src[:, ai]
        tv = tgt[:, bi] - tgt[:, ai]
        sn = np.linalg.norm(sv, axis=-1, keepdims=True)
        tn = np.linalg.norm(tv, axis=-1, keepdims=True)
        ok = (sn[:, 0] > 1e-6) & (tn[:, 0] > 1e-6) & valid_mask[:, ai] & valid_mask[:, bi]
        if ok.sum() == 0:
            return float("nan")
        sv_u = sv[ok] / sn[ok]
        tv_u = tv[ok] / tn[ok]
        cos = np.clip((sv_u * tv_u).sum(-1), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)).mean())

    per_bone_angular = {b: _angular_err(b) for b in ANGULAR_BONES}
    angular_valid = [v for v in per_bone_angular.values() if not math.isnan(v)]
    angular_mean = float(np.mean(angular_valid)) if angular_valid else float("nan")
    worst_bone = max(
        ((v, k) for k, v in per_bone_angular.items() if not math.isnan(v)),
        default=(0.0, "?"),
    )[1]

    # --- FLEX ANGLE comparison (the correct FK fidelity metric) ---
    # For each joint (e.g., elbow): compute the angle BETWEEN parent bone
    # and child bone on source vs target separately.  If FK is correct, these
    # angles should match (regardless of rest-pose axis differences).
    #
    # Flex angle = arccos(dot(parent_dir, child_dir))
    # where parent_dir = normalize(joint - parent_joint)
    #       child_dir  = normalize(child_joint - joint)
    FLEX_JOINTS = {
        # name: (parent_joint, this_joint, child_joint)
        "l_elbow_flex": ("l_shoulder", "l_elbow", "l_wrist"),
        "r_elbow_flex": ("r_shoulder", "r_elbow", "r_wrist"),
        "l_knee_flex":  ("l_hip",      "l_knee",  "l_ankle"),
        "r_knee_flex":  ("r_hip",      "r_knee",  "r_ankle"),
        "neck_flex":    ("spine_mid",  "neck",    "head"),
    }

    def _flex_err(name: str) -> float:
        p, j, c = FLEX_JOINTS[name]
        pi, ji, ci = joint_idx[p], joint_idx[j], joint_idx[c]
        # Source flexion: angle at j between (j-p) and (c-j)
        s_pj = src[:, ji] - src[:, pi]
        s_jc = src[:, ci] - src[:, ji]
        t_pj = tgt[:, ji] - tgt[:, pi]
        t_jc = tgt[:, ci] - tgt[:, ji]
        ok_s = (np.linalg.norm(s_pj, axis=-1) > 1e-6) & (np.linalg.norm(s_jc, axis=-1) > 1e-6)
        ok_t = (np.linalg.norm(t_pj, axis=-1) > 1e-6) & (np.linalg.norm(t_jc, axis=-1) > 1e-6)
        ok = ok_s & ok_t & valid_mask[:, pi] & valid_mask[:, ji] & valid_mask[:, ci]
        if ok.sum() == 0:
            return float("nan")
        s_pj_u = s_pj[ok] / np.linalg.norm(s_pj[ok], axis=-1, keepdims=True)
        s_jc_u = s_jc[ok] / np.linalg.norm(s_jc[ok], axis=-1, keepdims=True)
        t_pj_u = t_pj[ok] / np.linalg.norm(t_pj[ok], axis=-1, keepdims=True)
        t_jc_u = t_jc[ok] / np.linalg.norm(t_jc[ok], axis=-1, keepdims=True)
        src_flex = np.degrees(np.arccos(np.clip(
            (s_pj_u * s_jc_u).sum(-1), -1.0, 1.0)))
        tgt_flex = np.degrees(np.arccos(np.clip(
            (t_pj_u * t_jc_u).sum(-1), -1.0, 1.0)))
        return float(np.abs(src_flex - tgt_flex).mean())

    per_joint_flex = {k: _flex_err(k) for k in FLEX_JOINTS}
    flex_valid = [v for v in per_joint_flex.values() if not math.isnan(v)]
    flex_mean = float(np.mean(flex_valid)) if flex_valid else float("nan")

    # Bone-length drift on target rig
    tgt_bones = [spec[2] for spec in (jmap[j] for j in CANONICAL_JOINTS)]
    drift_pct = bone_length_drift(armature, frames, tgt_bones)

    ctx.cleanup()

    return {
        "clip": clip.path.name,
        "source": source,
        "n_frames_sampled": int(len(frames)),
        "n_valid_samples": n_valid,
        "pa_mpjpe_mean_mm": float(err_valid.mean()),
        "pa_mpjpe_p50_mm": float(np.percentile(err_valid, 50)),
        "pa_mpjpe_p95_mm": float(np.percentile(err_valid, 95)),
        "pa_mpjpe_p99_mm": float(np.percentile(err_valid, 99)),
        "worst_joint": worst_joint,
        "worst_joint_err_mm": float(max(v for v in per_joint_mean.values()
                                          if not math.isnan(v))),
        "per_joint_mean_mm": per_joint_mean,
        "angular_err_mean_deg": angular_mean,
        "per_bone_angular_deg": per_bone_angular,
        "worst_bone": worst_bone,
        "flex_err_mean_deg": flex_mean,
        "per_joint_flex_deg": per_joint_flex,
        "umeyama_scale": float(s_u),
        "bone_length_drift_pct": float(drift_pct),
    }


# ------------------------- Main -------------------------

def parse_argv():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    tag = args[0] if args else "baseline"
    max_per_source = int(args[1]) if len(args) > 1 else 3
    return tag, max_per_source


def main() -> int:
    tag, max_per_source = parse_argv()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_clips = load_all_clips(ASSETS)
    print(f"[validator] Loaded {len(all_clips)} clips total")
    # Bucket by source, limit per source
    by_source: dict[str, list] = {}
    for c in all_clips:
        by_source.setdefault(c.source, []).append(c)
    for src in by_source:
        by_source[src] = by_source[src][:max_per_source]

    total = sum(len(v) for v in by_source.values())
    print(f"[validator] Validating {total} clips "
          f"({max_per_source} per source): {list(by_source.keys())}")

    results = []
    errors = []
    t0 = time.time()
    seen = 0
    for src, clips in by_source.items():
        for clip in clips:
            seen += 1
            t1 = time.time()
            try:
                rep = validate_clip(clip, src, n_samples=15)
                rep["elapsed_sec"] = time.time() - t1
                if "error" in rep:
                    errors.append(rep)
                    print(f"  [{seen}/{total}] {clip.path.name}: ERROR {rep.get('error')}")
                else:
                    results.append(rep)
                    print(f"  [{seen}/{total}] {clip.path.name}: "
                          f"PA-MPJPE={rep['pa_mpjpe_mean_mm']:.0f}mm "
                          f"FLEX={rep['flex_err_mean_deg']:.1f}° "
                          f"ANG={rep['angular_err_mean_deg']:.1f}° "
                          f"in {rep['elapsed_sec']:.1f}s", flush=True)
            except Exception as e:
                errors.append({"clip": clip.path.name, "source": src, "error": str(e)})
                traceback.print_exc()

    # Aggregate
    by_src: dict[str, list[dict]] = {}
    for r in results:
        by_src.setdefault(r["source"], []).append(r)

    summary_per_source = {}
    for s, rs in by_src.items():
        per_bone_avg = {}
        for b in ANGULAR_BONES:
            vals = [r["per_bone_angular_deg"].get(b, float("nan")) for r in rs]
            vals = [v for v in vals if not math.isnan(v)]
            per_bone_avg[b] = float(np.mean(vals)) if vals else float("nan")
        summary_per_source[s] = {
            "n_clips_ok": len(rs),
            "pa_mpjpe_mean_avg_mm":      float(np.mean([r["pa_mpjpe_mean_mm"] for r in rs])),
            "pa_mpjpe_p95_avg_mm":       float(np.mean([r["pa_mpjpe_p95_mm"] for r in rs])),
            "angular_err_mean_avg_deg":  float(np.mean([r["angular_err_mean_deg"] for r in rs])),
            "flex_err_mean_avg_deg":    float(np.mean([r["flex_err_mean_deg"] for r in rs])),
            "bone_length_drift_avg_pct": float(np.mean([r["bone_length_drift_pct"] for r in rs])),
            "per_bone_angular_avg_deg":  per_bone_avg,
            "worst_joint_histogram":     _hist([r["worst_joint"] for r in rs]),
            "worst_bone_histogram":      _hist([r["worst_bone"] for r in rs]),
        }

    report = {
        "tag": tag,
        "n_clips_tested": total,
        "n_ok": len(results),
        "n_errors": len(errors),
        "wall_clock_sec": time.time() - t0,
        "per_source_summary": summary_per_source,
        "clips": results,
        "errors": errors,
    }
    out = OUT_DIR / f"report_{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[validator] Saved {out}")
    print(f"[validator] Per-source averages:")
    for s, st in summary_per_source.items():
        print(f"  {s:10s}: PA-MPJPE={st['pa_mpjpe_mean_avg_mm']:.0f}mm "
              f"FLEX={st['flex_err_mean_avg_deg']:.1f}° "
              f"ANG={st['angular_err_mean_avg_deg']:.1f}° "
              f"n={st['n_clips_ok']}")
    return 0


def _hist(values):
    h: dict[str, int] = {}
    for v in values:
        h[v] = h.get(v, 0) + 1
    return h


if __name__ == "__main__":
    sys.exit(main())

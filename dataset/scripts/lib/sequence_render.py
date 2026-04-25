"""Sequence-based rendering + decimation (BEDLAM recipe).

Rather than one render per character+motion+camera combo, we render a short
sequence (typically 3 s @ 24 fps = 72 frames) then keep every Nth frame for
the main training set (BEDLAM's 30->6 fps subsample).  Scene setup
(character build, HDRI load, camera rig) amortises across all kept frames.

Public API:
    render_sequence(config) -> SequenceResult

Config describes character seed, BVH motion, camera rig, HDRI, output dir,
frame step.  Internally:
  1. Build character (if not reused across sequences)
  2. Import BVH via RetargetContext; choose a contiguous frame window
  3. Configure camera_rig + HDRI
  4. For each rendered frame: apply BVH pose, compute keypoints, render
  5. Write per-frame annotation JSONs + a per-sequence manifest
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import bpy  # type: ignore

from .coco17 import (
    COCO17_NAMES, COCO17_SKELETON,
    get_coco17_world, project_to_pixels,
)
# FK retarget with rest-pose offset matrices (armature-space writes).
# Handles T-pose source → A-pose target rest-pose mismatch correctly.
# Copy Rotation POSE/LOCAL approaches both fail because source (CMU T-pose,
# SMPL T-pose) has arms horizontal while MPFB rest is A-pose (arms diagonal
# down); POSE-space copy forces target into source's rest orientation,
# producing splayed-out 45° arms for walking motions.
from .fk_retarget import RetargetContext, load_bvh
from . import camera_rig
from . import render_setup
from . import bvh_quality
from . import realism
from . import bvh_sanitizer
from . import pose_validator
from . import jump_parabola
from . import foot_lock


@dataclass
class SequenceConfig:
    """Inputs for a single sequence render."""
    sequence_id: int
    bvh_path: str
    hdri_path: str
    character_seed: int
    out_dir: Path
    # Sequence timing
    seq_frames: int = 72                # raw rendered frames spanned
    decimation: int = 6                 # (legacy; unused when adaptive_stride=True)
    bvh_window_start: Optional[int] = None   # None = auto-pick via window scorer
    # Quality controls
    adaptive_stride: bool = True        # velocity-adaptive stride (BEDLAM-style)
    stride_target_ms: float = 175.0     # ~BEDLAM 6fps effective
    max_kept_frames: int = 6            # hard cap on kept frames per seq
    window_candidates: int = 8          # candidates to score when auto-picking
    ground_lift: bool = True            # apply per-clip ground-plane offset
    apply_realism: bool = True          # AgX + SSS + compositor post-fx
    # SOTA plausibility validation
    sanitize_bvh: bool = True           # clip-level NaN/teleport/T-pose reject
    validate_poses: bool = True         # per-frame ROM/penetration/collision reject
    enable_foot_lock: bool = True       # velocity-threshold foot IK
    detect_real_jumps: bool = True      # parabola-fit jump discriminator
    pose_validation_cost_budget_ms: float = 30.0   # skip self-intersect if >budget
    # Video mode: when True, render a coherent sequence (single camera with
    # smooth motion across kept frames) for training temporal models like
    # MotionAGFormer.  When False (default), re-roll the camera per kept
    # frame for maximum per-frame viewpoint diversity (single-frame training).
    coherent_sequence: bool = False
    # Rendering
    resolution: tuple[int, int] = (1280, 720)
    engine: str = "BLENDER_EEVEE"
    samples: int = 32


@dataclass
class FrameAnnotation:
    frame_index: int            # global scene frame
    kept: bool                  # survived decimation
    png_path: str
    keypoints_2d_pixels: list
    keypoints_3d_world_m: list
    camera_intrinsics: dict
    camera_extrinsics: dict
    ground_lift_m: float = 0.0
    validation: dict = field(default_factory=dict)   # plausibility metrics


@dataclass
class SequenceResult:
    sequence_id: int
    config: SequenceConfig
    frames: list[FrameAnnotation] = field(default_factory=list)
    timing_sec: dict = field(default_factory=dict)


# ---------- Helpers ----------
def pick_bvh_window(ctx: RetargetContext, seq_frames: int, rng) -> int:
    """Choose a contiguous sub-window within the clip."""
    start, end = ctx.frame_range
    clip_len = end - start + 1
    if clip_len <= seq_frames:
        return start
    # Avoid the very first/last few frames (ramp-up / ramp-down artefacts).
    lo = start + 2
    hi = end - seq_frames - 2
    if hi <= lo:
        return max(start, lo)
    return rng.randint(lo, hi)


def _camera_extrinsics_dict(cam: object) -> dict:
    mw = cam.matrix_world
    return {
        "location_xyz": list(mw.translation),
        "rotation_euler_xyz": list(cam.rotation_euler),
        "matrix_world": [list(row) for row in mw],
    }


# ---------- Main entry point ----------
def render_sequence(cfg: SequenceConfig, rng) -> SequenceResult:
    """Render `cfg.seq_frames` contiguous frames, keep every `cfg.decimation`th.

    Assumes the character + armature are already built and present in the
    scene.  Returns per-frame annotations for the KEPT frames only.
    """
    t0 = time.time()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = cfg.out_dir / f"seq_{cfg.sequence_id:04d}"
    seq_dir.mkdir(exist_ok=True)

    # Identify the MPFB target armature.
    target_arm = None
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and "root" in o.pose.bones:
            target_arm = o
            break
    if target_arm is None:
        raise RuntimeError("No MPFB armature found in scene")

    # Load BVH + create retarget context
    t1 = time.time()
    bvh_arm = load_bvh(cfg.bvh_path)
    ctx = RetargetContext(bvh_arm, target_arm)
    t_retarget_init = time.time() - t1

    # SOTA gate 1: clip-level BVH sanitizer (reject whole clip if unusable)
    clip_report = None
    if cfg.sanitize_bvh:
        t_s = time.time()
        clip_report = bvh_sanitizer.sanitize_bvh_clip(bvh_arm, cfg.bvh_path)
        if not clip_report.ok:
            print(f"[seq {cfg.sequence_id}] {clip_report.describe()} -- skipping clip")
            ctx.cleanup()
            return SequenceResult(
                sequence_id=cfg.sequence_id, config=cfg, frames=[],
                timing_sec={"rejected_by_sanitizer": clip_report.rejection_reason,
                             "sanitizer_sec": time.time() - t_s,
                             "wall_clock": time.time() - t0},
            )

    # Window selection: pick the best-scoring candidate from several options
    if cfg.bvh_window_start is not None:
        bvh_start = cfg.bvh_window_start
        bvh_end = bvh_start + cfg.seq_frames - 1
        best_score = None
    else:
        best = bvh_quality.best_window(
            bvh_arm, cfg.seq_frames, rng,
            num_candidates=cfg.window_candidates,
        )
        bvh_start, bvh_end = best.start_frame, best.end_frame
        best_score = best

    # Ground-plane lift — percentile-based, applied to the target armature
    # (not the root bone) so the BVH action remains untouched.
    ground_lift_m = 0.0
    if cfg.ground_lift:
        g = bvh_quality.compute_ground_offset(bvh_arm, (bvh_start, bvh_end))
        ground_lift_m = g.lift_m
        bvh_quality.apply_ground_lift(target_arm, ground_lift_m)

    # Configure scene frame range
    scene = bpy.context.scene
    scene.frame_start = bvh_start
    scene.frame_end = bvh_end

    # Set rest pose at mid-window for framing calculation
    mid_frame = (bvh_start + bvh_end) // 2
    ctx.apply_pose(mid_frame)

    # Camera + HDRI.
    # Two modes:
    #  * coherent_sequence=False (default, single-frame training): re-roll
    #    camera pose PER KEPT FRAME so each rendered sample is an
    #    independent viewpoint draw.  Maximises per-frame diversity.
    #  * coherent_sequence=True (video/temporal training): build one camera
    #    with synthetic motion (pan/orbit/dolly) that smoothly evolves
    #    across the sequence.  Kept frames share the same camera trajectory,
    #    producing temporally-coherent video suitable for MotionAGFormer /
    #    SmoothNet training.
    t2 = time.time()
    cam = camera_rig.make_camera(f"seq{cfg.sequence_id:04d}_cam")
    render_setup.set_world_hdri(cfg.hdri_path,
                                 strength=rng.uniform(0.5, 2.0),
                                 rotation_z=rng.uniform(0, 2 * math.pi))
    t_cam_hdri = time.time() - t2

    if cfg.coherent_sequence:
        cam_sample = camera_rig.build_shot(
            cam, target_arm, rng,
            frame_start=bvh_start, frame_end=bvh_end,
            use_shake=True,
        )
    else:
        cam_sample = camera_rig.frame_armature_strict(
            cam, target_arm, rng, focal_mm=camera_rig.sample_focal_mm(rng))
        camera_rig.apply_dof(cam, focus_distance=cam_sample.distance,
                              fstop=cam_sample.dof_fstop)

    # Realism pack: AgX view + physical SSS + compositor phone-cam chain.
    if cfg.apply_realism:
        basemesh = None
        for o in bpy.data.objects:
            if o.type == "MESH" and "body" in o.name.lower():
                basemesh = o
                break
        if basemesh is None:
            for o in bpy.data.objects:
                if o.type == "MESH":
                    basemesh = o
                    break
        realism.apply_realism_pack(basemesh, rng)

    # Render + annotate per-frame
    render_setup.configure_render(
        resolution=cfg.resolution, engine=cfg.engine, samples=cfg.samples,
        output_path=str(seq_dir / "frame_####.png"),
    )

    # Capture rest lengths for per-frame bone-drift validation.
    rest_lengths = pose_validator.capture_rest_lengths(target_arm) if cfg.validate_poses else None

    # SOTA gate 2: detect airborne windows and install foot-lock on artifacts
    foot_pins: list[object] = []
    airborne_events: list = []
    if cfg.detect_real_jumps:
        airborne_events = jump_parabola.scan_airborne_windows(
            target_arm, bvh_start, bvh_end)
        # Any airborne span classified as NOT a real jump -> foot-lock as artifact mitigation
        if cfg.enable_foot_lock:
            for ev in airborne_events:
                if not ev.is_real_jump:
                    # Scan before + after the artifact window for plant opportunities
                    plants = foot_lock.detect_foot_plants(
                        target_arm, max(bvh_start, ev.start_frame - 10),
                        min(bvh_end, ev.end_frame + 10))
                    foot_pins.extend(foot_lock.apply_foot_plants(target_arm, plants))

    # Decide which frames to render.
    if cfg.adaptive_stride:
        kept_frames = bvh_quality.adaptive_keep_frames(
            bvh_arm, bvh_start, bvh_end,
            target_stride_ms=cfg.stride_target_ms,
            max_frames=cfg.max_kept_frames,
        )
    else:
        kept_frames = [f for f in range(bvh_start, bvh_end + 1)
                        if (f - bvh_start) % cfg.decimation == 0]
        kept_frames = kept_frames[: cfg.max_kept_frames]

    annotations: list[FrameAnnotation] = []
    rejected_annotations: list[FrameAnnotation] = []
    t_render_total = 0.0
    t_validation_total = 0.0
    rx, ry = scene.render.resolution_x, scene.render.resolution_y

    for f in kept_frames:
        # Apply this frame's pose FIRST so framing sees the actual posed body.
        ctx.apply_pose(f)
        scene.frame_set(f)
        bpy.context.view_layer.update()

        if cfg.coherent_sequence:
            # Video-coherent: keep the sequence-wide camera (Blender
            # animates its location/rotation via keyframes); re-use
            # cam_sample for the annotation intrinsics on every frame.
            per_frame_sample = cam_sample
        else:
            # Single-frame diverse: re-roll independent camera per frame.
            per_frame_sample = camera_rig.frame_armature_strict(
                cam, target_arm, rng, focal_mm=camera_rig.sample_focal_mm(rng))
            camera_rig.apply_dof(cam, focus_distance=per_frame_sample.distance,
                                  fstop=per_frame_sample.dof_fstop)
        bpy.context.view_layer.update()

        # Render one frame
        png_path = seq_dir / f"frame_{f:04d}.png"
        scene.render.filepath = str(png_path)
        t_r = time.time()
        bpy.ops.render.render(write_still=True)
        t_render_total += time.time() - t_r

        # Compute keypoints
        kps_world = get_coco17_world(target_arm)
        projections = project_to_pixels(scene, cam, kps_world)

        kps_2d, kps_3d = [], []
        for (w, proj) in zip(kps_world, projections):
            if w is None or proj is None:
                kps_2d.append({"u": None, "v": None, "px": None, "py": None,
                                "visible": False, "depth": None})
                kps_3d.append({"x": None, "y": None, "z": None})
                continue
            u, v, depth, inside = proj
            kps_2d.append({"u": u, "v": v,
                            "px": u * rx, "py": v * ry,
                            "visible": bool(inside), "depth": depth})
            kps_3d.append({"x": w[0], "y": w[1], "z": w[2]})

        # SOTA gate 3: per-frame plausibility validation
        val_dict = {}
        pose_ok = True
        if cfg.validate_poses:
            t_v = time.time()
            # Find the MPFB basemesh (the body mesh parented to the armature)
            basemesh = None
            for child in target_arm.children:
                if child.type == "MESH":
                    basemesh = child
                    break
            if basemesh is None:
                for o in bpy.data.objects:
                    if o.type == "MESH" and o.name.startswith("subject_"):
                        basemesh = o
                        break
            # Skip self-intersection if we're over the time budget on previous frames
            do_si = (t_validation_total / max(1, len(annotations) + len(rejected_annotations)) * 1000.0
                     ) < cfg.pose_validation_cost_budget_ms
            frame_val = pose_validator.validate_frame(
                target_arm, basemesh, rest_lengths, f,
                do_self_intersect=do_si,
            )
            pose_ok = frame_val.ok
            val_dict = {
                "ok": frame_val.ok,
                "rejections": frame_val.rejections,
                "metrics": {k: (v if not isinstance(v, list) or len(v) < 5 else len(v))
                             for k, v in frame_val.metrics.items()},
            }
            t_validation_total += time.time() - t_v

        ann = FrameAnnotation(
            frame_index=f, kept=pose_ok,
            png_path=png_path.name,
            keypoints_2d_pixels=kps_2d,
            keypoints_3d_world_m=kps_3d,
            camera_intrinsics=camera_rig.export_intrinsics(per_frame_sample),
            camera_extrinsics=_camera_extrinsics_dict(cam),
            ground_lift_m=float(ground_lift_m),
            validation=val_dict,
        )
        if pose_ok:
            annotations.append(ann)
        else:
            rejected_annotations.append(ann)

    # Cleanup foot-lock pins and the BVH armature (if clip survives).
    if foot_pins:
        foot_lock.cleanup_foot_plants(target_arm, foot_pins)

    # Write per-sequence manifest (AGORA-like, one JSON per sequence)
    labels = {
        "sequence_id": cfg.sequence_id,
        "character_seed": cfg.character_seed,
        "bvh": str(Path(cfg.bvh_path).name),
        "hdri": str(Path(cfg.hdri_path).name),
        "clip_report": (clip_report.__dict__ if clip_report is not None else None),
        "airborne_events": [a.__dict__ for a in airborne_events],
        "n_foot_pins_applied": len(foot_pins),
        "n_frames_rejected": len(rejected_annotations),
        "rejected_frames": [
            {"frame_index": r.frame_index, "rejections": r.validation.get("rejections", [])}
            for r in rejected_annotations
        ],
        "raw_frames": cfg.seq_frames,
        "decimation": cfg.decimation,
        "adaptive_stride": cfg.adaptive_stride,
        "stride_target_ms": cfg.stride_target_ms,
        "kept_frames": len(annotations),
        "bvh_window": [bvh_start, bvh_end],
        "ground_lift_m": float(ground_lift_m),
        "window_score": ({
            "pose_variance": best_score.pose_variance,
            "root_jerk": best_score.root_jerk,
            "foot_contact_ratio": best_score.foot_contact_ratio,
            "bone_length_drift": best_score.bone_length_drift,
            "score": best_score.score,
        } if best_score is not None else None),
        "resolution": [rx, ry],
        "keypoint_names": COCO17_NAMES,
        "skeleton_edges": COCO17_SKELETON,
        "camera_sample": {
            "motion": cam_sample.motion,
            "focal_mm": cam_sample.intrinsics.focal_mm,
            "distance_m": cam_sample.distance,
            "yaw_rad": cam_sample.yaw,
            "pitch_rad": cam_sample.pitch,
            "roll_rad": cam_sample.roll,
            "shake_amp_cm": cam_sample.shake_amp_cm,
            "shake_amp_deg": cam_sample.shake_amp_deg,
        },
        "frames": [asdict(a) for a in annotations],
    }
    (seq_dir / "labels.json").write_text(json.dumps(labels, indent=2, default=str))

    # Cleanup BVH armature
    ctx.cleanup()

    return SequenceResult(
        sequence_id=cfg.sequence_id,
        config=cfg,
        frames=annotations,
        timing_sec={
            "retarget_init": t_retarget_init,
            "camera_hdri": t_cam_hdri,
            "render_total": t_render_total,
            "validation_total": t_validation_total,
            "wall_clock": time.time() - t0,
        },
    )

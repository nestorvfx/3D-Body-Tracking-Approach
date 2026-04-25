"""Animation-batched variant of build_dataset.py.

Instead of calling bpy.ops.render.render(write_still=True) per frame, this
version keyframes pose + camera across N frames in one go and calls
bpy.ops.render.render(animation=True) once per clip. This amortizes the
fixed per-render overhead (Cycles session init/teardown, Python setup,
BVH build) across all N frames.

Experimental — used to A/B against build_dataset.py. Keeps the same
output schema so merge.py works unchanged.

Simplifications vs. the still-per-frame version:
  - Per-frame HDRI strength/rotation variation is dropped (HDRI is fixed
    per-clip). World-shader node-tree changes per-frame would invalidate
    persistent_data and defeat the whole point.
  - Camera motion type is ignored (the per-frame camera is re-sampled
    per animation frame, yielding a "cut" camera — different pose frame
    => different camera pose). apply_motion-based tween motions aren't
    relevant here since each frame is independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import mathutils

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ensure_cycles():
    try:
        import addon_utils
        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception as e:
        print(f"[build-anim] cycles addon enable failed: {e}")


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


ensure_cycles()
ensure_mpfb()

from lib.mpfb_build import build_character
from lib.fk_retarget import RetargetContext, load_bvh
from lib.render_setup import clear_scene, configure_render, set_world_hdri
from lib.source_mappings import detect_source_from_bvh
from lib.coco17 import get_coco17_world, project_to_pixels
from lib import camera_rig as crig


SEED_POOL = list(range(42, 542))
MASTER_SEED = 20260421
IMAGE_W = 256
IMAGE_H = 192
ASSETS_ROOT = (HERE.parent / "assets").resolve()


def build_plan(rng, n_clips, seeds_per_clip, frames_per_clip):
    cmu = sorted((ASSETS_ROOT / "bvh").glob("*.bvh"))
    style = sorted((ASSETS_ROOT / "bvh_100style" / "100STYLE").rglob("*.bvh"))
    aist = sorted((ASSETS_ROOT / "aist_plusplus" / "bvh").glob("*.bvh"))
    if not (cmu or style or aist):
        raise RuntimeError(f"No BVH assets found under {ASSETS_ROOT}")

    if n_clips >= 2:
        n_cmu = max(1, int(round(n_clips * 0.20)))
        n_style = max(0, n_clips - n_cmu)
    else:
        n_cmu, n_style = 0, n_clips

    def _sample_pool(pool, k):
        if not pool or k <= 0: return []
        if k <= len(pool): return rng.sample(pool, k)
        out = list(pool); rng.shuffle(out)
        while len(out) < k: out.append(rng.choice(pool))
        return out

    picks = []
    picks += _sample_pool(cmu, n_cmu)
    style_by_dir = {}
    for p in style:
        style_by_dir.setdefault(p.parent.name, []).append(p)
    chosen_styles = _sample_pool(list(style_by_dir.keys()), n_style)
    picks += [rng.choice(style_by_dir[s]) for s in chosen_styles]
    rng.shuffle(picks)

    plan = []
    for p in picks:
        seeds = rng.sample(SEED_POOL, min(seeds_per_clip, len(SEED_POOL)))
        rel = p.relative_to(ASSETS_ROOT).as_posix()
        for seed in seeds:
            plan.append((rel, [seed], frames_per_clip, 1))
    return plan


def sample_frame_indices(ctx, n, rng):
    start, end = ctx.frame_range
    if end <= start or n <= 0: return [start]
    if end - start + 1 >= n * 2:
        return sorted(rng.sample(range(start, end + 1), n))
    step = max(1, (end - start) // max(1, n - 1))
    return list(range(start, end + 1, step))[:n]


def split_for_id(sample_id):
    h = int(hashlib.sha1(sample_id.encode()).hexdigest()[:2], 16)
    return "val" if h < 26 else "train"


def compute_extrinsics(camera):
    mw = camera.matrix_world
    R_w2c_blender = mw.to_3x3().inverted()
    t_w2c_blender = -(R_w2c_blender @ mw.to_translation())
    flip = mathutils.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_cv = flip @ R_w2c_blender
    t_cv = flip @ t_w2c_blender
    return {
        "R": [list(R_cv.row[0]), list(R_cv.row[1]), list(R_cv.row[2])],
        "t": list(t_cv),
    }


def world_to_camera_3d(points_world, camera):
    extr = compute_extrinsics(camera)
    R = mathutils.Matrix(extr["R"])
    t = mathutils.Vector(extr["t"])
    out = []
    for p in points_world:
        if p is None: out.append(None); continue
        wp = mathutils.Vector(p)
        cam_p = R @ wp + t
        out.append([float(cam_p.x), float(cam_p.y), float(cam_p.z)])
    return out


def tight_bbox(kps_2d_pixels, visibility, pad=1.25, W=IMAGE_W, H=IMAGE_H):
    xs, ys = [], []
    for (kp, v) in zip(kps_2d_pixels, visibility):
        if v <= 0: continue
        u, vv = kp
        if 0 <= u < W and 0 <= vv < H:
            xs.append(u); ys.append(vv)
    if not xs: return [0, 0, W, H]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1.0, x_max - x_min); h = max(1.0, y_max - y_min)
    cx, cy = x_min + w/2, y_min + h/2
    w *= pad; h *= pad
    x = max(0.0, cx - w/2); y = max(0.0, cy - h/2)
    w = min(W - x, w); h = min(H - y, h)
    return [int(x), int(y), int(w), int(h)]


def _configure_cycles(samples=8, engine_mode="auto", threads=0):
    scene = bpy.context.scene
    if threads > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = threads
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPTIX" if engine_mode == "gpu" else "OPENIMAGEDENOISE"
    except Exception: pass
    try: scene.cycles.denoising_prefilter = "FAST"
    except AttributeError: pass
    try:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.1
        scene.cycles.adaptive_min_samples = 4
    except AttributeError: pass
    try: scene.render.use_persistent_data = True
    except AttributeError: pass
    try: scene.cycles.use_auto_tile = False
    except AttributeError: pass
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    try: scene.render.use_motion_blur = False
    except Exception: pass

    def _try_gpu(ctype):
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = ctype
            prefs.get_devices()
            active = [d for d in prefs.devices if d.type == ctype]
            if active:
                for d in prefs.devices:
                    d.use = d.type == ctype
                scene.cycles.device = "GPU"
                print(f"[cycles] {ctype}: {[d.name for d in active]}")
                return True
        except Exception as e:
            print(f"[cycles] {ctype} probe failed: {e}")
        return False

    if engine_mode == "cpu":
        scene.cycles.device = "CPU"
    elif engine_mode == "gpu":
        if not (_try_gpu("OPTIX") or _try_gpu("CUDA")):
            raise RuntimeError("engine_mode='gpu' but no CUDA/OptiX GPU found")
    else:
        if not (_try_gpu("OPTIX") or _try_gpu("CUDA")):
            scene.cycles.device = "CPU"


def _parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--target-count", type=int, default=500000)
    p.add_argument("--seeds-per-clip", type=int, default=2)
    p.add_argument("--frames-per-clip", type=int, default=5)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-start", type=int, default=None)
    p.add_argument("--shard-end", type=int, default=None)
    p.add_argument("--cycles-samples", type=int, default=8)
    p.add_argument("--engine", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def _clear_anim(obj):
    """Remove any animation data/keyframes on obj and obj.data (if object)."""
    try:
        if obj.animation_data:
            obj.animation_data_clear()
    except Exception: pass
    try:
        d = getattr(obj, "data", None)
        if d is not None and getattr(d, "animation_data", None):
            d.animation_data_clear()
    except Exception: pass


def _keyframe_armature_pose(arm, frame):
    """Insert keyframes for every pose bone's rotation_quaternion and location
    at the given animation frame."""
    for pb in arm.pose.bones:
        pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        pb.keyframe_insert(data_path="location", frame=frame)


def _keyframe_camera(cam, frame):
    cam.keyframe_insert(data_path="location", frame=frame)
    cam.keyframe_insert(data_path="rotation_euler", frame=frame)
    cam.data.keyframe_insert(data_path="lens", frame=frame)
    cam.data.keyframe_insert(data_path="shift_x", frame=frame)
    cam.data.keyframe_insert(data_path="shift_y", frame=frame)


def main():
    args = _parse_args()
    base_out = Path(args.out).resolve()
    out_dir = base_out / f"shard_{args.shard_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    manifest_path = out_dir / "manifest.csv"

    done_ids = set()
    if labels_path.exists():
        with labels_path.open() as fh:
            for line in fh:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[shard {args.shard_id}] resume: {len(done_ids)} done")

    hdris = sorted((ASSETS_ROOT / "hdris").glob("*.hdr"))
    if not hdris:
        print(f"[shard {args.shard_id}] FATAL: no HDRIs")
        sys.exit(1)

    rng_master = random.Random(MASTER_SEED)
    n_clips = max(1, math.ceil(args.target_count / (args.seeds_per_clip * args.frames_per_clip)))
    full_plan = build_plan(random.Random(MASTER_SEED), n_clips,
                            args.seeds_per_clip, args.frames_per_clip)
    full_plan.sort(key=lambda e: (e[1][0], e[0]))

    if args.shard_start is not None and args.shard_end is not None:
        shard_plan = full_plan[args.shard_start:args.shard_end]
        target_per_shard = math.ceil(
            args.target_count * (args.shard_end - args.shard_start) / max(1, len(full_plan)))
    else:
        shard_plan = [e for i, e in enumerate(full_plan)
                      if i % args.num_shards == args.shard_id]
        target_per_shard = math.ceil(args.target_count / args.num_shards)

    print(f"[shard {args.shard_id}/{args.num_shards}] "
          f"{len(shard_plan)} clips, target {target_per_shard} samples")

    labels_fh = labels_path.open("a", encoding="utf-8")
    manifest_fh = manifest_path.open(
        "w" if not done_ids else "a", encoding="utf-8", newline="")
    if not done_ids:
        manifest_fh.write("id,split,source,image_rel,bbox_x,bbox_y,bbox_w,bbox_h\n")

    stats = {"cmu": 0, "100style": 0, "aistpp": 0, "total": 0, "train": 0, "val": 0}
    sample_counter = 0
    t_start = time.time()
    current_seed = None
    current_arm = None
    cam = None

    for plan_i, (bvh_rel, seeds, n_frames, cams_per) in enumerate(shard_plan):
        bvh_abs = ASSETS_ROOT / bvh_rel
        if not bvh_abs.exists(): continue
        src_kind = detect_source_from_bvh(str(bvh_abs))
        clip_id = bvh_abs.stem

        for seed in seeds:
            if seed != current_seed:
                clear_scene()
                _bm, current_arm = build_character(seed, with_assets=True)
                current_seed = seed
                cam = crig.make_camera("DatasetCam")
                bpy.context.scene.camera = cam
                configure_render(resolution=(IMAGE_W, IMAGE_H), engine="CYCLES",
                                 samples=args.cycles_samples,
                                 output_path=str(out_dir / "images" / "_.png"))
                _configure_cycles(args.cycles_samples, args.engine, args.threads)
                cam.rotation_mode = "XYZ"
            arm = current_arm

            # Drop previous BVH armature.
            for _obj in list(bpy.data.objects):
                if _obj.type == "ARMATURE" and _obj.get("_mocap_source"):
                    bpy.data.objects.remove(_obj, do_unlink=True)
            bvh = load_bvh(str(bvh_abs), source=src_kind)
            ctx = RetargetContext(bvh, arm)

            hdri_pool_idx = int(hashlib.sha1(
                f"{seed}_{bvh_rel}".encode()).hexdigest()[:4], 16) % len(hdris)
            current_hdri_path = hdris[hdri_pool_idx]
            # Per-clip HDRI strength + rotation (no per-frame variation in
            # this approach to preserve persistent_data across animation frames).
            rng_clip = random.Random((seed * 100003) ^ hash(bvh_rel))
            set_world_hdri(str(current_hdri_path),
                           strength=rng_clip.uniform(0.7, 1.6),
                           rotation_z=rng_clip.uniform(0.0, 6.283185))

            frames = sample_frame_indices(ctx, n_frames, rng_clip)

            # Clear any leftover keyframes from previous clip.
            _clear_anim(arm)
            _clear_anim(cam)

            # PRE-PASS: keyframe pose + camera for each animation frame,
            # and compute labels now (while scene state matches).
            frame_records = []
            skipped = 0
            for anim_i, bvh_frame in enumerate(frames, start=1):
                sample_id = f"{src_kind}_{clip_id}_s{seed}_c0_f{anim_i:04d}"
                if sample_id in done_ids:
                    skipped += 1
                    continue
                ctx.apply_pose(bvh_frame)
                bpy.context.view_layer.update()
                _keyframe_armature_pose(arm, anim_i)

                rng_cam = random.Random(
                    rng_master.randrange(1 << 30) ^ (bvh_frame << 8))
                try:
                    focal = crig.sample_focal_mm(rng_cam)
                    cam_sample = crig.frame_armature_strict(
                        cam, arm, rng_cam, focal_mm=focal)
                except Exception as e:
                    print(f"  [skip cam] {sample_id}: {e}")
                    skipped += 1
                    continue
                bpy.context.view_layer.update()
                _keyframe_camera(cam, anim_i)

                # Compute labels while scene is in the right pose.
                kps3d_world = get_coco17_world(arm)
                proj = project_to_pixels(bpy.context.scene, cam, kps3d_world)
                kps2d_pixels, vis_flags = [], []
                for kp in proj:
                    if kp is None:
                        kps2d_pixels.append([0.0, 0.0]); vis_flags.append(0); continue
                    u, v, depth, inside = kp
                    kps2d_pixels.append([float(u*IMAGE_W), float(v*IMAGE_H)])
                    vis_flags.append(2 if inside else 1)

                kps3d_cam = world_to_camera_3d(kps3d_world, cam)
                intr_dict = cam_sample.intrinsics.K()
                extrinsics = compute_extrinsics(cam)
                bbox = tight_bbox(kps2d_pixels, vis_flags)
                split = split_for_id(sample_id)

                root = None
                if kps3d_cam[11] is not None and kps3d_cam[12] is not None:
                    l, r = kps3d_cam[11], kps3d_cam[12]
                    root = [(l[0]+r[0])/2, (l[1]+r[1])/2, (l[2]+r[2])/2]

                frame_records.append({
                    "anim_i": anim_i,
                    "sample_id": sample_id,
                    "bvh_frame": bvh_frame,
                    "label": {
                        "id": sample_id, "split": split,
                        "image_wh": [IMAGE_W, IMAGE_H],
                        "camera_K": intr_dict,
                        "camera_extrinsics": extrinsics,
                        "bbox_xywh": bbox,
                        "keypoints_2d": [[kp[0], kp[1], vf] for kp, vf in zip(kps2d_pixels, vis_flags)],
                        "keypoints_3d_cam": kps3d_cam,
                        "root_joint_cam": root,
                        "source": src_kind, "clip_id": clip_id,
                        "frame_idx": bvh_frame, "character_seed": seed,
                        "hdri": hdris[hdri_pool_idx].stem,
                        "focal_mm": float(cam_sample.intrinsics.focal_mm),
                        "render_engine": f"CYCLES_{bpy.context.scene.cycles.device}",
                        "shard_id": args.shard_id,
                    },
                    "bbox": bbox, "split": split,
                })

            if not frame_records:
                continue

            # Configure animation output. Filename pattern:
            #   {src}_{clip}_s{seed}_c0_f####.png
            # Blender replaces #### with zero-padded frame number.
            scene = bpy.context.scene
            scene.frame_start = frame_records[0]["anim_i"]
            scene.frame_end = frame_records[-1]["anim_i"]
            prefix = f"{src_kind}_{clip_id}_s{seed}_c0_f"
            scene.render.filepath = str(out_dir / "images" / f"{prefix}####")
            scene.render.use_file_extension = True

            # Render animation — one call produces N PNGs.
            bpy.ops.render.render(animation=True)

            # Write labels after successful render.
            for rec in frame_records:
                fname = f"{prefix}{rec['anim_i']:04d}.png"
                rec["label"]["image_rel"] = f"images/{fname}"
                labels_fh.write(json.dumps(rec["label"]) + "\n")
                manifest_fh.write(
                    f"{rec['sample_id']},{rec['split']},{src_kind},"
                    f"{rec['label']['image_rel']},{rec['bbox'][0]},{rec['bbox'][1]},"
                    f"{rec['bbox'][2]},{rec['bbox'][3]}\n")
                stats["total"] += 1
                stats[src_kind] = stats.get(src_kind, 0) + 1
                stats[rec["split"]] += 1
                sample_counter += 1
            labels_fh.flush(); manifest_fh.flush()

            dt = max(1e-3, time.time() - t_start)
            rate = sample_counter / dt
            rem = max(0, target_per_shard - stats["total"])
            eta = int(rem / rate) if rate > 0 else -1
            eta_s = f"{eta//3600:d}h{(eta%3600)//60:02d}m" if eta >= 0 else "?"
            print(f"[shard {args.shard_id}] "
                  f"[{stats['total']:6d}/{target_per_shard}] "
                  f"{rate:4.2f}/s ETA {eta_s}  (clip +{len(frame_records)})")

            if args.limit and stats["total"] >= args.limit:
                labels_fh.close(); manifest_fh.close(); return
            if stats["total"] >= target_per_shard:
                labels_fh.close(); manifest_fh.close(); return

    labels_fh.close(); manifest_fh.close()
    print(f"[shard {args.shard_id}] done — {stats['total']} samples")


if __name__ == "__main__":
    main()

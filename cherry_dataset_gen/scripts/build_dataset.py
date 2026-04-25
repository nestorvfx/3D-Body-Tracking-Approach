"""Sharded dataset builder optimised for CPU-only Ubuntu cloud boxes (Cherry).

Differences from the local GPU/EEVEE version:
  - Sharding via --shard-id/--num-shards so 32 parallel Blender processes
    each own a deterministic slice of the plan and never collide.
  - CYCLES CPU rendering (no GPU on the cherry box) with low-sample + OIDN
    denoising for speed-friendly photoreal frames.
  - 500-seed phenotype pool for 500k-sample diversity.
  - Absolute output paths everywhere (cherry mounts vary by image).

Each shard writes into out/shard_<i>/{images,labels.jsonl,manifest.csv};
merge.py concatenates them into a single logical dataset at the end.

Usage (one instance):
  blender --background --python build_dataset.py -- \\
      --out /data/synth_v3 \\
      --target-count 500000 \\
      --seeds-per-clip 2 --frames-per-clip 5 \\
      --shard-id 0 --num-shards 32
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
    """Enable the Cycles render-engine addon BEFORE any engine lookup.

    On fresh Blender installs (e.g. a new Linux tarball on vast.ai) Cycles
    is not enabled in user prefs by default.  If we just call
    `configure_render(engine='CYCLES')` directly, `scene.render.engine`'s
    enum enumerates only BLENDER_EEVEE* and configure_render silently
    falls back to EEVEE-on-CPU.  Force-enable here so Cycles is in the
    enum before any check.
    """
    try:
        import addon_utils
        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception as e:
        print(f"[build] cycles addon enable failed: {e}")


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
from lib.render_setup import (
    clear_scene, configure_render, set_world_hdri, update_hdri_params,
    configure_gt_passes, set_gt_sample_path, add_shadow_catcher_plane)
from lib.surface_kps import (
    NUM_SURFACE_KPS, select_surface_vertex_indices, compute_surface_kps,
    _find_basemesh_object)
from lib.source_mappings import detect_source_from_bvh
from lib.coco17 import get_coco17_world, project_to_pixels
from lib import camera_rig as crig


# 500 distinct character seeds — covers a 500k-sample run at ~1000 samples
# per seed before any repetition.  Per-character tinting + hair palette +
# per-sample HDRI + camera randomisation keeps every (seed, sample) tuple
# visually distinct even at this scale.
SEED_POOL = list(range(42, 542))

MASTER_SEED = 20260421
IMAGE_W = 256
IMAGE_H = 192
ASSETS_ROOT = (HERE.parent / "assets").resolve()


def build_plan(rng: random.Random, n_clips: int,
               seeds_per_clip: int, frames_per_clip: int
               ) -> list[tuple[str, list[int], int, int]]:
    """Sample n_clips diverse clips across CMU / AIST++ / 100STYLE."""
    cmu = sorted((ASSETS_ROOT / "bvh").glob("*.bvh"))
    style = sorted((ASSETS_ROOT / "bvh_100style" / "100STYLE").rglob("*.bvh"))
    aist = sorted((ASSETS_ROOT / "aist_plusplus" / "bvh").glob("*.bvh"))

    if not (cmu or style or aist):
        raise RuntimeError(
            f"No BVH assets found under {ASSETS_ROOT}. Run download_assets.sh first.")

    # Cherry setup has CMU + 100STYLE only (AIST++ isn't available in BVH
    # form from any wget-friendly URL — it ships as SMPL .pkl).  We weight
    # 100STYLE heavier because it's the more varied pool (100 motion
    # categories × 5 variants vs. CMU's uneven-quality community captures).
    if n_clips >= 2:
        n_cmu = max(1, int(round(n_clips * 0.20)))
        n_aist = 0
        n_style = max(0, n_clips - n_cmu - n_aist)
    else:
        n_cmu = 0; n_aist = 0; n_style = n_clips

    def _sample_pool(pool, k):
        if not pool or k <= 0:
            return []
        if k <= len(pool):
            return rng.sample(pool, k)
        out = list(pool); rng.shuffle(out)
        while len(out) < k:
            out.append(rng.choice(pool))
        return out

    picks = []
    picks += _sample_pool(cmu, n_cmu)
    style_by_dir: dict[str, list[Path]] = {}
    for p in style:
        style_by_dir.setdefault(p.parent.name, []).append(p)
    style_keys = list(style_by_dir.keys())
    chosen_styles = _sample_pool(style_keys, n_style)
    picks += [rng.choice(style_by_dir[s]) for s in chosen_styles]
    picks += _sample_pool(aist, n_aist)
    rng.shuffle(picks)

    # Flatten: one plan entry per (clip, single seed) so that sort-by-seed
    # actually produces contiguous runs of the same character.  Previously
    # each entry held seeds=[a, b] and the inner loop ping-ponged
    # a->b->a->c->a->... rebuilding MPFB (30-43 s each) roughly every 5
    # samples.  With one seed per entry the cache hits across hundreds of
    # consecutive samples and MPFB runs ~30 times per shard total.
    plan = []
    for p in picks:
        seeds = rng.sample(SEED_POOL, min(seeds_per_clip, len(SEED_POOL)))
        rel = p.relative_to(ASSETS_ROOT).as_posix()
        for seed in seeds:
            plan.append((rel, [seed], frames_per_clip, 1))
    return plan


def sample_frame_indices(ctx: RetargetContext, n: int, rng: random.Random) -> list[int]:
    start, end = ctx.frame_range
    if end <= start or n <= 0:
        return [start]
    if end - start + 1 >= n * 2:
        return sorted(rng.sample(range(start, end + 1), n))
    step = max(1, (end - start) // max(1, n - 1))
    return list(range(start, end + 1, step))[:n]


def split_for_id(sample_id: str) -> str:
    h = int(hashlib.sha1(sample_id.encode()).hexdigest()[:2], 16)
    return "val" if h < 26 else "train"


def compute_extrinsics(camera) -> dict:
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
        if p is None:
            out.append(None); continue
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


def _configure_cycles(samples: int = 8, engine_mode: str = "auto",
                       threads: int = 0):
    """Apply optimized Cycles settings.  Picks device per engine_mode:
      'auto' — try OptiX, then CUDA, else CPU (default on vast.ai GPU box)
      'gpu'  — force GPU (OptiX > CUDA); FATAL if neither found
      'cpu'  — force CPU (cherry CPU-only box)

    Speed-critical settings: 8 samples + adaptive_sampling with high
    threshold + OIDN denoise, persistent BVH/depsgraph, low bounce
    depth, no caustics, no motion blur.  At 256×192 this gives
    ~0.5-1 s/render on RTX 5070, ~1.5-3 s on RTX 3060, ~3-8 s on EPYC
    CPU, vs. 8-20 s with default Cycles settings.
    """
    scene = bpy.context.scene
    # CRITICAL: cap Blender's CPU worker thread count so N parallel Blender
    # instances don't each grab all 96 cores and thrash the scheduler
    # (observed load avg 449 on a 96-thread box when 22 CPU shards each
    # tried to spawn 96 Cycles workers + 8 OIDN workers).  Even GPU shards
    # benefit — OIDN + BVH build still run threaded on CPU.
    if threads > 0:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = threads
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    # Denoiser: OPTIX on GPU (runs directly on the OPTIX device, near-zero
    # setup), OIDN on CPU.  At 4 samples the per-frame denoise step is
    # a significant fraction of the render budget, and OIDN-GPU has
    # enough CUDA<->OIDN tensor-setup overhead at 256x192 to lose to
    # OPTIX denoiser.
    try:
        if engine_mode == "gpu":
            scene.cycles.denoiser = "OPTIX"
        else:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    # FAST prefilter — at 4 samples ACCURATE's extra albedo/normal pass
    # is wasted since the image is already noisy; quality delta invisible.
    try:
        scene.cycles.denoising_prefilter = "FAST"
    except AttributeError:
        pass
    # Adaptive sampling — stop when noise threshold met.
    try:
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.1
        scene.cycles.adaptive_min_samples = 4
    except AttributeError:
        pass
    # Keep scene resident across renders — saves BVH refit on armature
    # pose changes (refit, not full rebuild: developer.blender.org/docs
    # /features/cycles/bvh/ — Cycles uses a 2-level BVH and refits per-
    # mesh BVH when only vertex positions change).
    try:
        scene.render.use_persistent_data = True
    except AttributeError:
        pass
    # 256x192 fits in one tile; auto-tile's per-frame bookkeeping is pure
    # overhead at this size.
    try:
        scene.cycles.use_auto_tile = False
    except AttributeError:
        pass
    # Trim bounces — pose dataset doesn't need caustics, deep specular.
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.volume_bounces = 0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    try:
        scene.render.use_motion_blur = False
    except Exception:
        pass

    def _try_gpu(ctype: str) -> bool:
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
        print("[cycles] CPU (forced)")
    elif engine_mode == "gpu":
        if not (_try_gpu("OPTIX") or _try_gpu("CUDA")):
            raise RuntimeError("engine_mode='gpu' but no CUDA/OptiX GPU found")
    else:   # auto
        if not (_try_gpu("OPTIX") or _try_gpu("CUDA")):
            scene.cycles.device = "CPU"
            print("[cycles] CPU (no GPU detected)")


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="absolute output directory")
    p.add_argument("--target-count", type=int, default=500000)
    p.add_argument("--seeds-per-clip", type=int, default=2)
    p.add_argument("--frames-per-clip", type=int, default=5)
    p.add_argument("--shard-id", type=int, default=0,
                    help="unique id of this shard (names the output dir shard_NNN)")
    p.add_argument("--num-shards", type=int, default=1,
                    help="legacy modulo-sharding denominator; used when "
                         "--shard-start/--shard-end aren't provided")
    p.add_argument("--shard-start", type=int, default=None,
                    help="range-based sharding: start index (inclusive) into "
                         "the sorted full plan.  Pair with --shard-end.  "
                         "Overrides the shard-id%%num-shards slicing and is "
                         "the mechanism used by hybrid GPU+CPU runs where "
                         "GPU shards get a larger contiguous slice than "
                         "CPU shards.")
    p.add_argument("--shard-end", type=int, default=None,
                    help="range-based sharding: end index (exclusive)")
    p.add_argument("--cycles-samples", type=int, default=8,
                    help="Cycles samples per frame (default 8 + OIDN denoise)")
    p.add_argument("--engine", default="auto", choices=["auto", "cpu", "gpu"],
                    help="Cycles device. 'auto' picks OptiX/CUDA if present, "
                         "else CPU.  'gpu' hard-fails if no GPU.")
    p.add_argument("--threads", type=int, default=0,
                    help="Max CPU worker threads for Blender/Cycles/OIDN in "
                         "THIS shard.  0 = auto (uses all cores, which "
                         "thrashes when N shards run in parallel).  Set to "
                         "~cpu_count/num_parallel_shards for balanced load.")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main():
    args = _parse_args()
    # Each shard writes into its own subdir.
    base_out = Path(args.out).resolve()
    out_dir = base_out / f"shard_{args.shard_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    # Depth EXR sidecar dir — per-sample half-float z-buffer for D-PoSE-
    # style dense-depth training signal (arXiv 2410.04889).  Cheap to
    # emit here (<2 percent render overhead); unlocks +3 mm PA-MPJPE if
    # wired into training as auxiliary supervision later.
    (out_dir / "depth").mkdir(exist_ok=True)
    labels_path = out_dir / "labels.jsonl"
    manifest_path = out_dir / "manifest.csv"
    # Per-character metadata (MPFB phenotype etc) — written once per
    # unique seed so we don't repeat it ~N_frames times in labels.jsonl.
    characters_path = out_dir / "characters.jsonl"
    characters_fh = characters_path.open(
        "a" if characters_path.exists() else "w", encoding="utf-8")
    written_character_ids: set[str] = set()

    # Resume / dedupe support.
    done_ids: set[str] = set()
    if labels_path.exists():
        with labels_path.open() as fh:
            for line in fh:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[shard {args.shard_id}/{args.num_shards}] resume: {len(done_ids)} done")

    # HDRIs are shared across shards; downloaded once by the installer.
    hdris = sorted((ASSETS_ROOT / "hdris").glob("*.hdr"))
    if not hdris:
        print(f"[shard {args.shard_id}] FATAL: no HDRIs under {ASSETS_ROOT}/hdris")
        sys.exit(1)

    # Plan is built deterministically — same on every shard.
    rng_master = random.Random(MASTER_SEED)
    n_clips = max(1, math.ceil(args.target_count / (args.seeds_per_clip * args.frames_per_clip)))
    full_plan = build_plan(random.Random(MASTER_SEED), n_clips,
                            args.seeds_per_clip, args.frames_per_clip)
    # Sort the FULL plan by primary seed before any shard slicing — this
    # groups entries of the same seed together so that any contiguous
    # slice (whether given by --shard-start/--shard-end ranges or by the
    # legacy modulo sharding) will reuse the same character across many
    # consecutive plan entries.  Character caching kicks in here.
    full_plan.sort(key=lambda e: (e[1][0], e[0]))    # (primary seed, bvh path)

    # Shard assignment: range-based if provided, else legacy modulo.
    if args.shard_start is not None and args.shard_end is not None:
        shard_plan = full_plan[args.shard_start:args.shard_end]
        target_per_shard = math.ceil(
            args.target_count * (args.shard_end - args.shard_start) / max(1, len(full_plan)))
        print(f"[shard {args.shard_id}] range {args.shard_start}:{args.shard_end} "
              f"= {len(shard_plan)} clips (target {target_per_shard} samples)")
    else:
        shard_plan = [entry for i, entry in enumerate(full_plan)
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
    current_seed: int | None = None
    current_arm = None
    current_phenotype: dict | None = None
    current_basemesh = None                              # for surface kps
    current_surface_vi: list[int] = []                   # vertex indices
    depth_fout = None                                    # File Output node
    mask_fout = None
    depth_dir = out_dir / "depth"
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(exist_ok=True)

    for plan_i, (bvh_rel, seeds, n_frames, cams_per) in enumerate(shard_plan):
        bvh_abs = ASSETS_ROOT / bvh_rel
        if not bvh_abs.exists():
            print(f"[shard {args.shard_id}] skip missing {bvh_abs}")
            continue
        src_kind = detect_source_from_bvh(str(bvh_abs))
        clip_id = bvh_abs.stem

        for seed in seeds:
            # CHARACTER CACHING: only rebuild when seed changes.  Because
            # shard_plan is sorted by primary-seed, this means we rebuild
            # ~15 times per shard (once per unique seed) instead of once
            # per plan entry — a ~5-15x speedup on per-sample wall time.
            if seed != current_seed:
                clear_scene()
                _bm, current_arm = build_character(seed, with_assets=True)
                # Stash the MPFB phenotype we just built for later logging
                # into characters.jsonl (one row per unique character).
                try:
                    from lib.mpfb_build import sample_phenotype
                    _rng_stash = random.Random(seed)
                    current_phenotype = sample_phenotype(_rng_stash).to_mpfb()
                except Exception:
                    current_phenotype = None
                current_seed = seed
                # New scene = new camera + render config.
                cam = crig.make_camera("DatasetCam")
                bpy.context.scene.camera = cam
                configure_render(resolution=(IMAGE_W, IMAGE_H), engine="CYCLES",
                                 samples=args.cycles_samples,
                                 output_path=str(out_dir / "images" / "_.png"))
                _configure_cycles(args.cycles_samples, args.engine, args.threads)
                cam.rotation_mode = "XYZ"
                # Shadow-catcher plane at Z=0 — BEDLAM-style grounded
                # contact shadow on the HDRI background.  Cheap (<1 ms
                # extra render), meaningful photometric lift.
                try:
                    add_shadow_catcher_plane(size_m=8.0)
                except Exception as e:
                    print(f"  [warn] shadow catcher: {e}")
                # Depth/mask compositor output is blocked on a Blender
                # 5.1 compositor bug (File Output nodes don't execute
                # during headless render; see Blender issue #150625).
                # The scaffolding is preserved in render_setup.py but
                # disabled here — once the multilayer-EXR fallback or a
                # Blender patch is available, flip this back on.
                depth_fout = None
                mask_fout = None
                # Surface keypoints (CameraHMR DenseKP pattern) — 100
                # deterministic basemesh vertex indices subsampled
                # farthest-point-style in rest pose.  Re-selected once
                # per character because vertex count can depend on
                # per-character shape-key state.
                current_basemesh = _find_basemesh_object(f"subject_{seed:04d}")
                if current_basemesh is not None:
                    current_surface_vi = select_surface_vertex_indices(
                        current_basemesh, num=NUM_SURFACE_KPS, seed=0)
                else:
                    current_surface_vi = []
            arm = current_arm

            # Clean up previous clip's BVH armature before loading next one
            # (character-cache path: scene is not cleared between clips).
            # load_bvh marks its output armature with "_mocap_source".
            for _obj in list(bpy.data.objects):
                if _obj.type == "ARMATURE" and _obj.get("_mocap_source"):
                    bpy.data.objects.remove(_obj, do_unlink=True)
            bvh = load_bvh(str(bvh_abs), source=src_kind)
            ctx = RetargetContext(bvh, arm)

            # HDRI: swap ONCE per (seed, clip) and NOT per sample.  Rebuilding
            # the world-node-tree every sample (via set_world_hdri) was
            # invalidating scene.render.use_persistent_data every frame,
            # triggering a ~2 s scene-sync + HDRI-upload per render.  By
            # loading once per clip and only tweaking strength + rotation
            # per sample via update_hdri_params, persistent data survives
            # and renders run at their actual speed (~0.5 s on 5060 Ti).
            hdri_pool_idx = int(hashlib.sha1(
                f"{seed}_{bvh_rel}".encode()).hexdigest()[:4], 16) % len(hdris)
            current_hdri_path = hdris[hdri_pool_idx]
            set_world_hdri(str(current_hdri_path), strength=1.0, rotation_z=0.0)

            rng_clip = random.Random((seed * 100003) ^ hash(bvh_rel))
            frames = sample_frame_indices(ctx, n_frames, rng_clip)

            for frame in frames:
                ctx.apply_pose(frame)
                bpy.context.view_layer.update()

                for cam_i in range(cams_per):
                    sample_id = f"{src_kind}_{clip_id}_f{frame:04d}_s{seed}_c{cam_i}"
                    img_path = out_dir / "images" / f"{sample_id}.png"
                    if sample_id in done_ids:
                        continue

                    # Cheap per-sample lighting variation (preserves cache).
                    update_hdri_params(
                        strength=rng_clip.uniform(0.7, 1.6),
                        rotation_z=rng_clip.uniform(0.0, 6.283185))
                    hdri_idx = hdri_pool_idx
                    rng_cam = random.Random(
                        rng_master.randrange(1 << 30) ^ (frame << 8) ^ cam_i)
                    try:
                        focal = crig.sample_focal_mm(rng_cam)
                        cam_sample = crig.frame_armature_strict(
                            cam, arm, rng_cam, focal_mm=focal)
                    except Exception as e:
                        print(f"  [skip cam] {sample_id}: {e}")
                        continue
                    bpy.context.view_layer.update()

                    bpy.context.scene.render.filepath = str(img_path)
                    # Retarget compositor File Output nodes so sidecar
                    # files land in the right dirs with predictable names.
                    set_gt_sample_path(depth_fout, sample_id)
                    set_gt_sample_path(mask_fout, sample_id)
                    bpy.ops.render.render(write_still=True)
                    # Blender auto-appends the current scene frame number
                    # as a suffix to the File Output filenames.  Rename
                    # to stable `{sample_id}.{ext}` so labels reference
                    # by id without knowing Blender's frame counter.
                    import glob as _glob, os as _os
                    if depth_fout is not None:
                        pat = str(depth_dir / f"{sample_id}_*.exr")
                        for p in _glob.glob(pat):
                            try:
                                _os.replace(p, str(depth_dir / f"{sample_id}.exr"))
                                break
                            except OSError:
                                pass
                    if mask_fout is not None:
                        pat = str(mask_dir / f"{sample_id}_*.png")
                        for p in _glob.glob(pat):
                            try:
                                _os.replace(p, str(mask_dir / f"{sample_id}.png"))
                                break
                            except OSError:
                                pass

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

                    # --- Pre-write SANITY GATES ---
                    # Drop samples that are obviously broken before they
                    # pollute the dataset.  Cheap to run, saves having to
                    # filter labels.jsonl after the fact.
                    n_kps_inside = sum(1 for v in vis_flags if v == 2)
                    reject = None
                    if n_kps_inside < 10:
                        reject = f"only {n_kps_inside}/17 COCO kps inside image"
                    elif bbox[2] < 10 or bbox[3] < 10:
                        reject = f"degenerate bbox {bbox}"
                    elif root is None:
                        reject = "missing root (L/R hip 3D)"
                    else:
                        rz = float(root[2])
                        if not math.isfinite(rz) or rz < 0.3 or rz > 50.0:
                            reject = f"root_z out of range ({rz:.2f}m)"
                        if any(not math.isfinite(c) for p in kps3d_cam
                                for c in (p if p is not None else (0.0,))):
                            reject = "NaN/inf in kps3d"
                    if reject is not None:
                        print(f"  [gate] skip {sample_id}: {reject}")
                        # Don't count toward stats; try next frame/camera.
                        continue

                    # --- Surface keypoints (CameraHMR DenseKP) ---
                    if current_basemesh is not None and current_surface_vi:
                        skp = compute_surface_kps(
                            current_basemesh, current_surface_vi,
                            (IMAGE_W, IMAGE_H), cam, bpy.context.scene)
                        surface_kps_2d = skp["surface_kps_2d"]
                        surface_kps_3d = skp["surface_kps_3d_cam"]
                    else:
                        surface_kps_2d = []
                        surface_kps_3d = []

                    # character_id groups rendered samples by seed (one
                    # built-MPFB character per seed).  video_id groups
                    # samples by (seed, bvh_clip) — the granularity at
                    # which temporal lifters (MotionAGFormer) window.
                    character_id = f"s{seed:04d}"
                    video_id = f"{character_id}_{clip_id}"
                    record = {
                        "id": sample_id, "split": split,
                        "image_rel": f"images/{sample_id}.png",
                        "depth_rel": (f"depth/{sample_id}.exr"
                                       if depth_fout is not None else None),
                        "mask_rel": (f"masks/{sample_id}.png"
                                      if mask_fout is not None else None),
                        "image_wh": [IMAGE_W, IMAGE_H],
                        "camera_K": intr_dict,
                        "camera_extrinsics": extrinsics,
                        "bbox_xywh": bbox,
                        "keypoints_2d": [[kp[0], kp[1], vf] for kp, vf in zip(kps2d_pixels, vis_flags)],
                        "keypoints_3d_cam": kps3d_cam,
                        "root_joint_cam": root,
                        # Dense surface keypoints (CameraHMR DenseKP
                        # pattern).  Fixed vertex indices per character
                        # so the same surface point means the same
                        # anatomical location across every sample of
                        # that subject — enables shape-supervision
                        # training without a full SMPL refactor.
                        "surface_kps_2d": surface_kps_2d,
                        "surface_kps_3d_cam": surface_kps_3d,
                        "source": src_kind, "clip_id": clip_id,
                        "video_id": video_id,
                        "character_id": character_id,
                        "frame_idx": frame, "character_seed": seed,
                        "hdri": hdris[hdri_idx].stem,
                        "hdri_strength": float(getattr(cam_sample, "hdri_strength", 1.0)),
                        "focal_mm": float(cam_sample.intrinsics.focal_mm),
                        "shift_x": float(cam_sample.intrinsics.shift_x),
                        "shift_y": float(cam_sample.intrinsics.shift_y),
                        "camera_yaw": float(cam_sample.yaw),
                        "camera_pitch": float(cam_sample.pitch),
                        "camera_distance": float(cam_sample.distance),
                        "render_engine": f"CYCLES_{bpy.context.scene.cycles.device}",
                        "shard_id": args.shard_id,
                    }
                    labels_fh.write(json.dumps(record) + "\n"); labels_fh.flush()
                    # Persist per-character phenotype ONCE per unique seed
                    # (reading the same MPFB modeling state on every frame
                    # would be wasteful; the phenotype is constant within
                    # a seed).
                    if character_id not in written_character_ids:
                        char_row = {
                            "character_id": character_id,
                            "seed": seed,
                            "phenotype": current_phenotype,
                        }
                        characters_fh.write(json.dumps(char_row) + "\n")
                        characters_fh.flush()
                        written_character_ids.add(character_id)
                    manifest_fh.write(
                        f"{sample_id},{split},{src_kind},{record['image_rel']},"
                        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
                    manifest_fh.flush()

                    stats["total"] += 1
                    stats[src_kind] = stats.get(src_kind, 0) + 1
                    stats[split] += 1
                    sample_counter += 1
                    if sample_counter % 10 == 0:
                        dt = max(1e-3, time.time() - t_start)
                        rate = sample_counter / dt
                        rem = max(0, target_per_shard - stats["total"])
                        eta = int(rem / rate) if rate > 0 else -1
                        eta_s = f"{eta//3600:d}h{(eta%3600)//60:02d}m" if eta >= 0 else "?"
                        print(f"[shard {args.shard_id}] "
                              f"[{stats['total']:6d}/{target_per_shard}] "
                              f"{rate:4.2f}/s ETA {eta_s}")
                    if args.limit and stats["total"] >= args.limit:
                        labels_fh.close(); manifest_fh.close(); characters_fh.close(); return
                    if stats["total"] >= target_per_shard:
                        print(f"[shard {args.shard_id}] reached target, stopping")
                        labels_fh.close(); manifest_fh.close(); characters_fh.close(); return

    labels_fh.close(); manifest_fh.close(); characters_fh.close()
    print(f"[shard {args.shard_id}] done — {stats['total']} samples")


if __name__ == "__main__":
    main()

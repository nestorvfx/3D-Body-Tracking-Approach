"""Pilot: render N samples end-to-end and measure wall-clock time.

Usage:
  "/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" \
      --background --python dataset/scripts/pilot.py -- [N]

Pilot defaults to N=10 samples. Output goes to dataset/output/pilot_NN/.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

# --- Fix up sys.path so `lib` is importable regardless of cwd. ---
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --- Defer MPFB import until after the extension is guaranteed to be enabled. ---
import bpy  # type: ignore


def ensure_mpfb_enabled() -> None:
    """Enable the MPFB extension and register a top-level `mpfb` alias so that
    `from mpfb.services.* import *` works (Blender 5.x extensions expose the
    package as `bl_ext.user_default.mpfb`, not `mpfb`)."""
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except Exception as e:
        # Might already be enabled; non-fatal.
        print(f"[pilot] addon_enable note: {e}")

    import importlib, sys as _sys
    try:
        pkg = importlib.import_module("bl_ext.user_default.mpfb")
    except Exception as e:
        print(f"[pilot] cannot import bl_ext.user_default.mpfb: {e}")
        return

    # Alias the package and its submodules as `mpfb.*` so legacy imports work.
    _sys.modules["mpfb"] = pkg
    # Eagerly register common subpackages.
    for sub in ("services", "entities", "ui"):
        try:
            m = importlib.import_module(f"bl_ext.user_default.mpfb.{sub}")
            _sys.modules[f"mpfb.{sub}"] = m
        except Exception:
            pass
    # Register known service modules so `from mpfb.services.humanservice import ...`
    # resolves without triggering a fresh import under the legacy name.
    for svc in (
        "humanservice", "targetservice", "assetservice", "rigservice",
        "locationservice", "materialservice", "clothesservice",
        "objectservice", "animationservice", "logservice", "systemservice",
    ):
        try:
            m = importlib.import_module(f"bl_ext.user_default.mpfb.services.{svc}")
            _sys.modules[f"mpfb.services.{svc}"] = m
        except Exception:
            pass


ensure_mpfb_enabled()

from lib.coco17 import (  # noqa: E402
    COCO17_NAMES, COCO17_SKELETON,
    get_coco17_world, project_to_pixels,
)
from lib.mpfb_build import build_character  # noqa: E402
from lib.retarget import retarget_bvh_to_mpfb  # noqa: E402
from lib.render_setup import (  # noqa: E402
    configure_render, set_world_hdri, add_camera, frame_armature, clear_scene,
)


REPO = HERE.parent
BVH_DIR = REPO / "assets" / "bvh"
HDRI_DIR = REPO / "assets" / "hdris"


FOCAL_RANGE_MM = [18, 24, 35, 50, 85, 135]   # BEDLAM 2.0-style spread


def parse_argv() -> tuple[int, str]:
    # Blender passes user args after '--'
    if "--" in sys.argv:
        user_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        user_args = []
    n = int(user_args[0]) if user_args else 10
    out_tag = user_args[1] if len(user_args) > 1 else f"pilot_{n:02d}"
    return n, out_tag


def pick_assets(n_samples: int, seed: int = 0):
    bvh_files = sorted(BVH_DIR.glob("*.bvh"))
    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH in {BVH_DIR}")
    if not hdri_files:
        raise FileNotFoundError(f"No HDRIs in {HDRI_DIR}")
    rng = random.Random(seed)
    plans = []
    for i in range(n_samples):
        plans.append({
            "sample_id": i,
            "bvh": str(bvh_files[i % len(bvh_files)]),
            "hdri": str(hdri_files[i % len(hdri_files)]),
            "focal_mm": FOCAL_RANGE_MM[i % len(FOCAL_RANGE_MM)],
            "hdri_rotation": rng.uniform(0, 2 * math.pi),
            "cam_seed": rng.randrange(10**6),
            "char_seed": 1000 + i,
        })
    return plans


def render_one(plan: dict, out_dir: Path) -> dict:
    """Render a single pilot sample. Returns per-sample metadata dict."""
    t0 = time.time()
    clear_scene()
    t_clear = time.time() - t0

    # 1. Character
    t1 = time.time()
    basemesh, armature = build_character(plan["char_seed"], with_assets=True)
    if armature is None:
        raise RuntimeError("No armature produced by build_character")
    t_char = time.time() - t1

    # 2. Retarget BVH
    t2 = time.time()
    bake_start, bake_end = retarget_bvh_to_mpfb(plan["bvh"], armature)
    t_retarget = time.time() - t2

    # Pick a pose frame roughly in the middle of the clip.
    pose_frame = bake_start + (bake_end - bake_start) // 2
    bpy.context.scene.frame_set(pose_frame)
    bpy.context.view_layer.update()

    # 3. Camera + HDRI
    t3 = time.time()
    cam = add_camera(focal_length_mm=plan["focal_mm"])
    rng = random.Random(plan["cam_seed"])
    yaw, pitch, dist = frame_armature(cam, armature, rng=rng)
    set_world_hdri(plan["hdri"], strength=1.0, rotation_z=plan["hdri_rotation"])
    t_cam = time.time() - t3

    # 4. Render
    png_path = out_dir / f"sample_{plan['sample_id']:04d}.png"
    configure_render(
        resolution=(1280, 720),
        engine="BLENDER_EEVEE",
        samples=32,
        output_path=png_path,
    )
    t4 = time.time()
    bpy.ops.render.render(write_still=True)
    t_render = time.time() - t4

    # 5. Annotations: COCO-17 in 3D (world) + 2D (pixel)
    kps_world = get_coco17_world(armature)
    scene = bpy.context.scene
    projections = project_to_pixels(scene, cam, kps_world)

    rx = scene.render.resolution_x
    ry = scene.render.resolution_y
    kps_2d = []
    kps_3d = []
    for i, (w, proj) in enumerate(zip(kps_world, projections)):
        if w is None or proj is None:
            kps_2d.append({"u": None, "v": None, "px": None, "py": None,
                            "visible": False, "depth": None})
            kps_3d.append({"x": None, "y": None, "z": None})
            continue
        u, v, depth, inside = proj
        kps_2d.append({
            "u": u, "v": v,
            "px": u * rx, "py": v * ry,
            "visible": bool(inside),
            "depth": depth,
        })
        kps_3d.append({"x": w[0], "y": w[1], "z": w[2]})

    meta = {
        "sample_id": plan["sample_id"],
        "png": png_path.name,
        "bvh": Path(plan["bvh"]).name,
        "hdri": Path(plan["hdri"]).name,
        "focal_mm": plan["focal_mm"],
        "hdri_rotation_rad": plan["hdri_rotation"],
        "pose_frame": pose_frame,
        "bake_start": bake_start,
        "bake_end": bake_end,
        "camera": {
            "yaw_rad": yaw, "pitch_rad": pitch, "distance_m": dist,
            "location_xyz": list(cam.location),
            "rotation_euler_xyz": list(cam.rotation_euler),
        },
        "resolution": [rx, ry],
        "keypoint_names": COCO17_NAMES,
        "keypoints_3d_world_m": kps_3d,
        "keypoints_2d_pixels": kps_2d,
        "skeleton_edges": COCO17_SKELETON,
        "timing_sec": {
            "clear": t_clear, "character": t_char, "retarget": t_retarget,
            "camera_hdri": t_cam, "render": t_render,
            "total": time.time() - t0,
        },
    }
    return meta


def main() -> int:
    n, out_tag = parse_argv()
    out_dir = REPO / "output" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] rendering {n} samples → {out_dir}", flush=True)

    plans = pick_assets(n_samples=n, seed=42)
    all_meta = []
    timings = []
    errors = []

    t_all = time.time()
    for plan in plans:
        try:
            meta = render_one(plan, out_dir)
            all_meta.append(meta)
            timings.append(meta["timing_sec"]["total"])
            print(f"[pilot] sample {plan['sample_id']} total={timings[-1]:.2f}s "
                  f"render={meta['timing_sec']['render']:.2f}s", flush=True)
        except Exception as e:
            print(f"[pilot] sample {plan['sample_id']} FAILED: {e}", flush=True)
            traceback.print_exc()
            errors.append({"sample_id": plan["sample_id"], "error": str(e)})
    total = time.time() - t_all

    summary = {
        "n_samples": n,
        "n_ok": len(all_meta),
        "n_errors": len(errors),
        "wall_clock_sec": total,
        "per_sample_avg_sec": (sum(timings) / len(timings)) if timings else None,
        "errors": errors,
        "samples": all_meta,
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[pilot] DONE in {total:.1f}s ({summary['n_ok']}/{n} OK).", flush=True)
    return 0 if summary["n_ok"] == n else 3


if __name__ == "__main__":
    sys.exit(main())

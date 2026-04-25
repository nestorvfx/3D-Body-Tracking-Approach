"""Pilot v2: sequence-rendering pipeline with BEDLAM-style decimation,
camera-intrinsics diversity, intelligent framing, fast retarget.

Improvements over `pilot.py`:
  * Intelligent framing (`camera_rig.frame_armature_strict`) guarantees all 17
    COCO-17 keypoints land inside the frame with margin.
  * Camera diversity: lognormal focal 14-400mm, Perlin handheld shake,
    synthetic motions (static / pan / orbit / dolly / track / zoom),
    principal-point offset, DoF.
  * Fast retargeting: depsgraph-eval + direct pose bone matrix write,
    no `bpy.ops.nla.bake`.  <100 ms/frame vs ~16 s/sample in v1.
  * Short sequences (default 12 raw frames) decimated by 3 for a BEDLAM-
    style diverse-yet-amortised output stream.

Usage:
    blender --background --python dataset/scripts/pilot_v2.py -- N [out_tag]

Default N=10 sequences with raw=12 decimation=3  => 40 kept frames.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import bpy  # type: ignore

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def ensure_mpfb_enabled() -> None:
    try:
        bpy.ops.preferences.addon_enable(module="bl_ext.user_default.mpfb")
    except Exception as e:
        print(f"[pilot_v2] addon_enable note: {e}")
    import importlib, sys as _sys
    try:
        pkg = importlib.import_module("bl_ext.user_default.mpfb")
    except Exception as e:
        print(f"[pilot_v2] cannot import bl_ext.user_default.mpfb: {e}")
        return
    _sys.modules["mpfb"] = pkg
    for sub in ("services", "entities", "ui"):
        try:
            _sys.modules[f"mpfb.{sub}"] = importlib.import_module(
                f"bl_ext.user_default.mpfb.{sub}")
        except Exception:
            pass
    for svc in ("humanservice", "targetservice", "assetservice", "rigservice",
                "locationservice", "materialservice", "clothesservice",
                "objectservice", "animationservice", "logservice",
                "systemservice"):
        try:
            _sys.modules[f"mpfb.services.{svc}"] = importlib.import_module(
                f"bl_ext.user_default.mpfb.services.{svc}")
        except Exception:
            pass


ensure_mpfb_enabled()

from lib.mpfb_build import build_character  # noqa: E402
from lib.sequence_render import SequenceConfig, render_sequence  # noqa: E402
from lib.render_setup import clear_scene  # noqa: E402


REPO = HERE.parent
BVH_DIR = REPO / "assets" / "bvh"
HDRI_DIR = REPO / "assets" / "hdris"


def parse_argv():
    user_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    n = int(user_args[0]) if user_args else 10
    out_tag = user_args[1] if len(user_args) > 1 else f"pilot_v2_{n:02d}"
    return n, out_tag


def pick_assets(n_sequences: int, seed: int = 42):
    bvh_files = sorted(BVH_DIR.glob("*.bvh"))
    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))
    if not bvh_files:
        raise FileNotFoundError(f"No BVH in {BVH_DIR}")
    if not hdri_files:
        raise FileNotFoundError(f"No HDRIs in {HDRI_DIR}")
    rng = random.Random(seed)
    plans = []
    for i in range(n_sequences):
        plans.append({
            "sequence_id": i,
            "bvh": str(bvh_files[i % len(bvh_files)]),
            "hdri": str(hdri_files[i % len(hdri_files)]),
            "char_seed": 2000 + i,
            "rng_seed": rng.randrange(10**6),
        })
    return plans


def main() -> int:
    n, out_tag = parse_argv()
    out_dir = REPO / "output" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pilot_v2] {n} sequences -> {out_dir}", flush=True)

    plans = pick_assets(n)
    summaries = []
    errors = []
    t_all = time.time()

    for plan in plans:
        t0 = time.time()
        try:
            clear_scene()
            # Character
            t_c = time.time()
            basemesh, armature = build_character(plan["char_seed"], with_assets=True)
            if armature is None:
                raise RuntimeError("build_character returned no armature")
            t_char = time.time() - t_c

            cfg = SequenceConfig(
                sequence_id=plan["sequence_id"],
                bvh_path=plan["bvh"],
                hdri_path=plan["hdri"],
                character_seed=plan["char_seed"],
                out_dir=out_dir,
                # ~1 second at 120 fps BVH; with adaptive stride (175 ms target)
                # we get ~5 kept frames spaced evenly across the window.
                seq_frames=120,
                adaptive_stride=True,
                stride_target_ms=175.0,
                max_kept_frames=5,
                resolution=(1280, 720),
                engine="BLENDER_EEVEE",
                samples=32,
            )
            rng = random.Random(plan["rng_seed"])
            result = render_sequence(cfg, rng)
            tot = time.time() - t0
            print(f"[pilot_v2] seq {plan['sequence_id']}: {len(result.frames)} kept, "
                  f"char={t_char:.2f}s retarget_init={result.timing_sec['retarget_init']:.2f}s "
                  f"cam={result.timing_sec['camera_hdri']:.2f}s "
                  f"render={result.timing_sec['render_total']:.2f}s total={tot:.2f}s", flush=True)

            summaries.append({
                "sequence_id": plan["sequence_id"],
                "char_seed": plan["char_seed"],
                "bvh": Path(plan["bvh"]).name,
                "hdri": Path(plan["hdri"]).name,
                "kept_frames": len(result.frames),
                "timing_sec": {**result.timing_sec, "character": t_char, "total": tot},
            })
        except Exception as e:
            print(f"[pilot_v2] seq {plan['sequence_id']} FAILED: {e}", flush=True)
            traceback.print_exc()
            errors.append({"sequence_id": plan["sequence_id"], "error": str(e)})

    wall = time.time() - t_all
    total_kept = sum(s["kept_frames"] for s in summaries)
    summary = {
        "n_sequences": n,
        "n_ok": len(summaries),
        "n_errors": len(errors),
        "wall_clock_sec": wall,
        "total_kept_frames": total_kept,
        "per_sample_avg_sec": (wall / total_kept) if total_kept else None,
        "errors": errors,
        "sequences": summaries,
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[pilot_v2] DONE in {wall:.1f}s ({summary['n_ok']}/{n} seqs, "
          f"{total_kept} kept frames, {summary['per_sample_avg_sec'] or 0:.2f}s/kept).",
          flush=True)
    return 0 if summary["n_ok"] == n else 3


if __name__ == "__main__":
    sys.exit(main())

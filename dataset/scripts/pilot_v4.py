"""Pilot v4 — SOTA plausibility stack end-to-end.

Additions over v3:
  * Clip-level BVH sanitizer (NaN/teleport/T-pose/drift rejection)
  * Per-frame plausibility validator (OpenSim-derived ROM, ground penetration,
    bone-length drift, self-intersection via BVHTree)
  * Real-jump vs float-artifact parabola discriminator
  * Velocity-threshold foot-lock IK for detected artifacts
  * Multi-source motion library (CMU, 100STYLE, AIST++ when converted, MHAD
    when converted) via `motion_loader.py`
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
        print(f"[pilot_v4] addon_enable note: {e}")
    import importlib, sys as _sys
    try:
        pkg = importlib.import_module("bl_ext.user_default.mpfb")
    except Exception as e:
        print(f"[pilot_v4] cannot import bl_ext.user_default.mpfb: {e}")
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

from lib.mpfb_build import build_character                         # noqa: E402
from lib.sequence_render import SequenceConfig, render_sequence     # noqa: E402
from lib.render_setup import clear_scene                             # noqa: E402
from lib.motion_loader import load_all_clips, license_manifest       # noqa: E402
from lib.activity_tags import parse_cmu_index                        # noqa: E402
from lib.sampling import (                                           # noqa: E402
    SamplerConfig, build_catalog, sample_plans, report_distribution,
)


REPO = HERE.parent
ASSETS = REPO / "assets"
HDRI_DIR = ASSETS / "hdris"


def parse_argv():
    user_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    n = int(user_args[0]) if user_args else 10
    out_tag = user_args[1] if len(user_args) > 1 else f"pilot_v4_{n:02d}"
    video_mode = ("--video" in user_args) or ("video" in user_args)
    return n, out_tag, video_mode


def pick_assets(
    n_sequences: int,
    seed: int = 42,
    sampler_cfg: SamplerConfig | None = None,
):
    """Source + category stratified picks.

    Uses SamplerConfig to enforce per-source and per-category shares regardless
    of how many clips each source contributes to the corpus.  Picks `n_sequences`
    tagged clips, then attaches HDRI + char_seed + rng_seed metadata.
    """
    all_clips = load_all_clips(ASSETS)
    if not all_clips:
        raise FileNotFoundError(f"No motion clips found under {ASSETS}")
    hdri_files = sorted(HDRI_DIR.glob("*.hdr"))
    if not hdri_files:
        raise FileNotFoundError(f"No HDRIs in {HDRI_DIR}")

    # Load CMU descriptions for activity tagging (if available).
    cmu_index = ASSETS / "bvh" / "cmuindex.txt"
    cmu_descs: dict[str, str] = {}
    if cmu_index.exists():
        cmu_descs = parse_cmu_index(cmu_index.read_text(errors="ignore"))

    catalog = build_catalog(all_clips, cmu_descs)
    rng = random.Random(seed)
    picks = sample_plans(catalog, n_sequences, rng, sampler_cfg)

    plans = []
    for i, tc in enumerate(picks):
        plans.append({
            "sequence_id": i,
            "bvh": str(tc.clip.path),
            "hdri": str(hdri_files[i % len(hdri_files)]),
            "char_seed": 3000 + i,
            "rng_seed": rng.randrange(10**6),
            "source": tc.source,
            "category": tc.category,
            "description": tc.clip.description,
        })
    return plans, license_manifest(all_clips), report_distribution(picks)


def main() -> int:
    n, out_tag, video_mode = parse_argv()
    out_dir = REPO / "output" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "VIDEO (coherent sequences)" if video_mode else "SINGLE-FRAME (per-frame camera)"
    print(f"[pilot_v4] {n} sequences -> {out_dir}  [{mode}]", flush=True)

    plans, manifest, distribution = pick_assets(n)
    (out_dir / "license_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "sampling_distribution.json").write_text(json.dumps(distribution, indent=2))
    print(f"[pilot_v4] source shares: {distribution['source_shares']}")
    print(f"[pilot_v4] category shares: {distribution['category_shares']}")

    summaries = []
    errors = []
    t_all = time.time()

    for plan in plans:
        t0 = time.time()
        try:
            clear_scene()
            t_c = time.time()
            basemesh, armature = build_character(plan["char_seed"], with_assets=True)
            if armature is None:
                raise RuntimeError("build_character returned no armature")
            t_char = time.time() - t_c

            # Video mode: MotionAGFormer expects 27-frame windows at ~30fps
            # (≈0.9s coherent clips), so render 120 BVH frames (1s at 120fps)
            # with tighter stride to produce contiguous training samples.
            # Single-frame mode: wider stride, fewer kept frames, per-frame
            # camera diversity wins over motion coherence.
            if video_mode:
                seq_frames_cfg = 216           # 1.8s at 120fps BVH
                stride_ms = 33.3               # 30 kept effective fps
                max_kept = 27                  # MotionAGFormer window
            else:
                seq_frames_cfg = 180
                stride_ms = 280.0
                max_kept = 4

            cfg = SequenceConfig(
                sequence_id=plan["sequence_id"],
                bvh_path=plan["bvh"],
                hdri_path=plan["hdri"],
                character_seed=plan["char_seed"],
                out_dir=out_dir,
                seq_frames=seq_frames_cfg,
                adaptive_stride=True,
                stride_target_ms=stride_ms,
                max_kept_frames=max_kept,
                coherent_sequence=video_mode,
                # SOTA plausibility stack
                sanitize_bvh=True,
                validate_poses=True,
                enable_foot_lock=True,
                detect_real_jumps=True,
                ground_lift=True,
                apply_realism=True,
                resolution=(1280, 720),
                engine="BLENDER_EEVEE",
                samples=32,
            )
            rng = random.Random(plan["rng_seed"])
            result = render_sequence(cfg, rng)
            tot = time.time() - t0
            if result.frames:
                t_s = result.timing_sec
                print(f"[pilot_v4] seq {plan['sequence_id']} "
                      f"({plan['source']}/{plan.get('category', '?')}): "
                      f"{len(result.frames)} kept, char={t_char:.2f}s "
                      f"render={t_s.get('render_total', 0):.2f}s "
                      f"validation={t_s.get('validation_total', 0):.2f}s "
                      f"total={tot:.2f}s", flush=True)
            else:
                reason = result.timing_sec.get("rejected_by_sanitizer", "unknown")
                print(f"[pilot_v4] seq {plan['sequence_id']} ({plan['source']}): "
                      f"REJECTED clip ({reason}) in {tot:.2f}s", flush=True)
            summaries.append({
                "sequence_id": plan["sequence_id"],
                "source": plan["source"],
                "category": plan.get("category"),
                "description": plan["description"],
                "char_seed": plan["char_seed"],
                "bvh": Path(plan["bvh"]).name,
                "hdri": Path(plan["hdri"]).name,
                "kept_frames": len(result.frames),
                "timing_sec": {**result.timing_sec, "character": t_char, "total": tot},
            })
        except Exception as e:
            print(f"[pilot_v4] seq {plan['sequence_id']} FAILED: {e}", flush=True)
            traceback.print_exc()
            errors.append({"sequence_id": plan["sequence_id"], "error": str(e)})

    wall = time.time() - t_all
    total_kept = sum(s["kept_frames"] for s in summaries)
    summary = {
        "n_sequences": n,
        "n_ok": sum(1 for s in summaries if s["kept_frames"] > 0),
        "n_errors": len(errors),
        "wall_clock_sec": wall,
        "total_kept_frames": total_kept,
        "per_sample_avg_sec": (wall / total_kept) if total_kept else None,
        "source_distribution": {k: sum(1 for s in summaries if s["source"] == k)
                                 for k in {"cmu", "100style", "aistpp", "mhad"}},
        "license_manifest": manifest,
        "errors": errors,
        "sequences": summaries,
    }
    (out_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[pilot_v4] DONE in {wall:.1f}s ({summary['n_ok']}/{n} seqs, "
          f"{total_kept} kept, {(summary['per_sample_avg_sec'] or 0):.2f}s/kept).",
          flush=True)
    return 0 if summary["n_ok"] >= n * 0.5 else 3


if __name__ == "__main__":
    sys.exit(main())

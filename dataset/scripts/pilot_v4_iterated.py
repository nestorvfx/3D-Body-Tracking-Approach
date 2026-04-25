"""Blender-side entrypoint for the diversity iteration loop.

Reads sampler weights from env var BODYTRACK_SAMPLER_WEIGHTS (path to JSON),
invokes the same pipeline as pilot_v4.py, writes output under the tag given
as the second positional `--` arg.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Import pilot_v4 after path is set so MPFB init runs in its module.
import bpy   # type: ignore  # ensures Blender python

# Defer import until after sys.path manipulation
from pilot_v4 import ensure_mpfb_enabled, pick_assets, REPO, HDRI_DIR      # noqa: E402

ensure_mpfb_enabled()

from lib.mpfb_build import build_character                         # noqa: E402
from lib.sequence_render import SequenceConfig, render_sequence     # noqa: E402
from lib.render_setup import clear_scene                             # noqa: E402
from lib.sampling import SamplerConfig                               # noqa: E402
import random, time, traceback                                       # noqa: E402


def parse_argv():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    n = int(args[0]) if args else 10
    tag = args[1] if len(args) > 1 else "iter_00"
    return n, tag


def main() -> int:
    n, tag = parse_argv()
    weights_path = os.environ.get("BODYTRACK_SAMPLER_WEIGHTS")
    if weights_path and Path(weights_path).exists():
        weights_cfg = json.loads(Path(weights_path).read_text())
        scfg = SamplerConfig(
            source_weights=weights_cfg.get("source", {}),
            category_weights=weights_cfg.get("category", {}),
        )
    else:
        scfg = SamplerConfig()

    out_dir = REPO / "output" / "diversity_iter" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plans, manifest, distribution = pick_assets(n, sampler_cfg=scfg)
    (out_dir / "license_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "sampling_distribution.json").write_text(json.dumps(distribution, indent=2))
    print(f"[iter] source_shares={distribution['source_shares']}")
    print(f"[iter] category_shares={distribution['category_shares']}")

    summaries = []
    t_all = time.time()
    for plan in plans:
        t0 = time.time()
        try:
            clear_scene()
            basemesh, armature = build_character(plan["char_seed"], with_assets=True)
            if armature is None:
                continue
            cfg = SequenceConfig(
                sequence_id=plan["sequence_id"],
                bvh_path=plan["bvh"],
                hdri_path=plan["hdri"],
                character_seed=plan["char_seed"],
                out_dir=out_dir,
                seq_frames=120,
                adaptive_stride=True,
                stride_target_ms=175.0,
                max_kept_frames=3,           # smaller per iter for speed
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
            res = render_sequence(cfg, rng)
            summaries.append({
                "sequence_id": plan["sequence_id"],
                "source": plan["source"],
                "category": plan.get("category"),
                "kept_frames": len(res.frames),
                "total_sec": time.time() - t0,
            })
            print(f"[iter] seq {plan['sequence_id']} "
                  f"({plan['source']}/{plan.get('category')}): "
                  f"{len(res.frames)} kept in {time.time()-t0:.2f}s", flush=True)
        except Exception as e:
            print(f"[iter] seq {plan['sequence_id']} FAILED: {e}", flush=True)
            traceback.print_exc()

    (out_dir / "iter_summary.json").write_text(
        json.dumps({"distribution": distribution,
                     "wall_clock_sec": time.time() - t_all,
                     "sequences": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

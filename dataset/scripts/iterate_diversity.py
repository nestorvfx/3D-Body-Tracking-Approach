"""Diversity iteration loop.

Renders a small pilot batch, measures pose-diversity metrics, adjusts
sampler weights to boost under-represented categories, re-renders, repeats
until diversity targets are met or MAX_ITER exhausted.

Usage (from repo root, system Python — does NOT require Blender):
    python dataset/scripts/iterate_diversity.py --batch 10 --iters 3

Each iteration:
  1. Picks clip plans via sampler (current weights) and writes plan.json
  2. Invokes Blender headless to render the plans into a tagged output dir
  3. Loads the rendered annotations and computes diversity metrics
  4. Adjusts category_weights (Pareto boost) based on under-filled categories
  5. Logs iteration row to diversity_log.csv
  6. Stops when APD and log-volume metrics both plateau
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.sampling import SamplerConfig, DEFAULT_CATEGORY_WEIGHTS
from lib.diversity import diversity_report


BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
PIPELINE_SCRIPT = str(HERE / "pilot_v4_iterated.py")


def adjust_category_weights(
    current: dict[str, float],
    shares: dict[str, float],
    target: dict[str, float],
    step: float = 0.30,
) -> dict[str, float]:
    """Boost categories whose observed share is below the target by a fraction.

    Keeps the weights strictly positive and renormalised to sum = 1.
    """
    out = dict(current)
    for cat, target_share in target.items():
        observed = shares.get(cat, 0.0)
        if observed < target_share * 0.7:          # under-represented by >30%
            out[cat] = out.get(cat, 0.01) * (1.0 + step)
        elif observed > target_share * 1.3:        # over-represented by >30%
            out[cat] = out.get(cat, 0.01) * max(0.5, 1.0 - step)
    # Renormalise
    total = sum(out.values())
    if total <= 0:
        return current
    return {k: v / total for k, v in out.items()}


def run_iteration(
    iter_idx: int,
    out_root: Path,
    batch: int,
    sampler_weights: dict,
) -> tuple[dict, dict]:
    """Invoke Blender + pipeline for one batch, return (distribution, diversity)."""
    tag = f"iter_{iter_idx:02d}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write weights JSON so the Blender subprocess picks them up.
    weights_path = out_dir / "sampler_weights.json"
    weights_path.write_text(json.dumps(sampler_weights, indent=2))

    # Invoke the pilot in Blender (SUBPROCESS — we're running in system Python).
    env = dict(os.environ)
    env["BODYTRACK_SAMPLER_WEIGHTS"] = str(weights_path)
    env["BODYTRACK_OUT_TAG"] = tag
    env["BODYTRACK_BATCH"] = str(batch)
    t0 = time.time()
    proc = subprocess.run(
        [BLENDER_EXE, "--background", "--python", PIPELINE_SCRIPT,
         "--", str(batch), tag],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    elapsed = time.time() - t0
    last_lines = "\n".join(proc.stdout.splitlines()[-6:])
    print(f"  [iter {iter_idx}] Blender finished in {elapsed:.1f}s (rc={proc.returncode})")
    if proc.returncode != 0:
        print(f"  [iter {iter_idx}] stderr tail:\n    " + "\n    ".join(
            proc.stderr.splitlines()[-8:]))

    # Read back the distribution and compute diversity
    dist_path = out_dir / "sampling_distribution.json"
    distribution = json.loads(dist_path.read_text()) if dist_path.exists() else {}
    div = diversity_report(out_dir)
    return distribution, div


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10,
                     help="sequences per iteration")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--apd-target", type=float, default=1.2,
                     help="stop when APD ≥ target")
    ap.add_argument("--out", type=Path, default=Path("dataset/output/diversity_iter"))
    args = ap.parse_args()

    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # Starting weights: defaults
    weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    target = dict(DEFAULT_CATEGORY_WEIGHTS)
    source_weights = {
        "cmu": 0.40, "100style": 0.25, "aistpp": 0.25, "mhad": 0.10,
    }

    # CSV log
    csv_path = out_root / "diversity_log.csv"
    write_header = not csv_path.exists()
    csv_f = csv_path.open("a", newline="")
    w = csv.writer(csv_f)
    if write_header:
        w.writerow(["iter", "batch", "apd_m", "log_pca_volume",
                     "bone_dir_entropy", "n_frames",
                     *(f"cat_{c}" for c in target.keys())])

    last_apd = -float("inf")
    for it in range(args.iters):
        print(f"\n=== Iteration {it} ===  weights={weights}")
        sampler = {"source": source_weights, "category": weights}
        distribution, div = run_iteration(it, out_root, args.batch, sampler)
        print(f"  distribution: {distribution.get('category_shares', {})}")
        print(f"  diversity:    APD={div['apd_m']:.3f}m  "
              f"log_vol={div['log_pca_volume']:.2f}  "
              f"entropy={div['bone_dir_entropy']:.2f}  n={div['n_frames']}")
        w.writerow([it, args.batch, div["apd_m"], div["log_pca_volume"],
                     div["bone_dir_entropy"], div["n_frames"],
                     *[distribution.get("category_shares", {}).get(c, 0.0)
                        for c in target.keys()]])
        csv_f.flush()

        if div["apd_m"] >= args.apd_target:
            print(f"  ✓ APD target {args.apd_target}m reached, stopping.")
            break
        if abs(div["apd_m"] - last_apd) < 0.01 and it > 0:
            print("  ✓ APD plateaued, stopping.")
            break
        last_apd = div["apd_m"]

        # Adjust category weights for next iteration
        shares = distribution.get("category_shares", {})
        weights = adjust_category_weights(weights, shares, target)

    csv_f.close()
    print(f"\n[iterate_diversity] Log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

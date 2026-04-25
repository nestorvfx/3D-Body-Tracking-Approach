# Dataset Generator — Cherry CPU or Vast.ai GPU

Standalone folder for generating a 500k-sample 3D-pose synthetic dataset.
Same code works on **Cherry Servers 64-core CPU boxes** and **vast.ai multi-GPU
boxes** — auto-detects CUDA/OptiX and picks the fastest Cycles device.

Everything is **self-contained** — no files copied from your laptop. All bulk
assets (Blender, MPFB, BVH motion capture, HDRIs) are downloaded directly on
the server from their canonical sources.

## Budget — pick your box

| Provider / box | $/hr | Wall-clock | **Total for 500k** |
|---|---|---|---|
| **Vast.ai 4× RTX 5070** (Blackwell, 12 GB, 2 instances / GPU = 8 parallel) | $0.27 | **~9 h** | **~$2.40** ⬅ **best** |
| Vast.ai 4× RTX 3060 (Ampere, 12 GB, 8 parallel) | $0.19-0.23 | ~21 h | ~$4-5 |
| Vast.ai 1× RTX 4090 (24 GB, 2 parallel) | $0.25-0.40 | ~90 h | ~$22-36 |
| Cherry 64c/128t CPU (Cycles CPU, 32 parallel) | $0.70 | ~20-28 h | ~$14-20 |

**Pick RTX 5070 for best $/sample AND wall-clock.** AVOID 8 GB cards (3060 Ti,
3070) — VRAM is too tight for 2 Blender instances per GPU, halving parallelism.

Output archive size: **~15-20 GB compressed** (zstd level 19 on 500k PNG + JSON).
Download back to your laptop: ~15-30 min on most connections.

## Quickstart

On any fresh Ubuntu 22.04/24.04 box:

```bash
# Clone the repo (or unzip cherry_dataset_gen.zip)
git clone https://github.com/nestorvfx/3D-Body-Tracking-Approach.git
cd 3D-Body-Tracking-Approach/cherry_dataset_gen

# 1. Install everything + asset download + smoke test  (~30 min, ~$0.20)
bash setup.sh

# 2. Generate 500k samples (auto-detects GPU; uses 32 CPU shards if none)
#    On 4× RTX 5070: ~9 h, ~$2.40
#    On 64-core CPU: ~20-28 h, ~$14-20
./run.sh

# 3. Compress + prep download
./package.sh /data/synth_v3

# 4. Download to your laptop (rsync = resumable)
# From LOCAL (not remote):
rsync -avP --partial USER@REMOTE_IP:/data/synth_v3.tar.zst .
```

## What gets installed

- **Blender 5.1.1** (downloaded from blender.org) → `/opt/blender/blender`
  - Supports RTX 50-series Blackwell OptiX out of the box (4.4+ onwards)
- **MPFB2 extension** (from github.com/makehumancommunity/mpfb2)
  - Bundles skins, 8 suits + 6 shoes + 2 hats, 10 hairstyles
- **CMU Mocap BVH** (cgspeed conversion from archive.org, ~145 MB)
- **100STYLE BVH** (Zenodo, CC-BY 4.0, ~1.5 GB)
- **Poly Haven HDRIs** (API download, CC0, 300 × 1k-res, ~600 MB)

All licenses are commercial-use-permitted.

## Optimizations applied

The three speedups that matter most (stacked, all on):

1. **Character caching via plan sort-by-seed.** MPFB's single-threaded Python
   character build (~5 s per call) is now amortized over ~1000 samples per
   unique seed instead of every 5 samples. Saves ~1 s / sample.
2. **Cycles 8 samples + adaptive sampling + OIDN denoise + 4-bounce limit
   + no caustics/volumetrics + persistent_data.** Reduces per-render time
   ~3× vs default Cycles at the same visual quality for 256×192 crops.
3. **GPU device auto-detection** (OptiX preferred, CUDA fallback). Renders
   go from 3-8 s on CPU to 0.5-1 s on an RTX 5070.

## Monitoring progress

```bash
# live feed of one shard
tail -f ~/cherry_dataset_gen/logs/shard_000.log

# total images rendered so far
watch -n 10 'ls /data/synth_v3/shard_*/images 2>/dev/null | wc -l'

# GPU load (only on GPU boxes)
nvidia-smi
```

Expected aggregate throughput:
- 4× RTX 5070 with 8 parallel instances: **~15 samples/sec**
- 4× RTX 3060 with 8 parallel instances: ~6-7 samples/sec
- Cherry EPYC 64c with 32 parallel CPU instances: ~3-5 samples/sec

## Failure recovery

Each shard resumes automatically by reading its `labels.jsonl` on startup
and skipping already-done sample IDs. Safe to re-run:

```bash
./run.sh
```

## Output layout (after `merge.py` runs automatically)

```
/data/synth_v3/
├── images/                     # 500k PNGs, 256×192, <sample_id>.png
├── labels.jsonl                # one JSON per sample (COCO-17 2D + 3D, K, bbox)
├── manifest.csv                # compact training-loader index
└── dataset_stats.md            # source + train/val split summary
```

## Diversity at 500k

- **500 character seeds** — distinct phenotype × per-character dual-tint clothing + 10-colour hair palette. Each seed covers ~1000 samples but every sample has a unique (clip, frame, HDRI, camera, tint) combination.
- **2,700+ BVH clips** — CMU 20% / 100STYLE 80%. `plan()` samples uniformly across motion categories.
- **300 CC0 HDRIs** — per-sample rotation + strength jitter 0.7-1.6.
- **Camera**: yaw ±π, pitch ±20°, distance 2.5-6 m, focal 25-70 mm.

Combinatorially >> 500k unique configurations. Every sample differs.

## Override-able env vars

Set these before `./run.sh` to tweak behavior:

```bash
ENGINE=gpu              # force GPU (fails if none found)
ENGINE=cpu              # force CPU (even on GPU box)
CYCLES_SAMPLES=16       # bump quality (default 8 + OIDN)
INSTANCES_PER_GPU=2     # default 2; go 3 if VRAM allows
BLENDER=/path/to/blender
```

## Why 4× RTX 5070 beats cherry CPU

The Python-serial portion of each sample (MPFB build, scene setup, label
compute) is ~0.5 s. The render portion is:
- Cycles GPU RTX 5070: ~0.5 s
- Cycles CPU EPYC (even 64-core, at 8 samples + OIDN): ~3-5 s

After amortizing MPFB via character caching, render dominates. A 6-8× speedup
on render = 5-8× faster end-to-end wall-clock. The only reason cherry CPU is
competitive on $/sample is its cheaper hourly rate — but 5070 wins that too
because it finishes 3× faster and costs 0.4× per hour.

## Notes on AIST++

AIST++ dance motion is published only as SMPL `.pkl`, not native BVH. Skipped
entirely; plan is CMU 20% / 100STYLE 80%. If you add AIST++ BVH later
(via `smpl2bvh`), drop it at `~/cherry_dataset_gen/assets/aist_plusplus/bvh/`
and flip the 35% AIST++ allocation back on in `build_dataset.py`.

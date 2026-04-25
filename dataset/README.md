# dataset/ — synthetic-data generation + training pipeline

Reproducible pipeline for a **commercial-clean** 3D-body-pose dataset +
training + evaluation.  Respects the project's hard license constraints
(see `../README.md` Hard Exclusions): no SMPL/SMPL-X, no BEDLAM, no
AGORA, no Human3.6M / 3DPW / EMDB contamination.

## Quick reference

1. `PIPELINE_DESIGN.md` — dataset schema, model choice, eval strategy.
2. `RETARGETING.md` — BVH→MPFB retargeting algorithm + pitfalls.
3. `scripts/build_dataset.py` — generate the 1000-sample dataset.
4. `scripts/verify_dataset.py` — draw keypoints on N samples to spot-check.
5. `../training/` — model + train + eval for the dataset.

## Setup

Two environments are involved:

### Blender env (for dataset generation)
Blender 5.1 with the MPFB2 add-on.  The add-on zip is in
`assets/extensions/`.  Install via Blender preferences; our scripts
detect and enable it at startup.

### Training env (PyTorch + CUDA)

```bash
conda env create -f training/environment.yml
conda activate bodytrack
python -c "import torch; print(torch.cuda.get_device_name(0))"   # verify GPU
```

Tested on Windows 11 + RTX 4050 Laptop (6 GB, Ada CC 8.9) with CUDA 12.4.
See `training/environment.yml` for pinned versions.

## Generate a dataset

```bash
# 1000 samples, ~25-35 min on a single GPU / integrated Blender.
"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background \
    --python dataset/scripts/build_dataset.py -- \
    dataset/output/synth_v1

# Optional: cap samples for a smoke test
"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background \
    --python dataset/scripts/build_dataset.py -- \
    dataset/output/synth_v1 20

# Draw keypoints on 10 samples to verify labels
python dataset/scripts/verify_dataset.py dataset/output/synth_v1 10
```

Outputs:
- `dataset/output/synth_v1/images/` — rendered 256×192 PNGs.
- `dataset/output/synth_v1/labels.jsonl` — one JSON per sample with
  camera intrinsics/extrinsics, bbox, COCO-17 2D + 3D keypoints,
  source/clip/frame/seed/hdri/focal metadata.
- `dataset/output/synth_v1/manifest.csv` — compact index.
- `dataset/output/synth_v1/dataset_stats.md` — per-source / split
  counts.
- `dataset/output/synth_v1/verify/` — KP-overlay spot checks.

### Diversity design

Each sample randomizes along every axis (see
`PIPELINE_DESIGN.md §2`):

- **Clip**: 100 clips sampled from a 112-clip pool (CMU Mocap, 100STYLE
  motion-categories, AIST++ dance genres).  Walks, dances, crouches,
  jumps, stretches, balances, kicks.
- **Character**: 30-seed pool, 2 distinct seeds per clip, each producing
  a fresh phenotype (gender 0.05–0.95, age 0.25–0.80, weight 0.15–0.60,
  height 0.20–0.90, muscle 0.25–0.85 — full range).
- **Frames**: random across each clip's range (not uniform-spaced).
- **HDRI**: one of 8 environments per sample, id-hashed (warehouse,
  studio, puresky, ninomaru, rural_crossroads, shanghai_bund,
  symmetrical_garden, studio_small_09).
- **Camera**: BEDLAM 2.0–style — lognormal 14-400 mm focal, uniform
  360° yaw, ±15° pitch, binary-search distance for strict COCO-17
  framing, 36×20.25 mm DSLR sensor.

## Train a model

```bash
conda activate bodytrack

# Single-GPU AMP training loop.  About 1-2 hours for 20 epochs on RTX 4050 6GB.
python -m training.train \
    --dataset-dir dataset/output/synth_v1 \
    --out-dir training/runs/baseline_v1 \
    --epochs 20 --batch 16 --lr 3e-4

# Follow progress
tensorboard --logdir training/runs/baseline_v1/tb
```

Checkpoints go to `training/runs/baseline_v1/{last,best}.pt`.

## Evaluate

```bash
python -m training.eval \
    --dataset-dir dataset/output/synth_v1 \
    --ckpt training/runs/baseline_v1/best.pt \
    --report training/runs/baseline_v1/eval_report.md
```

The report contains:
- MPJPE / PA-MPJPE / PCK@50 / PCK@100 on the val split.
- Per-source breakdown.
- Bone-length std (label-free stability).
- Published SOTA reference table (HMR, CLIFF, 4DHumans, BEDLAM-HMR, NLF-L)
  for context.  **NOT** measured on our model — it's what a scaled-up
  model should approach.

## Known limits at 1000 samples

1000 samples proves the loop end-to-end.  Published SOTA trained on
hundreds of thousands of images; expect our MPJPE to be 2-4× worse.
Scaling plan (per the main README) is 5M+ synthetic frames + pseudo-
labelled real video.  This iteration is a scaffold.

## Licenses (commercial-clean)

- Code: this project, TBD but planned Apache-2.0.
- MPFB2 / MakeHuman base assets: GPLv3 code + CC0 assets; rendered
  output is unencumbered per their LICENSE.
- Motion: CMU Mocap (public domain via cgspeed mirror), 100STYLE (CC-BY
  4.0), AIST++ annotations (CC-BY 4.0).
- HDRIs: Poly Haven (CC0).
- Backbone: MobileNetV4 via timm (Apache-2.0).
- Head: RTMPose3D SimCC lifted from mmpose (Apache-2.0).

No SMPL / SMPL-X / AMASS / BEDLAM / AGORA / Human3.6M / 3DPW anywhere
in the chain.

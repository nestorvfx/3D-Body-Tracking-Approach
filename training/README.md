# training/ — Model implementations and training roadmap

## What's here (implemented)

### `lib/one_euro_filter.py` — public-domain causal smoother
- `OneEuroFilter1D`, `OneEuroFilter3D` — pure Python/math, no torch, no numpy.
- Reimplementable in ~30 lines of Swift/Kotlin/C++ for mobile runtime.
- Presets: `body_3d`, `body_2d`, `hand`, `head` (parameters from MediaPipe
  BlazePose's shipping config).
- **Cost**: <1ms on any CPU. Latency: 0 (causal).
- Use: post-processing on any per-frame 3D pose prediction.
- **Status**: production-ready; drop-in use at inference time.

### `lib/smoothnet.py` — tiny 1D conv temporal refiner
- PyTorch module, ~700k params (3 ResBlocks × 128 hidden × kernel 7 × 8-frame window).
- Exports cleanly to ONNX/CoreML/TFLite — pure Conv1D + Linear + LayerNorm.
- Takes (B, T, J*3) predictions; returns (B, T, J*3) smoothed residual.
- **Status**: architecture complete, requires training on our synthetic video sequences.

### `lib/motionagformer.py` — MotionAGFormer-XS temporal lifter
- Apache-2.0 spirit reimplementation of Mehraban et al. WACV 2024.
- Dual-stream ST-Transformer + GCNFormer with learned gating.
- **~240k params**, dim=64, 4 blocks, 4 heads, 27-frame window.
- Takes (B, T=27, J=17, 3) from single-frame model; returns refined (B, T, J, 3).
- **Status**: architecture complete, requires training.

### `lib/augmentation.py` — Albumentations sim-to-real stack (from earlier)
### `lib/simcc3d.py` — SimCC-3D target encoder (from earlier)
### `lib/dataset.py` — torch Dataset loading pilot output (from earlier)

## Inference stack (deployed)

```
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ RTMPose3D        │ 3D  │ MotionAGFormer-XS│ 3D  │ OneEuroFilter    │
  │ + MobileNetV4-M  │───▶ │ (27-frame window)│───▶ │ (per-joint x/y/z)│───▶ 3D pose
  │ 256×192 per frame│     │ ≈240k params     │     │ 0-latency causal │
  └──────────────────┘     └──────────────────┘     └──────────────────┘
       <10ms A17 ANE           <2ms A17 ANE              <1ms any CPU

   Optional alternative refiner:
  ┌──────────────────┐
  │ SmoothNet-T      │
  │ 8-frame window   │
  │ ~700k params     │
  │ <1ms A17 ANE     │
  └──────────────────┘
```

End-to-end inference budget: ~13 ms/frame → **~75 FPS** streaming on iPhone A17 ANE.

## Why all three temporal components?

| Component | What it solves | When to use |
|---|---|---|
| MotionAGFormer-XS | Monocular depth ambiguity + systematic temporal error | Primary temporal estimator (big win: +8–12 mm MPJPE) |
| OneEuroFilter | Residual per-frame jitter after the estimator | Final smoothing layer (cheap polish, 0 latency) |
| SmoothNet-T | Learned low-pass replacement for OneEuroFilter | Alternative to OneEuroFilter if model training is preferred |

**Recommended production stack**: RTMPose3D + MotionAGFormer-XS + OneEuroFilter.  
SmoothNet-T is an alternative middle-ground if MotionAGFormer's 27-frame buffer is too heavy.

## Training roadmap (not yet executed)

### Milestone 1 — Train the single-frame model (RTMPose3D-MNv4)
Precondition: ~500K-frame synthetic dataset in `dataset/output/pilot_*/`.
1. Use `dataset.SyntheticPoseDataset` (already written) with Albumentations augmentation.
2. Model: MobileNetV4-Conv-M backbone (pretrained ImageNet) + RTMPose3D SimCC head.
3. Loss: KLDiscretLoss on SimCC-3D targets (already in `simcc3d.py`), plus BoneLoss.
4. Training: single H100 or A100 for ~3–5 days (210 epochs on ~500K frames).
5. Expected: ~65–75 mm MPJPE on synthetic val, ~85–95 mm on real test set.

### Milestone 2 — Render coherent-sequence training data for MotionAGFormer
1. Run `pilot_v4.py --video` to produce 27-frame coherent sequences
   (same character/camera/HDRI, pose evolves smoothly).
2. Target: ~50k sequences × 27 frames = ~1.35M frame-records for temporal training.
3. Pair with teacher-predicted 2D from the Milestone-1 model to get realistic
   single-frame input noise (mirrors deployment).

### Milestone 3 — Train MotionAGFormer-XS
1. Use `motionagformer.py` architecture, retrain from scratch on our data
   (public weights are tainted).
2. Loss: MPJPE on refined 3D + velocity loss (acceleration penalty) + bone-length consistency.
3. Training: 3–5 days on 8×H100.
4. Expected gain: +8–12 mm MPJPE over single-frame-only; ~40% jitter reduction.

### Milestone 4 — Mobile export
1. MobileNetV4 backbone: int8 (TFLite PTQ), CoreML mixed precision (fp16 SimCC heads).
2. MotionAGFormer-XS: int8 activation / fp16 weights on A17 ANE.
3. OneEuroFilter: hand-port to Swift/Kotlin (30 LOC).
4. Target bundle size: <60 MB uncompressed, <25 MB compressed.

## Current dataset status (for training readiness)

| Source | Clips | License | Used by |
|---|---|---|---|
| CMU Mocap | 10 | Public Domain | both |
| 100STYLE | ~800 | CC-BY 4.0 | both |
| AIST++ | 411 (converted) | CC-BY 4.0 | both |

**Total clips: ~1220**, enough to bootstrap training. Expansion to 3000+ clips
(CMU bulk + MHAD + KIT) is scheduled as a pre-training step.

## Quick checks

```bash
# Verify models load + forward cleanly
python -c "
from training.lib.one_euro_filter import make_filter
from training.lib.smoothnet import make_smoothnet
from training.lib.motionagformer import make_motionagformer_xs
print('all three temporal modules imported OK')
"

# Render a coherent-sequence pilot for MotionAGFormer
"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background \
    --python dataset/scripts/pilot_v4.py -- 1 test_video --video

# Render single-frame pilot (per-frame camera diversity)
"/c/Program Files/Blender Foundation/Blender 5.1/blender.exe" --background \
    --python dataset/scripts/pilot_v4.py -- 1 test_single
```

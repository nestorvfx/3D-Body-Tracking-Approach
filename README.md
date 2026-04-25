# Body Tracking — Commercial-Clean 3D Pose Estimation

A from-scratch real-time monocular 3D body pose estimation model for commercial mobile deployment. Targets iPhone A17/A18 ANE and Snapdragon 8 Gen 3 / 8 Elite HTP at 100+ FPS, with every component (code, weights, training data) cleared for commercial use without paid licenses or negotiation.

## Goal

Beat MediaPipe BlazePose GHUM Heavy on accuracy while matching or exceeding its mobile inference speed, using only commercial-permissive components.

**Current % of SOTA estimates** (where NLF-L ≈ 62mm MPJPE on 3DPW = 100%):
- BlazePose GHUM Heavy: ~48–56%
- This project (target): ~65–73%
- Expected improvement: ~1.3–1.5× lower MPJPE than BlazePose, real-time on mobile.

## Architecture

**Single-stage end-to-end image-to-3D**:
- Detector: RTMDet-nano (Apache-2.0), ~2ms on mobile NPU
- Backbone: **MobileNetV4-Conv-M** (Apache-2.0, timm), stride 16+32 features
- Neck: 1×1 channel projection + 2-level mini-PAFPN
- Head: **RTMW3DHead with SimCC-3D** (Apache-2.0, lifted from mmpose)
  - Input: 256×192 person crop
  - Bins: W=384 / H=512 / D=512 (split_ratio=2.0)
  - Z-range: ±2.17m relative to mid-hip (root joint)
  - Loss: KLDiscretLoss (β=10, label_softmax) + BoneLoss (weight 2.0)
- Output: COCO-17 body skeleton in metric 3D (root-relative)
- Postprocessing: OneEuroFilter per-joint (MediaPipe's filter)

**Mobile export**: int8 backbone + fp16 SimCC heads (mixed precision in CoreML/QNN). PixelShuffle replaced with Upsample+conv for ANE compatibility. Argmax in Swift/Kotlin postprocessing, not in graph.

## Body Model: Anny + MPFB2

Replaces SMPL-X entirely with commercial-clean components.

**MPFB2** (https://github.com/makehumancommunity/mpfb2) — GPLv3 code + CC0 assets. Output (meshes, renders, exports) explicitly unencumbered per their LICENSE. Source of base mesh assets and shape variation.

**Anny** (https://github.com/naver/anny) — Apache-2.0 + CC0. Differentiable PyTorch parametric body model with 564 WHO-calibrated blendshape parameters (age, gender, height, weight, muscle, body proportions). Built on MPFB2 assets. HMR models trained on Anny match SMPL-X within ~1mm MPJPE on 3DPW (Anny: 86.5 / 49.4 vs SMPL-X: 86.0 / 52.0). Significantly better on children and edge body types.

⚠ **DO NOT USE**:
- Anny's `smplx` topology add-on (v0.3, non-commercial)
- Anny-One pretrained dataset (uses AMASS poses → tainted)

We regenerate an Anny-One-equivalent dataset with commercial-clean motion sources.

## Data Sources (all GREEN, commercial-permissive)

### Motion (clean replacements for AMASS) — zero-gate, anonymous HTTP
- **CMU Mocap** via archive.org cgspeed mirror (https://archive.org/download/CMU_Mocap_bvh_cgspeed/) — ~2,548 BVH, free for any use
- **ACCAD** (CC-BY 3.0) — dance, cultural performance, everyday actions
- **HDM05** (CC-BY-SA 3.0) — 70+ motion classes, 3+ hours
- **Berkeley MHAD** (BSD-2-Clause) — 660 sequences, 11 action categories
- **AIST++ keypoint annotations only** (CC-BY 4.0) — 10.1M frames of dance 3D keypoints (skip images & SMPL params)
- **SFU Mocap** (https://mocap.cs.sfu.ca) — anonymous HTTP, BVH available, commercial OK
- **MoCapAct** (CDLA-Permissive-2.0) — 2,500 processed CMU clips with RL rollouts
- **MuJoCo + Isaac Lab** RL-synthesized motion (Apache-2.0 / BSD-3) for sports / extreme poses

⚠ Excluded from pipeline (gated / signup-required, even though license would otherwise allow):
- Rokoko Free Motion Library (signup wall)
- Mixamo (Adobe account + FAQ forbids ML training)
- Sketchfab CC0 (account required for download)
- RenderPeople free samples (per-asset EULA click)

### Bodies / Characters
- **Anny** model (Apache-2.0 + CC0) — primary parametric body source
- **MPFB2 / MakeHuman CC0 assets** — direct mesh export
- **Microsoft Rocketbox** (MIT) — 115 fixed-shape avatars for appearance variety
- **Sketchfab CC0** human scans — supplementary diversity

### Rendering
- **Blender** (GPL tool, output is yours per Blender FAQ)
- **BlenderProc2** (Apache-2.0) — automated domain randomization
- **Unity Perception** (Apache-2.0) — for bulk rendering (⚠ swap out PeopleSansPeople's RenderPeople sample assets — Unity Asset Store EULA bans ML training)

### Environments / Textures
- **Poly Haven** HDRIs/textures/models (CC0) — 750+ HDRIs available
- **Stable Diffusion XL** for fabric textures — CreativeML Open RAIL++-M permits using outputs as ML training data
- Procedural materials in Blender

### 2D Real Pretraining / Pseudo-Labels
- **COCO Keypoints 2017** base set (CC-BY 4.0) — 250K person instances, 2D pretraining
- **InfiniteRep + InfiniteForm** (CC-BY 4.0) — 1K fitness videos + 60K fitness images
- **Kinetics-400** `is_cc=True` subset (CC-BY annotations) — for self-pseudo-labeling
- **Wikimedia Commons** video — all CC-BY/SA/0 by policy

### Pseudo-Label Teachers (all commercially clean)
- **GEM-X** (NVIDIA Open Model License, commercial use OK) — 77-joint 3D from monocular video
- **ViTPose / ViTPose++** (Apache-2.0) — SOTA 2D pose
- **RTMPose / RTMW3D** (Apache-2.0) — fast 2D/3D pose
- **MediaPipe BlazePose** (Apache-2.0) — for ensemble agreement

## Hard Exclusions

Never use these (license violations or SMPL contamination):
- **SMPL / SMPL-X / SMPL+H / STAR / SUPR** — CC-BY-NC, forbids commercial training
- **AMASS, BEDLAM, BEDLAM2, AGORA, SURREAL** — all SMPL-derived
- **THuman, HumanSC3D, Fit3D, ChiSCan, MOYO, BABEL, HumanML3D, Motion-X**
- **Human3.6M, MPI-INF-3DHP, 3DPW, CMU Panoptic, EMDB, TotalCapture** — research-only
- **MPII, COCO-WholeBody, CrowdPose, PoseTrack, JHMDB, FLIC** — non-commercial / image copyright issues
- **Mixamo** — Adobe FAQ explicitly bans ML training
- **SynthMoCap** (Microsoft) — non-commercial
- **Unity Asset Store assets** — EULA bans ML training (only the Perception package code is safe)
- **MakeHuman GUI used as library / server / programmatic export** — triggers AGPL (use MPFB2 directly instead)
- **Anny-One pretrained dataset** — AMASS poses → tainted
- **Anny v0.3 smplx topology** — non-commercial
- **PeopleSansPeople's bundled RenderPeople characters** — verify license per asset

## Skeleton

**COCO-17 body keypoints** (no patent, used as open standard). Bridge layer maps from Anny's 163-bone rig → COCO-17 via deterministic joint regressor (~50 LoC). NOT SMPL's 24-joint layout.

## Compute Targets

- **Local prototyping**: RTX 3090/4090 for code correctness, overfit-a-batch tests
- **Kaggle free tier**: 30 hrs/week T4×2 or P100 — hyperparameter sweeps, ablations
- **Vast.ai** for full runs: H100 SXM ($1.49–1.99/hr) or A100 80GB ($0.67–1.20/hr), with R2/B2 storage for checkpoints

**Estimated total cost**: ~$800 (compute only) for full pipeline:
- Synthetic dataset generation (5M frames): ~$150
- Training (210 epochs + 30 epoch distillation): ~$550
- Pseudo-label generation: ~$100

## Evaluation

Cannot ship MPJPE numbers measured on Human3.6M / 3DPW / EMDB (research-only data implies training contamination risk). Eval strategy:

- **Self-collected iPhone ARKit/LiDAR multi-view** — primary public benchmark
- **BEDLAM test set** — internal only (research license fine for in-house eval)
- **Bone-length constancy + jitter / acceleration** — label-free production metrics
- **Kinetics CC subset + multi-view triangulation** — pseudo-3D consistency
- **PCK@0.05 on self-collected 2D** + pairwise depth-order accuracy

## Status

- [x] Architecture finalized
- [x] Licensing chain audited
- [x] Dataset sources confirmed
- [x] Pilot v1: 10 single-frame renders (retarget bottleneck identified)
- [x] Pilot v2: sequence rendering + intelligent framing + camera diversity + fast retarget
- [x] Training data loader (`training/lib/{augmentation, simcc3d, dataset}.py`)
- [x] Pseudo-labeled real-video pipeline scaffold (`real_video/`)
- [ ] Scale to 5M-frame dataset (single 4090, estimated ~1 week)
- [ ] Pseudo-labeled real video: download → teacher ensemble → filter → pair with synth
- [ ] RTMPose3D + MobileNetV4-Conv-M implementation and training
- [ ] Mobile export (CoreML + LiteRT/QNN)
- [ ] On-device benchmarking vs BlazePose

## Pilot v1 → v2 comparison (10 samples / 10 sequences × 4 kept frames)

| Metric | Pilot v1 | Pilot v2 | Change |
|---|---|---|---|
| Wall-clock | 189s | 59s | **3.2× faster** |
| Per-kept-frame | 19.0s | 1.48s | **12.8× faster** |
| Kept frames | 10 | 40 | **4× more data** |
| Retarget per sample | 16.6s | 0.08s | **200× faster** |
| 17/17 keypoint visibility | 7/10 samples | 40/40 frames | **100% of samples** |
| Camera diversity | uniform 50mm | lognormal 14-400mm + shake + motions | ✓ |
| Sequence-aware | no | 3-sec amortised + 3× decimation | ✓ |

## Directory layout

```
Body Tracking/
├── README.md                     # this file
├── dataset/                      # synthetic data generation (Blender)
│   ├── scripts/
│   │   ├── lib/
│   │   │   ├── coco17.py         # COCO-17 skeleton + MPFB bone mapping
│   │   │   ├── cmu_bvh.py        # CMU BVH → MPFB rig map
│   │   │   ├── mpfb_build.py     # seeded MPFB character generator
│   │   │   ├── fast_retarget.py  # live-constraint retargeting, no nla.bake
│   │   │   ├── camera_rig.py     # BEDLAM 2.0-style camera + intelligent framing
│   │   │   ├── render_setup.py   # engine / HDRI / scene clear
│   │   │   ├── retarget.py       # LEGACY v1 (constraints + bake)
│   │   │   └── sequence_render.py # sequence + decimate, per-frame annotations
│   │   ├── pilot.py              # LEGACY v1 single-frame pilot
│   │   ├── pilot_v2.py           # sequence-based pilot (current)
│   │   ├── qa_overlay.py         # LEGACY v1 QA
│   │   ├── qa_overlay_v2.py      # sequence-aware QA + contact sheet
│   │   └── install_env.py        # MPFB2 install + asset pack extract
│   ├── assets/                   # BVH, HDRIs, MPFB zip, extension zip
│   └── output/                   # rendered frames + labels.json
├── training/                     # training-time code (no Blender imports)
│   └── lib/
│       ├── augmentation.py       # sim-to-real Albumentations + FDA + flip
│       ├── simcc3d.py            # SimCC-3D target encoder/decoder
│       └── dataset.py            # torch Dataset wrapping pilot output
└── real_video/                   # pseudo-labeled real-video pipeline scaffold
    ├── README.md                 # legal chain + teacher selection
    ├── download/                 # Kinetics CC filter + Wikimedia + yt-dlp
    ├── detect/                   # RTMDet person detector
    ├── teachers/                 # ViTPose + RTMPose + MediaPipe
    ├── ensemble/                 # multi-teacher agreement
    ├── temporal/                 # OneEuroFilter + bone-length + velocity
    ├── lifter/                   # synthetic-trained 2D→3D lifter
    ├── quality/                  # per-frame quality scoring
    ├── legal/                    # attribution tracking (SQLite)
    └── pipeline/                 # orchestrator
```

## License

Project code: TBD. All third-party components used per their respective licenses (see `THIRD_PARTY_LICENSES.md` — to be created).

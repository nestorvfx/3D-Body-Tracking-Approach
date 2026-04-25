# Pipeline design — 1000-sample dataset + training + eval

This document locks the dataset schema, model architecture, benchmark choice,
and metric targets BEFORE any code.  It respects the project's hard constraints
from `README.md`:

- **Commercial-clean licenses only.**  No SMPL/SMPL-X/BEDLAM/AMASS/AGORA/
  Human3.6M/3DPW/CMU-Panoptic/EMDB anywhere in the training or shipped-eval
  path.  Those datasets, their derivatives, and their pretrained weights
  are forbidden.
- **Skeleton: COCO-17.**  No SMPL topology.
- **Body model: MPFB2 + Anny** (Apache-2.0 + CC0).  No SMPL-X parameters
  anywhere.
- **Mocap sources (green):** CMU cgspeed mirror (PD), 100STYLE (CC-BY 4.0),
  AIST++ keypoint annotations only (CC-BY 4.0).
- **Hardware:** single RTX 4050 mobile, 6 GB VRAM.  Model + batch size must
  fit, so we use a SMALLER backbone than the README's MobileNetV4-Conv-M
  headline target: MobileNetV4-Conv-S (≈5.5 M params) as the first-iteration
  baseline, with a config switch to scale up on larger GPUs.

---

## 1. Dataset schema (per sample)

Each rendered image produces one row in a JSONL manifest plus one PNG.

```
id                 : str                 # e.g. "cmu_02_01_f0150_s42_c0"
split              : "train" | "val"
image_rel          : str                 # "images/0000123.png"
image_wh           : [w, h]              # pixels
camera_intrinsics  : {fx, fy, cx, cy}    # pinhole, pixels
camera_extrinsics  : {R: 3x3, t: [x,y,z]} # world -> camera, metres
bbox_xywh          : [x, y, w, h]        # person bbox in pixels (tight ±25%)
keypoints_2d       : [[u, v, v_flag], x17]  # COCO-17 pixel + visibility (0/1/2)
keypoints_3d_cam   : [[x, y, z], x17]    # camera-frame, metres, root-relative flag separate
root_joint         : [x, y, z]           # camera-frame mid-hip for reference
source             : "cmu" | "100style" | "aistpp"
clip_id            : str                 # BVH stem
frame_idx          : int
character_seed     : int
phenotype          : {gender, age, muscle, weight, height, proportions}
render_engine      : "BLENDER_EEVEE"
skin_mask_rel      : str | null          # optional silhouette PNG
```

**Storage:** PNG images at **256×192** (W×H; matches RTMPose3D input and
our existing `training/lib/simcc3d.py` `SimCC3DConfig.input_size = (192, 256)`
in `(W, H, D)` convention — note mmpose uses W×H, we follow them).  Labels
in `labels.jsonl` (one JSON per line) + `manifest.csv` with a compact view.

**Split:** deterministic 90/10 train/val by `id` hashed to an 8-bit bucket.
Master seed = 20260420 (today) for reproducibility.

**Keypoint convention:** COCO-17 body.  Derived from MPFB rig head positions
via our existing `training/lib/dataset.py` pattern.  Visibility flag: 0 =
not labeled, 1 = labeled but occluded, 2 = visible (following COCO spec).

---

## 2. Diversity budget for 1000 samples

Following the README's multi-source strategy:

| source      | clips  | seeds/clip | frames/clip | cameras/frame | samples |
|-------------|--------|------------|-------------|---------------|---------|
| CMU         | 5      | 4          | 10          | 1             | 200     |
| 100STYLE    | 5      | 4          | 10          | 1             | 200     |
| AIST++      | 3      | 4          | 10          | 2             | 240     |
| CMU-extra   | 4      | 4          | 10          | 1             | 160     |
| 100STYLE-extra | 4   | 4          | 10          | 1             | 160     |
| AIST++-extra| 2      | 4          | 10          | 1             | 80      |
| **total**   |        |            |             |               | **1040** |

Frames per clip: 10 evenly-spaced across the clip's range (plus the
`find_worst_frames` picks for joint-extreme coverage).  Cameras are sampled
from a lognormal focal-length distribution (14–400 mm) at randomized yaw
(±180°) and pitch (−15°…+15°) to match BEDLAM 2.0 / `camera_rig.py` diversity.

---

## 3. Model architecture

**Locked per README.md:** RTMPose3D SimCC-3D head + lightweight backbone.

**First-iteration baseline (fits 6 GB VRAM):**

- **Backbone:** MobileNetV4-Conv-S (≈5.5 M params) — Apache-2.0 via `timm`.
  Stride 16 features at 256×192 input → 16×12 feature map, 960 channels.
  README's MobileNetV4-Conv-M (9.7 M) is the target; we start with -S for
  VRAM fit and switch via config when moving to vast.ai.
- **Neck:** 1×1 projection 960 → 384 channels.  No PAFPN at this iteration
  (paper's multi-scale fusion is a phase-3 ablation).
- **Head:** RTMPose3D SimCC-3D head, 17 joints, bins `W=384 H=512 D=512`
  per `simcc3d.SimCC3DConfig(split_ratio=2.0)` — already implemented in
  `training/lib/simcc3d.py`.
- **Loss:** KLDiscretLoss (β=10, `label_softmax=True`) for each of the
  three 1-D classifications, plus BoneLoss (weight 2.0) on the 16 COCO-17
  skeleton bones — standard RTMPose3D config.
- **Postprocess:** Soft-argmax decode (bin centroid) → pixel X/Y + depth Z
  (metres, root-centred).

Total params: ≈5.5 M backbone + ≈0.8 M neck/head = **≈6.3 M params**.

**Precision:** fp32 parameters, **AMP (torch.cuda.amp.autocast)** on
forward + backward, fp32 optimizer state.  VRAM budget for RTX 4050 6 GB:
- params + grads + Adam optimizer state in fp32 = 6.3M × 16 bytes ≈ 100 MB
- Activations + intermediate tensors at batch=16 ≈ 1.5 GB in AMP
- Headroom for the SimCC target tensors (17 × 384 + 17 × 512 + 17 × 512 floats
  per sample) ≈ 100 MB at batch=16
- Leaves ≈4 GB for CUDA kernels / cuDNN scratch.

Target batch size: **16**, gradient accumulation 4 → effective batch 64.

---

## 4. Benchmark strategy

**Primary problem:** the README explicitly forbids shipping MPJPE numbers
measured on Human3.6M / 3DPW / EMDB because research-only data implies a
training-contamination risk.  But for INTERNAL ranking against the published
SOTA, those datasets are the only standard.

**Ranking strategy adopted (internal-only, not for publication):**

1. **Self-validation (synthetic val split, 100 samples):** primary iteration
   signal.  Report MPJPE, PA-MPJPE, PCK@5cm.  Produced fresh by our
   generation pipeline on samples held out by split hash.  *No contamination
   risk because we own the data.*

2. **Label-free production metrics on Kinetics CC subset:**
   - Bone-length constancy (std of per-joint bone-length / mean over time).
   - Jitter / acceleration (L2 of second-derivative of 3D predictions).
   - Left-right symmetry consistency.
   These are SHIPPABLE metrics — no ground truth is needed, and they're
   the canonical quality indicators for monocular 3D tracking models that
   can't use benchmark ground truth.

3. **Future ship-eval (NOT in this iteration):** self-collected iPhone
   ARKit/LiDAR multi-view (per README), paired with PCK@0.05 on
   self-collected 2D.

**For THIS iteration the ranking we report is:**
- (a) Our-val MPJPE / PA-MPJPE (own data, reproducible).
- (b) Bone-length constancy on Kinetics CC val-subset.
- (c) Jitter/acceleration on the same.

Note: research-only benchmarks (Human3.6M, MPI-INF-3DHP, 3DPW, AGORA,
BEDLAM, EMDB, RICH, MoYo) are NOT used for validation.  Their licenses
forbid use in a commercial product or service, including use of the
data to validate a model that ships in a commercial product.

---

## 5. Metric targets (this iteration)

Because 1000 samples is *two orders of magnitude* smaller than BEDLAM's
~350k or 4DHumans' mix, we do NOT expect state-of-the-art numbers.  This
iteration proves the loop end-to-end.

| metric                        | target (iter 1) | reference (published SOTA) |
|-------------------------------|-----------------|---------------------------|
| Self-val MPJPE                | < 250 mm        | n/a (our data)            |
| Self-val PA-MPJPE             | < 120 mm        | CLIFF on 3DPW ≈ 68 mm     |
| Self-val PCK@5 cm             | > 35 %          | n/a                       |
| Train-loss monotonic decrease | yes             | sanity check              |
| Overfit-a-batch (4 samples)   | MPJPE < 15 mm   | sanity check              |

Scaling to README targets (65-73 % of NLF-L ≈ 62 mm MPJPE on 3DPW) requires
~500k samples per that doc, not 1000.  This iteration is a scaffold; the
scaling run is next.

---

## 6. Research sources consulted

- Project `README.md` (commercial-clean licensing policy, architecture).
- Project `training/README.md` (model roadmap, VRAM targets).
- Existing `training/lib/simcc3d.py`, `augmentation.py`, `dataset.py`
  scaffolds (what we build on).
- RTMPose paper + mmpose repo (Apache-2.0): `github.com/open-mmlab/mmpose`
  — RTMPose SimCC head, KLDiscretLoss, BoneLoss.
- RTMPose3D deepwiki + `b-arac/rtmpose3d` HF repo (Apache-2.0): 3D
  extension of SimCC; 133-keypoint whole-body model — we use the 17-joint
  body subset only.
- MobileNetV4 via `timm` (Apache-2.0): backbone choice per README.
- COCO-17 skeleton (public standard, no license).

---

## 7. Deliverables this iteration

1. **`dataset/scripts/build_dataset.py`** — generates 1000 samples, writes
   manifest + labels + images.
2. **`dataset/output/synth_v1/`** — the 1000-sample dataset (manifest.csv,
   labels.jsonl, images/*.png).
3. **`training/model/rtmpose3d_mnv4s.py`** — MobileNetV4-Conv-S + RTMPose3D
   head assembly.
4. **`training/train.py`** — single-GPU training loop with AMP, TensorBoard
   logging, checkpoint save.
5. **`training/eval.py`** — MPJPE / PA-MPJPE / PCK on self-val split;
   bone-length constancy / jitter on Kinetics CC sample.
6. **`training/config/baseline.yaml`** — all hyperparameters.
7. **`dataset/README.md`** — reproduction docs (setup, generate, train,
   eval).
8. **`eval_report.md`** — measured numbers on self-val + published-SOTA
   reference table.

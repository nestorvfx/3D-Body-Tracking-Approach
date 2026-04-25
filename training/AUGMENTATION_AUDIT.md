# Augmentation audit — canonical SOTA stack for 3D body-pose training

Source: mmpose `projects/rtmpose/.../rtmpose-l_8xb256-420e_coco-256x192.py`
(Apache-2.0), cross-checked against RTMPose tech report (arXiv 2303.07399),
BEDLAM paper (CVPR 2023 Sec 4), 4DHumans config, and sim-to-real best
practices from FDA (Yang & Soatto 2020) and CLIFF (ECCV 2022).

## Canonical RTMPose Stage-1 pipeline (body_2d_keypoint, 256×192 input)

Applied in order, per the mmpose config:

| step | transform | parameters |
|------|-----------|------------|
| 1 | LoadImage | |
| 2 | GetBBoxCenterScale | from labelled bbox |
| 3 | RandomFlip horizontal | p=0.5; joint-pair remap |
| 4 | RandomHalfBody | p=0.3 (default) |
| 5 | RandomBBoxTransform | scale_factor `[0.6, 1.4]`, rotate_factor `±80°` |
| 6 | TopdownAffine | warp to `(192, 256)` |
| 7 | YOLOXHSVRandomAug | HSV jitter |
| 8 | Albumentations A.Blur | p=0.1 |
| 9 | Albumentations A.MedianBlur | p=0.1 |
| 10 | Albumentations A.CoarseDropout | 1 hole, up to 40×40% of crop, `p=1.0` |
| 11 | GenerateTarget | SimCC Gaussian encode |

## Canonical RTMPose Stage-2 (final 30 epochs)

Gentler: scale `[0.75, 1.25]`, rotate `±60°`, shift 0, CoarseDropout `p=0.5`.
Not applied in our 20-epoch first run — we use Stage-1 aug throughout.

## BEDLAM additions (for sim-to-real)

BEDLAM relies on rendering realism (HDRIs, physics-sim cloth, scanned
clothing, depth of field).  Data-loader augmentation mirrors HMR baseline:
flip, crop, colour jitter.  They report **no FDA / no CutOut**; the
photoreal render IS the augmentation.  Our MPFB2 renders are less photoreal,
so we add stronger sim-to-real aug than BEDLAM needed.

## CLIFF additions

CLIFF uses HMR2's aug: scale `[0.7, 1.3]`, rotation `±30°`, occlusion
patches `p=0.3`, colour jitter.  Lighter on rotation than RTMPose.

## Our adopted stack (training/data.py v2)

Reconciling the three references + our sim-to-real needs:

| # | transform | probability / range | source |
|---|-----------|---------------------|--------|
| 1 | bbox_jitter — scale `[0.65, 1.35]` | always | RTMPose (wider than CLIFF) |
| 2 | bbox_jitter — shift `±12%` | always | RTMPose |
| 3 | half_body_bbox | `p=0.25` | RTMPose (mmpose default) |
| 4 | TopdownAffine + **rotation** `±45°` | always (rotation sampled uniform in range) | RTMPose |
| 5 | RandomFlip horizontal + COCO-17 pair remap | `p=0.5` | RTMPose |
| 6 | FDA against real reference corpus | `p=0.3` if refs provided | Yang 2020; critical for sim-to-real |
| 7 | ColorJitter (brightness 0.3, contrast 0.3, saturation 0.3, hue 0.05) | `p=0.8` | RTMPose HSV + CLIFF |
| 8 | HueSaturationValue | `p=0.4` | RTMPose HSV |
| 9 | RandomGamma `[0.7, 1.5]` | `p=0.4` | our sim2real |
| 10 | CLAHE | `p=0.1` | our sim2real |
| 11 | ToGray | `p=0.1` | our sim2real |
| 12 | ISONoise / GaussNoise / MultiplicativeNoise (OneOf) | `p=0.3` | our sim2real |
| 13 | MotionBlur / GaussianBlur / Defocus (OneOf) | `p=0.4` | RTMPose `Blur` + our sim2real |
| 14 | ImageCompression quality `[30, 95]` | `p=0.9` | phone realism |
| 15 | Downscale | `p=0.2` | phone realism |
| 16 | ImageCompression quality `[50, 90]` | `p=0.3` | double-JPEG |
| 17 | CoarseDropout (1-3 holes, 10-40% of crop) | `p=0.5` | RTMPose (reduced from p=1 to match mixed real+synth reality) |
| 18 | RandomFog / RandomShadow / RandomSunFlare (OneOf) | `p=0.1` | outdoor rarity |

Rotation range softened to ±45° (from RTMPose's ±80°) because our
synth-only dataset has limited pose extremes that would stay plausible
after large rotation; ±45° matches CLIFF-range.

## What's explicitly NOT in the stack

- **MixUp / CutMix**: not used in keypoint tasks per RTMPose ablation
  (unstable gradient on spatial targets).
- **Heavy perspective warp**: breaks pinhole-camera back-projection for 3D.
- **Vertical flip**: breaks gravity prior for 3D.

## Implementation plan

1. Rewrite `training/lib/augmentation.py` on Albumentations 2.x API
   (some calls changed, e.g. `ImageCompression(quality_range=…)`).
2. Add `A.Affine(rotate=…)` or equivalent BEFORE the photometric stack.
3. Wire into `training/data.py` replacing the ad-hoc `_build_photometric_aug`.
4. Produce `dataset/output/aug_debug/` with each aug step visualised on
   10 samples, 2D keypoints overlaid, to verify label transforms work.

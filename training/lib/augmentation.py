"""Sim-to-real Albumentations pipeline for RTMPose3D-style training.

Updated for Albumentations 2.x API.  Canonical stack documented in
training/AUGMENTATION_AUDIT.md.  Bridges clean-EEVEE-synth vs
real-phone-video gap via:
  * Geometric: bbox jitter, half-body crop, rotation, TopdownAffine, flip.
  * Photometric: ColorJitter + HSV + Gamma + CLAHE + ToGray.
  * Sensor: GaussNoise / ISONoise / MultiplicativeNoise, blur variants.
  * Compression: JPEG + Downscale + double-JPEG.
  * Occlusion: CoarseDropout with 1-3 holes at 10-40% of crop.
  * Weather: fog / shadow / sun-flare.
  * (Optional) FDA: Fourier Domain Adaptation to real reference corpus.

Applied at DataLoader time.  Never imports Blender.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import albumentations as A
    import cv2  # type: ignore
except ImportError:
    A = None
    cv2 = None


# COCO-17 flip index pairs (left-right swap during horizontal flip)
COCO17_FLIP_PAIRS: list[tuple[int, int]] = [
    (1, 2),   # eyes
    (3, 4),   # ears
    (5, 6),   # shoulders
    (7, 8),   # elbows
    (9, 10),  # wrists
    (11, 12), # hips
    (13, 14), # knees
    (15, 16), # ankles
]


def build_sim2real_aug(
    *,
    p_color: float = 0.8,
    p_blur: float = 0.4,
    p_noise: float = 0.3,
    p_jpeg: float = 0.9,
    p_dropout: float = 0.5,
    p_fog_shadow: float = 0.1,
    fda_reference_images: Sequence[np.ndarray] | None = None,
    p_fda: float = 0.3,
    occluders_active: bool = False,
):
    """Return an Albumentations 2.x Compose with keypoint-aware aug for pose.

    Consumes uint8 RGB images.  Keypoints format `xy` (pixel coords).  We
    set `remove_invisible=False` so out-of-frame joints are still returned
    (their coords become off-image) — the downstream code computes
    visibility masks from the bbox.
    """
    if A is None:
        raise RuntimeError("albumentations>=2.0 required")

    transforms = []

    # FDA is handled OUTSIDE this Compose now (see data.py _transform).
    # Albumentations 2.x's A.FDA requires per-call metadata kwargs, which
    # is awkward when we want a fixed reference corpus.  We instead call
    # the standalone numpy function `albumentations.fourier_domain_adaptation`
    # directly, with a manually-sampled reference per call.  The kwargs
    # `fda_reference_images` and `p_fda` are kept on this signature for
    # API compatibility but ignored here.
    _ = fda_reference_images, p_fda  # silence unused-arg linter

    # Photometric / colour
    transforms.extend([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                       hue=0.05, p=p_color),
        A.RandomGamma(gamma_limit=(70, 150), p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                              val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.1),
        A.ToGray(p=0.1),
    ])

    # Sensor noise
    transforms.append(A.OneOf([
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.GaussNoise(std_range=(0.04, 0.12), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
    ], p=p_noise))

    # Blur / defocus — phone camera motion
    transforms.append(A.OneOf([
        A.MotionBlur(blur_limit=(3, 11), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.Defocus(radius=(1, 3), p=1.0),
    ], p=p_blur))

    # Compression — real phones always JPEG + re-upload
    transforms.extend([
        A.ImageCompression(quality_range=(30, 95), p=p_jpeg),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.ImageCompression(quality_range=(50, 90), p=0.3),       # double-JPEG
    ])

    # Occlusion: prefer Sárándi-style realistic-object pasting (handled
    # outside Albumentations in data.py via sim2real_aug.occlude_with_objects).
    # When that's active (`occluders_active=True`) we drop CoarseDropout
    # because realistic occluders strictly subsume rectangular-hole dropout.
    # Fall back to RTMPose's CoarseDropout otherwise so the recipe doesn't
    # silently regress on setups that haven't installed an occluder corpus.
    if not occluders_active:
        transforms.append(A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.10, 0.40),     # fractions of image
            hole_width_range=(0.10, 0.40),
            fill=0,                             # pad with black
            p=p_dropout,
        ))

    # Outdoor rarities
    transforms.append(A.OneOf([
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=1.0),
        A.RandomShadow(p=1.0),
        A.RandomSunFlare(src_radius=80, p=1.0),
    ], p=p_fog_shadow))

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def horizontal_flip(
    img: np.ndarray,
    kps2d: np.ndarray,        # [17, 2] pixels
    kps3d: np.ndarray,        # [17, 3] camera space (x, y, z)
    visibility: np.ndarray,   # [17]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flip image horizontally and swap left/right COCO-17 joint indices.

    On 3D: negate X in camera space (standard mmpose flip).
    """
    H, W = img.shape[:2]
    img = np.ascontiguousarray(img[:, ::-1])

    kps2d = kps2d.copy()
    kps3d = kps3d.copy()
    vis = visibility.copy()

    kps2d[:, 0] = W - 1 - kps2d[:, 0]
    kps3d[:, 0] = -kps3d[:, 0]

    for l, r in COCO17_FLIP_PAIRS:
        kps2d[[l, r]] = kps2d[[r, l]]
        kps3d[[l, r]] = kps3d[[r, l]]
        vis[[l, r]] = vis[[r, l]]
    return img, kps2d, kps3d, vis


def jitter_bbox(
    bbox: tuple[int, int, int, int],
    *,
    rng,
    scale_range: tuple[float, float] = (0.65, 1.35),
    shift_range: float = 0.12,
) -> tuple[int, int, int, int]:
    """RTMPose bbox jitter: scale ±35%, translation ±12%."""
    x, y, w, h = bbox
    s = rng.uniform(*scale_range)
    new_w, new_h = w * s, h * s
    dx = rng.uniform(-shift_range, shift_range) * w
    dy = rng.uniform(-shift_range, shift_range) * h
    cx, cy = x + w / 2 + dx, y + h / 2 + dy
    return int(cx - new_w / 2), int(cy - new_h / 2), int(new_w), int(new_h)


def half_body_bbox(
    kps2d: np.ndarray,        # [17, 2]
    visibility: np.ndarray,   # [17]
    rng,
    padding: float = 1.5,
) -> tuple[int, int, int, int] | None:
    """RandomHalfBody: occasionally crop to upper or lower body only."""
    upper = list(range(0, 11))   # nose..wrists
    lower = list(range(11, 17))  # hips..ankles
    half = rng.choice([upper, lower])
    valid = [i for i in half if visibility[i] > 0.5]
    if len(valid) < 3:
        return None
    pts = kps2d[valid]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w, h = max(1, x_max - x_min), max(1, y_max - y_min)
    cx, cy = x_min + w / 2, y_min + h / 2
    w *= padding
    h *= padding
    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


def rotation_affine(
    img: np.ndarray,
    kps2d: np.ndarray,       # [17, 2] pixel coords in crop frame
    kps3d: np.ndarray,       # [17, 3] camera-frame metres
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate image + 2D keypoints by `angle_deg` around image centre.
    Also rotates the 3D Z=0 plane coordinates in-plane (since the camera
    rotated in view space, 3D X/Y in camera frame rotate by the same angle).
    """
    if cv2 is None:
        raise RuntimeError("opencv required for rotation_affine")
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W * 0.5, H * 0.5), angle_deg, 1.0)
    img_rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderValue=(114, 114, 114))
    # Rotate 2D keypoints.
    ones = np.ones((kps2d.shape[0], 1), dtype=np.float32)
    hpt = np.concatenate([kps2d, ones], axis=1)
    kps2d_new = hpt @ M.T
    # Rotate 3D X/Y by the same in-plane angle (camera frame).  Since
    # Blender-style camera has +X right, +Y down, +Z forward, rotating
    # the image by θ rotates X/Y in the OPPOSITE direction.
    th = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)
    R = np.array([[cos_t, -sin_t, 0],
                   [sin_t,  cos_t, 0],
                   [0,      0,     1]], dtype=np.float32)
    kps3d_rot = kps3d @ R.T
    return img_rot, kps2d_new.astype(np.float32), kps3d_rot.astype(np.float32)


def build_visibility_masks(
    kps2d_aug: np.ndarray,     # [17, 2] post-aug pixel coords
    img_hw: tuple[int, int],   # (H, W) of augmented crop
    original_vis: np.ndarray,  # [17] pre-aug
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-joint 2D and 3D loss masks.  2D loss masked to joints
    inside the crop; 3D stays at original visibility (CLIFF/HMR2 pattern)."""
    H, W = img_hw
    in_frame = ((kps2d_aug[:, 0] >= 0) & (kps2d_aug[:, 0] < W) &
                (kps2d_aug[:, 1] >= 0) & (kps2d_aug[:, 1] < H))
    vis_2d = (original_vis.astype(bool) & in_frame).astype(np.float32)
    vis_3d = original_vis.astype(np.float32)
    return vis_2d, vis_3d

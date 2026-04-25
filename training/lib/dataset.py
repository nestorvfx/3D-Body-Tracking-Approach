"""Training dataset: loads a sequence directory produced by our Blender pilot,
applies bbox-jitter + augmentations + SimCC-3D encoding, returns tensors.

This is the glue layer between `dataset/output/pilot_v2_*/seq_*/` and a torch
DataLoader.  It is intentionally torch-only here; no Blender imports.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

from .augmentation import (
    build_sim2real_aug, horizontal_flip, jitter_bbox, half_body_bbox,
    build_visibility_masks,
)
from .simcc3d import SimCC3DConfig, encode as simcc3d_encode


@dataclass
class DatasetConfig:
    pilot_dirs: list[str]                          # one or more output/<tag>/ dirs
    input_size: tuple[int, int] = (192, 256)       # W, H after TopdownAffine
    training: bool = True
    halfbody_prob: float = 0.25
    flip_prob: float = 0.5
    bbox_scale_range: tuple[float, float] = (0.75, 1.25)
    bbox_shift_range: float = 0.15


def _load_seq_manifests(pilot_dirs: list[str]) -> list[dict]:
    items = []
    for d in pilot_dirs:
        for seq_dir in sorted(Path(d).glob("seq_*")):
            labels_path = seq_dir / "labels.json"
            if not labels_path.exists():
                continue
            labels = json.loads(labels_path.read_text())
            for fa in labels["frames"]:
                png = seq_dir / fa["png_path"]
                if not png.exists():
                    continue
                items.append({
                    "image_path": str(png),
                    "sequence_id": labels["sequence_id"],
                    "frame_index": fa["frame_index"],
                    "resolution": labels["resolution"],
                    "kps2d": [(k["px"] if k["px"] is not None else 0.0,
                              k["py"] if k["py"] is not None else 0.0)
                             for k in fa["keypoints_2d_pixels"]],
                    "kps3d": [(k["x"] or 0.0, k["y"] or 0.0, k["z"] or 0.0)
                             for k in fa["keypoints_3d_world_m"]],
                    "visibility": [1.0 if (k and k["px"] is not None) else 0.0
                                   for k in fa["keypoints_2d_pixels"]],
                    "camera_intrinsics": fa["camera_intrinsics"],
                    "camera_extrinsics": fa["camera_extrinsics"],
                })
    return items


def _bbox_from_keypoints(
    kps2d: np.ndarray, visibility: np.ndarray, *, pad: float = 1.25
) -> tuple[int, int, int, int]:
    """Tight bbox around labeled keypoints, expanded by `pad`."""
    valid = kps2d[visibility > 0.5]
    if len(valid) == 0:
        return 0, 0, 1, 1
    x_min, y_min = valid.min(axis=0)
    x_max, y_max = valid.max(axis=0)
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    cx, cy = x_min + w / 2, y_min + h / 2
    w *= pad
    h *= pad
    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


def _topdown_affine(
    img: np.ndarray,
    kps2d: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract bbox crop, resize to out_size=(W,H), remap keypoints."""
    if cv2 is None:
        raise RuntimeError("opencv-python required for _topdown_affine")
    x, y, w, h = bbox
    H_img, W_img = img.shape[:2]

    # Source points (corners of bbox, no rotation)
    src = np.array([[x, y], [x + w, y], [x, y + h]], dtype=np.float32)
    W_out, H_out = out_size
    dst = np.array([[0, 0], [W_out, 0], [0, H_out]], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (W_out, H_out),
                           flags=cv2.INTER_LINEAR,
                           borderValue=(114, 114, 114))
    kps_new = np.concatenate([kps2d, np.ones((kps2d.shape[0], 1), dtype=np.float32)], axis=1)
    kps_new = kps_new @ M.T
    return crop, kps_new


class SyntheticPoseDataset(Dataset):
    """Per-frame COCO-17 3D pose dataset from a Blender pilot output dir."""

    def __init__(self, cfg: DatasetConfig, fda_reference_images=None):
        self.cfg = cfg
        self.records = _load_seq_manifests(cfg.pilot_dirs)
        if not self.records:
            raise RuntimeError(f"No frames found in {cfg.pilot_dirs}")
        self.aug = build_sim2real_aug(fda_reference_images=fda_reference_images)
        self.simcc_cfg = SimCC3DConfig(
            input_size=(cfg.input_size[0], cfg.input_size[1], cfg.input_size[1]),
        )
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        rng = random.Random(idx * 1_000_003 + (0 if self.cfg.training else 1))

        img = cv2.imread(r["image_path"])                 # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kps2d = np.asarray(r["kps2d"], dtype=np.float32)  # [17, 2]
        kps3d = np.asarray(r["kps3d"], dtype=np.float32)  # [17, 3]
        vis = np.asarray(r["visibility"], dtype=np.float32)

        # bbox: half-body or full
        bbox = None
        if self.cfg.training and rng.random() < self.cfg.halfbody_prob:
            bbox = half_body_bbox(kps2d, vis, rng)
        if bbox is None:
            bbox = _bbox_from_keypoints(kps2d, vis)
        if self.cfg.training:
            bbox = jitter_bbox(bbox, rng=rng,
                                scale_range=self.cfg.bbox_scale_range,
                                shift_range=self.cfg.bbox_shift_range)

        # Topdown crop + keypoint warp
        crop, kps2d_crop = _topdown_affine(img, kps2d, bbox, self.cfg.input_size)

        # Horizontal flip
        if self.cfg.training and rng.random() < self.cfg.flip_prob:
            crop, kps2d_crop, kps3d, vis = horizontal_flip(crop, kps2d_crop, kps3d, vis)

        # Albumentations photometric/noise/JPEG
        if self.cfg.training:
            out = self.aug(image=crop, keypoints=kps2d_crop.tolist())
            crop = out["image"]
            kps2d_crop = np.asarray(out["keypoints"], dtype=np.float32)

        # Visibility masks
        vis_2d, vis_3d = build_visibility_masks(kps2d_crop, crop.shape[:2], vis)

        # SimCC-3D target encoding
        targets = simcc3d_encode(kps2d_crop, kps3d, vis, self.simcc_cfg)

        # Normalize + tensorize
        img_t = (crop.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).contiguous()
        return {
            "image": img_t,
            "target_x": torch.from_numpy(targets["target_x"]),
            "target_y": torch.from_numpy(targets["target_y"]),
            "target_z": torch.from_numpy(targets["target_z"]),
            "root_z": torch.tensor(float(targets["root_z"])),
            "keypoints_2d": torch.from_numpy(kps2d_crop),
            "keypoints_3d": torch.from_numpy(kps3d),
            "vis_2d": torch.from_numpy(vis_2d),
            "vis_3d": torch.from_numpy(vis_3d),
            "sequence_id": r["sequence_id"],
            "frame_index": r["frame_index"],
        }

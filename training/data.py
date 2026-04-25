"""PyTorch Dataset for the JSONL labels produced by dataset/scripts/build_dataset.py.

Applies the RTMPose-canonical + sim-to-real augmentation stack documented
in training/AUGMENTATION_AUDIT.md:
  1. bbox jitter (scale 0.65-1.35, shift ±12 %)
  2. half-body crop (p=0.25)
  3. TopdownAffine to crop
  4. rotation ±45° (training only)
  5. horizontal flip + COCO-17 pair remap + 3D-X mirror (p=0.5)
  6. build_sim2real_aug (FDA-ready, ColorJitter, HSV, Gamma, CLAHE, Gray,
     Noise, Blur, JPEG compression x2, Downscale, CoarseDropout,
     Fog/Shadow/SunFlare)
  7. ImageNet normalisation
  8. SimCC-3D target encoding

Returns a dict of tensors (+ some meta) for each sample.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
from simcc3d import SimCC3DConfig, encode as simcc3d_encode   # noqa: E402
from augmentation import (   # noqa: E402
    build_sim2real_aug, horizontal_flip, jitter_bbox, half_body_bbox,
    build_visibility_masks, rotation_affine,
)


@dataclass
class DataConfig:
    dataset_dir: str
    split: Literal["train", "val"] = "train"
    input_wh: tuple[int, int] = (192, 256)   # (W, H) — RTMPose3D default
    bbox_scale_range: tuple[float, float] = (0.65, 1.35)
    bbox_shift_frac: float = 0.12
    half_body_prob: float = 0.25
    flip_prob: float = 0.5
    rotation_deg: float = 45.0
    photometric: bool = True
    training: bool = True


def _load_manifest(dataset_dir: Path, split: str) -> list[dict]:
    out = []
    with (dataset_dir / "labels.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            out.append(rec)
    return out


def _topdown_affine(img: np.ndarray, bbox_xywh, out_wh: tuple[int, int]):
    x, y, w, h = bbox_xywh
    src = np.array([[x, y], [x + w, y], [x, y + h]], dtype=np.float32)
    W, H = out_wh
    dst = np.array([[0, 0], [W, 0], [0, H]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img, M, (W, H),
                           flags=cv2.INTER_LINEAR,
                           borderValue=(114, 114, 114))
    return crop, M


def _warp_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    hpt = np.concatenate([pts.astype(np.float32), ones], axis=1)
    return hpt @ M.T


class SynthPoseDataset(Dataset):
    """Yields RGB + SimCC-3D targets for the JSONL dataset."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.root = Path(cfg.dataset_dir)
        self.records = _load_manifest(self.root, cfg.split)
        if not self.records:
            raise RuntimeError(
                f"No samples found in {cfg.dataset_dir} split={cfg.split}")
        self.simcc = SimCC3DConfig(
            input_size=(cfg.input_wh[0], cfg.input_wh[1], cfg.input_wh[1]))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.photo = build_sim2real_aug() if (cfg.training and cfg.photometric) else None

    def __len__(self) -> int:
        return len(self.records)

    def _transform(self, img_full, kps_2d_full, kps_3d, bbox, rng, training: bool,
                   return_stages=False):
        """Run the aug stack. Optionally collect per-stage intermediates
        for the aug_debug viewer.  Returns (crop, kps2d_crop, kps3d, vis01,
        stages-or-None)."""
        stages = {}
        vis = kps_2d_full[:, 2].copy()
        if return_stages:
            stages["orig"] = (img_full.copy(), kps_2d_full.copy(), vis.copy())

        # --- bbox jitter + half-body ---
        use_bbox = list(bbox)
        if training:
            if rng.random() < self.cfg.half_body_prob:
                hb = half_body_bbox(kps_2d_full[:, :2], vis, rng)
                if hb is not None:
                    use_bbox = list(hb)
            use_bbox = list(jitter_bbox(
                use_bbox, rng=rng,
                scale_range=self.cfg.bbox_scale_range,
                shift_range=self.cfg.bbox_shift_frac))

        # --- crop (TopdownAffine) ---
        crop, M = _topdown_affine(img_full, use_bbox, self.cfg.input_wh)
        kps2d = _warp_points(kps_2d_full[:, :2], M)
        if return_stages:
            stages["after_bbox"] = (crop.copy(), kps2d.copy(), vis.copy())

        # --- rotation (training only) ---
        if training and self.cfg.rotation_deg > 0:
            angle = rng.uniform(-self.cfg.rotation_deg, self.cfg.rotation_deg)
            crop, kps2d, kps_3d = rotation_affine(crop, kps2d, kps_3d, angle)
            if return_stages:
                stages["after_rotate"] = (crop.copy(), kps2d.copy(), vis.copy())

        # --- horizontal flip ---
        if training and rng.random() < self.cfg.flip_prob:
            crop, kps2d, kps_3d, vis = horizontal_flip(crop, kps2d, kps_3d, vis)
            if return_stages:
                stages["after_flip"] = (crop.copy(), kps2d.copy(), vis.copy())

        # --- sim2real photometric / noise / occlusion ---
        if training and self.photo is not None:
            out = self.photo(image=crop, keypoints=kps2d.tolist())
            crop = out["image"]
            kps2d = np.array(out["keypoints"], dtype=np.float32)
            if return_stages:
                stages["after_photo"] = (crop.copy(), kps2d.copy(), vis.copy())

        vis01 = (vis > 0).astype(np.float32)
        return crop, kps2d, kps_3d, vis01, use_bbox, stages

    def __getitem__(self, idx):
        rec = self.records[idx]
        rng = random.Random(
            idx * 1_000_003 + (0 if self.cfg.training else 1))

        img_path = self.root / rec["image_rel"]
        img_bgr = cv2.imread(str(img_path))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        kps_2d_full = np.array(rec["keypoints_2d"], dtype=np.float32)   # [17, 3] (u,v,vis)
        kps_3d = np.array(rec["keypoints_3d_cam"], dtype=np.float32)    # [17, 3]
        bbox = tuple(rec["bbox_xywh"])

        crop, kps2d, kps3d, vis01, use_bbox, _ = self._transform(
            img, kps_2d_full, kps_3d, bbox, rng, self.cfg.training)

        # SimCC-3D target encoding.
        targets = simcc3d_encode(kps2d, kps3d, vis01, self.simcc)

        # ImageNet normalise + tensor.
        img_t = (crop.astype(np.float32) / 255.0 - self.mean) / self.std
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).contiguous()

        # Camera intrinsics (full-image K matrix, 3x3, in pixels).
        # Needed at eval time to back-project 2D+Z predictions into the
        # camera 3D frame.  We keep the ORIGINAL K (full-image) and store
        # the cropped-frame bbox so eval can un-warp the crop pixel back
        # to original-image pixel before applying K^-1.
        K = np.array(rec["camera_K"], dtype=np.float32)

        # CLIFF-style conditioning features: bbox (normalised by full-image
        # size) + camera focals (normalised by full-image diagonal).  Giving
        # the network the crop's position in the full frame + camera focal
        # lets it predict absolute root depth instead of relying on a
        # weak-perspective guess at inference time (see CLIFF, ECCV 2022).
        #
        # The bbox used here is the POST-jitter bbox (the crop's actual
        # source region in the full image), not the raw label bbox — that's
        # what the model needs to geometrically reason about.
        img_w_full = float(rec["image_wh"][0])
        img_h_full = float(rec["image_wh"][1])
        diag = float((img_w_full ** 2 + img_h_full ** 2) ** 0.5)
        bx, by, bw, bh = use_bbox
        cond = np.array([
            (bx + bw / 2.0) / img_w_full,   # bbox centre x (frac of image W)
            (by + bh / 2.0) / img_h_full,   # bbox centre y (frac of image H)
            bw / img_w_full,                # bbox width   (frac of image W)
            bh / img_h_full,                # bbox height  (frac of image H)
            float(K[0, 0]) / diag,          # fx / image diagonal
            float(K[1, 1]) / diag,          # fy / image diagonal
        ], dtype=np.float32)

        # RootNet k_prior: geometric depth prior assuming an "average" human
        # occupies a 2m × 2m bounding area (A_real = 4 m² canonical).  The
        # network only learns multiplicative corrections (gamma_size for
        # body-size deviation, gamma_pose for pose compactness) on top of
        # this analytical geometric anchor.
        #
        #   k = sqrt(fx · fy · A_real / A_bbox)
        #
        # Clamp bbox area to at least 1 pixel² to avoid division blow-up
        # on pathological samples.  Clamp k itself to [0.3, 100] metres to
        # guard against numerically-degenerate training labels — outliers
        # (CMU BVH bugs with root_z 10000+m) produce valid but huge k_priors;
        # the log-gamma head absorbs the remainder gracefully, but a hard
        # clamp prevents fp16 overflow in autocast.
        A_REAL_M2 = 4.0   # (2 m)^2 — RootNet canonical human area prior
        A_bbox_px2 = max(1.0, float(bw * bh))
        k_prior = float(
            (float(K[0, 0]) * float(K[1, 1]) * A_REAL_M2 / A_bbox_px2) ** 0.5)
        k_prior = max(0.3, min(100.0, k_prior))

        return {
            "image": img_t,
            "target_x": torch.from_numpy(targets["target_x"]),
            "target_y": torch.from_numpy(targets["target_y"]),
            "target_z": torch.from_numpy(targets["target_z"]),
            "root_z": torch.tensor(float(targets["root_z"])),
            "kps2d": torch.from_numpy(kps2d.astype(np.float32)),
            "kps3d": torch.from_numpy(kps3d.astype(np.float32)),
            "vis": torch.from_numpy(vis01),
            "camera_K": torch.from_numpy(K),                   # [3,3]
            "image_wh_full": torch.tensor(rec["image_wh"], dtype=torch.float32),
            # NB: `kps2d` above is in CROP frame (after all augs).  For
            # val we bypass aug (photometric=False), but keypoints still
            # live in crop frame because of TopdownAffine.  We carry the
            # pre-crop bbox so eval can un-crop.
            "bbox_pre_crop": torch.tensor(list(use_bbox), dtype=torch.float32),
            "cond": torch.from_numpy(cond),                    # [6]
            "k_prior": torch.tensor(k_prior, dtype=torch.float32),
            "id": rec["id"],
        }

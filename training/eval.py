"""Evaluate a trained checkpoint on the synth_v1 val split.

Reports:
  - MPJPE and PA-MPJPE (root-relative) in millimetres.
  - PCK@50mm, PCK@100mm.
  - Per-source breakdown (CMU / 100STYLE / AIST++).
  - Bone-length constancy std (label-free stability indicator — small = good).

Usage:
    python -m training.eval \
        --dataset-dir dataset/output/synth_v1 \
        --ckpt training/runs/baseline_v1/best.pt \
        --report eval_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from training.data import DataConfig, SynthPoseDataset   # noqa: E402
from training.model import build_model, decode_simcc     # noqa: E402
from training.losses import pa_mpjpe                      # noqa: E402


def backproject_to_cam(kps_uv_crop: torch.Tensor, depth_z: torch.Tensor,
                         K: torch.Tensor, bbox_xywh: torch.Tensor,
                         input_wh: tuple[int, int]) -> torch.Tensor:
    """Back-project [B,J,2] crop-pixel keypoints + [B,J] depth to 3D
    camera-frame coordinates using the ORIGINAL image intrinsics K.

    The model operates on a bbox crop of size input_wh, but K was defined
    on the full render.  We first un-warp the crop-pixel to full-image
    pixel (invert the affine), then back-project.
    """
    B, J, _ = kps_uv_crop.shape
    x_crop, y_crop = kps_uv_crop[..., 0], kps_uv_crop[..., 1]
    W_in, H_in = input_wh
    bx = bbox_xywh[:, 0].unsqueeze(-1)
    by = bbox_xywh[:, 1].unsqueeze(-1)
    bw = bbox_xywh[:, 2].unsqueeze(-1)
    bh = bbox_xywh[:, 3].unsqueeze(-1)
    u_full = bx + x_crop / W_in * bw
    v_full = by + y_crop / H_in * bh

    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)
    x = (u_full - cx) / fx * depth_z
    y = (v_full - cy) / fy * depth_z
    return torch.stack([x, y, depth_z], dim=-1)   # [B, J, 3]


def pck_at(thresh_m: float, pred_3d, gt_3d, vis) -> float:
    """Percent of joints within thresh_m of GT (root-relative)."""
    err = (pred_3d - gt_3d).norm(dim=-1)     # [B, J]
    hits = (err < thresh_m).float() * vis
    return float(hits.sum() / vis.sum().clamp(min=1.0))


def bone_length_std_mm(pred_3d: torch.Tensor) -> float:
    """Standard deviation of bone lengths across the batch — a
    label-free stability signal when the same character appears across
    multiple frames.  In this eval we approximate by treating each sample
    independently; the cross-sample std includes normal anatomical
    variation so the number is informative but not tight.
    """
    from training.losses import COCO17_BONES
    parents = torch.tensor([b[0] for b in COCO17_BONES])
    children = torch.tensor([b[1] for b in COCO17_BONES])
    lens = (pred_3d[:, children] - pred_3d[:, parents]).norm(dim=-1)  # [B, nBones]
    return float(lens.std(dim=0).mean() * 1000.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--report", default="eval_report.md")
    p.add_argument("--backbone", default="mnv4s")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--input-w", type=int, default=192)
    p.add_argument("--input-h", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_wh = (args.input_w, args.input_h)

    val_cfg = DataConfig(dataset_dir=args.dataset_dir, split="val",
                          input_wh=input_wh, training=False, photometric=False)
    ds = SynthPoseDataset(val_cfg)
    ld = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"[eval] val={len(ds)} samples")

    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    per_src_mpjpe: dict[str, list[float]] = {}
    per_src_pa: dict[str, list[float]] = {}
    all_pred = []
    all_gt = []
    all_vis = []

    with torch.no_grad():
        for batch in ld:
            img = batch["image"].to(device)
            root_z = batch["root_z"].to(device)
            cond = batch["cond"].to(device)
            k_prior = batch["k_prior"].to(device)
            gt3d = batch["kps3d"].to(device)
            vis = batch["vis"].to(device)
            K = batch["camera_K"].to(device)
            bbox = batch["bbox_pre_crop"].to(device)
            out = model(img, cond, k_prior=k_prior)
            # Use the model's learned root-z if present; else fall back to GT
            # (legacy models); useful to ablate the head's contribution.
            root_z_use = out.get("root_z", root_z)
            kps2d, kps_z = decode_simcc(
                out["x_logits"], out["y_logits"], out["z_logits"],
                input_wh=input_wh, root_z=root_z_use, mode="argmax")

            # Back-project crop-frame 2D + Z through intrinsics → cam3D.
            from training.train import backproject_crop_to_cam3d
            pred3d = backproject_crop_to_cam3d(
                kps2d, kps_z, K, bbox, input_wh)
            err = (pred3d - gt3d).norm(dim=-1)   # [B, J]
            mpjpe_per = (err * vis).sum(dim=-1) / vis.sum(dim=-1).clamp(min=1.0)
            for i, sid in enumerate(batch["id"]):
                src = sid.split("_", 1)[0]
                per_src_mpjpe.setdefault(src, []).append(float(mpjpe_per[i]))
                per_src_pa.setdefault(src, []).append(
                    float(pa_mpjpe(pred3d[i:i+1], gt3d[i:i+1])))
            all_pred.append(pred3d.cpu())
            all_gt.append(gt3d.cpu())
            all_vis.append(vis.cpu())

    pred_cat = torch.cat(all_pred, dim=0)
    gt_cat = torch.cat(all_gt, dim=0)
    vis_cat = torch.cat(all_vis, dim=0)

    mpjpe_all = ((pred_cat - gt_cat).norm(dim=-1) * vis_cat).sum() / vis_cat.sum().clamp(min=1.0)
    pa_all = pa_mpjpe(pred_cat, gt_cat)
    pck50 = pck_at(0.05, pred_cat, gt_cat, vis_cat)
    pck100 = pck_at(0.10, pred_cat, gt_cat, vis_cat)
    bone_std = bone_length_std_mm(pred_cat)

    report = []
    report.append("# Eval report — synth_v1 val split\n")
    report.append(f"Checkpoint: `{args.ckpt}`\n")
    report.append(f"Epoch:      `{ckpt.get('epoch', '?')}`\n")
    report.append(f"Val size:   `{len(ds)}`\n\n")
    report.append("## Aggregate metrics\n\n")
    report.append(f"- MPJPE:     **{float(mpjpe_all)*1000:.1f} mm**\n")
    report.append(f"- PA-MPJPE:  **{float(pa_all)*1000:.1f} mm**\n")
    report.append(f"- PCK@50mm:  {pck50*100:.1f} %\n")
    report.append(f"- PCK@100mm: {pck100*100:.1f} %\n")
    report.append(f"- Bone-length std (cross-sample): {bone_std:.1f} mm\n\n")
    report.append("## Per-source MPJPE\n\n")
    for src, lst in per_src_mpjpe.items():
        arr = np.array(lst)
        report.append(f"- **{src}** ({len(arr)}): MPJPE {arr.mean()*1000:.1f} mm, "
                      f"PA-MPJPE {np.mean(per_src_pa[src])*1000:.1f} mm\n")

    Path(args.report).write_text("".join(report))
    print(f"[eval] wrote {args.report}")
    print(f"  MPJPE {float(mpjpe_all)*1000:.1f}mm  "
          f"PA-MPJPE {float(pa_all)*1000:.1f}mm  "
          f"PCK@50 {pck50*100:.1f}%  "
          f"PCK@100 {pck100*100:.1f}%")


if __name__ == "__main__":
    main()

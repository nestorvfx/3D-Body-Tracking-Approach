"""Generic benchmark runner.

Loads a checkpoint, iterates any Benchmark (from this package), runs the
model on each sample, computes MPJPE / PA-MPJPE / PCK on the benchmark's
COCO-17-mapped subset, writes a report with per-subject breakdown.

Usage:
    python -m training.benchmarks.run \
        --benchmark <module-name> \
        --data-root <path-to-test-set> \
        --ckpt training/runs/<run>/best.pt \
        --report reports/<run>_<benchmark>.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))          # project root

from base import (   # noqa: E402
    BenchmarkSample, pick_matched, valid_joint_mask, COCO17_NAMES,
)

from training.model import build_model, decode_simcc   # noqa: E402
from training.losses import pa_mpjpe                    # noqa: E402
from training.train import backproject_crop_to_cam3d   # noqa: E402
from training.lib.pose_anchor import itrr_refine_root  # noqa: E402


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("args", {}).get("backbone", "mnv4s")
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def preprocess_crop(img_rgb: np.ndarray, bbox_xywh, input_wh=(192, 256)):
    """Return a [1,3,H,W] normalised tensor + the affine used."""
    x, y, w, h = bbox_xywh
    src = np.array([[x, y], [x + w, y], [x, y + h]], dtype=np.float32)
    W_out, H_out = input_wh
    dst = np.array([[0, 0], [W_out, 0], [0, H_out]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img_rgb, M, (W_out, H_out),
                           flags=cv2.INTER_LINEAR,
                           borderValue=(114, 114, 114))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (crop.astype(np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    return t, M


def estimate_root_z(bbox_xywh, K) -> float:
    """Legacy weak-perspective depth estimate: Z ≈ focal * body_height / bbox_height.
    Retained as a fallback for models without a learned root-depth head.
    """
    _, _, _, h = bbox_xywh
    fy = float(K[1, 1])
    # Average human height ~1.7m subtending bbox height.
    return fy * 1.7 / max(1.0, float(h))


def compute_cond(bbox_xywh, K, full_wh) -> torch.Tensor:
    """CLIFF-style [6]-vec: bbox + focal normalised to full-image frame.
    MUST match training (training/data.py __getitem__)."""
    W, H = full_wh
    diag = float((W * W + H * H) ** 0.5)
    bx, by, bw, bh = bbox_xywh
    return torch.tensor([
        (bx + bw / 2.0) / W,
        (by + bh / 2.0) / H,
        bw / W,
        bh / H,
        float(K[0, 0]) / diag,
        float(K[1, 1]) / diag,
    ], dtype=torch.float32)


def compute_k_prior(bbox_xywh, K) -> float:
    """RootNet geometric depth prior: sqrt(fx · fy · A_real / A_bbox) metres,
    with A_real = 4 m² (2m × 2m canonical human).  Must match the formula
    used in training/data.py __getitem__ exactly."""
    A_REAL_M2 = 4.0
    _, _, bw, bh = bbox_xywh
    A_bbox_px2 = max(1.0, float(bw * bh))
    k = float((float(K[0, 0]) * float(K[1, 1]) * A_REAL_M2 / A_bbox_px2) ** 0.5)
    return max(0.3, min(100.0, k))


def run_benchmark(benchmark, model, device, input_wh=(192, 256),
                   use_pose_anchor: bool = False):
    """Run the model over a benchmark and return per-sample predictions + GT.

    If ``use_pose_anchor`` is True, applies PoseAnchor ITRR root-translation
    refinement (Kim et al., ICCV 2025) as a CPU post-processing step after
    the model's root_z prediction.  Published improvement: +5-15 mm MPJPE
    on H3.6M / 3DPW with zero retraining.  ~0.5 ms per frame on CPU.
    """
    pred_list, gt_list, mask_list, sid_list = [], [], [], []
    for s in benchmark.iter_samples():
        # Preprocess crop.
        img_t, _M = preprocess_crop(s.image_rgb, s.bbox_xywh, input_wh)
        img_t = img_t.to(device)
        K_t = torch.from_numpy(s.camera_K).unsqueeze(0).to(device)
        bbox_t = torch.tensor(list(s.bbox_xywh), dtype=torch.float32,
                                device=device).unsqueeze(0)
        # CLIFF-style conditioning: bbox + focal normalised to full image.
        full_wh = (s.image_rgb.shape[1], s.image_rgb.shape[0])
        cond = compute_cond(s.bbox_xywh, s.camera_K, full_wh
                             ).unsqueeze(0).to(device)
        # RootNet geometric depth prior (pinhole + canonical 2m×2m).
        k_prior = torch.tensor([compute_k_prior(s.bbox_xywh, s.camera_K)],
                                device=device)

        with torch.no_grad():
            out = model(img_t, cond, k_prior=k_prior)
            # Prefer the model's learned root-depth head.  If the checkpoint
            # doesn't have one (legacy train_v1), fall back to the weak-
            # perspective estimate.
            if "root_z" in out:
                root_z_use = out["root_z"]
            else:
                root_z_use = torch.tensor(
                    [estimate_root_z(s.bbox_xywh, s.camera_K)], device=device)
            kps2d, kps_z = decode_simcc(
                out["x_logits"], out["y_logits"], out["z_logits"],
                input_wh=input_wh, root_z=root_z_use,
                mode="argmax")
            pred_cam = backproject_crop_to_cam3d(
                kps2d, kps_z, K_t, bbox_t, input_wh)[0].cpu().numpy()

        # Optional: PoseAnchor ITRR refinement (CPU-only, ~0.5 ms).
        # Runs OUTSIDE the torch.no_grad block because it's numpy/lstsq.
        if use_pose_anchor:
            # 2D crop-pixels -> full-image pixels (inverse of crop affine).
            kps2d_crop_np = kps2d[0].cpu().numpy()
            x0, y0, w0, h0 = s.bbox_xywh
            kps2d_full = np.stack([
                x0 + kps2d_crop_np[:, 0] * w0 / input_wh[0],
                y0 + kps2d_crop_np[:, 1] * h0 / input_wh[1],
            ], axis=-1)
            # Root-relative 3D (subtract the L/R-hip midpoint).
            root3d = 0.5 * (pred_cam[11] + pred_cam[12])
            kps3d_rel = pred_cam - root3d[None, :]
            # Solve for a refined root translation T_ref.
            T_ref = itrr_refine_root(
                kps2d_full, kps3d_rel, s.camera_K,
                n_iters=5, support_frac=0.7)
            # Apply the refinement: re-centre relative 3D at T_ref.
            pred_cam = kps3d_rel + T_ref[None, :]

        # Pick COCO-matched joints from GT.
        gt_matched = pick_matched(s.gt_kps3d_cam, benchmark.coco17_to_bench)
        gt_vis17 = np.zeros(17, dtype=np.float32)
        for i, j in enumerate(benchmark.coco17_to_bench):
            if j >= 0:
                gt_vis17[i] = s.gt_vis[j]
        valid = valid_joint_mask(gt_matched, gt_vis17)

        pred_list.append(pred_cam.astype(np.float32))
        gt_list.append(gt_matched.astype(np.float32))
        mask_list.append(valid)
        sid_list.append(s.sample_id)

    pred = np.stack(pred_list, axis=0)   # [N, 17, 3]
    gt = np.stack(gt_list, axis=0)        # may contain NaN for -1 joints
    mask = np.stack(mask_list, axis=0)    # [N, 17] bool
    return pred, gt, mask, sid_list


def mpjpe_root_relative(pred: np.ndarray, gt: np.ndarray,
                         mask: np.ndarray,
                         root_coco_pair=(11, 12)) -> float:
    """Root-relative MPJPE averaged over valid joints, in millimetres.
    Root = mean of L/R hips per sample (COCO indices 11, 12)."""
    # Root per sample.
    hips_valid = mask[:, root_coco_pair[0]] & mask[:, root_coco_pair[1]]
    root_p = (pred[:, root_coco_pair[0]] + pred[:, root_coco_pair[1]]) / 2
    root_g = (gt[:, root_coco_pair[0]] + gt[:, root_coco_pair[1]]) / 2
    p_rel = pred - root_p[:, None, :]
    g_rel = gt - root_g[:, None, :]
    err = np.linalg.norm(p_rel - g_rel, axis=-1)    # [N, 17]
    err = err[mask & hips_valid[:, None]]
    if err.size == 0:
        return float("nan")
    return float(err.mean() * 1000.0)


def pa_mpjpe_np(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Procrustes-aligned MPJPE, per-sample, averaged over samples.
    Uses only jointly-valid joints in each sample."""
    import torch as T
    errs = []
    for p, g, m in zip(pred, gt, mask):
        if m.sum() < 3:
            continue
        p_sub = T.from_numpy(p[m][None])
        g_sub = T.from_numpy(g[m][None])
        err = pa_mpjpe(p_sub, g_sub)
        errs.append(float(err))
    if not errs:
        return float("nan")
    return float(np.mean(errs) * 1000.0)


def pck_at(thresh_mm: float, pred: np.ndarray, gt: np.ndarray,
            mask: np.ndarray) -> float:
    """Percentage of joints within `thresh_mm` of GT (root-relative)."""
    hip_l, hip_r = 11, 12
    root_p = (pred[:, hip_l] + pred[:, hip_r]) / 2
    root_g = (gt[:, hip_l] + gt[:, hip_r]) / 2
    err = np.linalg.norm(
        (pred - root_p[:, None]) - (gt - root_g[:, None]), axis=-1)
    hits = (err * 1000.0 < thresh_mm) & mask
    denom = mask.sum()
    return float(hits.sum() / max(1, denom))


def write_report(report_path: Path, bench_name: str, ckpt_info,
                 mpjpe_mm: float, pa_mm: float, pck150: float, pck100: float,
                 per_subject: dict, n_samples: int, n_matched_joints: int):
    lines = []
    lines.append(f"# Benchmark report — {bench_name}\n")
    lines.append(f"Checkpoint: `{ckpt_info['path']}`  (epoch {ckpt_info.get('epoch', '?')}, "
                 f"backbone `{ckpt_info.get('backbone', '?')}`)\n")
    lines.append(f"Benchmark samples: {n_samples}\n")
    lines.append(f"Matched COCO joints: {n_matched_joints} / 17 "
                  "(some benchmarks lack face-keypoint correspondences)\n\n")

    lines.append("## Our numbers on this benchmark\n\n")
    lines.append(f"- MPJPE (root-relative): **{mpjpe_mm:.1f} mm**\n")
    lines.append(f"- PA-MPJPE (Procrustes):  **{pa_mm:.1f} mm**\n")
    lines.append(f"- PCK@150mm:              **{pck150*100:.1f} %**\n")
    lines.append(f"- PCK@100mm:              **{pck100*100:.1f} %**\n\n")

    if per_subject:
        lines.append("## Per-subject MPJPE\n\n")
        for sub, m in per_subject.items():
            lines.append(f"- {sub}: {m:.1f} mm\n")
        lines.append("\n")

    report_path.write_text("".join(lines))
    print(f"[report] wrote {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                    help="benchmark module name in this package "
                          "(e.g. a commercial-clean evaluation set you've added)")
    p.add_argument("--data-root", required=True,
                    help="path to benchmark test-set root (see README)")
    p.add_argument("--ckpt", required=True, help="trained model checkpoint (.pt)")
    p.add_argument("--report", required=True, help="output markdown path")
    p.add_argument("--stride", type=int, default=5,
                    help="sample every N-th frame (default 5)")
    p.add_argument("--input-w", type=int, default=192)
    p.add_argument("--input-h", type=int, default=256)
    p.add_argument("--use-pose-anchor", action="store_true",
                    help="Apply PoseAnchor ITRR root-position refinement "
                         "(Kim ICCV 2025) as CPU post-processing.  "
                         "Typically +5-15 mm MPJPE over the raw head output; "
                         "costs ~0.5 ms per frame on a performance CPU core.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    # Import benchmark module dynamically.
    bench_mod = importlib.import_module(args.benchmark)
    benchmark = bench_mod.build_benchmark(args.data_root, frame_stride=args.stride)
    print(f"[benchmark] {benchmark.name}: {len(benchmark)} samples")

    model, ckpt = load_model(Path(args.ckpt), device)
    ckpt_info = {"path": args.ckpt,
                 "epoch": ckpt.get("epoch", "?"),
                 "backbone": ckpt.get("args", {}).get("backbone", "?")}
    print(f"[model] backbone={ckpt_info['backbone']}, epoch={ckpt_info['epoch']}")

    input_wh = (args.input_w, args.input_h)
    pred, gt, mask, sids = run_benchmark(
        benchmark, model, device, input_wh,
        use_pose_anchor=args.use_pose_anchor)

    # Metrics.
    mpjpe_mm = mpjpe_root_relative(pred, gt, mask)
    pa_mm = pa_mpjpe_np(pred, gt, mask)
    pck150 = pck_at(150.0, pred, gt, mask)
    pck100 = pck_at(100.0, pred, gt, mask)

    # Per-group breakdown using the prefix before "/" in each sample_id
    # (benchmarks typically encode subject/clip there).
    per_subject = {}
    for subj in sorted({s.split("/")[0] for s in sids}):
        idx = [i for i, s in enumerate(sids) if s.startswith(subj + "/")]
        if not idx:
            continue
        per_subject[subj] = mpjpe_root_relative(
            pred[idx], gt[idx], mask[idx])

    write_report(Path(args.report), benchmark.name, ckpt_info,
                 mpjpe_mm, pa_mm, pck150, pck100, per_subject,
                 n_samples=len(sids),
                 n_matched_joints=sum(1 for j in benchmark.coco17_to_bench if j >= 0))

    print(f"[summary] MPJPE {mpjpe_mm:.1f}mm  PA-MPJPE {pa_mm:.1f}mm  "
          f"PCK@150 {pck150*100:.1f}%")


if __name__ == "__main__":
    main()

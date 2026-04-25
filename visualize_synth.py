"""Run train_v1/best.pt on one image from OUR synthetic training dataset
and draw predicted vs GT keypoints.

If the model works here (synthetic-domain image, same distribution as training)
but fails on real-world data, the problem is domain-gap / preprocessing —
not the model.  If the model fails here too, it's undertrained or buggy
at its core.

Output: training/runs/train_v1/viz_synth.png
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from training.model import build_model, decode_simcc


SYNTH_DIR = Path("dataset/output/synth_v2")
CKPT = Path("training/runs/train_v1/best.pt")
OUT_PNG = Path("training/runs/train_v1/viz_synth.png")

COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (1, 3), (0, 2), (2, 4),
]


def preprocess_crop(img_rgb, bbox_xywh, input_wh=(192, 256)):
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
    xn = (crop.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(xn.transpose(2, 0, 1)).unsqueeze(0), M


def estimate_root_z(bbox_xywh, K):
    _, _, _, h = bbox_xywh
    fy = float(K[1, 1])
    return fy * 1.7 / max(1.0, float(h))


def backproject_crop_to_cam3d(kps_uv_crop, depth_z, K, bbox_xywh, input_wh):
    W_in, H_in = input_wh
    bx = bbox_xywh[:, 0].unsqueeze(-1)
    by = bbox_xywh[:, 1].unsqueeze(-1)
    bw = bbox_xywh[:, 2].unsqueeze(-1)
    bh = bbox_xywh[:, 3].unsqueeze(-1)
    u_full = bx + kps_uv_crop[..., 0] / W_in * bw
    v_full = by + kps_uv_crop[..., 1] / H_in * bh
    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)
    x = (u_full - cx) / fx * depth_z
    y = (v_full - cy) / fy * depth_z
    return torch.stack([x, y, depth_z], dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Pick a random val-split record from synth_v2.
    records = []
    with (SYNTH_DIR / "labels.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("split") == "val":
                records.append(r)
    print(f"[data] {len(records)} val records in {SYNTH_DIR}")
    rec = random.Random(7).choice(records)
    print(f"[sample] id={rec['id']}")

    img_path = SYNTH_DIR / rec["image_rel"]
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    bbox = tuple(rec["bbox_xywh"])
    K = np.array(rec["camera_K"], dtype=np.float32)
    kps2d_gt = np.array(rec["keypoints_2d"], dtype=np.float32)   # [17, 3] u,v,vis
    print(f"[sample] image={W}x{H} bbox={bbox} bbox_aspect={bbox[2]/bbox[3]:.3f}")

    # Load model.
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    backbone = ckpt.get("args", {}).get("backbone", "mnv4s")
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] backbone={backbone} epoch={ckpt.get('epoch')}")

    input_wh = (192, 256)
    img_t, _M = preprocess_crop(img_rgb, bbox, input_wh)
    img_t = img_t.to(device)
    root_z = torch.tensor([estimate_root_z(bbox, K)], device=device)
    K_t = torch.from_numpy(K).unsqueeze(0).to(device)
    bbox_t = torch.tensor(list(bbox), dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model(img_t)
        # Soft-argmax (default decoder).
        kps2d_soft, _ = decode_simcc(
            out["x_logits"], out["y_logits"], out["z_logits"],
            input_wh=input_wh, root_z=root_z)
        # Argmax decode (no uniform-bias pull toward center).
        split_ratio = 2.0
        bx = out["x_logits"].argmax(dim=-1).float() / split_ratio   # [B, J]
        by = out["y_logits"].argmax(dim=-1).float() / split_ratio
        kps2d_argmax = torch.stack([bx, by], dim=-1)                  # [B, J, 2]

    kps2d_soft = kps2d_soft[0].cpu().numpy()
    kps2d_argmax = kps2d_argmax[0].cpu().numpy()
    # Use argmax as primary prediction; keep soft for comparison in output.
    kps2d_crop = kps2d_argmax
    x0, y0, w0, h0 = bbox
    W_in, H_in = input_wh
    kps2d_pred = np.stack([
        x0 + kps2d_crop[:, 0] * w0 / W_in,
        y0 + kps2d_crop[:, 1] * h0 / H_in,
    ], axis=-1)

    # Upscale the 256x192 synth image so we can see joints clearly.
    SCALE = 4
    vis = cv2.resize(img_rgb, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)

    def draw_kps(img, kps, skel, color_line, color_pt, scale=SCALE):
        for a, b in skel:
            pa, pb = kps[a] * scale, kps[b] * scale
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                     color_line, 3)
        for i, pt in enumerate(kps):
            cv2.circle(img, tuple((pt * scale).astype(int)),
                       6, color_pt, -1)

    # Green: ground truth (from labels, already 2D pixels).
    draw_kps(vis, kps2d_gt[:, :2], COCO17_SKELETON, (0, 255, 0), (0, 200, 0))
    # Red: soft-argmax predicted (current decoder).
    # Unwarp soft too for fair compare.
    kps2d_soft_full = np.stack([
        x0 + kps2d_soft[:, 0] * w0 / W_in,
        y0 + kps2d_soft[:, 1] * h0 / H_in,
    ], axis=-1)
    draw_kps(vis, kps2d_soft_full, COCO17_SKELETON, (255, 0, 0), (255, 50, 50))
    # Yellow: argmax predicted (no uniform-bias pull).
    draw_kps(vis, kps2d_pred, COCO17_SKELETON, (255, 255, 0), (200, 200, 0))

    # Bbox.
    cv2.rectangle(vis,
                  (int(x0 * SCALE), int(y0 * SCALE)),
                  (int((x0 + w0) * SCALE), int((y0 + h0) * SCALE)),
                  (0, 0, 255), 3)

    # Legend.
    txt = [
        f"synth val  id={rec['id'][:40]}  epoch={ckpt.get('epoch','?')}",
        "GREEN  = ground truth",
        "RED    = soft-argmax (current decoder)",
        "YELLOW = argmax (no center-bias)",
        "BLUE   = bbox from label",
    ]
    for i, line in enumerate(txt):
        cv2.putText(vis, line, (20, 40 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (20, 40 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), bgr)
    print(f"[viz] wrote {OUT_PNG}")

    # Sanity: per-joint error (ignores vis for simple reporting).
    errs = np.linalg.norm(kps2d_pred - kps2d_gt[:, :2], axis=-1)
    print(f"[sanity] 2D px err (all 17): mean={errs.mean():.1f} median={np.median(errs):.1f} max={errs.max():.1f}")
    print(f"[sanity] image is {W}x{H}, err as % width: {errs.mean()/W*100:.1f}%")


if __name__ == "__main__":
    main()

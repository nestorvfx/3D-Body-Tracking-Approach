"""Side-by-side comparison: BlazePose (Google MediaPipe) vs our trained model.

Reads a video, runs both models per-frame, composites a side-by-side
output video with each model's keypoint overlay drawn over the original
footage on its half.

Layout:
    ┌─────────────────────────┬─────────────────────────┐
    │    BlazePose (33 kpts)  │  Ours (17 COCO kpts)    │
    │    on original frame    │  on original frame      │
    └─────────────────────────┴─────────────────────────┘

Defaults:
    --video       assets/testvideo.mp4
    --ckpt        training/runs/sota_20260424_2304/best.pt
    --out         assets/testvideo_comparison.mp4
    --mp-model    auto-download pose_landmarker_heavy.task on first run

Usage:
    python compare_blazepose_vs_ours.py
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from training.model import build_model, decode_simcc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# MediaPipe BlazePose 33-landmark connections (subset of standard).  Each
# tuple is (a, b) of indices into the 33-landmark output.
BLAZEPOSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),                     # face left
    (0, 4), (4, 5), (5, 6), (6, 8),                     # face right
    (9, 10),                                              # mouth
    (11, 12),                                             # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),     # left arm + hand
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),     # right arm + hand
    (18, 20),
    (11, 23), (12, 24), (23, 24),                         # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),     # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),     # right leg
]

# COCO-17 skeleton (our model output)
COCO17_CONNECTIONS = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (1, 3), (0, 2), (2, 4),
]

POSE_LANDMARKER_HEAVY_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


# ---------------------------------------------------------------------------
# Inference helpers (mirror training/benchmarks/run.py)
# ---------------------------------------------------------------------------

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


def estimate_root_z(bbox_xywh, K) -> float:
    _, _, _, h = bbox_xywh
    fy = float(K[1, 1])
    return fy * 1.7 / max(1.0, float(h))


def compute_cond(bbox_xywh, K, full_wh) -> torch.Tensor:
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
    A_REAL_M2 = 4.0
    _, _, bw, bh = bbox_xywh
    A_bbox_px2 = max(1.0, float(bw * bh))
    k = float((float(K[0, 0]) * float(K[1, 1]) * A_REAL_M2 / A_bbox_px2) ** 0.5)
    return max(0.3, min(100.0, k))


def default_intrinsics(W, H):
    """Reasonable default K for an unknown phone-camera video.
    Wide-angle phone equivalent, fx=fy≈image_width.  Used only for the
    cond/k_prior conditioning vector — the 2D overlay doesn't depend on K."""
    fx = fy = float(W)
    cx, cy = float(W) / 2.0, float(H) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                     dtype=np.float32)


# ---------------------------------------------------------------------------
# BlazePose
# ---------------------------------------------------------------------------

def ensure_pose_landmarker_model(target: Path) -> Path:
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[blazepose] downloading model to {target} …")
    urllib.request.urlretrieve(POSE_LANDMARKER_HEAVY_URL, target)
    print(f"[blazepose] downloaded ({target.stat().st_size / 1024 / 1024:.1f} MB)")
    return target


def make_pose_landmarker(model_path: Path):
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def blazepose_landmarks_to_pixels(landmarks, W, H):
    """Convert 33 NormalizedLandmark to [33, 3] pixel array (x, y, visibility)."""
    if not landmarks:
        return None
    arr = np.zeros((33, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i, 0] = lm.x * W
        arr[i, 1] = lm.y * H
        arr[i, 2] = getattr(lm, "visibility", 1.0)
    return arr


def bbox_from_blazepose(kps2d_pixels, pad_frac=0.15, W=None, H=None):
    """Tight bbox over visible BlazePose landmarks, padded to be safe.
    Returns (x, y, w, h) in image pixels, clamped to image bounds."""
    if kps2d_pixels is None:
        return None
    visible = kps2d_pixels[kps2d_pixels[:, 2] > 0.3]
    if len(visible) < 5:
        return None
    xs, ys = visible[:, 0], visible[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw = x1 - x0
    bh = y1 - y0
    px = bw * pad_frac
    py = bh * pad_frac
    x0 -= px; y0 -= py; x1 += px; y1 += py
    if W is not None and H is not None:
        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(float(W) - 1, x1); y1 = min(float(H) - 1, y1)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


# ---------------------------------------------------------------------------
# Our model
# ---------------------------------------------------------------------------

def load_our_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt.get("args", {}).get("backbone", "mnv4s")
    model = build_model(backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def run_our_model(model, img_rgb, bbox_xywh, K, device, input_wh=(192, 256)):
    """Returns kps2d_full [17, 2] in image pixels, or None if bbox invalid."""
    if bbox_xywh is None or bbox_xywh[2] < 10 or bbox_xywh[3] < 10:
        return None
    img_t, _M = preprocess_crop(img_rgb, bbox_xywh, input_wh)
    img_t = img_t.to(device)
    full_wh = (img_rgb.shape[1], img_rgb.shape[0])
    cond = compute_cond(bbox_xywh, K, full_wh).unsqueeze(0).to(device)
    k_prior = torch.tensor([compute_k_prior(bbox_xywh, K)], device=device)
    with torch.no_grad():
        out = model(img_t, cond, k_prior=k_prior)
        if "root_z" in out:
            root_z = out["root_z"]
        else:
            root_z = torch.tensor([estimate_root_z(bbox_xywh, K)], device=device)
        kps2d_crop, _ = decode_simcc(
            out["x_logits"], out["y_logits"], out["z_logits"],
            input_wh=input_wh, root_z=root_z, mode="argmax")
    kps2d_crop = kps2d_crop[0].cpu().numpy()  # [17, 2]
    x0, y0, w0, h0 = bbox_xywh
    W_in, H_in = input_wh
    kps2d_full = np.stack([
        x0 + kps2d_crop[:, 0] * w0 / W_in,
        y0 + kps2d_crop[:, 1] * h0 / H_in,
    ], axis=-1)
    return kps2d_full


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton(img, kps2d, connections, joint_color, edge_color,
                   joint_radius=4, edge_thickness=2, vis_threshold=0.0,
                   visibility=None):
    """Draw keypoints + skeleton edges on img in-place."""
    if kps2d is None:
        return img
    for a, b in connections:
        if a >= len(kps2d) or b >= len(kps2d):
            continue
        if visibility is not None:
            if visibility[a] < vis_threshold or visibility[b] < vis_threshold:
                continue
        pa = tuple(int(v) for v in kps2d[a, :2])
        pb = tuple(int(v) for v in kps2d[b, :2])
        cv2.line(img, pa, pb, edge_color, edge_thickness, cv2.LINE_AA)
    for i in range(len(kps2d)):
        if visibility is not None and visibility[i] < vis_threshold:
            continue
        p = tuple(int(v) for v in kps2d[i, :2])
        cv2.circle(img, p, joint_radius, joint_color, -1, cv2.LINE_AA)
    return img


def draw_label(img, text, org=(20, 40), bg_color=(0, 0, 0),
                fg_color=(255, 255, 255), font_scale=0.7, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, bg_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg_color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="assets/testvideo.mp4")
    p.add_argument("--ckpt", default="training/runs/sota_20260424_2304/best.pt")
    p.add_argument("--out", default="assets/testvideo_comparison.mp4")
    p.add_argument("--mp-model",
                    default="assets/pose_landmarker_heavy.task",
                    help="MediaPipe pose_landmarker .task file (auto-downloaded)")
    p.add_argument("--input-w", type=int, default=192)
    p.add_argument("--input-h", type=int, default=256)
    p.add_argument("--max-frames", type=int, default=0,
                    help="0 = process whole video; otherwise stop after N frames")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Open input video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[error] video not found: {video_path}", file=sys.stderr)
        return 1
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[input] {video_path}  {W}x{H}  {fps:.1f}fps  {n_total} frames")

    # Load BlazePose
    mp_model_path = ensure_pose_landmarker_model(Path(args.mp_model))
    landmarker = make_pose_landmarker(mp_model_path)

    # Load our model
    ckpt_path = Path(args.ckpt)
    model, ckpt = load_our_model(ckpt_path, device)
    backbone = ckpt.get("args", {}).get("backbone", "mnv4s")
    print(f"[ours] {ckpt_path}  backbone={backbone}  epoch={ckpt.get('epoch','?')}")

    K = default_intrinsics(W, H)
    input_wh = (args.input_w, args.input_h)

    # Output writer (side-by-side, same height, 2x width)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W * 2, H))
    if not writer.isOpened():
        print(f"[error] cannot open writer for {out_path}", file=sys.stderr)
        return 2
    print(f"[output] {out_path}  {W*2}x{H}  {fps:.1f}fps")

    frame_idx = 0
    n_processed = 0
    n_blazepose_hits = 0
    n_ours_hits = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(round(frame_idx * 1000.0 / fps))

        # --- BlazePose ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            bp_kps = blazepose_landmarks_to_pixels(
                result.pose_landmarks[0], W, H)
            n_blazepose_hits += 1
        else:
            bp_kps = None

        bbox = bbox_from_blazepose(bp_kps, pad_frac=0.15, W=W, H=H)

        # --- Our model ---
        our_kps = run_our_model(model, frame_rgb, bbox, K, device, input_wh)
        if our_kps is not None:
            n_ours_hits += 1

        # --- Render LEFT: BlazePose ---
        left = frame_bgr.copy()
        if bp_kps is not None:
            draw_skeleton(left, bp_kps[:, :2], BLAZEPOSE_CONNECTIONS,
                           joint_color=(50, 50, 255),    # red joints (BGR)
                           edge_color=(0, 220, 0),         # green edges
                           visibility=bp_kps[:, 2],
                           vis_threshold=0.3)
        draw_label(left, "BlazePose (Google MediaPipe, 33 landmarks)")
        if bp_kps is None:
            draw_label(left, "no pose detected", (20, H - 20),
                        fg_color=(0, 255, 255))

        # --- Render RIGHT: Ours ---
        right = frame_bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(right, (x, y), (x + w, y + h), (255, 80, 0), 2)
        if our_kps is not None:
            draw_skeleton(right, our_kps, COCO17_CONNECTIONS,
                           joint_color=(50, 50, 255),
                           edge_color=(255, 80, 0))     # blue edges (BGR)
        draw_label(right, f"Ours ({backbone}+SimCC3D, 17 COCO kpts)")
        if our_kps is None:
            draw_label(right, "no pose (no bbox)", (20, H - 20),
                        fg_color=(0, 255, 255))

        # Frame counter footer (across both halves)
        canvas = np.hstack([left, right])
        draw_label(canvas, f"frame {frame_idx+1}/{n_total}",
                    (20, H - 20), font_scale=0.6, thickness=1)
        writer.write(canvas)

        frame_idx += 1
        n_processed += 1
        if frame_idx % 30 == 0 or frame_idx == 1:
            pct = 100.0 * frame_idx / max(1, n_total)
            print(f"  frame {frame_idx}/{n_total} ({pct:.1f}%)  "
                  f"blazepose_hits={n_blazepose_hits}  ours_hits={n_ours_hits}",
                  flush=True)

    cap.release()
    writer.release()
    landmarker.close()

    print(f"[done] processed {n_processed} frames  "
          f"blazepose_hits={n_blazepose_hits}  ours_hits={n_ours_hits}")
    print(f"[done] wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

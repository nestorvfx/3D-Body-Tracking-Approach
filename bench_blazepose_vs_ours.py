"""Bench inference time + parameter count: BlazePose vs our model.

Reports:
  - Param count for our model (PyTorch state-dict)
  - Param count for BlazePose Heavy (extracted from .task → .tflite, summed)
  - Median ms/frame across many trials, with warmup, for:
      * BlazePose Heavy via MediaPipe Python (CPU TFLite, XNNPACK)
      * Our model on CPU (mobile-equivalent path)
      * Our model on GPU (if CUDA available)

Usage:
    conda activate bodytrack
    python bench_blazepose_vs_ours.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from training.model import build_model


# ---------------------------------------------------------------------------
# Param counting
# ---------------------------------------------------------------------------

def count_pytorch_params(state_dict) -> int:
    return sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))


def count_tflite_params(tflite_path: Path) -> int:
    """Count weight + bias tensor params in a .tflite model."""
    try:
        import tensorflow as tf
    except ImportError:
        try:
            from tflite_runtime import interpreter as tflite_interp
            interp = tflite_interp.Interpreter(model_path=str(tflite_path))
        except Exception:
            return -1
    else:
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    n_params = 0
    for d in interp.get_tensor_details():
        if d.get("dtype") in (np.float32, np.float16, np.int8, np.uint8):
            shape = d.get("shape", [])
            if len(shape) >= 1 and shape[0] > 0 and len(shape) > 1:
                # Heuristic: only count tensors that look like weights
                # (multi-dim, name suggests a weight var).
                name = d.get("name", "").lower()
                if any(k in name for k in ("kernel", "weight", "depthwise",
                                              "conv", "matmul", "bias",
                                              "filter", "embedding")):
                    n_params += int(np.prod(shape))
    return n_params


def extract_tflite_from_task(task_path: Path) -> list[Path]:
    """Extract any .tflite files inside the .task zip and return their paths."""
    out_dir = Path(tempfile.gettempdir()) / "blazepose_extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    try:
        with zipfile.ZipFile(task_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".tflite"):
                    target = out_dir / Path(name).name
                    target.write_bytes(zf.read(name))
                    extracted.append(target)
    except zipfile.BadZipFile:
        # Older .task formats may not be zip; skip param counting in that case
        pass
    return extracted


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_pytorch(model, img_t, cond, k_prior, device, n_warm=20, n_run=200):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warm):
            _ = model(img_t, cond, k_prior=k_prior)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            _ = model(img_t, cond, k_prior=k_prior)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def time_blazepose(landmarker, frame_rgb, fps_for_video_mode=30.0,
                    n_warm=20, n_run=200):
    """detect_for_video requires monotonically-increasing timestamps."""
    H, W = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    # warmup
    t_ms = 0
    dt = int(round(1000.0 / fps_for_video_mode))
    for _ in range(n_warm):
        _ = landmarker.detect_for_video(mp_image, t_ms)
        t_ms += dt
    # timed runs
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        _ = landmarker.detect_for_video(mp_image, t_ms)
        times.append((time.perf_counter() - t0) * 1000.0)
        t_ms += dt
    return times


def stats_ms(times):
    if not times:
        return {"median": 0.0, "mean": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(times)
    n = len(s)
    return {
        "median": statistics.median(s),
        "mean": statistics.mean(s),
        "p95": s[int(0.95 * (n - 1))],
        "min": s[0],
        "max": s[-1],
        "fps": 1000.0 / statistics.median(s),
    }


def fmt(stats):
    return (f"{stats['median']:6.2f} ms median  ({stats['fps']:5.1f} fps)   "
            f"mean {stats['mean']:6.2f}   p95 {stats['p95']:6.2f}   "
            f"range [{stats['min']:.2f}, {stats['max']:.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="training/runs/sota_20260424_2304/best.pt")
    p.add_argument("--mp-model",
                    default="assets/pose_landmarker_heavy.task")
    p.add_argument("--frame-from",
                    default="assets/testvideo.mp4",
                    help="Sample one frame from this video to use as bench input")
    p.add_argument("--frame-idx", type=int, default=200,
                    help="Frame index to use as bench input")
    p.add_argument("--n-warm", type=int, default=20)
    p.add_argument("--n-run",  type=int, default=200)
    p.add_argument("--input-w", type=int, default=192)
    p.add_argument("--input-h", type=int, default=256)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.frame_from)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        print(f"[error] could not read frame {args.frame_idx} from {args.frame_from}",
              file=sys.stderr)
        return 1
    H, W = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"[bench] sampled frame {args.frame_idx} from {args.frame_from}  ({W}x{H})")
    print(f"[bench] warmup={args.n_warm}, timed runs={args.n_run}")
    print()

    # ---------- Param counts ----------
    print("=" * 70)
    print("PARAMETER COUNTS")
    print("=" * 70)

    # Our model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_ours = count_pytorch_params(ckpt["model"])
    backbone = ckpt.get("args", {}).get("backbone", "mnv4s")
    print(f"  Ours ({backbone}+SimCC3D)     : {n_ours:>12,d} params  "
          f"({n_ours/1e6:5.2f} M)")

    # BlazePose: extract .tflite files, count params per file
    tflite_files = extract_tflite_from_task(Path(args.mp_model))
    if tflite_files:
        n_bp_total = 0
        for tf in tflite_files:
            n = count_tflite_params(tf)
            label = tf.stem
            if n > 0:
                print(f"  BlazePose: {label:<30s}: {n:>12,d} params  "
                      f"({n/1e6:5.2f} M)")
                n_bp_total += n
            else:
                print(f"  BlazePose: {label:<30s}: (could not introspect)")
        if n_bp_total > 0:
            print(f"  BlazePose: TOTAL                  : {n_bp_total:>12,d} "
                  f"params  ({n_bp_total/1e6:5.2f} M)")
    else:
        print("  BlazePose: could not extract .tflite from .task")
    # Reference values from MediaPipe / BlazePose paper:
    print("  (Reference: BlazePose Heavy ≈ 6.6M params per MediaPipe model card)")
    print()

    # ---------- Inference timing ----------
    print("=" * 70)
    print("INFERENCE TIME (ms/frame)")
    print("=" * 70)

    # 1. BlazePose Heavy on CPU (TFLite XNNPACK — default on Windows w/o GPU delegate)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.mp_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    bp_times = time_blazepose(landmarker, frame_rgb,
                                n_warm=args.n_warm, n_run=args.n_run)
    landmarker.close()
    print(f"  BlazePose Heavy  (CPU/TFLite/XNNPACK)  "
          f"{fmt(stats_ms(bp_times))}")

    # 2. Our model — preprocess once
    bbox = (W // 4, H // 4, W // 2, H // 2)  # rough centred bbox
    input_wh = (args.input_w, args.input_h)
    src = np.array([[bbox[0], bbox[1]],
                     [bbox[0] + bbox[2], bbox[1]],
                     [bbox[0], bbox[1] + bbox[3]]], dtype=np.float32)
    dst = np.array([[0, 0], [input_wh[0], 0], [0, input_wh[1]]],
                    dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(frame_rgb, M, input_wh,
                           flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = ((crop.astype(np.float32) / 255.0) - mean) / std
    img_np = img_np.transpose(2, 0, 1)[None]

    diag = float((W * W + H * H) ** 0.5)
    cond_np = np.array([
        (bbox[0] + bbox[2] / 2.0) / W,
        (bbox[1] + bbox[3] / 2.0) / H,
        bbox[2] / W, bbox[3] / H,
        float(W) / diag, float(W) / diag,
    ], dtype=np.float32)[None]
    k_prior_np = np.array([1.0], dtype=np.float32)

    # CPU
    device_cpu = torch.device("cpu")
    model_cpu = build_model(backbone, pretrained=False).to(device_cpu)
    model_cpu.load_state_dict(ckpt["model"])
    img_t_cpu = torch.from_numpy(img_np).to(device_cpu)
    cond_t_cpu = torch.from_numpy(cond_np).to(device_cpu)
    kpr_t_cpu = torch.from_numpy(k_prior_np).to(device_cpu)
    cpu_times = time_pytorch(model_cpu, img_t_cpu, cond_t_cpu, kpr_t_cpu,
                               device_cpu,
                               n_warm=args.n_warm, n_run=args.n_run)
    print(f"  Ours ({backbone})   (CPU/PyTorch)            "
          f"{fmt(stats_ms(cpu_times))}")
    del model_cpu

    # GPU
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        model_gpu = build_model(backbone, pretrained=False).to(device_gpu)
        model_gpu.load_state_dict(ckpt["model"])
        img_t_gpu = torch.from_numpy(img_np).to(device_gpu)
        cond_t_gpu = torch.from_numpy(cond_np).to(device_gpu)
        kpr_t_gpu = torch.from_numpy(k_prior_np).to(device_gpu)
        gpu_times = time_pytorch(model_gpu, img_t_gpu, cond_t_gpu, kpr_t_gpu,
                                   device_gpu,
                                   n_warm=args.n_warm, n_run=args.n_run)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Ours ({backbone})   (GPU/CUDA, {gpu_name}) "
              f"{fmt(stats_ms(gpu_times))}")
    else:
        print("  Ours (GPU)        : (CUDA not available — skipping)")

    print()
    print("=" * 70)
    print("NOTES")
    print("=" * 70)
    print("- BlazePose runs full-frame; our model runs on a single bbox crop.")
    print("- Our timing excludes person detection (in real deployment, you would")
    print("  add ~5-15 ms for a YOLO/RetinaFace detector or use prev-frame bbox).")
    print("- BlazePose includes its own internal detector + landmarker + smoother.")
    print("- MediaPipe defaults to CPU even with CUDA available on Python; the")
    print("  mobile-GPU delegate is not exposed in mediapipe-python.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

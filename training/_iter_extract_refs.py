"""One-off: extract real-image FDA + BG references from assets/testvideo.mp4.

Outputs:
  assets/sim2real_refs/fda/  — frames at synth resolution (256x192, RGB) for FDA
  assets/sim2real_refs/bg/   — full-res frames for background compositing

Strategy: stride-sample frames across the video so we cover lighting/scene
variation, not 200 near-duplicates of the same shot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(r"c:/Users/Mihajlo/Documents/Body Tracking")
VIDEO = ROOT / "assets" / "testvideo.mp4"
FDA_OUT = ROOT / "assets" / "sim2real_refs" / "fda"
BG_OUT = ROOT / "assets" / "sim2real_refs" / "bg"
N_FRAMES = 240
SYNTH_WH = (256, 192)   # (W, H)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    FDA_OUT.mkdir(parents=True, exist_ok=True)
    BG_OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"[error] cannot open {VIDEO}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {total} frames @ {fps:.1f} fps")

    if total <= 0:
        print("[error] zero-frame video")
        return 2

    stride = max(1, total // N_FRAMES)
    n_written = 0
    for i in range(0, total, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        # FDA: resize to synth crop size; FDA needs source/target same shape.
        fda_img = cv2.resize(frame_bgr, SYNTH_WH, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(FDA_OUT / f"frame_{n_written:04d}.png"), fda_img)
        # BG: keep at native res; we'll random-crop to synth crop at training time.
        # But to keep storage moderate, downscale long edge to 720.
        H, W = frame_bgr.shape[:2]
        if max(H, W) > 720:
            f = 720.0 / max(H, W)
            frame_bgr = cv2.resize(
                frame_bgr, (int(W * f), int(H * f)),
                interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(BG_OUT / f"frame_{n_written:04d}.jpg"),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_written += 1
        if n_written >= N_FRAMES:
            break

    cap.release()
    print(f"wrote {n_written} fda + {n_written} bg refs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

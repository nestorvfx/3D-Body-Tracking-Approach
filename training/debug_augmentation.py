"""Visualise each augmentation stage on N samples with 2D keypoints overlaid.

Produces out_dir/<sample_id>_<stage>.png for stages:
  orig, after_bbox, after_bg_composite, after_rotate, after_flip,
  after_occluder, after_photo

Run:
    python -m training.debug_augmentation --dataset-dir dataset/output/synth_iter \
           --out-dir dataset/output/aug_debug_v2 --n 12 \
           --occluder-dir assets/voc_occluders \
           --bg-dir assets/sim2real_refs/bg \
           --matte-dir dataset/output/synth_iter/mattes \
           --fda-dir assets/sim2real_refs/fda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from training.data import DataConfig, SynthPoseDataset   # noqa: E402


COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (1, 3), (0, 2), (2, 4),
]


def draw_kps(img_rgb: np.ndarray, kps2d: np.ndarray,
             vis: np.ndarray | None = None) -> Image.Image:
    if kps2d.ndim == 2 and kps2d.shape[1] >= 2:
        pts = kps2d[:, :2]
    else:
        pts = kps2d
    img = Image.fromarray(img_rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    for a, b in COCO17_SKELETON:
        if vis is not None and (vis[a] <= 0 or vis[b] <= 0):
            continue
        draw.line([(pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1])],
                  fill=(80, 200, 255), width=2)
    for i in range(pts.shape[0]):
        if vis is not None and vis[i] <= 0:
            continue
        u, v = float(pts[i, 0]), float(pts[i, 1])
        r = 3
        draw.ellipse([(u - r, v - r), (u + r, v + r)], fill=(255, 80, 80),
                     outline=(255, 255, 255))
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--occluder-dir", default="")
    p.add_argument("--bg-dir", default="")
    p.add_argument("--matte-dir", default="")
    p.add_argument("--fda-dir", default="")
    p.add_argument("--p-occluder", type=float, default=1.0,
                   help="For viz, default 1.0 so we see the effect on every sample")
    p.add_argument("--p-bg-composite", type=float, default=1.0)
    p.add_argument("--p-fda", type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataConfig(
        dataset_dir=args.dataset_dir, split="train", training=True,
        occluder_dir=args.occluder_dir,
        bg_corpus_dir=args.bg_dir,
        matte_dir=args.matte_dir,
        fda_refs_dir=args.fda_dir,
        p_occluder=args.p_occluder,
        p_bg_composite=args.p_bg_composite,
        p_fda=args.p_fda,
    )
    ds = SynthPoseDataset(cfg)
    print(f"[aug-debug] loaded {len(ds)} samples, rendering {args.n}")
    print(f"[aug-debug] occluders={len(ds.occluders)}, "
          f"bg_corpus={len(ds.bg_corpus)}, fda_refs={len(ds.fda_refs)}")

    # Read raw record fields directly from the parallel arrays (no
    # SynthPoseDataset.records anymore — we use ds.arrs).
    n = min(args.n, len(ds))
    for i in range(n):
        sid = ds.arrs["id"][i].decode("utf-8")
        img_rel = ds.arrs["image_rel"][i].decode("utf-8")
        img_bgr = cv2.imread(str(ds.root / img_rel))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kps_2d_full = ds.arrs["kps2d"][i].copy()         # [17, 3] (u,v,vis)
        kps_3d = ds.arrs["kps3d"][i].copy()
        bbox = tuple(ds.arrs["bbox"][i].tolist())
        rng = random.Random(i * 1_000_003)

        _, _, _, _, _, stages = ds._transform(
            img, kps_2d_full, kps_3d, bbox, rng,
            training=True, return_stages=True, sample_id=sid)

        # Composite a horizontal strip showing each stage.
        panels: list[Image.Image] = []
        labels: list[str] = []
        for stage_name, (img_s, kps_s, vis_s) in stages.items():
            pil = draw_kps(img_s, kps_s, vis_s)
            panels.append(pil)
            labels.append(stage_name)

        if panels:
            H = max(p.height for p in panels)
            strip = Image.new(
                "RGB",
                (sum(p.width + 16 for p in panels) + 16, H + 28),
                (20, 22, 26))
            draw = ImageDraw.Draw(strip)
            x = 8
            for pil, lbl in zip(panels, labels):
                strip.paste(pil, (x, 28))
                draw.text((x + 4, 6), lbl, fill=(220, 220, 220))
                x += pil.width + 16
            strip.save(out_dir / f"{sid}_aug.png")
            print(f"  {sid}: {labels}")

    print(f"[aug-debug] wrote {n} strips to {out_dir}")


if __name__ == "__main__":
    main()

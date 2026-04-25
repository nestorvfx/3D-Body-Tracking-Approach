"""Overlay surface keypoints + COCO-17 on a few random 100-sample renders."""
from __future__ import annotations
import json, random
from pathlib import Path
import cv2
import numpy as np

import os
ROOT = Path(os.path.expandvars("$LOCALAPPDATA/Temp/synth100/shard_000"))
OUT = Path("training/runs/500k_prep_qa.png")

with (ROOT / "labels.jsonl").open() as f:
    records = [json.loads(l) for l in f]

random.seed(1)
picks = random.sample(records, 4)

tiles = []
for r in picks:
    img = cv2.imread(str(ROOT / r["image_rel"]))
    if img is None:
        continue
    # COCO-17 in blue
    for kp in r["keypoints_2d"]:
        u, v, vis = kp
        if vis > 0:
            cv2.circle(img, (int(u), int(v)), 2, (255, 100, 0), -1)
    # Surface kps in green
    for kp in r["surface_kps_2d"]:
        u, v, vis = kp
        if vis == 2:
            cv2.circle(img, (int(u), int(v)), 1, (0, 255, 0), -1)
    # bbox
    x, y, w, h = r["bbox_xywh"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    tag = f"f={r['focal_mm']:.0f}mm  z={r['root_joint_cam'][2]:.1f}m"
    cv2.putText(img, tag, (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(img)

# 2x2 grid
row1 = np.hstack(tiles[:2])
row2 = np.hstack(tiles[2:])
grid = np.vstack([row1, row2])
OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), grid)
print(f"wrote {OUT}  shape={grid.shape}")

"""
Build a NON-DESTRUCTIVE 'training view' over /workspace/synth_v3/shard_*/.

- Reads all shard labels.jsonl (append-only safe)
- Applies the same outlier filter as pack_and_upload.py
- Rewrites each row's image_rel from 'images/{id}.png' to
  '../shard_NNN/images/{id}.png' so a single `dataset_dir` can resolve
  images across all shards without copying/symlinking
- Writes merged labels.jsonl to <out>/labels.jsonl
- Source data is NEVER touched — pause/resume of generation is safe

Usage:
    python3 cherry_dataset_gen/scripts/prepare_training_view.py \
        --data /workspace/synth_v3 \
        --out  /workspace/synth_v3_train
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def is_outlier(row: dict) -> tuple[bool, str]:
    rz = row.get("root_joint_cam", [0, 0, 0])[2]
    if not (0.3 <= rz <= 50.0):
        return True, "root_z"

    f = row.get("focal_mm", 35.0)
    if not (10.0 <= f <= 500.0):
        return True, "focal"

    bb = row.get("bbox_xywh", [0, 0, 0, 0])
    if bb[2] < 10 or bb[3] < 10:
        return True, "bbox"

    for field in (
        "keypoints_2d", "keypoints_3d_cam",
        "surface_kps_2d", "surface_kps_3d_cam",
        "camera_K", "bbox_xywh", "root_joint_cam",
    ):
        v = row.get(field)
        if v is None:
            continue
        try:
            arr = np.asarray(v, dtype=float)
        except Exception:
            return True, f"{field}_parse"
        if arr.size and not np.isfinite(arr).all():
            return True, f"{field}_nan"

    kps2d = np.asarray(row.get("keypoints_2d", []), dtype=float)
    if kps2d.size and (kps2d[:, 2] >= 2).sum() < 10:
        return True, "visible<10"

    skp3d = np.asarray(row.get("surface_kps_3d_cam", []), dtype=float)
    if skp3d.size:
        z = skp3d[:, 2]
        if z.max() > 60.0 or z.min() < -10.0:
            return True, "surface_z"

    return False, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/workspace/synth_v3")
    p.add_argument("--out",  default="/workspace/synth_v3_train")
    args = p.parse_args()

    src = Path(args.data)
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    out_labels = dst / "labels.jsonl"

    kept = 0
    reasons: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    shard_dirs = sorted(src.glob("shard_*"))
    if not shard_dirs:
        print(f"no shard_* dirs under {src}")
        return 1

    with out_labels.open("w", encoding="utf-8") as fo:
        for shard in shard_dirs:
            labels = shard / "labels.jsonl"
            images = shard / "images"
            if not labels.exists() or not images.exists():
                continue
            # Relative path from dst dir to this shard's images dir.
            # os.path.relpath handles arbitrary src/dst layouts (sibling,
            # nested, disjoint) correctly — '..' hop count depends on both.
            rel_prefix = os.path.relpath(images, dst)
            with labels.open(encoding="utf-8") as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        reasons["parse_err"] += 1
                        continue
                    sid = r.get("id")
                    if not sid or sid in seen_ids:
                        reasons["dup_or_no_id"] += 1
                        continue
                    img_rel = r.get("image_rel", "")
                    img_name = Path(img_rel).name if img_rel else f"{sid}.png"
                    img_abs = images / img_name
                    try:
                        if img_abs.stat().st_size <= 0:
                            reasons["img_empty"] += 1
                            continue
                    except FileNotFoundError:
                        reasons["img_missing"] += 1
                        continue
                    bad, why = is_outlier(r)
                    if bad:
                        reasons[why] += 1
                        continue
                    r["image_rel"] = f"{rel_prefix}/{img_name}".replace(os.sep, "/")
                    r.setdefault("schema_version", "1.0")
                    fo.write(json.dumps(r, separators=(",", ":")) + "\n")
                    seen_ids.add(sid)
                    split_counts[r.get("split", "train")] += 1
                    kept += 1

    print(f"kept    : {kept:,}")
    print(f"splits  : {dict(split_counts)}")
    print(f"rejects : {dict(reasons)}")
    print(f"wrote   : {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

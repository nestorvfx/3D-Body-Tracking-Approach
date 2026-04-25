"""Merge all shard_NNN/ subdirectories into a single unified dataset.

After run.sh finishes, the output looks like:
  <out>/shard_000/{images, labels.jsonl, manifest.csv}
  <out>/shard_001/{...}
  ...

This script produces:
  <out>/images/            — hardlinked from each shard (no copy, no extra disk)
  <out>/labels.jsonl       — deduped concat of all shard labels
  <out>/manifest.csv       — concat of all shard manifests (one header)
  <out>/dataset_stats.md   — summary

Hardlinking means the merge step is ~1 sec for 200k images instead of
minutes of copy time.  The shard_NNN/ dirs can be deleted after merge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="the base output dir used by run.sh")
    p.add_argument("--keep-shards", action="store_true",
                   help="keep the shard_NNN/ subdirs after merge (default: delete)")
    args = p.parse_args()

    base = Path(args.out).resolve()
    if not base.exists():
        print(f"[merge] fatal: {base} does not exist", file=sys.stderr); sys.exit(1)

    shards = sorted(d for d in base.glob("shard_*") if d.is_dir())
    if not shards:
        print(f"[merge] fatal: no shard_*/ dirs under {base}", file=sys.stderr); sys.exit(1)
    print(f"[merge] {len(shards)} shards to combine")

    img_out = base / "images"
    img_out.mkdir(exist_ok=True)
    labels_out = base / "labels.jsonl"
    manifest_out = base / "manifest.csv"

    seen: set[str] = set()
    stats = Counter()
    n_labels = 0; n_images = 0

    # ----- labels + manifest -----
    with labels_out.open("w", encoding="utf-8") as lh, \
         manifest_out.open("w", encoding="utf-8", newline="") as mh:
        mh.write("id,split,source,image_rel,bbox_x,bbox_y,bbox_w,bbox_h\n")
        for shard in shards:
            sp = shard / "labels.jsonl"
            if not sp.exists(): continue
            with sp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try: rec = json.loads(line)
                    except Exception: continue
                    sid = rec["id"]
                    if sid in seen: continue
                    seen.add(sid)
                    # Rewrite image_rel to the unified images/ dir.
                    rec["image_rel"] = f"images/{sid}.png"
                    lh.write(json.dumps(rec) + "\n")
                    n_labels += 1
                    stats[rec["source"]] += 1
                    stats[rec["split"]] += 1
            # Manifest: skip header, concat body.
            mpath = shard / "manifest.csv"
            if mpath.exists():
                with mpath.open("r", encoding="utf-8") as fh:
                    next(fh, None)
                    for line in fh:
                        # Rewrite image_rel column too (4th field).
                        parts = line.rstrip("\n").split(",")
                        if len(parts) >= 4 and parts[0] in seen:
                            parts[3] = f"images/{parts[0]}.png"
                            mh.write(",".join(parts) + "\n")

    # ----- images (hardlink to avoid copy) -----
    for shard in shards:
        src_imgs = shard / "images"
        if not src_imgs.is_dir(): continue
        for img in src_imgs.iterdir():
            dst = img_out / img.name
            if dst.exists(): continue
            try:
                os.link(img, dst)
            except OSError:
                # Fallback to copy if hardlink unsupported (e.g. crossed FS).
                import shutil; shutil.copy2(img, dst)
            n_images += 1

    # ----- stats doc -----
    with (base / "dataset_stats.md").open("w", encoding="utf-8") as fh:
        fh.write("# Dataset stats\n\n")
        fh.write(f"Total samples: {n_labels}\n")
        fh.write(f"Total images:  {n_images}\n\n")
        fh.write("Per source:\n")
        for src in ("cmu", "100style", "aistpp"):
            fh.write(f"- {src}: {stats.get(src, 0)}\n")
        fh.write("\nSplit:\n")
        fh.write(f"- train: {stats.get('train', 0)}\n")
        fh.write(f"- val:   {stats.get('val', 0)}\n")

    print(f"[merge] labels:   {n_labels}")
    print(f"[merge] images:   {n_images}")
    print(f"[merge] per source: {dict(stats)}")

    if not args.keep_shards:
        import shutil
        for shard in shards:
            shutil.rmtree(shard)
        print(f"[merge] removed {len(shards)} shard dirs")


if __name__ == "__main__":
    main()

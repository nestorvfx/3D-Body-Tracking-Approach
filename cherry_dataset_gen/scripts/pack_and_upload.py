"""
Batch pack + upload of synthetic pose samples to Hugging Face.

Runs alongside build_dataset.py without interfering:
- reads labels.jsonl append-only (no lock)
- only ingests samples whose PNG is fully written (stat + non-zero size)
- applies outlier filter (NaN/Inf, bad root_z, bad focal, empty bbox, too
  few visible COCO kps, depth-out-of-range surface kps)
- packs into ZIP_STORED (PNG is already DEFLATE'd internally — no
  re-compression, no quality loss)
- uploads to HF, deletes local zip, updates state, continues

Run once to process the next batch, or with --watch to loop until target.
State is kept in <release>/_state.json so the script is safe to re-run
after crashes.

Usage:
    # one-time auth on the box
    huggingface-cli login

    # one-shot: pack + upload the next available 50k
    python3 cherry_dataset_gen/scripts/pack_and_upload.py

    # watch mode: loop until 500k uploaded, polling every 5 min
    python3 cherry_dataset_gen/scripts/pack_and_upload.py --watch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, get_token


DEFAULTS = {
    "data":       "/workspace/synth_v3",
    "release":    "/workspace/synth_v3_release",
    "repo":       "nestorvfx/3DBodyTrackingDatabase",
    "batch_size": 50_000,
    "target":     500_000,
}


def gather_completed_rows(data_root: Path) -> list[dict]:
    """Read labels.jsonl from every shard, keep rows with present PNGs."""
    rows: list[dict] = []
    for shard_dir in sorted(data_root.glob("shard_*")):
        labels = shard_dir / "labels.jsonl"
        images_dir = shard_dir / "images"
        if not labels.exists() or not images_dir.exists():
            continue
        with labels.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img_rel = r.get("image_rel")
                if not img_rel:
                    continue
                img_abs = images_dir / Path(img_rel).name
                try:
                    st = img_abs.stat()
                except FileNotFoundError:
                    continue
                if st.st_size <= 0:
                    continue
                r["_img_abs"] = str(img_abs)
                rows.append(r)
    rows.sort(key=lambda r: r["id"])
    return rows


def is_outlier(row: dict) -> tuple[bool, str]:
    """Return (reject?, reason)."""
    rz = row.get("root_joint_cam", [0, 0, 0])[2]
    if not (0.3 <= rz <= 50.0):
        return True, f"root_z={rz:.2f}"

    f = row.get("focal_mm", 35.0)
    if not (10.0 <= f <= 500.0):
        return True, f"focal={f:.1f}mm"

    bb = row.get("bbox_xywh", [0, 0, 0, 0])
    if bb[2] < 10 or bb[3] < 10:
        return True, f"bbox={bb}"

    # NaN/Inf sweep across all numeric-array fields.
    # Schema uses "camera_K" (not "K") — nested 3x3 list.
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
            return True, f"{field} unparseable"
        if arr.size and not np.isfinite(arr).all():
            return True, f"{field} NaN/Inf"

    # Visibility sanity on COCO-17.  Convention from build_dataset.py:
    #   0 = unprojectable, 1 = projected but outside frame, 2 = inside frame.
    # The render-time gate already enforces n_inside >= 10; we re-check for
    # defense-in-depth.
    kps2d = np.asarray(row.get("keypoints_2d", []), dtype=float)
    if kps2d.size and (kps2d[:, 2] >= 2).sum() < 10:
        return True, "visible_coco<10"

    # Surface-kps 3D depth sanity (catches compositor depth blow-ups).
    skp3d = np.asarray(row.get("surface_kps_3d_cam", []), dtype=float)
    if skp3d.size:
        z = skp3d[:, 2]
        if z.max() > 60.0 or z.min() < -10.0:
            return True, f"surface_z_range=[{z.min():.1f},{z.max():.1f}]"

    return False, ""


def pack_zip(rows: list[dict], zip_path: Path) -> None:
    """Write rows into a STORE-mode zip (no recompression of PNG bytes)."""
    tmp = zip_path.with_suffix(".zip.partial")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED,
                          allowZip64=True) as zf:
        for r in rows:
            img_abs = r["_img_abs"]
            sid = r["id"]
            clean = {k: v for k, v in r.items()
                     if k not in ("_img_abs", "image_rel")}
            clean.setdefault("schema_version", "1.0")
            zf.write(img_abs, arcname=f"{sid}.png")
            zf.writestr(f"{sid}.json", json.dumps(clean,
                                                    separators=(",", ":")))
    tmp.rename(zip_path)


def upload(api: HfApi, zip_path: Path, repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=f"data/{zip_path.name}",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add {zip_path.name}",
    )


def human(n: int) -> str:
    return f"{n:,}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default=DEFAULTS["data"])
    p.add_argument("--release",    default=DEFAULTS["release"])
    p.add_argument("--repo",       default=DEFAULTS["repo"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--target",     type=int, default=DEFAULTS["target"])
    p.add_argument("--watch",      action="store_true",
                    help="Poll every 5 min until target is reached.")
    p.add_argument("--poll-sec",   type=int, default=300)
    p.add_argument("--keep-zip",   action="store_true",
                    help="Don't delete zip after upload (debug).")
    args = p.parse_args()

    data = Path(args.data)
    release = Path(args.release)
    release.mkdir(parents=True, exist_ok=True)
    state_path = release / "_state.json"
    state = (json.loads(state_path.read_text())
              if state_path.exists()
              else {"packed_cursor": 0, "batches_uploaded": []})

    token = get_token()
    if not token:
        print("[packer] no HF token found — run `hf auth login` first.",
              file=sys.stderr)
        return 2

    api = HfApi(token=token)

    while True:
        rows_all = gather_completed_rows(data)

        # Apply outlier filter once, log reasons.
        rows_clean: list[dict] = []
        reasons: dict[str, int] = {}
        for r in rows_all:
            bad, why = is_outlier(r)
            if bad:
                reasons[why.split("=")[0] if "=" in why else why] = \
                    reasons.get(why.split("=")[0] if "=" in why else why, 0) + 1
                continue
            rows_clean.append(r)

        total_clean = len(rows_clean)
        cursor = state["packed_cursor"]
        next_end = cursor + args.batch_size

        # Final partial batch when target is reached.
        near_target = total_clean >= args.target
        can_pack_full = next_end <= total_clean
        if not can_pack_full:
            if near_target and cursor < total_clean:
                next_end = total_clean
            else:
                msg = (f"[packer] clean={human(total_clean)}  "
                       f"cursor={human(cursor)}  "
                       f"need ≥ {human(next_end)} for next batch  "
                       f"(rejected: {dict(reasons)})")
                if args.watch and cursor < args.target:
                    print(msg + f"  waiting {args.poll_sec}s…")
                    time.sleep(args.poll_sec)
                    continue
                print(msg)
                return 0

        batch = rows_clean[cursor:next_end]
        batch_no = len(state["batches_uploaded"]) + 1
        zip_path = release / f"batch_{batch_no:02d}.zip"

        print(f"[packer] batch {batch_no:02d}: packing samples "
              f"[{human(cursor)}..{human(next_end)})  ({len(batch):,} rows)")
        t0 = time.time()
        pack_zip(batch, zip_path)
        t_pack = time.time() - t0
        size_mb = zip_path.stat().st_size / 1024 / 1024
        print(f"[packer] packed {zip_path.name}: {size_mb:,.0f} MB in "
              f"{t_pack:.1f}s  → uploading to {args.repo}")

        t0 = time.time()
        upload(api, zip_path, args.repo)
        t_up = time.time() - t0
        print(f"[packer] uploaded in {t_up:.1f}s  "
              f"({size_mb / t_up:,.1f} MB/s)")

        state["packed_cursor"] = next_end
        state["batches_uploaded"].append(batch_no)
        state["batches_uploaded_sorted"] = sorted(state["batches_uploaded"])
        state["last_reject_reasons"] = reasons
        state_path.write_text(json.dumps(state, indent=2))

        if not args.keep_zip:
            zip_path.unlink(missing_ok=True)

        if next_end >= args.target:
            print("[packer] target reached. All batches uploaded.")
            return 0
        if not args.watch:
            print("[packer] batch done. Re-run to process the next one "
                  "(or pass --watch to loop).")
            return 0


if __name__ == "__main__":
    sys.exit(main())

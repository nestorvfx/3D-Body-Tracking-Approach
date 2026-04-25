"""Download CC0 HDRIs from Poly Haven to expand the dataset's lighting pool.

Poly Haven hosts 750+ CC0 HDRIs (https://polyhaven.com/hdris); all are
free for any use including commercial.  Public API:
  https://api.polyhaven.com/assets?t=hdris
  https://api.polyhaven.com/files/<slug>
Downloads go to dataset/assets/hdris/<slug>.hdr.

Usage:
    python dataset/scripts/download_hdris.py --count 50
    python dataset/scripts/download_hdris.py --count 100 --res 1k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "assets" / "hdris"


def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "bodytracking-dataset"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download_file(url: str, out_path: Path, timeout: int = 300) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "bodytracking-dataset"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, out_path.open("wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            read = 0
            chunk = 1 << 20  # 1 MB
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if total:
                    pct = read / total * 100
                    print(f"\r    {out_path.name}: {pct:5.1f}% "
                          f"({read/1024/1024:.1f}/{total/1024/1024:.1f} MB)",
                          end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"\n    FAIL {out_path.name}: {e}")
        if out_path.exists():
            out_path.unlink()
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=50,
                    help="number of HDRIs to download (default 50)")
    p.add_argument("--res", default="1k", choices=["1k", "2k", "4k"],
                    help="HDRI resolution; 1k is plenty for background use (default 1k)")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output directory")
    p.add_argument("--seed", type=int, default=20260420)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[hdri] listing Poly Haven HDRIs…")
    index = fetch_json("https://api.polyhaven.com/assets?t=hdris")
    all_slugs = sorted(index.keys())
    print(f"[hdri] {len(all_slugs)} CC0 HDRIs available")

    # Skip slugs we already have locally.
    existing = {p.stem for p in out_dir.glob("*.hdr")}
    candidates = [s for s in all_slugs if s not in existing]
    print(f"[hdri] {len(existing)} already in {out_dir}; {len(candidates)} candidates")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    saved = 0
    for slug in candidates:
        if saved >= args.count:
            break
        info = fetch_json(f"https://api.polyhaven.com/files/{slug}")
        try:
            url = info["hdri"][args.res]["hdr"]["url"]
        except KeyError:
            print(f"[hdri] skip {slug}: no {args.res} hdr file")
            continue
        out_path = out_dir / f"{slug}.hdr"
        print(f"[hdri] {saved+1}/{args.count}  {slug}  ({args.res})")
        if download_file(url, out_path):
            saved += 1
        else:
            continue
        # Be polite to the API.
        time.sleep(0.2)

    total = sum(1 for _ in out_dir.glob("*.hdr"))
    print(f"[hdri] done — {saved} new; total in {out_dir}: {total}")


if __name__ == "__main__":
    main()

"""Kinetics-700 CC-BY manifest filter (scaffold).

Inputs:
  The official Kinetics manifests from
  https://github.com/cvdfoundation/kinetics-dataset — CSV rows of
  `label,youtube_id,time_start,time_end,split,is_cc`.

Outputs:
  `kinetics_ccby_manifest.csv` containing only `is_cc == 1` rows, plus per-row
  uploader/license resolution (via yt-dlp metadata extraction) for attribution
  tracking.

This is a scaffold; actual download requires the user to place the raw manifest
CSVs and install yt-dlp.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def filter_ccby(input_csv: Path, output_csv: Path) -> int:
    kept = 0
    with input_csv.open(newline="", encoding="utf-8") as fin, \
         output_csv.open("w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        w = csv.DictWriter(fout, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if str(row.get("is_cc", "")).strip() in {"1", "True", "true"}:
                w.writerow(row)
                kept += 1
    return kept


def fetch_uploader(youtube_id: str) -> dict:
    """Use yt-dlp --print-json to extract uploader/license metadata (dry-run)."""
    try:
        p = subprocess.run(
            ["yt-dlp", "--skip-download", "--print-json",
             f"https://www.youtube.com/watch?v={youtube_id}"],
            capture_output=True, text=True, timeout=20,
        )
        import json
        j = json.loads(p.stdout.strip().splitlines()[0])
        return {
            "uploader": j.get("uploader"),
            "license": j.get("license"),
            "webpage_url": j.get("webpage_url"),
            "duration_s": j.get("duration"),
            "title": j.get("title"),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", type=Path, help="Kinetics-700 original CSV")
    ap.add_argument("output_csv", type=Path, help="Filtered CC-BY CSV out")
    args = ap.parse_args()
    n = filter_ccby(args.input_csv, args.output_csv)
    print(f"Kept {n} rows with is_cc=1 -> {args.output_csv}")


if __name__ == "__main__":
    sys.exit(main())

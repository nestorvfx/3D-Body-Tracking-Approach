# real_video — Pseudo-labeled real video pipeline

Commercial-clean pseudo-labeled 3D body pose dataset from Kinetics-700 CC-BY and
Wikimedia Commons videos.  Produces crops + JSON labels in the same schema as
`dataset/output/pilot_v2_*/seq_*/`.

## Status: scaffold only
Modules are stubs.  Downloads + teacher inference + temporal filtering are
sequential engineering tasks (~1 week).  Start order is `download/` → `detect/`
→ `teachers/` → `ensemble/` → `temporal/` → `quality/` → `pipeline/run.py`.

## Licensing chain of custody

All sources below are CC-BY (with attribution tracking) or CC0.  **Explicitly
excluded**:
- Pixabay (TOS Jan 2024 prohibits ML training)
- Any CC-BY-SA (viral copyleft — would force dataset + weights under SA)
- Research-only datasets (H3.6M, 3DPW, EMDB, AMASS-derived, Mixamo)

Per-video attribution tracked in an SQLite ledger
(`legal/attribution.sqlite`) with columns:
`video_id, source_url, uploader, license, license_url, retrieved_at`.

## Directory layout

```
real_video/
  download/            kinetics_cc_filter.py   wikimedia_scraper.py   video_fetcher.py
  detect/              rtmdet_person.py                              # Apache-2.0 detector
  teachers/            vitpose.py  rtmpose.py  mediapipe_pose.py    # 2D teachers
  ensemble/            agreement.py  consensus.py
  temporal/            tracker.py (ByteTrack)  smoothing.py (OneEuroFilter)
  lifter/              model.py  train_lifter.py  infer_lifter.py   # synth-trained 2D->3D
  quality/             score.py  dedup.py (pose-hash via PCA)
  legal/               attribution.py  license_audit.py
  pipeline/            run.py  config.yaml
  out/                 crops/  labels/  manifest.parquet  ATTRIBUTION.md
```

## Target scale

Kinetics-700 CC-BY subset + Wikimedia Commons human-motion → **~300K frames**
after multi-teacher agreement (≥2 teachers agree on ≥10 joints within 8px
OKS), temporal consistency (bone-length constancy <15%, velocity <3σ), and
quality scoring (top 50%).

## Teacher ensemble (all commercial-clean)

| Teacher | License | Skeleton | Training data |
|---|---|---|---|
| **ViTPose-B COCO-only** | Apache 2.0 | COCO-17 | COCO (CC-BY 4.0) |
| **RTMPose-L COCO-only** | Apache 2.0 | COCO-17 | COCO (CC-BY 4.0) |
| MediaPipe BlazePose (tiebreaker) | Apache 2.0 | 33 kps | opaque, Google-indemnified |

**Excluded** because training data is research-only:
- ViTPose-H (MPII-pretrained), DWPose (CrowdPose), MotionBERT (H3.6M),
  HMR2 (SMPL-derivative), RTMW3D (H3WB).

## 2D→3D lifter

Do **not** distill from tainted 3D teachers.  Instead:

1. Train a small 2D→3D lifter on our synthetic MPFB+CMU data
   (target ~40mm MPJPE on synthetic test split).
2. Run it on real videos to produce pseudo-3D.
3. Filter by teacher ensemble 2D agreement + temporal consistency.
4. Retrain the lifter on synthetic + pseudo-real (70/30 mix) — iterate 2×.

This keeps the 3D supervision chain pure-synthetic-derivative.

## Compute estimate

- ViTPose-B + RTMPose-L ensemble, ~15 fps combined on RTX 4090
- 40K CC-BY videos × ~300 sampled frames = 12M raw → ~3.6M after detection
  → ~1M after ensemble agreement → ~400K after temporal + quality filter
- **~240 GPU-hours total, ~$75** on Vast.ai 4090 spot.

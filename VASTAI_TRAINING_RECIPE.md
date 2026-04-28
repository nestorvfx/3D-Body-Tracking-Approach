# Vast.ai Training Recipe — synth pretrain + Ego-Exo4D fine-tune

End-to-end pipeline that takes a fresh Vast.ai instance from zero to a 3D
body-pose model trained on synthetic data + real Ego-Exo4D fine-tune.
Two-stage: 20-epoch synth+aug pretrain → 12-epoch 50/50 mixed fine-tune.

## Instance specs (recommended)

- **GPU**: 4× RTX 5070 Ti (or equivalent ≥ 16 GB VRAM each, Blackwell preferred)
- **CPU**: ≥ 32 cores
- **RAM**: ≥ 64 GB
- **Disk**: ≥ 116 GB
- **Driver**: ≥ 570 (required for sm_120 / Blackwell)

Total wall time end-to-end: ~12–18 hours.

---

## Phase 0 — verify instance specs

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h /workspace
free -h | head -2
python3 --version
```

Need: GPU(s) detected, ≥ 100 GB free on `/workspace`, Python ≥ 3.10.

## Phase 1 — clone repo + install deps + Open Images V7 corpus

```bash
cd /workspace
git clone https://github.com/nestorvfx/3D-Body-Tracking-Approach.git
cd 3D-Body-Tracking-Approach
bash training/vastai_setup.sh
```

`vastai_setup.sh` runs in order:
1. PyTorch nightly cu128 + project deps (timm, albumentations, opencv, tensorboard, tqdm, scipy, mat73, h5py, numpy)
2. Open Images V7 metadata + 16 mask-zip download (~110 MB)
3. Build the F1 occluder corpus (~8000 RGBA cutouts) + F2b bg corpus (~5000 person-free 256×192 crops)

Verify:
```bash
ls assets/sim2real_refs/occluders | wc -l   # expect ~8000
ls assets/sim2real_refs/bg        | wc -l   # expect ~5000
```

## Phase 2 — install HF CLI + auth

```bash
pip install -U "huggingface_hub[hf_xet]" hf_transfer
hf auth login    # paste read token, "n" to git credential prompt
```

Or env-only:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Phase 3 — download synth dataset (10 zips × ~4 GB)

```bash
mkdir -p /workspace/3D-Body-Tracking-Approach/dataset/synth_500k/data
cd /workspace/3D-Body-Tracking-Approach/dataset/synth_500k

export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Download 3-at-a-time in strict groups (uses curl directly to avoid hf-cli hangs)
ALL=(01 02 03 04 05 06 07 08 09 10)
for ((i=0; i<${#ALL[@]}; i+=3)); do
    GROUP=("${ALL[@]:i:3}")
    echo "=== group: ${GROUP[*]} ==="
    for N in "${GROUP[@]}"; do
        (
            curl -L -C - -fsS -H "Authorization: Bearer $HF_TOKEN" \
                -o "data/batch_${N}.zip" \
                "https://huggingface.co/datasets/nestorvfx/3DBodyTrackingDatabase/resolve/main/data/batch_${N}.zip" \
            && echo "[done] batch_${N}" || echo "[FAIL] batch_${N}"
        ) &
    done
    wait
done

# Verify integrity
for z in data/*.zip; do unzip -tq "$z" >/dev/null 2>&1 && echo "OK $z" || echo "BAD $z"; done
```

If any BAD or undersized, re-run the loop — `curl -C -` resumes partial files.

## Phase 4 — extract zips (sequentially — disk-pressure-safe)

```bash
cd /workspace/3D-Body-Tracking-Approach/dataset/synth_500k
for z in data/batch_*.zip; do
    echo "=== $z ==="
    unzip -q -o "$z" -d .
    df -h /workspace | tail -1
done
```

Cross-zip duplicates are byte-identical (verified via sha256), so flat
`-o` extract dedupes naturally to ~149,292 unique samples (the dataset's
true size, despite 10 zips totalling 999k entries with 47% cross-batch
overlap).

```bash
find . -maxdepth 1 -name "*.png"  -type f | wc -l   # ~149292
find . -maxdepth 1 -name "*.json" -type f | wc -l   # ~149292
```

## Phase 5 — build labels.jsonl from per-image JSONs

```bash
cd /workspace/3D-Body-Tracking-Approach/dataset/synth_500k

python - <<'PY'
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

R = Path(".")
files = list(R.glob("*.json"))
print(f"found {len(files)} JSON files")

def parse(p):
    try:
        rec = json.loads(p.read_text())
        rec["image_rel"] = f"{rec['id']}.png"
        rec.pop("mask_rel", None)
        rec.pop("depth_rel", None)
        if not (R / rec["image_rel"]).exists():
            return None
        return json.dumps(rec, separators=(",", ":"))
    except Exception:
        return None

n_ok = n_bad = 0
with open("labels.jsonl", "w") as out, ProcessPoolExecutor(max_workers=8) as ex:
    for line in ex.map(parse, files, chunksize=500):
        if line is None:
            n_bad += 1; continue
        out.write(line + "\n"); n_ok += 1
print(f"DONE: {n_ok} written, {n_bad} skipped")
PY

wc -l labels.jsonl
```

## Phase 6 — filter outliers

```bash
cd /workspace/3D-Body-Tracking-Approach
sed -i 's|^root = .*|root = "dataset/synth_500k"|' filter_outliers.py
python filter_outliers.py
```

Drops records with NaN root_z or root_z outside [0.3, 50] m. On clean
synth data this typically drops 0.

Optional tighter filter (drop bbox < 16 px — 0.5% of records, marginal
training stability gain):
```bash
cd /workspace/3D-Body-Tracking-Approach/dataset/synth_500k
python - <<'PY'
import json, os
src, tmp = "labels.jsonl", "labels.jsonl.tmp"
kept = dropped = 0
with open(src) as fi, open(tmp, "w") as fo:
    for line in fi:
        rec = json.loads(line)
        bw, bh = rec["bbox_xywh"][2], rec["bbox_xywh"][3]
        if bw < 16 or bh < 16: dropped += 1; continue
        fo.write(line); kept += 1
os.replace(tmp, src)
print(f"kept={kept} dropped={dropped}")
PY
```

## Phase 7 — reorganize into project-canonical layout

QA tool expects `dataset_dir/images/*.png` + `dataset_dir/labels.jsonl`.
Move PNGs into `images/`, drop redundant per-image JSONs, fix `image_rel`.

```bash
cd /workspace/3D-Body-Tracking-Approach/dataset/synth_500k

mkdir -p images
find . -maxdepth 1 -name "*.png" -type f -exec mv -t images/ {} +
find . -maxdepth 1 -name "*.json" -type f -delete

python - <<'PY'
import json, os
src, tmp = "labels.jsonl", "labels.jsonl.tmp"
n = 0
with open(src) as fi, open(tmp, "w") as fo:
    for line in fi:
        rec = json.loads(line)
        if not rec["image_rel"].startswith("images/"):
            rec["image_rel"] = "images/" + rec["image_rel"]
        fo.write(json.dumps(rec, separators=(",", ":")) + "\n"); n += 1
os.replace(tmp, src)
print(f"updated {n} records")
PY
```

## Phase 8 — compute MediaPipe mattes (parallel, ~32 CPU)

The matte script is hard-coded to `synth_iter`; patch it for `synth_500k`:

```bash
cd /workspace/3D-Body-Tracking-Approach
sed -i 's|^ROOT = .*|ROOT = Path("/workspace/3D-Body-Tracking-Approach")|' \
    training/_iter_compute_mattes.py
sed -i 's|^ITER_DIR = .*|ITER_DIR = ROOT / "dataset" / "synth_500k"|' \
    training/_iter_compute_mattes.py
```

Use the parallel runner (16-way CPU) — much faster than the single-thread
`_iter_compute_mattes.py --mattes-only`:

```bash
mkdir -p dataset/synth_500k/mattes
python training/_iter_compute_mattes_parallel.py 16
```

The parallel runner (`training/_iter_compute_mattes_parallel.py`) is a
~80-line wrapper that distributes the work over `multiprocessing.Pool`
with each worker holding its own MediaPipe instance. Writes to the same
shared `mattes/` dir — no race conditions because each record has a
unique id.

Wall time: ~5–10 min for 149k images on a 32-core box.

## Phase 9 — Stage 1 training: 20 epochs synth + aug, DDP×4

```bash
cd /workspace/3D-Body-Tracking-Approach

python -m torch.distributed.run --standalone --nproc_per_node 4 training/train.py \
    --dataset-dir dataset/synth_500k \
    --out-dir training/runs/sota_500k_simreal_v1 \
    --epochs 20 \
    --batch 96 \
    --lr 1e-4 \
    --workers 8 \
    --occluder-dir assets/sim2real_refs/occluders \
    --bg-corpus-dir assets/sim2real_refs/bg \
    --matte-dir dataset/synth_500k/mattes \
    --p-occluder 0.6 \
    --p-bg-composite 0.5 \
    --p-fda 0
```

Expected wall time: ~30–45 min on 4× RTX 5070 Ti.
Expected final val MPJPE on synth val: 75–95 mm.

Use `python -m torch.distributed.run` instead of `torchrun` if the latter
is not on `$PATH` (the `--user` pip install puts it at `~/.local/bin`).

## Phase 10 — Ego-Exo4D auth

```bash
pip install awscli
aws configure
# Paste the AWS Access Key ID, Secret, region (typically us-east-2) emailed by Ego4D consortium
```

## Phase 11 — Ego-Exo4D CLI + projectaria-tools

```bash
pip install ego4d --upgrade
pip install projectaria-tools
```

## Phase 12 — download Ego-Exo4D annotations + metadata only (~10 GB)

```bash
mkdir -p /workspace/3D-Body-Tracking-Approach/dataset/egoexo4d
cd /workspace/3D-Body-Tracking-Approach/dataset/egoexo4d

egoexo -o . \
    --parts metadata annotations \
    --benchmarks body_pose \
    --splits train val
```

Yields:
- `metadata/` + `takes.json` (registry mapping take_uid → root_dir)
- `annotations/ego_pose/{train,val}/body/annotation/<take_uid>.json` (864 train + 218 val takes)
- `annotations/ego_pose/{train,val}/camera_pose/<take_uid>.json` (per-camera intrinsics + extrinsics)

## Phase 13 — Ego-Exo4D body-pose frame extraction (stream)

The script `training/_iter_extract_egoexo_frames.py` (committed to the
repo) streams one take at a time:
1. Downloads just that take's 4 exo videos at 448 px (~200 MB)
2. Decodes only the annotated frames (sparse — ~99 frames per take avg)
3. Transforms 3D world → camera, computes bbox from 2D, saves JPG + record
4. Deletes the videos (peak transient disk: ~5 GB)

Net steady-state disk: ~6–12 GB final.

```bash
cd /workspace/3D-Body-Tracking-Approach

# 5-take sanity test first
python training/_iter_extract_egoexo_frames.py --workers 2 --max-takes 5

# Verify the test worked
ls dataset/egoexo4d/images | wc -l                 # ~600-800
wc -l dataset/egoexo4d/labels.jsonl                # same number
head -1 dataset/egoexo4d/labels.jsonl | python -c "import json,sys; print(list(json.loads(sys.stdin.read()).keys()))"

# Full run (~20-40 min with 12 workers)
python training/_iter_extract_egoexo_frames.py --workers 12
```

Expected: ~1,082 takes → ~400k records (4 cams × ~99 frames per take avg).

## Phase 14 — patch data.py for source-aware augmentation

Real records get gentler photometric (over-aug on real adds noise without
benefit, per the 2025 CVPR synth-to-real-gap paper). Synth records keep
full F1+F2 strength.

```bash
cd /workspace/3D-Body-Tracking-Approach
python - <<'PY'
from pathlib import Path
p = Path("training/data.py")
src = p.read_text()

old_init = '''        self.photo = (
            build_sim2real_aug(
                fda_reference_images=self.fda_refs or None,
                p_fda=cfg.p_fda,
                occluders_active=bool(self.occluders),
            )
            if (cfg.training and cfg.photometric)
            else None)'''

new_init = '''        self.photo_synth = (
            build_sim2real_aug(
                fda_reference_images=self.fda_refs or None,
                p_fda=cfg.p_fda,
                occluders_active=bool(self.occluders),
            )
            if (cfg.training and cfg.photometric)
            else None)
        self.photo_real = (
            build_sim2real_aug(
                p_color=0.4, p_blur=0.15, p_noise=0.15, p_jpeg=0.0,
                p_dropout=0.3, p_fog_shadow=0.0,
                fda_reference_images=None, p_fda=0.0,
                occluders_active=bool(self.occluders),
            )
            if (cfg.training and cfg.photometric)
            else None)
        self.photo = self.photo_synth'''

src = src.replace(old_init, new_init)

old_call = '''        if training and self.photo is not None:
            out = self.photo(image=crop, keypoints=kps2d.tolist())'''
new_call = '''        if training and self.photo_synth is not None:
            is_real = (sample_id is not None) and sample_id.startswith("egoexo_")
            photo = self.photo_real if is_real else self.photo_synth
            out = photo(image=crop, keypoints=kps2d.tolist())'''

src = src.replace(old_call, new_call)
p.write_text(src)
print("data.py patched OK")
PY
```

## Phase 15 — build merged dataset (synth + egoexo)

Prefixed `image_rel` paths so a single `dataset_dir` can host both
sources without renaming conflicts.

```bash
cd /workspace/3D-Body-Tracking-Approach

rm -rf dataset/mixed && mkdir -p dataset/mixed
ln -sn /workspace/3D-Body-Tracking-Approach/dataset/synth_500k/images dataset/mixed/synth_images
ln -sn /workspace/3D-Body-Tracking-Approach/dataset/egoexo4d/images   dataset/mixed/egoexo_images
ln -sn /workspace/3D-Body-Tracking-Approach/dataset/synth_500k/mattes dataset/mixed/mattes

python - <<'PY'
import json
n_s = n_e = 0
with open("dataset/mixed/labels.jsonl", "w") as out:
    for line in open("dataset/synth_500k/labels.jsonl"):
        rec = json.loads(line)
        if rec["image_rel"].startswith("images/"):
            rec["image_rel"] = "synth_" + rec["image_rel"]
        out.write(json.dumps(rec, separators=(",", ":")) + "\n"); n_s += 1
    for line in open("dataset/egoexo4d/labels.jsonl"):
        rec = json.loads(line)
        if rec["image_rel"].startswith("images/"):
            rec["image_rel"] = "egoexo_" + rec["image_rel"]
        out.write(json.dumps(rec, separators=(",", ":")) + "\n"); n_e += 1
print(f"merged: {n_s} synth + {n_e} egoexo = {n_s + n_e} total "
      f"({n_s/(n_s+n_e)*100:.0f}% / {n_e/(n_s+n_e)*100:.0f}%)")
PY
```

## Phase 16 — rebalance to ~50/50

Per BEDLAM Tab. 5, 50/50 synth+real beats both 100% real and 100% synth by
~6 mm PA-MPJPE. Random-drop egoexo records to match synth count.

```bash
cd /workspace/3D-Body-Tracking-Approach

python - <<'PY'
import json, random, os
random.seed(42)
n_s = n_e = 0
with open("dataset/mixed/labels.jsonl") as f:
    for line in f:
        if json.loads(line).get("source") == "egoexo4d": n_e += 1
        else: n_s += 1
keep_prob = n_s / n_e
n_kept_s = n_kept_e = 0
src, tmp = "dataset/mixed/labels.jsonl", "dataset/mixed/labels.jsonl.tmp"
with open(src) as fi, open(tmp, "w") as fo:
    for line in fi:
        rec = json.loads(line)
        if rec.get("source") == "egoexo4d":
            if random.random() < keep_prob: fo.write(line); n_kept_e += 1
        else:
            fo.write(line); n_kept_s += 1
os.replace(tmp, src)
print(f"rebalanced: {n_kept_s} synth + {n_kept_e} egoexo = {n_kept_s + n_kept_e}")
PY

wc -l dataset/mixed/labels.jsonl
```

## Phase 17 — sanity check merged dataset

```bash
python - <<'PY'
import json, os, math
n = nan = broken = 0
sources = {}
for line in open("dataset/mixed/labels.jsonl"):
    r = json.loads(line)
    n += 1
    sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1
    if not os.path.exists(os.path.join("dataset/mixed", r["image_rel"])): broken += 1
    for kp in r.get("keypoints_3d_cam", []):
        if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in kp):
            nan += 1; break
print(f"records: {n}  broken_images: {broken}  nan_3d: {nan}")
print(f"sources: {sources}")
PY
```

Need: `broken=0`, `nan_3d=0`, `sources` shows both synth (cmu/100style/aist) and egoexo4d.

## Phase 18 — Stage 2 training: 12 epochs 50/50 mix, DDP×4

```bash
cd /workspace/3D-Body-Tracking-Approach

python -m torch.distributed.run --standalone --nproc_per_node 4 training/train.py \
    --dataset-dir dataset/mixed \
    --out-dir training/runs/sota_mixed_v1 \
    --resume training/runs/sota_500k_simreal_v1/best.pt \
    --fresh-schedule \
    --epochs 12 \
    --batch 96 \
    --lr 1e-4 \
    --workers 8 \
    --occluder-dir assets/sim2real_refs/occluders \
    --bg-corpus-dir assets/sim2real_refs/bg \
    --matte-dir dataset/mixed/mattes \
    --p-occluder 0.6 \
    --p-bg-composite 0.5 \
    --p-fda 0
```

Expected output in the first 30 sec — confirms warm-start from Stage 1:
```
[lr] scaling base 1.0e-04 -> 4.0e-04 (world_size=4)
[model] mnv4s, params=12,188,386
[resume] fresh schedule — keeping only model weights from epoch 18
[data] train=N_train, val=N_val
```

Expected wall time: ~45–60 min on 4× RTX 5070 Ti.

## Expected metric trajectory

Epoch 1's `val_mpjpe` will look very high (~290 mm) because the val set
contains never-before-seen real images. The number drops fast as the
model adapts.

| Ep | val_mpjpe | val_pa | bone | rz_err |
|---|---|---|---|---|
| 1 | ~290 mm | ~150 mm | ~130 mm | ~510 mm |
| 3 | ~230 | ~130 | ~107 | ~405 |
| 6 | ~210 | ~110 | ~85 | ~340 |
| 9 | ~190 | ~85 | ~65 | ~280 |
| 12 | ~170 | ~70 | ~55 | ~250 |

Watch:
- **`val_pa`** (Procrustes-aligned) — cross-distribution comparable; should drop fastest
- **`bone`** (skeleton consistency) — should drop steadily
- **`val_mpjpe`** plateau ≈ converged
- `val_mpjpe` numerically higher than Stage 1's 86 mm is expected — the val set composition changed (now includes real)

## Output artefacts

After Stage 2 completes:

```
training/runs/sota_mixed_v1/
├── best.pt                 ← shippable model (best val_mpjpe seen)
├── last.pt                 ← last epoch's weights
├── final_metrics.json
└── tb/                     ← TensorBoard event files
```

The `best.pt` is the model trained on synth + real, source-aware
augmentation, 50/50 balanced mix, fresh-schedule fine-tune from the
synth-only Stage 1.

## Attribution / licence chain

All components used here are commercial-clean per project's
[`dataset/LICENSE_AUDIT.md`](dataset/LICENSE_AUDIT.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md):
- Synth dataset: CMU mocap (PD) + 100STYLE (CC-BY-4.0) + AIST++ (CC-BY-4.0)
- Sim-to-real corpus: Open Images V7 (images CC-BY-2.0, annotations Apache-2.0)
- MediaPipe Pose Heavy (mattes only, preprocessing): Apache-2.0
- Ego-Exo4D: licence covers research + commercial use with attribution +
  no-redistribution per signed agreement; verify the exact terms of your
  signed copy before deployment.

Per-deliverable attribution lives in `THIRD_PARTY_NOTICES.md` at repo root.

## Reuse checklist for new instances

1. Phase 0 (specs check)
2. Phase 1 (clone + setup)
3. Phase 2 (HF auth)
4. Phases 3 + 4 + 5 + 6 + 7 (synth dataset)
5. Phase 8 (mattes)
6. Phase 9 (Stage 1)
7. Phases 10 + 11 + 12 + 13 (Ego-Exo4D)
8. Phase 14 (data.py patch — only on a fresh clone where the patch hasn't been applied)
9. Phases 15 + 16 + 17 (merge + balance + verify)
10. Phase 18 (Stage 2)

If Phase 9's `best.pt` exists from a previous run, scp it onto the new
instance to skip Phase 9. Vast.ai SSH connect details are in the web
console under each instance.

```bash
# Local machine → Vast.ai (replace PORT and HOST):
scp -P PORT \
    "training/runs/sota_500k_simreal_v1/best.pt" \
    root@HOST:/workspace/3D-Body-Tracking-Approach/training/runs/sota_500k_simreal_v1/best.pt
```

## Troubleshooting

**`hf download` hangs**: switch to direct curl with `-C -` resume (Phase 3 above).

**`curl: (23) Failure writing output to destination`**: missing target directory; `mkdir -p data` first.

**`401 AccessDenied` on `aws s3 ls s3://ego4d-consortium-sharing/`**: expected — your Ego-Exo4D credentials only authorize the `egoexo` CLI's bucket. Use `egoexo -o . --parts metadata` to test instead.

**`torchrun: command not found`**: `torchrun` lives at `~/.local/bin/torchrun` after `pip install --user`; either `export PATH="$HOME/.local/bin:$PATH"` or use the equivalent `python -m torch.distributed.run`.

**Mid-extract crash at unzip with "No space left on device"**: extract sequentially (Phase 4 above), not in parallel — peak disk on parallel-10 unzip can hit 80+ GB transiently.

**Stage 2 epoch 1 `val_mpjpe` ~290 mm**: expected. The val set now includes ~28k Ego-Exo4D val records the model has never seen. The number drops fast — by epoch 12 you should see ~170 mm.

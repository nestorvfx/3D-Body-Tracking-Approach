#!/bin/bash
# ---------------------------------------------------------------------------
# Launch multi-GPU DDP training on the vast.ai box.  Detects GPU count
# automatically and runs `torchrun --nproc_per_node=N` with NCCL backend.
#
# Usage:
#   ./vastai_train.sh [DATASET_DIR] [OUT_DIR] [EPOCHS]
# Defaults: /data/synth_v3, training/runs/vastai_v1, 50
#
# Env overrides:
#   BATCH=16               per-GPU batch (effective = BATCH * num_gpus)
#   LR=3e-3                per-GPU base LR (scaled linearly with world size)
#   WORKERS=8              dataloader workers PER RANK (not total)
#   WARMUP=1000            linear LR warmup iters before cosine
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DATASET="${1:-/data/synth_v3}"
OUT_DIR="${2:-$REPO_ROOT/training/runs/vastai_v1}"
EPOCHS="${3:-50}"
BATCH="${BATCH:-16}"
LR="${LR:-3e-3}"
WORKERS="${WORKERS:-8}"
WARMUP="${WARMUP:-1000}"

[ -d "$DATASET/images" ] || { echo "[train] $DATASET/images not found. Run dataset-gen first."; exit 1; }

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
[ "$GPU_COUNT" -gt 0 ] || { echo "[train] no GPUs found"; exit 1; }

echo "=========================================================="
echo "  dataset:       $DATASET  ($(ls $DATASET/images | wc -l) images)"
echo "  out:           $OUT_DIR"
echo "  GPUs:          $GPU_COUNT  (DDP, NCCL backend)"
echo "  per-GPU batch: $BATCH       (effective: $(( BATCH * GPU_COUNT )))"
echo "  LR:            $LR          (scaled: $(python3 -c "print($LR * $GPU_COUNT)"))"
echo "  workers/rank:  $WORKERS"
echo "  epochs:        $EPOCHS"
echo "  warmup iters:  $WARMUP"
echo "=========================================================="

cd "$REPO_ROOT"

# torchrun handles rdzv + env vars for each rank.
exec torchrun \
    --nproc_per_node="$GPU_COUNT" \
    --master-port=29500 \
    -m training.train \
    --dataset-dir "$DATASET" \
    --out-dir "$OUT_DIR" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --workers "$WORKERS" \
    --warmup-iters "$WARMUP"

#!/bin/bash
# ---------------------------------------------------------------------------
# Launch parallel Blender shards to generate the full dataset.
#
# HYBRID GPU+CPU MODE:
# If the box has NVIDIA GPUs, runs:
#   * GPU shards  — 2 Blender processes per GPU, Cycles GPU (OptiX), pinned
#                   via CUDA_VISIBLE_DEVICES.  Each GPU shard is assigned a
#                   LARGER contiguous slice of the plan (GPU_WEIGHT factor)
#                   so GPU and CPU shards finish around the same time.
#   * CPU shards  — Cycles CPU Blender processes using remaining cores.
#                   Each gets a smaller slice (CPU_WEIGHT=1).
# The slice sizes are computed so the total wall-clock is minimised.
#
# Usage:
#   ./run.sh [TARGET_COUNT] [NUM_SHARDS] [OUT_DIR]
# Defaults: TARGET=500000, NUM_SHARDS=auto, OUT=/data/synth_v3
#
# Env-var overrides:
#   ENGINE=auto|cpu|gpu      force Cycles device (default auto-detect)
#   GPU_WEIGHT=5             GPU shard slice size vs CPU (default 5)
#   CPU_SHARDS=16            # of Cycles-CPU shards when GPUs are present
#   INSTANCES_PER_GPU=2      parallel Blender processes sharing each GPU
#   CYCLES_SAMPLES=8         path-tracing samples + OIDN denoise
#   BLENDER=/opt/blender/blender
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-500000}"
OUT="${3:-/data/synth_v3}"
ENGINE="${ENGINE:-auto}"
CYCLES_SAMPLES="${CYCLES_SAMPLES:-8}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-2}"
GPU_WEIGHT="${GPU_WEIGHT:-5}"
# CPU_SHARDS default is computed below based on CPU count, because
# hardcoding 16 caused 96-thread boxes to thrash: each Cycles CPU shard
# spawns ~threads equal to core count, so 22 shards × 96 threads each
# = thousands of runnable threads.  We cap per-shard threads and size
# the shard count so total threads ≈ CPU count.
CPU_SHARDS="${CPU_SHARDS:-0}"           # 0 = auto
CPU_THREADS_PER_SHARD="${CPU_THREADS_PER_SHARD:-8}"
GPU_THREADS_PER_SHARD="${GPU_THREADS_PER_SHARD:-4}"    # GPU shards still use CPU for Python + OIDN
BLENDER="${BLENDER:-/opt/blender/blender}"
SEEDS_PER_CLIP="${SEEDS_PER_CLIP:-2}"
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-5}"

[ -x "$BLENDER" ] || { echo "[run] Blender not found at $BLENDER — run ./setup.sh"; exit 1; }

# ---------------------------------------------------------------------------
# Detect GPUs + compute plan size
# ---------------------------------------------------------------------------
GPU_COUNT=0
if [ "$ENGINE" != "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    fi
fi

# Plan size = total number of clips (each yields seeds_per_clip * frames_per_clip samples).
PLAN_SIZE=$(( (TARGET + (SEEDS_PER_CLIP * FRAMES_PER_CLIP) - 1) / (SEEDS_PER_CLIP * FRAMES_PER_CLIP) ))

# ---------------------------------------------------------------------------
# Shard allocation
# ---------------------------------------------------------------------------
CPU_THREADS_TOTAL=$(nproc)
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_SHARDS=$(( GPU_COUNT * INSTANCES_PER_GPU ))
    if [ "$CPU_SHARDS" -eq 0 ]; then
        # Auto-size so (GPU_SHARDS * GPU_THREADS_PER_SHARD) + (CPU_SHARDS * CPU_THREADS_PER_SHARD) <= CPU_THREADS_TOTAL
        remaining=$(( CPU_THREADS_TOTAL - GPU_SHARDS * GPU_THREADS_PER_SHARD ))
        CPU_SHARDS=$(( remaining / CPU_THREADS_PER_SHARD ))
        if [ "$CPU_SHARDS" -lt 2 ]; then CPU_SHARDS=2; fi
    fi
    TOTAL_WEIGHT=$(( GPU_SHARDS * GPU_WEIGHT + CPU_SHARDS ))   # CPU weight = 1
    TOTAL_SHARDS=$(( GPU_SHARDS + CPU_SHARDS ))
else
    GPU_SHARDS=0
    if [ "$CPU_SHARDS" -eq 0 ]; then
        CPU_SHARDS=$(( CPU_THREADS_TOTAL / CPU_THREADS_PER_SHARD ))
    fi
    TOTAL_WEIGHT="$CPU_SHARDS"
    TOTAL_SHARDS="$CPU_SHARDS"
fi

# Override num_shards from arg if given (advanced use).
if [ -n "${2:-}" ]; then
    TOTAL_SHARDS="$2"
    GPU_SHARDS=0
    CPU_SHARDS="$TOTAL_SHARDS"
    TOTAL_WEIGHT="$TOTAL_SHARDS"
fi

mkdir -p "$OUT" "$HERE/logs"

echo "=========================================================="
echo "  target:               $TARGET samples ($PLAN_SIZE clips)"
echo "  out:                  $OUT"
echo "  CPU threads avail:    $CPU_THREADS_TOTAL"
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "  GPUs:                 $GPU_COUNT"
    echo "  GPU shards:           $GPU_SHARDS  ($INSTANCES_PER_GPU per GPU, $GPU_THREADS_PER_SHARD threads each, weight=$GPU_WEIGHT)"
    echo "  CPU shards:           $CPU_SHARDS  ($CPU_THREADS_PER_SHARD threads each, weight=1)"
    echo "  Total thread budget:  $(( GPU_SHARDS * GPU_THREADS_PER_SHARD + CPU_SHARDS * CPU_THREADS_PER_SHARD )) / $CPU_THREADS_TOTAL"
else
    echo "  GPUs:                 none"
    echo "  CPU shards:           $CPU_SHARDS  ($CPU_THREADS_PER_SHARD threads each)"
fi
echo "  cycles samples:       $CYCLES_SAMPLES + OIDN denoise"
echo "  log dir:              $HERE/logs/"
echo "=========================================================="
echo

# ---------------------------------------------------------------------------
# Launch shards with weighted contiguous plan slices.
# ---------------------------------------------------------------------------
pids=()
cum_weight=0
shard_idx=0

launch_shard() {
    local weight=$1
    local engine=$2
    local gpu_id=$3
    local threads=$4
    local start=$(( cum_weight * PLAN_SIZE / TOTAL_WEIGHT ))
    cum_weight=$(( cum_weight + weight ))
    local end=$(( cum_weight * PLAN_SIZE / TOTAL_WEIGHT ))
    local log="$HERE/logs/shard_$(printf '%03d' $shard_idx).log"

    # Also cap OpenMP + TBB + OIDN thread pools so even non-Cycles CPU
    # work respects the per-shard thread budget.
    local env_caps="OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
OPENBLAS_NUM_THREADS=$threads TBB_NUM_THREADS=$threads \
OIDN_NUM_THREADS=$threads"

    if [ "$engine" = "gpu" ]; then
        echo "  shard $shard_idx -> GPU $gpu_id  plan[$start:$end]  threads=$threads  log $log"
        env $env_caps CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$BLENDER" --background --factory-startup --disable-autoexec -noaudio --python "$HERE/scripts/build_dataset.py" -- \
            --out "$OUT" --target-count "$TARGET" \
            --seeds-per-clip "$SEEDS_PER_CLIP" --frames-per-clip "$FRAMES_PER_CLIP" \
            --shard-id "$shard_idx" --num-shards "$TOTAL_SHARDS" \
            --shard-start "$start" --shard-end "$end" \
            --cycles-samples "$CYCLES_SAMPLES" --engine gpu \
            --threads "$threads" \
            >"$log" 2>&1 &
    else
        echo "  shard $shard_idx -> CPU       plan[$start:$end]  threads=$threads  log $log"
        env $env_caps \
        "$BLENDER" --background --factory-startup --disable-autoexec -noaudio --python "$HERE/scripts/build_dataset.py" -- \
            --out "$OUT" --target-count "$TARGET" \
            --seeds-per-clip "$SEEDS_PER_CLIP" --frames-per-clip "$FRAMES_PER_CLIP" \
            --shard-id "$shard_idx" --num-shards "$TOTAL_SHARDS" \
            --shard-start "$start" --shard-end "$end" \
            --cycles-samples "$CYCLES_SAMPLES" --engine cpu \
            --threads "$threads" \
            >"$log" 2>&1 &
    fi
    pids+=($!)
    shard_idx=$(( shard_idx + 1 ))
    sleep 0.5    # stagger Blender startup
}

# Launch GPU shards first (they pick up the fast slices at the start of
# the sorted plan, which helps character caching lock in early).
if [ "$GPU_SHARDS" -gt 0 ]; then
    for g in $(seq 0 $(( GPU_SHARDS - 1 ))); do
        gpu_id=$(( g / INSTANCES_PER_GPU ))
        launch_shard "$GPU_WEIGHT" "gpu" "$gpu_id" "$GPU_THREADS_PER_SHARD"
    done
fi

# CPU shards.
for c in $(seq 0 $(( CPU_SHARDS - 1 ))); do
    launch_shard 1 "cpu" "" "$CPU_THREADS_PER_SHARD"
done

echo
echo "  all $TOTAL_SHARDS shards launched — monitor with:"
echo "    tail -f $HERE/logs/shard_000.log"
echo "    watch -n 10 'ls $OUT/*/images 2>/dev/null | wc -l'"
[ "$GPU_COUNT" -gt 0 ] && echo "    nvidia-smi         # GPU load"
echo

# ---------------------------------------------------------------------------
# Wait and track failures.
# ---------------------------------------------------------------------------
fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "  shard $i FAILED (pid ${pids[$i]})"
        fail=$(( fail + 1 ))
    fi
done

echo
if [ "$fail" -gt 0 ]; then
    echo "  $fail/$TOTAL_SHARDS shards failed.  Re-run the same ./run.sh — "
    echo "  each shard's resume logic skips already-done IDs."
else
    echo "  all $TOTAL_SHARDS shards done"
fi

# ---------------------------------------------------------------------------
# Merge shards into unified dataset.
# ---------------------------------------------------------------------------
echo
echo "  merging shards..."
python3 "$HERE/merge.py" --out "$OUT"
echo
echo "  DONE — dataset at $OUT (see $OUT/dataset_stats.md)"

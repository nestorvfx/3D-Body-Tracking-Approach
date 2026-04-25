#!/bin/bash
# ---------------------------------------------------------------------------
# Install training deps on the vast.ai box so the same machine used for
# dataset generation can train.  Installs PyTorch NIGHTLY cu128 (required
# for RTX 5070 Blackwell — stable PyTorch 2.6+cu124 wheels ship NO sm_120
# kernels and will silently CPU-fallback), plus our model/data deps.
#
# Run once after dataset gen completes.  Idempotent.
# ---------------------------------------------------------------------------
set -euo pipefail

log() { echo -e "\n\033[1;34m[train-setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[train-setup:fatal]\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System check: GPU present, driver >= 570.
# ---------------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi — is this a GPU box?"
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
log "GPU: $GPU_NAME   driver: $DRIVER"

driver_major=${DRIVER%%.*}
if [ "${driver_major:-0}" -lt 570 ] && echo "$GPU_NAME" | grep -qiE "RTX 50"; then
    die "RTX 50-series detected but driver $DRIVER is below 570 — training will CPU-fallback. Update driver: sudo apt install nvidia-driver-570-open"
fi

# ---------------------------------------------------------------------------
# 2. Python + pip hygiene.
# ---------------------------------------------------------------------------
log "python deps"
sudo apt-get install -y -qq python3 python3-pip python3-venv libsm6 libxext6 libglib2.0-0 >/dev/null
python3 -m pip install --user --quiet --upgrade pip

# ---------------------------------------------------------------------------
# 3. PyTorch nightly cu128 — required for Blackwell (sm_120).
#    stable 2.6+cu124 does NOT ship sm_120 kernels; your training would
#    silently CPU-fall-back and take 100x longer.  Nightly has sm_120.
# ---------------------------------------------------------------------------
log "PyTorch nightly cu128 (Blackwell-capable)"
python3 -m pip install --user --quiet --pre --upgrade \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify the wheel is Blackwell-capable.
python3 - <<'PY'
import torch
print(f"torch:       {torch.__version__}")
print(f"cuda:        {torch.version.cuda}")
print(f"gpu count:   {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu {i}:    {p.name}  sm_{p.major}{p.minor}")
    # Allocate a tiny tensor to force kernel load — this will fail early
    # with "no kernel image available" if the wheel doesn't support sm_XY.
    x = torch.randn(4, 4, device=f"cuda:{i}")
    y = x @ x.T
    print(f"           sanity matmul OK  ({y.sum().item():.3f})")
PY

# ---------------------------------------------------------------------------
# 4. Our training deps.
# ---------------------------------------------------------------------------
log "project deps"
python3 -m pip install --user --quiet \
    timm==1.0.26 \
    albumentations==2.0.8 \
    opencv-python \
    tensorboard \
    tqdm \
    scipy \
    mat73 \
    h5py \
    numpy

log "DONE — run ./vastai_train.sh"

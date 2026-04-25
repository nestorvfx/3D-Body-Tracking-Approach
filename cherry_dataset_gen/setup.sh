#!/bin/bash
# ---------------------------------------------------------------------------
# One-shot installer for the dataset-gen box (Ubuntu 22.04 / 24.04).
# Works on BOTH CPU boxes (cherry EPYC) and GPU boxes (vast.ai RTX 5070 etc.).
# Auto-detects GPUs via nvidia-smi and verifies OptiX path before running.
# Installs Blender 5.1.1 + MPFB2, populates Python deps, downloads assets.
# Idempotent — safe to re-run; skips what's already present.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="$HERE/assets"
BLENDER_DIR="${BLENDER_DIR:-/opt/blender}"
# Blender 5.1.1 — fully supports RTX 50-series Blackwell via OptiX 9.x.
# Matches what the local dev setup uses (consistency across envs).
BLENDER_VER="${BLENDER_VER:-5.1.1}"
BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_VER%.*}/blender-${BLENDER_VER}-linux-x64.tar.xz"
# MPFB is installed from git HEAD (not the v2.0.x GitHub release zips).
# Reason: the release zips are stale — v2.0.8 (Jan 2025) still references
# ShaderNodeCombineRGB which was removed in Blender 4.1+, so the skin
# material fails to apply with "Node type ShaderNodeCombineRGB undefined"
# on Blender 5.1.  The main branch has been maintained and works with
# current Blender.  We clone --depth 1 and package src/mpfb into an
# extension zip on the fly.
MPFB_GIT="${MPFB_GIT:-https://github.com/makehumancommunity/mpfb2.git}"

log() { echo -e "\n\033[1;34m[setup]\033[0m $*"; }
die() { echo -e "\033[1;31m[setup:fatal]\033[0m $*" >&2; exit 1; }

# Ensure sibling scripts are executable — git over HTTPS sometimes drops
# the +x bit on the clone so `./run.sh` fails with Permission denied.
chmod +x "$HERE"/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# 1. System packages (Blender headless + HTTPS tooling + compression)
# ---------------------------------------------------------------------------
log "1/7  apt packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    wget curl ca-certificates unzip xz-utils zstd git \
    libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libxrender1 libxrandr2 \
    libegl1 libdbus-1-3 libsm6 libgl1 libglu1-mesa \
    python3 python3-pip parallel pigz \
    >/dev/null

# ---------------------------------------------------------------------------
# 2. GPU detection + driver sanity check
# ---------------------------------------------------------------------------
GPU_MODE="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        log "2/7  detected $GPU_COUNT GPU(s): $GPU_NAME  driver $DRIVER"
        # RTX 50-series (Blackwell) needs driver 570+; Ada 525+; Ampere 470+.
        # Warn, don't fail, on old driver — user may have newer GPU with newer driver.
        driver_major="${DRIVER%%.*}"
        if [ "${driver_major:-0}" -lt 525 ]; then
            echo "    WARN: driver $DRIVER is old; 570+ recommended for RTX 50-series"
        fi
        GPU_MODE="gpu"
    else
        log "2/7  nvidia-smi present but not functional — treating as CPU box"
    fi
else
    log "2/7  no NVIDIA GPU — CPU rendering mode"
fi

# ---------------------------------------------------------------------------
# 3. Blender 5.1.1 — prebuilt Linux tarball from blender.org
# ---------------------------------------------------------------------------
if [ -x "$BLENDER_DIR/blender" ]; then
    log "3/7  Blender already at $BLENDER_DIR (skipping)"
else
    log "3/7  Downloading Blender $BLENDER_VER"
    wget -q --show-progress -c "$BLENDER_URL" -O /tmp/blender.tar.xz
    sudo mkdir -p "$BLENDER_DIR"
    sudo tar --strip-components=1 -xf /tmp/blender.tar.xz -C "$BLENDER_DIR"
    rm /tmp/blender.tar.xz
fi
export BLENDER="$BLENDER_DIR/blender"
"$BLENDER" --version

# ---------------------------------------------------------------------------
# 4. MPFB2 extension  (core addon + CC0 asset pack).
#
# The MPFB GitHub release zip is CORE ONLY — it lacks the skins/clothes/
# hair/eyes/teeth asset directories that MPFB needs to build a textured
# character.  Those are distributed via makehumancommunity.org as a
# separate 268 MB zip ("makehuman_system_assets_cc0.zip", CC0 licensed),
# which we extract into MPFB's data directory.
# ---------------------------------------------------------------------------
MPFB_STAMP="$HOME/.mpfb_installed"
MPFB_ASSETS_STAMP="$HOME/.mpfb_assets_installed"
MPFB_ASSETS_URL="${MPFB_ASSETS_URL:-https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip}"

# 4a) Core extension — installed from git HEAD (see reasoning above).
if [ -f "$MPFB_STAMP" ]; then
    log "4a/7 MPFB2 core already installed (skipping)"
else
    log "4a/7 Installing MPFB2 core extension from git HEAD"
    apt-get install -y zip >/dev/null 2>&1 || sudo apt-get install -y zip >/dev/null 2>&1 || true
    rm -rf /tmp/mpfb2 /tmp/mpfb.zip
    git clone --depth 1 "$MPFB_GIT" /tmp/mpfb2 \
        || die "git clone $MPFB_GIT failed"
    [ -f /tmp/mpfb2/src/mpfb/blender_manifest.toml ] \
        || die "unexpected MPFB git layout — blender_manifest.toml not at src/mpfb/"
    (cd /tmp/mpfb2/src && zip -qr /tmp/mpfb.zip mpfb -x '*/__pycache__/*' -x '*.pyc') \
        || die "zip failed"
    "$BLENDER" --command extension install-file -r user_default -e /tmp/mpfb.zip \
        || die "extension install-file failed (Blender refused our rebuilt zip)"
    touch "$MPFB_STAMP"
    rm -rf /tmp/mpfb2 /tmp/mpfb.zip
fi

# 4b) CC0 asset pack (clothes, hair, skins, eyes, teeth, etc.)
if [ -f "$MPFB_ASSETS_STAMP" ]; then
    log "4b/7 MPFB CC0 asset pack already extracted (skipping)"
else
    log "4b/7 Downloading MPFB CC0 asset pack (~268 MB)"
    # Find MPFB's installed data dir via Blender itself — robust across
    # Blender versions / config path variations.
    MPFB_DATA_DIR="$(
        "$BLENDER" --background --python-expr "
import sys
try:
    from bl_ext.user_default.mpfb.services.locationservice import LocationService
    print('MPFB_DATA_DIR=' + LocationService.get_user_data())
except Exception as e:
    print('MPFB_DATA_DIR_ERR=' + str(e), file=sys.stderr)
" 2>&1 | grep -oE 'MPFB_DATA_DIR=[^ ]+' | head -1 | cut -d= -f2- )"
    if [ -z "$MPFB_DATA_DIR" ] || [ ! -d "$MPFB_DATA_DIR" ]; then
        # Fallback — usual Blender 5.x Linux location
        for candidate in \
            "$HOME/.config/blender/5.1/extensions/user_default/mpfb/data" \
            "$HOME/.config/blender/5.0/extensions/user_default/mpfb/data" \
            "$HOME/.config/blender/4.5/extensions/user_default/mpfb/data"; do
            if [ -d "$candidate" ]; then MPFB_DATA_DIR="$candidate"; break; fi
        done
    fi
    [ -d "$MPFB_DATA_DIR" ] || die "MPFB data dir not found. Searched ~/.config/blender/*/extensions/user_default/mpfb/data"
    echo "[setup]      target dir: $MPFB_DATA_DIR"
    echo "[setup]      URL: $MPFB_ASSETS_URL"
    wget --show-progress -c "$MPFB_ASSETS_URL" -O /tmp/mpfb_assets.zip \
        || die "MPFB CC0 asset pack download failed"
    unzip -oq /tmp/mpfb_assets.zip -d "$MPFB_DATA_DIR"
    rm -f /tmp/mpfb_assets.zip
    n_clothes=$(find "$MPFB_DATA_DIR/clothes" -name "*.mhclo" 2>/dev/null | wc -l)
    n_skins=$(find "$MPFB_DATA_DIR/skins" -name "*.mhmat" 2>/dev/null | wc -l)
    n_hair=$(find "$MPFB_DATA_DIR/hair" -name "*.mhclo" 2>/dev/null | wc -l)
    echo "[setup]      extracted: clothes=$n_clothes  skins=$n_skins  hair=$n_hair"
    [ "$n_clothes" -gt 0 ] || die "Asset extraction completed but no clothes found — check MPFB_DATA_DIR"
    touch "$MPFB_ASSETS_STAMP"
fi

# ---------------------------------------------------------------------------
# 5. Python deps (for the orchestrator / downloader scripts, NOT Blender's
#    embedded python — Blender ships its own interpreter).
# ---------------------------------------------------------------------------
log "5/7  Python deps"
python3 -m pip install --user --quiet --upgrade pip
python3 -m pip install --user --quiet tqdm

# ---------------------------------------------------------------------------
# 6. Assets (delegated to download_assets.sh so it can be re-run)
# ---------------------------------------------------------------------------
log "6/7  Downloading assets (BVH motion + HDRIs)"
bash "$HERE/download_assets.sh"

# ---------------------------------------------------------------------------
# 7. Sanity check — a 2-sample render to confirm everything wires up,
#    including GPU OptiX path if available.
# ---------------------------------------------------------------------------
log "7/7  Sanity-check render (2 samples, ~30s)"
SMOKE_OUT="$HERE/_smoke_test"
rm -rf "$SMOKE_OUT"
"$BLENDER" --background --python "$HERE/scripts/build_dataset.py" -- \
    --out "$SMOKE_OUT" \
    --target-count 2 \
    --seeds-per-clip 1 --frames-per-clip 2 \
    --shard-id 0 --num-shards 1 \
    --cycles-samples 8 \
    --engine "$GPU_MODE"
n_imgs=$(ls "$SMOKE_OUT/shard_000/images" 2>/dev/null | wc -l)
if [ "$n_imgs" -lt 1 ]; then
    die "Smoke test produced no images. See log above."
fi
log "    smoke-test rendered $n_imgs images with engine=$GPU_MODE — pipeline OK"
rm -rf "$SMOKE_OUT"

log "DONE — mode: $GPU_MODE.  Ready to run ./run.sh"

#!/bin/bash
# ---------------------------------------------------------------------------
# Fetch BVH motion capture + Poly Haven HDRIs into ./assets/ from their
# canonical sources.  No uploads from your laptop — everything is pulled
# directly on the cherry box.  Idempotent: re-run to resume partial
# downloads (wget -c).  Total download: ~3 GB, 10-15 min on Gb line.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS="$HERE/assets"
HDRI_COUNT="${HDRI_COUNT:-300}"           # how many HDRIs to fetch (200-750 available)
HDRI_RES="${HDRI_RES:-1k}"                # 1k plenty for background; 2k if you want more detail

mkdir -p "$ASSETS"/{bvh,bvh_100style,hdris}

log() { echo -e "\n\033[1;34m[assets]\033[0m $*"; }

# ---------------------------------------------------------------------------
# 1. CMU Mocap — cgspeed Poser-friendly BVH conversion, ~145 MB zip,
#    ~2500 .bvh clips.  Hosted on archive.org (public, stable, CC-BY-SA).
# ---------------------------------------------------------------------------
CMU_URL="https://archive.org/download/cmu-ecstasy-motion-bvh-poser-friendly-2012/CMU_EcstasyMotion_BVH-Poser-friendly-2012.zip"
CMU_ZIP="/tmp/cmu_bvh.zip"
if [ -z "$(ls -A "$ASSETS/bvh" 2>/dev/null)" ]; then
    log "1/3  CMU Mocap BVH (145 MB)"
    wget -q --show-progress -c "$CMU_URL" -O "$CMU_ZIP"
    # Zip root directory name is long; we want the BVH files directly under
    # assets/bvh/.  Extract to temp, flatten, move.
    tmp_cmu="$(mktemp -d)"
    unzip -q "$CMU_ZIP" -d "$tmp_cmu"
    # cgspeed zip layout: CMU/<subject>/<take>.bvh.  We flatten to
    # assets/bvh/<subject>_<take>.bvh so the existing plan loader's
    # `.glob("*.bvh")` picks them all up in one directory.
    find "$tmp_cmu" -name "*.bvh" | while read -r f; do
        fname="$(basename "$f")"
        # Prefix with parent-dir name when uniqueness requires (subject_XX_take.bvh).
        parent="$(basename "$(dirname "$f")")"
        out="$ASSETS/bvh/${parent}_${fname}"
        # If the bvh filename already starts with "parent_", don't double-prefix.
        if [[ "$fname" == "${parent}"* ]]; then out="$ASSETS/bvh/$fname"; fi
        cp "$f" "$out"
    done
    rm -rf "$tmp_cmu" "$CMU_ZIP"
    log "    CMU: $(ls "$ASSETS/bvh" | wc -l) BVH files"
else
    log "1/3  CMU BVH already present ($(ls "$ASSETS/bvh" | wc -l) files)"
fi

# ---------------------------------------------------------------------------
# 2. 100STYLE — Zenodo DOI-backed BVH pack, ~1.47 GB, 100 styles × 5 variants
#    = 500 clips.  CC-BY 4.0.
# ---------------------------------------------------------------------------
STYLE_URL="https://zenodo.org/api/records/8127870/files/100STYLE.zip/content"
STYLE_ZIP="/tmp/100style.zip"
if [ -z "$(ls -A "$ASSETS/bvh_100style" 2>/dev/null)" ]; then
    log "2/3  100STYLE BVH pack (1.5 GB)"
    wget -q --show-progress -c "$STYLE_URL" -O "$STYLE_ZIP"
    unzip -q "$STYLE_ZIP" -d "$ASSETS/bvh_100style"
    rm "$STYLE_ZIP"
    log "    100STYLE: $(find "$ASSETS/bvh_100style" -name '*.bvh' | wc -l) BVH files"
else
    log "2/3  100STYLE already present"
fi

# ---------------------------------------------------------------------------
# 3. HDRIs — Poly Haven CC0 via API.  Script is idempotent (skips anything
#    already in assets/hdris/).
# ---------------------------------------------------------------------------
n_hdri=$(ls "$ASSETS/hdris" 2>/dev/null | wc -l)
if [ "$n_hdri" -ge "$HDRI_COUNT" ]; then
    log "3/3  HDRIs already present ($n_hdri files)"
else
    log "3/3  Downloading Poly Haven HDRIs ($HDRI_COUNT requested, $HDRI_RES)"
    python3 "$HERE/scripts/download_hdris.py" \
        --count "$HDRI_COUNT" --res "$HDRI_RES" \
        --out "$ASSETS/hdris"
    log "    HDRIs: $(ls "$ASSETS/hdris" | wc -l) files"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
log "ASSETS READY"
echo "  CMU BVH:      $(ls "$ASSETS/bvh" 2>/dev/null | wc -l) files"
echo "  100STYLE:     $(find "$ASSETS/bvh_100style" -name '*.bvh' 2>/dev/null | wc -l) files"
echo "  HDRIs:        $(ls "$ASSETS/hdris" 2>/dev/null | wc -l) files"
echo "  Total size:   $(du -sh "$ASSETS" 2>/dev/null | awk '{print $1}')"

#!/bin/bash
# ---------------------------------------------------------------------------
# Package the finished dataset into a single download-friendly archive.
# Splits into 2 GB parts if desired (useful for flaky connections).
#
# Usage:
#   ./package.sh [OUT_DIR] [SPLIT_SIZE]
# Defaults: /data/synth_v3, no split (single .tar.zst).
# ---------------------------------------------------------------------------
set -euo pipefail

OUT="${1:-/data/synth_v3}"
SPLIT_SIZE="${2:-}"       # e.g. "2G" to split into 2 GB parts; empty = no split

[ -d "$OUT/images" ] || { echo "[package] $OUT/images not found — run ./run.sh first"; exit 1; }

n_imgs=$(ls "$OUT/images" | wc -l)
size=$(du -sh "$OUT" | awk '{print $1}')
echo "[package] $n_imgs images, $size on disk"

cd "$(dirname "$OUT")"
name="$(basename "$OUT")"

# zstd (installed via setup.sh) compresses PNG+JSON well and decompresses
# ~5x faster than xz.  Use tar for portability.
ARCHIVE="${name}.tar.zst"
echo "[package] creating $ARCHIVE"
tar -cf - "$name" | zstd -T0 -q -19 -o "$ARCHIVE"
echo "[package] size: $(du -sh "$ARCHIVE" | awk '{print $1}')"

if [ -n "$SPLIT_SIZE" ]; then
    echo "[package] splitting into $SPLIT_SIZE parts"
    split -b "$SPLIT_SIZE" -d --additional-suffix=".part" "$ARCHIVE" "${ARCHIVE}."
    rm "$ARCHIVE"
    ls -lh "${ARCHIVE}."*
    echo
    echo "[package] to reconstruct on your laptop:"
    echo "    cat ${ARCHIVE}.*.part > $ARCHIVE"
    echo "    zstd -d $ARCHIVE -o ${name}.tar"
    echo "    tar -xf ${name}.tar"
else
    echo
    echo "[package] archive ready: $(pwd)/$ARCHIVE"
    echo
    echo "Download from your laptop (pick one):"
    echo "  # rsync (resumable, recommended)"
    echo "  rsync -avP --partial USER@CHERRY_IP:$(pwd)/$ARCHIVE ."
    echo
    echo "  # scp (simple but no resume)"
    echo "  scp USER@CHERRY_IP:$(pwd)/$ARCHIVE ."
    echo
    echo "  # sshfs (mount remote, cp)"
    echo "  sshfs USER@CHERRY_IP:$(pwd) /mnt/cherry && cp /mnt/cherry/$ARCHIVE ."
    echo
    echo "Then unpack:"
    echo "  zstd -d $ARCHIVE -o ${name}.tar && tar -xf ${name}.tar"
fi

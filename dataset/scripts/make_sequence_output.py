"""From a 100-frame sequence folder produce:
  - stills/frame_XX.png: 10 highlight frames (every 10th)
  - motion.gif: animated GIF of all 100 frames
"""
import sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
seq_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "sequence"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output" / "sequence_output"
out_dir.mkdir(parents=True, exist_ok=True)

frames = sorted(seq_dir.glob("F_*.png"))
print(f"Found {len(frames)} frames in {seq_dir}")
if not frames:
    sys.exit(1)

# 10 highlight stills (evenly spaced)
stills_dir = out_dir / "stills"
stills_dir.mkdir(exist_ok=True)
n = len(frames)
step = max(1, n // 10)
picked = [frames[i * step] for i in range(10) if i * step < n]
for i, f in enumerate(picked):
    img = Image.open(f)
    out = stills_dir / f"still_{i:02d}.png"
    img.save(out)
    print(f"  still: {out.name} <- {f.name}")

# Animated GIF of ALL frames
# Scale down a bit so file isn't too large (max 480 wide).
max_w = 480
first = Image.open(frames[0])
if first.width > max_w:
    scale = max_w / first.width
    new_size = (int(first.width * scale), int(first.height * scale))
else:
    new_size = first.size

# Convert RGB -> palette 'P' mode per frame with a shared adaptive palette
# (reduces GIF size dramatically: 19MB -> ~2-4MB typical).
imgs_rgb = [Image.open(f).convert("RGB").resize(new_size, Image.LANCZOS) for f in frames]
# Derive palette from first frame, apply to all for color consistency.
palette_ref = imgs_rgb[0].quantize(colors=128, method=Image.MEDIANCUT)
imgs_p = [im.quantize(colors=128, palette=palette_ref, dither=Image.Dither.NONE)
          for im in imgs_rgb]
# CMU 02_01 source is 120fps.  At 24fps the 100 frames represent ~0.83s of
# real motion; we stretch playback to 40ms/frame for a 4s smooth loop.
duration_ms = 40
gif_path = out_dir / "motion.gif"
imgs_p[0].save(
    gif_path,
    save_all=True,
    append_images=imgs_p[1:],
    duration=duration_ms,
    loop=0,
    optimize=True,
    disposal=2,
)
size_kb = gif_path.stat().st_size // 1024
print(f"GIF saved: {gif_path}  ({len(imgs_p)} frames @ {int(1000/duration_ms)}fps, "
      f"{new_size}, {size_kb} KB)")

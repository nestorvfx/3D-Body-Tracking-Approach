"""Composite 16 renders (4 sources x 4 frames) into a single grid image."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

grid_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "grid"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output" / "grid.png"

ROWS = [
    ("CMU 02_01 walk",    grid_dir / "cmu",     ["F_0100", "F_0142", "F_0220", "F_0260"]),
    ("100STYLE Neutral",  grid_dir / "100style", ["F_0100", "F_0142", "F_0220", "F_0260"]),
    ("AIST++ Breakdance", grid_dir / "aistpp",  ["F_0100", "F_0150", "F_0250", "F_0400"]),
    ("CMU 56_01",         grid_dir / "cmu2",    ["F_0100", "F_0150", "F_0200", "F_0260"]),
]

# Load images, determine cell size from first.
first = Image.open(ROWS[0][1] / f"{ROWS[0][2][0]}.png")
cw, ch = first.size
label_h = 60

cols = len(ROWS[0][2])
rows = len(ROWS)
total_w = cw * cols
total_h = (ch + label_h) * rows + label_h

grid = Image.new("RGB", (total_w, total_h), (20, 20, 22))
draw = ImageDraw.Draw(grid)

# Font
try:
    font = ImageFont.truetype("arial.ttf", 28)
    small = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()
    small = ImageFont.load_default()

# Column headers
for c, frame in enumerate(ROWS[0][2]):
    draw.text((cw * c + 20, 15), frame, fill=(230, 230, 230), font=font)

# Each row
for r, (label, folder, frames) in enumerate(ROWS):
    y0 = label_h + r * (ch + label_h)
    draw.text((20, y0 - label_h + 15), label, fill=(200, 230, 255), font=font)
    for c, frame in enumerate(frames):
        p = folder / f"{frame}.png"
        if p.exists():
            img = Image.open(p).convert("RGB")
            grid.paste(img, (c * cw, y0))
        else:
            draw.rectangle([(c*cw, y0), ((c+1)*cw-1, y0+ch)], fill=(40, 20, 20))
            draw.text((c*cw + 10, y0 + ch//2), "missing", fill=(255, 80, 80), font=font)

grid.save(out_path)
print(f"saved grid to {out_path} ({grid.size[0]}x{grid.size[1]})")

"""Composite a 4x4 side-by-side grid: for each source/frame, show source-
skeleton and retargeted character side by side in one cell."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sbs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "sbs"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output" / "sbs_grid.png"

# Row spec: (label, folder, frames)
ROWS = [
    ("CMU 02_01 walk",    sbs_dir / "cmu",      ["0100", "0142", "0220", "0260"]),
    ("100STYLE Neutral",  sbs_dir / "100style", ["0100", "0142", "0220", "0260"]),
    ("AIST++ Breakdance", sbs_dir / "aistpp",   ["0100", "0150", "0250", "0400"]),
    ("CMU 56_01",         sbs_dir / "cmu2",     ["0100", "0150", "0200", "0260"]),
]

# Probe cell size from first existing tgt image.
def find_first_img():
    for _, folder, frames in ROWS:
        for f in frames:
            p = folder / f"tgt_{f}.png"
            if p.exists():
                return Image.open(p).size
    return (640, 960)

cw, ch = find_first_img()
label_h = 50
pair_pad = 4           # spacing between src and tgt in a cell
cell_w = cw * 2 + pair_pad
cell_h = ch

rows = len(ROWS)
cols = len(ROWS[0][2])
total_w = cell_w * cols
total_h = (ch + label_h) * rows + label_h

grid = Image.new("RGB", (total_w, total_h), (18, 20, 22))
draw = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("arial.ttf", 24)
except Exception:
    font = ImageFont.load_default()

# column headers
for c, frame in enumerate(ROWS[0][2]):
    draw.text((cell_w * c + 30, 15), f"frame {frame}", fill=(230, 230, 230), font=font)

for r, (label, folder, frames) in enumerate(ROWS):
    y0 = label_h + r * (ch + label_h)
    draw.text((18, y0 - label_h + 15), label, fill=(180, 210, 255), font=font)
    for c, frame in enumerate(frames):
        src_p = folder / f"src_{frame}.png"
        tgt_p = folder / f"tgt_{frame}.png"
        if src_p.exists():
            grid.paste(Image.open(src_p).convert("RGB"),
                       (c * cell_w, y0))
        if tgt_p.exists():
            grid.paste(Image.open(tgt_p).convert("RGB"),
                       (c * cell_w + cw + pair_pad, y0))
        # Divider line between src and tgt
        x_line = c * cell_w + cw + pair_pad // 2
        draw.line([(x_line, y0), (x_line, y0 + ch)], fill=(90, 90, 100), width=1)

grid.save(out_path)
print(f"saved {out_path} ({grid.size[0]}x{grid.size[1]})")

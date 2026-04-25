"""Composite LBS-vs-DQS close-ups into a single grid image.

Layout: each row is (joint, angle); each row has N columns of
[LBS | DQS] pairs — one pair per source.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
lbs_dqs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "lbs_dqs"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output" / "lbs_dqs_grid.png"

SOURCES = [("cmu", "CMU 02_01"),
           ("100style", "100STYLE Neutral"),
           ("aistpp", "AIST++ Breakdance")]

# Show the most diagnostic joints + angles (hip_L/R, knee_L, shoulder_L, elbow_L)
ROWS = [
    ("hip_L",      "side"),
    ("hip_L",      "front"),
    ("hip_R",      "side"),
    ("knee_L",     "side"),
    ("shoulder_L", "side"),
    ("shoulder_L", "front"),
    ("elbow_L",    "side"),
]

CELL = 384
label_w = 180
label_h = 40
pair_pad = 3
cell_w = CELL * 2 + pair_pad
col_sep = 30

src_count = len(SOURCES)
cols_per_row = src_count
total_w = label_w + (cell_w + col_sep) * cols_per_row
total_h = label_h * 2 + (CELL + label_h) * len(ROWS)

grid = Image.new("RGB", (total_w, total_h), (16, 18, 22))
draw = ImageDraw.Draw(grid)

try:
    font = ImageFont.truetype("arial.ttf", 22)
    bigfont = ImageFont.truetype("arial.ttf", 28)
    small = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()
    bigfont = ImageFont.load_default()
    small = ImageFont.load_default()

# Title strip
draw.text((20, 10), "LBS (linear blend)  vs  DQS (preserve volume)", fill=(230, 230, 230), font=bigfont)

# Per-source column headers
for ci, (_tag, label) in enumerate(SOURCES):
    x0 = label_w + ci * (cell_w + col_sep)
    draw.text((x0 + CELL - 25, label_h + 5), "LBS", fill=(255, 180, 160), font=font)
    draw.text((x0 + CELL + pair_pad + CELL - 25, label_h + 5), "DQS", fill=(160, 255, 180), font=font)
    draw.text((x0 + 5, label_h - 20), label, fill=(200, 210, 255), font=font)

y = 2 * label_h
for joint, angle in ROWS:
    draw.text((10, y + CELL // 2 - 12), f"{joint}  {angle}", fill=(230, 230, 230), font=font)
    for ci, (tag, _label) in enumerate(SOURCES):
        x0 = label_w + ci * (cell_w + col_sep)
        lbs_p = lbs_dqs_dir / tag / f"{tag}_{joint}_lbs_{angle}.png"
        dqs_p = lbs_dqs_dir / tag / f"{tag}_{joint}_dqs_{angle}.png"
        if lbs_p.exists():
            grid.paste(Image.open(lbs_p).convert("RGB"), (x0, y))
        else:
            draw.rectangle([(x0, y), (x0 + CELL, y + CELL)], fill=(50, 20, 20))
        if dqs_p.exists():
            grid.paste(Image.open(dqs_p).convert("RGB"), (x0 + CELL + pair_pad, y))
        else:
            draw.rectangle([(x0 + CELL + pair_pad, y),
                             (x0 + 2 * CELL + pair_pad, y + CELL)],
                            fill=(50, 20, 20))
        draw.line([(x0 + CELL + pair_pad // 2, y),
                    (x0 + CELL + pair_pad // 2, y + CELL)],
                   fill=(90, 90, 100), width=1)
    y += CELL + label_h

grid.save(out_path)
print(f"saved {out_path} ({grid.size[0]}x{grid.size[1]})")

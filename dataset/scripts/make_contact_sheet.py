"""Compose N stills into a single horizontal contact sheet so the full
motion arc is visible in one image."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
stills_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "output" / "sequence_output" / "stills"
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output" / "sequence_output" / "contact_sheet.png"

pngs = sorted(stills_dir.glob("still_*.png"))
if not pngs:
    print("no stills found")
    sys.exit(1)

imgs = [Image.open(p).convert("RGB") for p in pngs]
w, h = imgs[0].size
label_h = 36

total = Image.new("RGB", (w * len(imgs), h + label_h), (18, 20, 22))
draw = ImageDraw.Draw(total)

try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()

for i, img in enumerate(imgs):
    total.paste(img, (i * w, label_h))
    draw.text((i * w + 10, 8), f"f{i*10:02d}", fill=(220, 220, 230), font=font)
    # thin vertical divider
    if i > 0:
        draw.line([(i * w, 0), (i * w, h + label_h)], fill=(60, 60, 65), width=1)

total.save(out_path)
print(f"contact sheet saved: {out_path}  ({total.size[0]}x{total.size[1]})")
